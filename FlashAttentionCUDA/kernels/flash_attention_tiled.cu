// flash_attention_tiled.cu
//
// Kernel CUDA com tiling + softmax online (Nivel 1 - contribuicao central
// do projeto). Calcula exatamente a mesma funcao que naive_attention.cu
// (forward-only, nao causal, FP32):
//
//     output = softmax(Q K^T / sqrt(D)) V
//
// mas SEM materializar a matriz de scores N x N em memoria global. Em vez
// disso, Q, K e V sao processados em blocos (tiles) que cabem na memoria
// compartilhada do SM, e a normalizacao do softmax e mantida de forma
// incremental (online softmax / log-sum-exp), com reescala das saidas
// parciais a cada novo bloco de K/V processado (ver item "b" das
// dificuldades de paralelizacao na proposta).
//
// Convencao de forma dos tensores: [B, H, N, D] (igual ao kernel ingenuo).
//
// Paralelizacao: um bloco de threads por linha de consulta (b,h,i) - igual
// ao kernel ingenuo, para manter a comparacao de desempenho justa (mesma
// quantidade de work-items). O ganho aqui nao vem de mais paralelismo, e
// sim de trafego de memoria global muito menor: cada elemento de K e V e
// lido uma vez por bloco de K/V (nao uma vez por elemento da matriz N x N),
// e a matriz de scores nunca existe inteira, nem em memoria global nem em
// memoria compartilhada (so um tile de BLOCK_N scores por vez).
//
// Observacao de escopo: dentro de cada tile, a soma sobre D (produto
// escalar) e a soma sobre j (acumulacao da saida) ainda sao loops
// sequenciais simples - essa e a mesma simplificacao proposital do kernel
// ingenuo. O ajuste fino de BLOCK_N / numero de threads para melhor
// ocupacao fica para o Nivel 2 (tile-size tuning), conforme o plano de
// desenvolvimento.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 128
// Tamanho do tile de K/V (numero de colunas j processadas por vez).
// Mantido pequeno de proposito para a memoria compartilhada caber
// folgadamente no limite de 48KB por bloco mesmo com D maior (ate 128),
// sem precisar de opt-in de memoria compartilhada dinamica maior.
#define BLOCK_N 32

__global__ void flash_attention_tiled_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D,
    float scale)
{
    // Layout da memoria compartilhada dinamica (um unico bloco contiguo):
    //   q_sh   [D]             - linha de Q desta thread-block (carregada 1x)
    //   k_tile [BLOCK_N * D]   - tile de K atual
    //   v_tile [BLOCK_N * D]   - tile de V atual
    //   s_tile [BLOCK_N]       - scores do tile; reaproveitado para guardar
    //                            os pesos exp(score - max) depois da 1a fase
    //   red    [blockDim.x]    - buffer de reducao (maximo e soma)
    //   acc    [D]             - acumulador de saida NAO normalizado
    extern __shared__ float smem[];
    float* q_sh   = smem;
    float* k_tile = q_sh + D;
    float* v_tile = k_tile + (size_t)BLOCK_N * D;
    float* s_tile = v_tile + (size_t)BLOCK_N * D;
    float* red    = s_tile + BLOCK_N;
    float* acc    = red + blockDim.x;

    // Estado escalar do softmax online, compartilhado por todo o bloco.
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float new_max;
    __shared__ float alpha;
    __shared__ float tile_sum;

    int row = blockIdx.x;              // 0 .. B*H*N - 1
    int b = row / (H * N);
    int h = (row / N) % H;
    int i = row % N;

    const float* q_row = Q + ((long)(b * H + h) * N + i) * D;
    const float* k_base = K + (long)(b * H + h) * N * D;
    const float* v_base = V + (long)(b * H + h) * N * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        q_sh[d] = q_row[d];
        acc[d] = 0.0f;
    }
    if (threadIdx.x == 0) {
        running_max = -INFINITY;
        running_sum = 0.0f;
    }
    __syncthreads();

    int num_tiles = (N + BLOCK_N - 1) / BLOCK_N;

    for (int t = 0; t < num_tiles; t++) {
        int tile_start = t * BLOCK_N;
        int tile_len = min(BLOCK_N, N - tile_start);

        // Carrega K_tile e V_tile cooperativamente (cada thread carrega
        // varios elementos via stride loop sobre o bloco "achatado").
        for (int idx = threadIdx.x; idx < tile_len * D; idx += blockDim.x) {
            int jj = idx / D;
            int dd = idx % D;
            k_tile[idx] = k_base[(long)(tile_start + jj) * D + dd];
            v_tile[idx] = v_base[(long)(tile_start + jj) * D + dd];
        }
        __syncthreads();

        // Fase A: scores do tile, S[j] = (Q_i . K_tile[j]) * scale,
        // e maximo local (preparacao para o softmax online).
        float local_max = -INFINITY;
        for (int j = threadIdx.x; j < tile_len; j += blockDim.x) {
            const float* k_row = k_tile + (long)j * D;
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q_sh[d] * k_row[d];
            }
            float sc = dot * scale;
            s_tile[j] = sc;
            local_max = fmaxf(local_max, sc);
        }
        red[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float tile_max = red[0];
            // running_max comeca em -inf: na 1a iteracao alpha = exp(-inf) = 0,
            // o que e coerente com acc/running_sum ainda zerados.
            new_max = fmaxf(running_max, tile_max);
            alpha = __expf(running_max - new_max);
        }
        __syncthreads();

        // Fase B: p[j] = exp(S[j] - new_max) (reescala para o novo maximo);
        // soma local dos pesos do tile. Reaproveita s_tile[] para os pesos.
        float local_sum = 0.0f;
        for (int j = threadIdx.x; j < tile_len; j += blockDim.x) {
            float p = __expf(s_tile[j] - new_max);
            s_tile[j] = p;
            local_sum += p;
        }
        red[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            tile_sum = red[0];
            running_sum = running_sum * alpha + tile_sum;
            running_max = new_max;
        }
        __syncthreads();

        // Reescala o acumulador de saida pelo fator do novo maximo e soma
        // a contribuicao deste tile: acc[d] += soma_j p[j] * V_tile[j, d].
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float rescaled = acc[d] * alpha;
            float sum_pv = 0.0f;
            for (int j = 0; j < tile_len; j++) {
                sum_pv += s_tile[j] * v_tile[(long)j * D + d];
            }
            acc[d] = rescaled + sum_pv;
        }
        // sincroniza antes do proximo tile sobrescrever k_tile/v_tile/s_tile
        __syncthreads();
    }

    // Normalizacao final: output[i,d] = acc[d] / running_sum.
    float* out_row = O + (long)row * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out_row[d] = acc[d] / running_sum;
    }
}

torch::Tensor flash_attention_tiled_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K e V precisam estar na GPU");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "este kernel so suporta FP32");
    TORCH_CHECK(Q.dim() == 4, "esperado tensor com forma [B, H, N, D]");

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);
    float scale = 1.0f / sqrtf((float)D);

    auto output = torch::empty({B, H, N, D}, Q.options());

    int num_rows = B * H * N;
    int threads = THREADS_PER_BLOCK;

    size_t smem_bytes = (size_t)(2 * D + 2 * BLOCK_N * D + BLOCK_N + threads) * sizeof(float);
    const size_t SMEM_LIMIT = 48 * 1024; // limite conservador, sem opt-in de smem dinamica maior
    TORCH_CHECK(smem_bytes <= SMEM_LIMIT,
        "memoria compartilhada necessaria (", smem_bytes,
        " bytes) excede o limite de 48KB para D=", D,
        " com BLOCK_N=", BLOCK_N,
        ". Reduza BLOCK_N no kernel (flash_attention_tiled.cu) para este D.");

    flash_attention_tiled_kernel<<<num_rows, threads, smem_bytes>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(), B, H, N, D, scale);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_tiled_forward,
          "FlashAttention (tiling + softmax online) - forward only, nao causal, FP32");
}
