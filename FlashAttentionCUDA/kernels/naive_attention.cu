// naive_attention.cu
//
// Kernel CUDA "ingenuo" (Nivel 0 / Nivel 1 - versao base de comparacao).
//
// Calcula a atencao padrao, forward-only, nao causal:
//     output = softmax(Q K^T / sqrt(D)) V
//
// De proposito, materializa a matriz de scores N x N inteira na memoria
// global da GPU (um tensor "scores" de tamanho [B, H, N, N]) em vez de
// processar os blocos de K/V de forma incremental. Essa e exatamente a
// limitacao discutida na proposta (item "a" das dificuldades de
// paralelizacao): o gargalo aqui e volume de leituras/escritas na memoria
// global, nao o numero de operacoes de ponto flutuante.
//
// Convencao de forma dos tensores: [B, H, N, D]
//   B = tamanho do batch
//   H = numero de cabecas de atencao
//   N = comprimento da sequencia
//   D = dimensao por cabeca (head_dim)

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

// Kernel 1: scores[b,h,i,j] = (Q[b,h,i,:] . K[b,h,j,:]) / sqrt(D)
//
// Um bloco de threads por linha de consulta (b,h,i); cada thread calcula
// um subconjunto das colunas j via stride loop. O produto escalar em si
// e um loop sequencial sobre D (sem uso de memoria compartilhada - essa
// otimizacao fica para o kernel com tiling, no Nivel 1).
__global__ void compute_scores_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ scores,
    int B, int H, int N, int D,
    float scale)
{
    int row = blockIdx.x;              // 0 .. B*H*N - 1
    int b = row / (H * N);
    int h = (row / N) % H;
    int i = row % N;

    const float* q_row = Q + ((long)(b * H + h) * N + i) * D;
    const float* k_base = K + (long)(b * H + h) * N * D;
    float* scores_row = scores + (long)row * N;

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        const float* k_row = k_base + (long)j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += q_row[d] * k_row[d];
        }
        scores_row[j] = dot * scale;
    }
}

// Kernel 2: softmax numericamente estavel, em cada linha de "scores".
//
// Tres passagens por linha (maximo, soma dos exponenciais, normalizacao),
// cada uma com uma reducao em memoria compartilhada. Essa e a mesma
// matematica do softmax online do Nivel 1 - a diferenca e que aqui as
// tres passagens leem/escrevem a linha inteira, ja materializada em
// memoria global, em vez de processar blocos de K/V incrementalmente.
__global__ void softmax_kernel(
    float* __restrict__ scores,
    int N)
{
    extern __shared__ float shared[];
    int row = blockIdx.x;
    float* scores_row = scores + (long)row * N;

    // passo 1: maximo da linha
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        local_max = fmaxf(local_max, scores_row[j]);
    }
    shared[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float row_max = shared[0];
    __syncthreads();

    // passo 2: soma dos exponenciais (e guarda e^(x - max) no lugar)
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float e = __expf(scores_row[j] - row_max);
        scores_row[j] = e;
        local_sum += e;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float row_sum = shared[0];
    __syncthreads();

    // passo 3: normalizar
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        scores_row[j] /= row_sum;
    }
}

// Kernel 3: output[b,h,i,d] = soma_j attn[b,h,i,j] * V[b,h,j,d]
__global__ void compute_output_kernel(
    const float* __restrict__ attn,
    const float* __restrict__ V,
    float* __restrict__ output,
    int B, int H, int N, int D)
{
    int row = blockIdx.x; // 0 .. B*H*N - 1
    int b = row / (H * N);
    int h = (row / N) % H;

    const float* attn_row = attn + (long)row * N;
    const float* v_base = V + (long)(b * H + h) * N * D;
    float* out_row = output + (long)row * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) {
            acc += attn_row[j] * v_base[(long)j * D + d];
        }
        out_row[d] = acc;
    }
}

torch::Tensor naive_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K e V precisam estar na GPU");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "este kernel ingenuo so suporta FP32");
    TORCH_CHECK(Q.dim() == 4, "esperado tensor com forma [B, H, N, D]");

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);
    float scale = 1.0f / sqrtf((float)D);

    auto scores = torch::empty({B, H, N, N}, Q.options());
    auto output = torch::empty({B, H, N, D}, Q.options());

    int num_rows = B * H * N;
    int threads = THREADS_PER_BLOCK;

    compute_scores_kernel<<<num_rows, threads>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), scores.data_ptr<float>(),
        B, H, N, D, scale);

    softmax_kernel<<<num_rows, threads, threads * sizeof(float)>>>(
        scores.data_ptr<float>(), N);

    compute_output_kernel<<<num_rows, threads>>>(
        scores.data_ptr<float>(), V.data_ptr<float>(), output.data_ptr<float>(),
        B, H, N, D);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &naive_attention_forward,
          "Atencao ingenua (materializa a matriz N x N) - forward only, nao causal");
}
