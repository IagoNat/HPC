"""
Benchmark minimo do Nivel 1 (ver plano de desenvolvimento).

Compara tempo de execucao e memoria de pico das tres versoes descritas na
Secao 1 da proposta, em funcao do comprimento da sequencia N:

    1. kernel ingenuo (materializa a matriz N x N)       -> naive_attention
    2. kernel com tiling + softmax online (este projeto) -> flash_attention_tiled
    3. implementacao nativa do PyTorch (referencia superior de desempenho)
       -> torch.nn.functional.scaled_dot_product_attention

Medicao de tempo: eventos CUDA (torch.cuda.Event), nao profiler de
contadores de hardware (Nsight Compute / nvprof tipicamente nao funcionam
em GPUs compartilhadas na nuvem - ver item "d" da proposta e a celula de
confirmacao do ERR_NVGPUCTRPERM no notebook do Nivel 0).

Medicao de memoria: torch.cuda.max_memory_allocated(), resetado antes de
cada medicao com torch.cuda.reset_peak_memory_stats().

Uso (dentro de um runtime com GPU, ex. Colab):
    python python/benchmark.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from python.naive_attention import naive_attention
from python.flash_attention_tiled import flash_attention_tiled

# Comprimentos de sequencia avaliados. Mantidos pequenos por padrao (o
# kernel ingenuo materializa N x N em FP32: N=4096 ja sao 4 * 4096^2 * 4
# bytes = 256 MB so para os scores de UMA (batch, cabeca) - em T4 (16GB)
# cabe, mas sobe rapido; ajuste livremente para o relatorio final).
SEQ_LENS = [128, 512, 1024, 2048]
B, H, D = 1, 8, 64
WARMUP_ITERS = 3
TIMED_ITERS = 10

IMPLEMENTATIONS = {
    "ingenuo": naive_attention,
    "tiled (este projeto)": flash_attention_tiled,
    "pytorch nativo (sdpa)": lambda Q, K, V: F.scaled_dot_product_attention(Q, K, V),
}


def _time_and_memory(fn, Q, K, V):
    for _ in range(WARMUP_ITERS):
        fn(Q, K, V)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(TIMED_ITERS):
        fn(Q, K, V)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / TIMED_ITERS
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return elapsed_ms, peak_mb


def run_benchmark(seq_lens=SEQ_LENS):
    if not torch.cuda.is_available():
        print("Nenhuma GPU CUDA disponivel - rode isto no Colab (Ambiente de execucao > Alterar tipo de ambiente de execucao > GPU).")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}  B={B} H={H} D={D}\n")
    header = f"{'N':>6} | {'versao':<22} | {'tempo (ms)':>12} | {'memoria de pico (MB)':>20}"
    print(header)
    print("-" * len(header))

    results = []
    for N in seq_lens:
        torch.manual_seed(0)
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

        for name, fn in IMPLEMENTATIONS.items():
            try:
                elapsed_ms, peak_mb = _time_and_memory(fn, Q, K, V)
                print(f"{N:>6} | {name:<22} | {elapsed_ms:>12.3f} | {peak_mb:>20.1f}")
                results.append((N, name, elapsed_ms, peak_mb))
            except RuntimeError as e:
                print(f"{N:>6} | {name:<22} | {'ERRO':>12} | {str(e)[:40]}")
    return results


if __name__ == "__main__":
    run_benchmark()
