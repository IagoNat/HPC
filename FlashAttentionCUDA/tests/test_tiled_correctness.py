"""
Harness de corretude do Nivel 1.

Compara o kernel com tiling + softmax online (CUDA) com a mesma
implementacao de referencia usada no Nivel 0 (python/reference.py), por
tolerancia numerica - nunca por igualdade bit a bit (a ordem das somas de
ponto flutuante difere entre as tres implementacoes: ingenua, tiled e
referencia).

Tambem compara o kernel tiled diretamente contra o kernel ingenuo
(mesma GPU, mesma chamada), como checagem cruzada adicional: os dois
devem concordar dentro da mesma tolerancia, ja que calculam exatamente a
mesma funcao matematica por caminhos de implementacao diferentes.

Uso (dentro de um runtime com GPU, ex. Colab):
    python tests/test_tiled_correctness.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from python.reference import reference_attention
from python.naive_attention import naive_attention
from python.flash_attention_tiled import flash_attention_tiled

# Inclui N maiores que BLOCK_N (32) do kernel tiled, para exercitar o loop
# de multiplos tiles (e nao so o caso degenerado de 1 tile so).
TEST_CASES = [
    (1, 1, 16, 32),    # N < BLOCK_N: 1 tile parcial
    (1, 1, 32, 32),    # N == BLOCK_N: exatamente 1 tile
    (1, 1, 65, 32),    # N > BLOCK_N, nao multiplo: ultimo tile parcial
    (2, 2, 128, 64),
    (2, 4, 256, 64),
]

RTOL = 1e-3
ATOL = 1e-4


def run_case(B, H, N, D, device="cuda"):
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    expected = reference_attention(Q, K, V)
    actual_tiled = flash_attention_tiled(Q, K, V)
    actual_naive = naive_attention(Q, K, V)

    err_ref = (actual_tiled - expected).abs().max().item()
    err_naive = (actual_tiled - actual_naive).abs().max().item()

    ok_ref = torch.allclose(actual_tiled, expected, rtol=RTOL, atol=ATOL)
    ok_naive = torch.allclose(actual_tiled, actual_naive, rtol=RTOL, atol=ATOL)
    ok = ok_ref and ok_naive

    status = "OK  " if ok else "FAIL"
    print(f"[{status}] B={B} H={H} N={N} D={D}  "
          f"erro_vs_referencia={err_ref:.3e}  erro_vs_ingenuo={err_naive:.3e}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("Nenhuma GPU CUDA disponivel - rode isto no Colab (Ambiente de execucao > Alterar tipo de ambiente de execucao > GPU).")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results = [run_case(*case) for case in TEST_CASES]

    if all(results):
        print(f"\nTodos os {len(results)} casos passaram (rtol={RTOL}, atol={ATOL}).")
    else:
        n_fail = results.count(False)
        print(f"\n{n_fail} de {len(results)} casos falharam.")
        sys.exit(1)


if __name__ == "__main__":
    main()
