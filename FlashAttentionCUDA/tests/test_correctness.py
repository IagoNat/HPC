"""
Harness de corretude do Nivel 0.

Compara o kernel ingenuo (CUDA) com a implementacao de referencia
(PyTorch puro, float64 internamente) por TOLERANCIA numerica - nunca por
igualdade bit a bit, ja que a ordem das somas de ponto flutuante difere
entre as duas implementacoes (ver item "b" das dificuldades de
paralelizacao na proposta).

Uso (dentro de um runtime com GPU, ex. Colab):
    python tests/test_correctness.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from python.reference import reference_attention
from python.naive_attention import naive_attention

# (B, H, N, D) - tamanhos pequenos de proposito: o objetivo aqui e
# corretude, nao desempenho (isso fica para o benchmark do Nivel 1/2).
TEST_CASES = [
    (1, 1, 16, 32),
    (1, 1, 64, 32),
    (2, 2, 32, 64),
    (2, 4, 128, 64),
]

RTOL = 1e-3
ATOL = 1e-4


def run_case(B, H, N, D, device="cuda"):
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    K = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    V = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    expected = reference_attention(Q, K, V)
    actual = naive_attention(Q, K, V)

    max_abs_err = (actual - expected).abs().max().item()
    ok = torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)

    status = "OK  " if ok else "FAIL"
    print(f"[{status}] B={B} H={H} N={N} D={D}  erro_abs_max={max_abs_err:.3e}")
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
