"""
Implementacao de referencia (PyTorch puro) da atencao padrao:

    softmax(Q K^T / sqrt(D)) V

Usada como "fonte da verdade" para os testes de corretude dos kernels CUDA
deste projeto (ver tests/test_correctness.py). O calculo e feito
internamente em float64 e convertido de volta para o dtype de entrada, de
forma que o proprio erro de arredondamento da referencia nao contamine a
comparacao por tolerancia numerica contra os kernels em FP32.

Esta NAO e a implementacao "PyTorch nativo" usada como referencia de
desempenho na Secao 1 da proposta (essa e
torch.nn.functional.scaled_dot_product_attention, otimizada); esta aqui e
so para checar corretude.
"""
import torch


def reference_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Q, K, V: tensores [B, H, N, D], mesmo dtype e device entre si.
    Retorna: tensor [B, H, N, D], no mesmo dtype de entrada.
    """
    orig_dtype = Q.dtype
    Qd = Q.double()
    Kd = K.double()
    Vd = V.double()

    D = Qd.shape[-1]
    scale = 1.0 / (D ** 0.5)

    scores = torch.matmul(Qd, Kd.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, Vd)

    return out.to(orig_dtype)
