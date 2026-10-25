"""
Wrapper Python para o kernel CUDA com tiling + softmax online
(Nivel 1 - kernels/flash_attention_tiled.cu).

Mesma logica de compilacao sob demanda do wrapper do kernel ingenuo
(python/naive_attention.py).
"""
import os
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_SRC = os.path.join(_THIS_DIR, "..", "kernels", "flash_attention_tiled.cu")

_ext = None


def _get_extension():
    global _ext
    if _ext is None:
        _ext = load(
            name="flash_attention_tiled_cuda",
            sources=[_KERNEL_SRC],
            verbose=True,
        )
    return _ext


def flash_attention_tiled(Q: "torch.Tensor", K: "torch.Tensor", V: "torch.Tensor"):
    """
    Q, K, V: tensores CUDA [B, H, N, D], float32.
    Retorna: tensor CUDA [B, H, N, D], float32.

    Matematicamente identico a naive_attention() - a diferenca e so de
    implementacao: aqui a matriz de scores N x N nunca e materializada.
    """
    ext = _get_extension()
    return ext.forward(Q, K, V)
