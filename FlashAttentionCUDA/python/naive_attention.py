"""
Wrapper Python para o kernel CUDA ingenuo (Nivel 0 / kernels/naive_attention.cu).

Compila a extensao sob demanda com torch.utils.cpp_extension.load(): isso
evita um passo de build separado e funciona bem num notebook Colab, onde
o ambiente e recriado a cada sessao (sem cache de build persistente entre
sessoes, a nao ser que o diretorio de cache do PyTorch seja preservado).
"""
import os
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_SRC = os.path.join(_THIS_DIR, "..", "kernels", "naive_attention.cu")

_ext = None


def _get_extension():
    global _ext
    if _ext is None:
        _ext = load(
            name="naive_attention_cuda",
            sources=[_KERNEL_SRC],
            verbose=True,
        )
    return _ext


def naive_attention(Q: "torch.Tensor", K: "torch.Tensor", V: "torch.Tensor"):
    """
    Q, K, V: tensores CUDA [B, H, N, D], float32.
    Retorna: tensor CUDA [B, H, N, D], float32.
    """
    ext = _get_extension()
    return ext.forward(Q, K, V)
