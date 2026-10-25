# FlashAttentionCUDA

Projeto final de Programação e Arquitetura de Computadores Paralelos
(DCC030/049) — reimplementação, em CUDA e a partir do zero, da ideia
algorítmica central do FlashAttention (Dao et al., NeurIPS 2022).

## Ambiente

Sem GPU local — desenvolvimento e testes rodam no Google Colab (tier
grátis, GPU T4/Turing).

## Estrutura

```
kernels/     kernels CUDA (.cu), expostos ao Python via torch.utils.cpp_extension
python/      wrappers Python das extensões + referência + benchmark
tests/       harness de corretude (tolerância numérica contra a referência)
notebooks/   notebooks Colab — ponto de entrada para rodar tudo com GPU
```

## Nível 0

- `kernels/naive_attention.cu` — kernel ingênuo: materializa a matriz N×N.
- `python/reference.py`, `python/naive_attention.py`, `tests/test_correctness.py`.
- `notebooks/00_setup_and_naive.ipynb`.
- **Status:** validado no Colab (T4) — harness passou em 4/4 casos, erro máximo ~1e-7. Nsight Compute funcionou (ao contrário do previsto na proposta) e confirmou o kernel como memory-bound.

## Nível 1 (atual)

- `kernels/flash_attention_tiled.cu` — tiling (`BLOCK_N=32`) + softmax online, nunca materializa a matriz N×N. Testamos `BLOCK_N=64` também (hipótese: melhor ocupação de threads dentro do bloco) — mas o benchmark mostrou que ficou mais lento em todos os N, provavelmente porque tiles maiores usam mais memória compartilhada e isso reduz quantos blocos cabem residentes por SM ao mesmo tempo, o que importa mais aqui do que ocupação interna ao bloco. Voltamos para 32.
- `python/flash_attention_tiled.py`, `tests/test_tiled_correctness.py`, `python/benchmark.py`.
- `notebooks/01_tiled_flash_attention.ipynb`.
- Ajuste fino de `BLOCK_N` fica para o Nível 2.

## Como rodar (Colab)

Abrir `00_setup_and_naive.ipynb` e depois `01_tiled_flash_attention.ipynb`, GPU em *Ambiente de execução > Alterar tipo*, rodar células em ordem.
