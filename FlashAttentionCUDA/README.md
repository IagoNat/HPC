# FlashAttentionCUDA

Projeto final de Programação e Arquitetura de Computadores Paralelos
(DCC030/049) — reimplementação, em CUDA e a partir do zero, da ideia
algorítmica central do FlashAttention (Dao et al., NeurIPS 2022): um
kernel com tiling e softmax online para o mecanismo de atenção, evitando
materializar a matriz de scores N×N inteira na memória global da GPU.

Proposta completa: ver `proposta_projeto_final_PACP.pdf` / `.tex` (fora
deste diretório). Plano de desenvolvimento por níveis: documento do
projeto no Claude ("Plano de Desenvolvimento — FlashAttention CUDA").

## Ambiente

Sem GPU local — todo o desenvolvimento e todos os testes rodam no Google
Colab (tier grátis, GPU T4/Turing). Não há máquina com GPU fornecida pela
disciplina.

## Estrutura

```
kernels/     kernels CUDA (.cu), expostos ao Python via torch.utils.cpp_extension
python/      wrappers Python das extensões + implementação de referência + benchmark
tests/       harness de corretude (tolerância numérica contra a referência)
notebooks/   notebooks Colab — ponto de entrada para rodar tudo com GPU
```