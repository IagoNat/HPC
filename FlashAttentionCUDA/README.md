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
python/      wrappers Python das extensões + implementação de referência
tests/       harness de corretude (tolerância numérica contra a referência)
notebooks/   notebooks Colab — ponto de entrada para rodar tudo com GPU
```

## Nível 0 (atual)

- `kernels/naive_attention.cu` — kernel ingênuo: materializa a matriz de
  scores N×N inteira, três passes (scores → softmax → output).
- `python/reference.py` — referência em PyTorch puro (float64 interno),
  usada como fonte da verdade para os testes de corretude.
- `python/naive_attention.py` — wrapper que compila e expõe o kernel.
- `tests/test_correctness.py` — compara o kernel ingênuo contra a
  referência por tolerância numérica (rtol=1e-3, atol=1e-4), não por
  igualdade bit a bit.
- `notebooks/00_setup_and_naive.ipynb` — notebook Colab: compila a
  extensão, roda o harness de corretude, confirma a limitação de
  profiling (`ERR_NVGPUCTRPERM`) e valida eventos CUDA como alternativa
  de medição.

## Como rodar (Colab)

Abrir `notebooks/00_setup_and_naive.ipynb` no Colab, garantir GPU em
*Ambiente de execução > Alterar tipo de ambiente de execução*, e rodar as
células em ordem.
