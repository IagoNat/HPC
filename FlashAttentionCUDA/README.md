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

## Nível 0

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

## Nível 1 (atual)

- `kernels/flash_attention_tiled.cu` — kernel com tiling (`BLOCK_N = 32`)
  e softmax online (log-sum-exp incremental): nunca materializa a matriz
  N×N, nem em memória global nem em memória compartilhada (só um tile de
  scores por vez). Mesma paralelização do kernel ingênuo (um bloco de
  threads por linha de consulta), para manter a comparação de desempenho
  justa — o ganho vem de menos tráfego de memória, não de mais paralelismo.
- `python/flash_attention_tiled.py` — wrapper que compila e expõe o kernel.
- `tests/test_tiled_correctness.py` — compara o kernel tiled contra a
  referência *e* contra o kernel ingênuo, incluindo casos com N menor,
  igual e maior que `BLOCK_N` (exercita o caso de 1 tile e o de múltiplos
  tiles, inclusive último tile parcial).
- `python/benchmark.py` — benchmark mínimo: tempo de execução (eventos
  CUDA) e memória de pico (`torch.cuda.max_memory_allocated`) das três
  versões da proposta (ingênuo, tiled, `scaled_dot_product_attention`
  nativo do PyTorch), em função de N.
- `notebooks/01_tiled_flash_attention.ipynb` — compila a extensão tiled,
  roda o harness de corretude e o benchmark mínimo.
- Ajuste fino de `BLOCK_N` / ocupação da GPU (item "c" da proposta) fica
  para o Nível 2.

## Como rodar (Colab)

Abrir `notebooks/00_setup_and_naive.ipynb` primeiro (Nível 0) e depois
`notebooks/01_tiled_flash_attention.ipynb` (Nível 1) no Colab, garantir
GPU em *Ambiente de execução > Alterar tipo de ambiente de execução*, e
rodar as células em ordem.
