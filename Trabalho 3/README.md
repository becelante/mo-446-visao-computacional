# Trabalho 3: detecção de copy-move em imagens biomédicas

Dados da competição [Recod.ai/LUC](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection).

```bash
uv sync                       # ambiente (PyTorch com CUDA)
./download_data.sh            # baixa e extrai o dataset em data/ (precisa de ~/.kaggle/kaggle.json)
# Etapa 1: abrir analise_exploratoria/analise_exploratoria.ipynb (kernel do .venv); figuras em results/eda/
uv run python tests_metrics.py  # testa a metrica oficial e as complementares
uv run python evaluate.py predicoes_teste.csv solucao_teste.csv   # S, deteccao, localizacao e baseline

```

## Métrica

[pipeline/recodai_f1.py](pipeline/recodai_f1.py) é a cópia **sem modificações** da métrica oficial
([kaggle.com/code/metric/recodai-f1](https://www.kaggle.com/code/metric/recodai-f1), versão 6), com `rle_encode`,
`rle_decode` e `score`. [pipeline/metrics.py](pipeline/metrics.py) calcula em volta dela o que o relatório pede:
S, baseline `authentic`, detecção no nível da imagem (acurácia, precisão, revocação, F1, matriz de confusão) e
oF1 médio nas forjadas detectadas. `evaluate.py` lê um CSV de predições (`case_id,annotation`) e um de
solução (`case_id,annotation,shape`, com `shape = "[H, W]"`).
