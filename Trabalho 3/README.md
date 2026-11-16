# Trabalho 3: detecção de copy-move em imagens biomédicas

Dados da competição [Recod.ai/LUC](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection).

```bash
uv sync                       # ambiente (PyTorch com CUDA)
./download_data.sh            # baixa e extrai o dataset em data/ (precisa de ~/.kaggle/kaggle.json)
# Etapa 1: abrir analise_exploratoria/analise_exploratoria.ipynb (kernel do .venv); figuras em results/eda/
uv run python make_split.py     # particoes 70/15/15 por case_id -> splits/{train,val,test}.csv
uv run python tests_metrics.py  # testa a metrica oficial e as complementares
uv run python evaluate.py predicoes_teste.csv test   # S, deteccao, localizacao e baseline
./train.sh unet_base             # treina a U-Net (results/unet/unet_base/checkpoints/melhor.ckpt)
./predict.sh unet_base val       # escolhe os limiares na validacao e avalia
uv run tensorboard --logdir results/unet   # acompanhar o treino em http://localhost:6006

```

## Métrica

[pipeline/recodai_f1.py](pipeline/recodai_f1.py) é a cópia **sem modificações** da métrica oficial
([kaggle.com/code/metric/recodai-f1](https://www.kaggle.com/code/metric/recodai-f1), versão 6), com `rle_encode`,
`rle_decode` e `score`. [pipeline/metrics.py](pipeline/metrics.py) calcula em volta dela o que o relatório pede:
S, baseline `authentic`, detecção no nível da imagem (acurácia, precisão, revocação, F1, matriz de confusão) e
oF1 médio nas forjadas detectadas. `evaluate.py` lê um CSV de predições (`case_id,annotation`) e o nome da
partição (`train`, `val` ou `test`). A verdade é montada a partir das máscaras originais, e o `case_id` das
predições é o `id` da imagem em `splits/<partição>.csv`, porque um mesmo `case_id` pode ter até três imagens.

## Rede de segmentação

- [pipeline/dataset.py](pipeline/dataset.py): cada imagem tem o lado maior reduzido ou ampliado para `unet.size` px e
  é completada com zeros até ficar quadrada. Assim a imagem inteira entra na rede e o par origem/cópia nunca é cortado.
  O alvo é a união das instâncias. As versões reduzidas ficam em `results/cache/<size>/` (cerca de 10 GB para 768 px).
- [pipeline/segmenter.py](pipeline/segmenter.py): U-Net com encoder pré-treinado no ImageNet, em PyTorch Lightning,
  com perda BCE + Dice. A cada época a validação é avaliada com a métrica oficial, e o melhor checkpoint é o de maior S.
  A predição usa p = maior probabilidade da imagem. Se p < `p_thr`, a imagem é `authentic`. Senão, os pixels com
  probabilidade > `pix_thr` formam uma única instância.
- `predict.py` escolhe `pix_thr`, `p_thr` e `min_area` na validação (grade em `config.yaml`) e aplica no teste.
