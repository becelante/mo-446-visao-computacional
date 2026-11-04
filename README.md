# MO446 - Visão Computacional

## Trabalho 1: Panoramas

Coloque os `.ARW` do [Drive](https://drive.google.com/drive/folders/1qkxJ3tsieq7-eGgap9bC-92S4rU1gImP?usp=sharing) em `D1/`, `A1/`, `B1/` e `C1/` e rode:

```bash
uv sync
uv run python 1_revelar_raw.py
uv run python 2_panorama.py
uv run python 3_extras.py
uv run python 4_comparacao_x4.py
```

## Trabalho 2: Reconstrução 3D de ambiente interno

Coloque o vídeo `.MTS` do [Drive](https://drive.google.com/drive/folders/1AJ1TIS4Ky__nHI-AqC2CHUOg5BjUT_gt) em `videos/`, escolha-o em `VIDEO` no [src/config.py](Trabalho%202/src/config.py) e rode (ou use `./run.sh`, que roda tudo):

```bash
uv sync
uv run python 1_extrair_quadros.py
uv run python 2_comparar_detectores.py
uv run python 3_reconstrucao.py --detector SIFT --densa
uv run python 3_reconstrucao.py --detector ORB
uv run python 4_ablacao.py --detector SIFT
uv run python visualizar.py
```
