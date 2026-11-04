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

Coloque os vídeos `.MTS` do [Drive](https://drive.google.com/drive/folders/1AJ1TIS4Ky__nHI-AqC2CHUOg5BjUT_gt) em `videos/`, escolha um em `video` no [config.yaml](Trabalho%202/config.yaml) (ou rode `VIDEO=A ./run.sh`) e rode (ou use `./run.sh`, que roda tudo):

```bash
uv sync
uv run python run_pipeline.py   # Etapas 1 a 5, ablação, vídeo de navegação e GLB -> results/<vídeo>/
uv run python visualize.py      # janela 3D: nuvem, câmeras e trajetória
uv run python extras.py         # opcional: ângulo entre paredes, casos difíceis, PatchMatch x SGBM, malha
```
