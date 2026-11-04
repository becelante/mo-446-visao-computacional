#!/usr/bin/env bash
# Pipeline completo da Variante A. Coloque o video (.MTS) em videos/ e rode: ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
uv sync
uv run python 1_extrair_quadros.py        # Etapa 1: quadros mais nitidos do video
uv run python 2_comparar_detectores.py    # Etapa 2: SIFT x ORB em pares de quadros
uv run python 3_reconstrucao.py --detector ORB           # modelo completo com o ORB (comparacao)
uv run python 3_reconstrucao.py --detector SIFT --densa  # modelo final: SfM, escala, planta, densa
uv run python 4_ablacao.py --detector SIFT                # Etapa 5: ablacao (pares e numero de quadros)
VIDEO_STEM=$(uv run python -c "from src import config; print(config.video_path().stem)")
uv run python visualizar.py --densa --video results/$VIDEO_STEM/sift/navegacao.mp4 --duracao 45
# Extras opcionais: descomente para rodar
# uv run python 5_colmap.py                 # referencia: o mesmo video no COLMAP (pycolmap, CPU)
# uv run python 6_malha.py                  # malha colorida (Poisson) da nuvem densa, em .ply e .glb
# uv run python visualizar.py --malha --video results/$VIDEO_STEM/sift/navegacao_malha.mp4
echo "Pronto. Para explorar o modelo: uv run python visualizar.py [--detector ORB] [--densa] [--malha]"
