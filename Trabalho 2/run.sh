#!/usr/bin/env bash
# Variante A completa no video de config.yaml (ou VIDEO=A ./run.sh), em results/<video>/,
# e a visualizacao 3D do resultado
set -euo pipefail
cd "$(dirname "$0")"
uv sync
uv run python run_pipeline.py
uv run python visualize.py
