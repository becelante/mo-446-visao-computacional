#!/usr/bin/env bash
# Escolhe os limiares na validacao e gera as predicoes. Uso: ./predict.sh [nome] [val|test]
set -euo pipefail
cd "$(dirname "$0")"
uv run python predict.py "results/unet/${1:-unet_base}" "${2:-val}"
