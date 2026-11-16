#!/usr/bin/env bash
# Treina a rede (split de treino, melhor epoca pelo S da validacao). Uso: ./train.sh [nome]
set -euo pipefail
cd "$(dirname "$0")"
uv run python train.py --name "${1:-unet_base}"
