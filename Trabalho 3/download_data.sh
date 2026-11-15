#!/usr/bin/env bash
# Baixa o dataset da competicao para data/ (~10 GB).
# Precisa do token da API do Kaggle em ~/.kaggle/kaggle.json
# (Kaggle -> Settings -> API -> Create New Token) e de ter aceitado as regras da competicao no site.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
if [ ! -f data/sample_submission.csv ]; then
  uv run kaggle competitions download -c recodai-luc-scientific-image-forgery-detection -p data
  unzip -q -o data/recodai-luc-scientific-image-forgery-detection.zip -d data
  rm -f data/recodai-luc-scientific-image-forgery-detection.zip
fi
ls data
