#!/usr/bin/env bash
# Pipeline completo da Variante A (ambiente interno).
#
#   ./run.sh video_da_sala.mp4 [saida] [opcoes do run_pipeline.py]
#   ./run.sh pasta_de_fotos/   [saida] [opcoes]
#   ./run.sh demo                          # gera uma sala sintetica 5,0 x 4,0 x 2,7 m e roda tudo
#
# Opcoes uteis: --no-ablation --no-dense --no-video --no-analysis --reuse-sfm --config outro.yaml
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
INPUT="${1:?uso: ./run.sh <video|pasta|demo> [saida] [opcoes]}"
shift
# o 2o argumento e a pasta de saida, a menos que ja seja uma opcao (--no-dense etc.)
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then OUT="$1"; shift; else OUT="saida"; fi
PY="${PYTHON:-python3}"

case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
CONFIG_ARGS=()
if [ "$INPUT" = "demo" ]; then
  mkdir -p "$HERE/dados"
  [ -f "$HERE/dados/sala_sintetica.mp4" ] || "$PY" "$HERE/tests/synthetic_room.py" --out "$HERE/dados/sala_sintetica.mp4"
  INPUT="$HERE/dados/sala_sintetica.mp4"
  CONFIG_ARGS=(--config "$HERE/tests/config_demo.yaml")
else
  case "$INPUT" in /*) ;; *) INPUT="$PWD/$INPUT" ;; esac
fi

if [ -d "$INPUT" ]; then SRC=(--images "$INPUT"); else SRC=(--video "$INPUT"); fi
cd "$HERE"
[ ${#CONFIG_ARGS[@]} -eq 0 ] && CONFIG_ARGS=(--config "$HERE/config.yaml")
"$PY" run_pipeline.py "${SRC[@]}" --out "$OUT" "${CONFIG_ARGS[@]}" "$@"

echo
echo "Resultados em: $OUT"
echo "  metricas:         $OUT/metricas.md"
echo "  planta baixa:     $OUT/planta_baixa.png"
echo "  video:            $OUT/video_navegacao.mp4"
echo "Visualizar em 3D:   $PY visualize.py \"$OUT\""
