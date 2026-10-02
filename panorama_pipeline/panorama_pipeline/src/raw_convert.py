"""
Conversao de imagens RAW (.ARW, formato Sony) para PNG, para uso no pipeline
de panoramas (as demais etapas trabalham com PNG/JPG via OpenCV).

Instalacao (uma vez):
    pip install rawpy imageio

Uso:
    python -m src.raw_convert --input_dir fotos_arw --output_dir fotos_png

Observacao sobre a Etapa 4.1 do enunciado ("nomes embaralhados, sem pista de
ordem de captura"): este script preserva os nomes originais (que ja saem da
camera fora de ordem alfabetica "natural" da captura). Se quiser garantir que
os nomes de saida nao carreguem nenhuma pista, use --anonymize para renomear
os arquivos de saida para um hash aleatorio.
"""
import argparse
import os
import random
import string

import rawpy
import imageio.v2 as imageio
import cv2


def random_name(length=8):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def convert_arw_to_png(input_dir, output_dir, max_dim=1600, anonymize=False):
    os.makedirs(output_dir, exist_ok=True)
    arw_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".arw")]
    if not arw_files:
        print(f"Nenhum arquivo .ARW encontrado em {input_dir}")
        return []

    out_paths = []
    for filename in sorted(arw_files):
        path = os.path.join(input_dir, filename)
        with rawpy.imread(path) as raw:
            # postprocess faz demosaicagem + balanco de branco da propria camera
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)

        h, w = rgb.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        base = random_name() if anonymize else os.path.splitext(filename)[0]
        out_name = base + ".png"
        out_path = os.path.join(output_dir, out_name)
        imageio.imwrite(out_path, rgb)
        out_paths.append(out_path)
        print(f"Convertido: {filename} -> {out_name}  ({rgb.shape[1]}x{rgb.shape[0]})")

    return out_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte imagens .ARW para .PNG")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_dim", type=int, default=1600,
                         help="Dimensao maxima (lado maior) apos redimensionamento")
    parser.add_argument("--anonymize", action="store_true",
                         help="Renomeia as saidas para nomes aleatorios (remove qualquer pista de ordem)")
    args = parser.parse_args()
    convert_arw_to_png(args.input_dir, args.output_dir, args.max_dim, args.anonymize)
