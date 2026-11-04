import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda"  # cai para "cpu" sozinho se nao houver GPU (ver `device()`)

# Etapa 1: video (.MTS) e selecao de quadros
# Video usado: VIDEO abaixo ou a variavel de ambiente (p.ex. `VIDEO=B.mts ./run.sh`). Quadros,
# cache e resultados de cada video ficam separados: frames/<video>/ e results/<video>/
VIDEO = os.environ.get("VIDEO", "C.mts")
VIDEO_DIR = ROOT / "videos"
FRAMES_DIR = ROOT / "frames"
# quadros escolhidos: o mais nitido de cada janela de duracao / N_FRAMES. Os videos tem giros rapidos:
# com ~500 quadros (1 a cada 0,5 s) a sequencia se parte em pedacos, ate no COLMAP; ~1 a cada 0,1 s
N_FRAMES = 2300
RESULTS = ROOT / "results"

# Referencia de escala: o PDF impresso. O nome do arquivo diz os cantos internos ("...-8x6...":
# 9 x 7 quadrados -> 8 x 6 cantos internos) e o lado do quadrado no PDF (25 mm). A impressao saiu
# reduzida: o quadrado medido na folha tem 22 mm, e e essa medida que vale (None = a do nome)
CHECKERBOARD_PDF = ROOT / "images" / "Checkerboard-A4-25mm-8x6 (1).pdf"
CHECKERBOARD_SQUARE_M = 0.022

# Medidas reais (trena) de cada sala, em metros, por video. Preencha; None = nao comparada. As
# chaves alem de "largura" e "comprimento" sao medidas a mao no modelo com
# `uv run python visualizar.py --medir <chave>`
REAL_MEASUREMENTS_BY_VIDEO = {
    "A": {"largura": 6.95, "comprimento": 12.75, "altura_porta": 2.12},  # laboratorio Recod
    "B": {"largura": 6.20, "comprimento": 9.00, "altura_porta": 2.16},   # sala 90 (porta com batente)
    "C": {"largura": 5.70, "comprimento": 5.80, "altura_porta": 2.15},   # copa (porta com batente)
}
# angulo entre paredes vizinhas: medido 89-92 graus nas salas; assume-se 90

# Etapa 2: detectores. SIFT/ORB usam vizinho mais proximo exato (forca bruta, na GPU) + ratio
# test. Tambem ha SUPERPOINT, DISK e ALIKED (redes, casadas com o LightGlue), fora do enunciado:
# para incluir na comparacao, acrescente-os em DETECTORS_COMPARED
DETECTOR = "SIFT"
DETECTORS_COMPARED = ("SIFT", "ORB")
MAX_KEYPOINTS = 4096
SIFT_CONTRAST = 0.01      # o padrao (0.04) acha ~300 pontos por quadro neste video comprimido
EQUALIZE = True           # CLAHE antes da deteccao
RATIO_THRESH = 0.8
# verificacao cruzada (so matches mutuos): com o contraste baixo, o SIFT pega muitos cantos quase
# iguais em parede lisa; sem ela, centenas de pontos de um quadro casam com os mesmos poucos do
# outro, e o RANSAC de E aceita esse caso degenerado (epipolo sobre o ponto repetido)
CROSS_CHECK = True

# Etapa 3: pares e geometria epipolar
# Quadros-chave: os N_FRAMES quadros densos so servem para medir a sobreposicao (pares sequenciais
# a ate DENSE_WINDOW quadros); o proximo quadro-chave e o mais distante que ainda tem
# KEYFRAME_MIN_INLIERS inliers com o atual. Mais proximos nos giros rapidos, espacados nos lentos.
# Nao funcionou neste video: com 40 inliers (~420 quadros-chave) o SfM registrou 24 cameras; com
# 100 (~750), 2. Escolhendo o quadro mais distante que ainda passa do limiar, quadros-chave vizinhos
# compartilham pouco e os tracks nao atravessam 3 deles. Com todos os quadros densos (~390 inliers
# entre consecutivos) o SfM registra a maior parte; o BA esparso da conta de milhares de cameras
KEYFRAMES = False
KEYFRAME_MIN_INLIERS = 100
DENSE_WINDOW = 20
PAIRING = "sequential_loop"  # "exhaustive", "sequential" ou "sequential_loop"
SEQ_WINDOW = 10           # cada quadro(-chave) casa com os SEQ_WINDOW seguintes
LOOP_TOP_K = 5            # candidatos a laco por quadro (bag of visual words), fora da janela
RANSAC_THRESH = 2.0       # px; 1 px descarta matches bons nos quadros comprimidos
MIN_MATCHES = 15

# Etapa 4: SfM, limpeza e densa
MIN_INLIERS_PNP = 30
MAX_REPROJ_ERROR = 4.0
CLEAN = {"nb_neighbors": 20, "std_ratio": 2.0, "voxel_size": None, "crop_percentile": 0.5}
DENSE = {"max_side": 640, "n_sources": 4, "n_depths": 128}


def checkerboard_spec(path=CHECKERBOARD_PDF, square_m=CHECKERBOARD_SQUARE_M):
    """
    (cantos internos por linha, por coluna), lidos do nome do PDF, e o lado do quadrado em
    metros: `square_m` (medido na folha impressa) ou, sem ele, o do nome do PDF.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF do tabuleiro nao encontrado: {path}")
    found = re.search(r"(\d+(?:\.\d+)?)mm-(\d+)x(\d+)", path.name)
    if not found:
        raise ValueError(f"O nome do PDF deve conter '<lado>mm-<cols>x<linhas>': {path.name}")
    size_mm, cols, rows = found.groups()
    return (int(cols), int(rows)), square_m if square_m is not None else float(size_mm) / 1000


def video_path():
    path = VIDEO_DIR / VIDEO
    if not path.exists():
        found = ", ".join(p.name for p in sorted(VIDEO_DIR.iterdir()) if p.suffix.lower() == ".mts")
        raise FileNotFoundError(f"{path} nao existe; videos em {VIDEO_DIR}: {found or 'nenhum'}")
    return path


def frames_dir():
    return FRAMES_DIR / video_path().stem


def video_results():
    """results/<video>/: tudo o que sai de um video."""
    return RESULTS / video_path().stem


def cache_dir():
    return video_results() / "cache"


def results_dir(detector=None):
    """Saidas da reconstrucao de cada detector: results/<video>/sift/, results/<video>/orb/..."""
    return video_results() / (detector or DETECTOR).lower()


def real_measurements():
    return REAL_MEASUREMENTS_BY_VIDEO.get(video_path().stem, {})


def device():
    import torch
    return torch.device(DEVICE if torch.cuda.is_available() else "cpu")
