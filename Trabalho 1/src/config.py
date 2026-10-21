from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 0  # semente do embaralhamento da leitura (Etapa 4.1)

# Passo 1: revelacao do RAW. (pasta com os .ARW, pasta de saida, lado maior em px (0 = total), bits,
# nomes aleatorios (Etapa 4.1, so no conjunto do pipeline principal))
RAW_SETS = [
    (ROOT / "D1", ROOT / "D1_png", 1600, 8, True),
    (ROOT / "A1", ROOT / "A1_png", 0, 16, False),
    (ROOT / "B1", ROOT / "B1_png", 0, 16, False),
    (ROOT / "C1", ROOT / "C1_png", 0, 16, False),
]

DETECTOR = "SIFT"
DETECTORS_COMPARED = ("SIFT", "ORB")
RATIO_THRESH = 0.75
MIN_MATCHES = 8

# Passo 2: pipeline principal (Secoes 1 a 6 do relatorio), com projecao plana. So o D1: o A1
# (213 graus) e o B1 (360 graus) nao cabem num plano, e o C1 tem paralaxe demais para homografias.
BASE_INPUT = ROOT / "D1_png"
BASE_RESULTS = ROOT / "D1_results"
BASE_RANSAC_THRESH = 1.0  # px
BASE_MIN_INLIERS = 10
BETA = 0.3                # Brown & Lowe: inliers > 8 + BETA * matches
FEATHER_WIDTH = 15        # px

# Passo 3: extras X1 a X3 (bundle adjustment, projecao cilindrica/esferica)
RANSAC_THRESH = 4.0       # px, na escala de trabalho
MIN_INLIERS = 20
WORK_DIM = 3000           # lado maior das fotos nas etapas 2 a 5
COMPOSE_SCALE = 1.0       # 1.0 = resolucao total; 0.25 = previa
NUM_BANDS = 7

# ba: corte de pares e escala da perda robusta (px); align: alinhamento local nas costuras
LOW_PARALLAX = {"ba": {"max_pair_error": 4.0, "loss_scale": 1.0},
                "align": {"fade": 48, "max_flow": 12.0, "smooth": 3.0}}
HIGH_PARALLAX = {"ba": {"max_pair_error": float("inf"), "loss_scale": 10.0},
                 "align": {"fade": 150, "max_flow": 80.0, "smooth": 12.0}}

# beta menor no C1: com folhagem ao vento, so 10 a 30% dos matches de um par real sao inliers
def _extra(name, projection, settings, beta=BETA):
    return {"name": name, "png": ROOT / f"{name}_png", "results": ROOT / "extras" / name,
            "projection": projection, "beta": beta, "ba": settings["ba"], "align": settings["align"]}

EXTRAS = [
    _extra("D1", "cylindrical", LOW_PARALLAX),
    _extra("A1", "cylindrical", LOW_PARALLAX),
    _extra("B1", "spherical", HIGH_PARALLAX),
    _extra("C1", "cylindrical", HIGH_PARALLAX, beta=0.05),
]

# Passo 4: extra X4. SIFT x LightGlue nos pares do X4_DATASET; cv2.Stitcher x LightGlue no D1
X4_DATASET = "A1"
X4_RESULTS = ROOT / "extras" / "x4"
X4_SEEDS = 10             # o emparelhador do cv2.Stitcher e aleatorio: as metricas usam todas as sementes
X4_FIGURE_SEED = 6        # semente dos panoramas e detalhes salvos
X4_DETAIL_ANCHOR = ("DSC03739", (806.0, 165.0))  # (foto, pixel) no centro do recorte: topo da coluna
