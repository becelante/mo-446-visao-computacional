from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# beta: Brown & Lowe (inliers > 8 + beta * matches); ba: corte de pares e escala da
# perda robusta (px); align: alinhamento local nas costuras (px da escala de trabalho)
LOW_PARALLAX = {"ba": {"max_pair_error": 4.0, "loss_scale": 1.0},
                "align": {"fade": 48, "max_flow": 12.0, "smooth": 3.0}}
HIGH_PARALLAX = {"ba": {"max_pair_error": float("inf"), "loss_scale": 10.0},
                 "align": {"fade": 150, "max_flow": 80.0, "smooth": 12.0}}

def _dataset(name, projection, beta, settings):
    return {"name": name, "raw": ROOT / name, "png": ROOT / f"{name}_png",
            "results": ROOT / f"{name}_results", "projection": projection, "beta": beta,
            "ba": settings["ba"], "align": settings["align"]}

DATASETS = [
    _dataset("A1", "cylindrical", 0.3, LOW_PARALLAX),
    _dataset("B1", "spherical", 0.3, HIGH_PARALLAX),
    _dataset("D1", "cylindrical", 0.3, LOW_PARALLAX),
    _dataset("C1", "cylindrical", 0.05, HIGH_PARALLAX),
]

RAW_MAX_DIM = 0       # 0 = resolucao total
RAW_BITS = 16
RAW_ANONYMIZE = True

DETECTOR = "SIFT"
DETECTORS_COMPARED = ("SIFT", "ORB")
RATIO_THRESH = 0.75
RANSAC_THRESH = 4.0       # px, na escala de trabalho
MIN_MATCHES = 8
MIN_INLIERS = 20
WORK_DIM = 3000           # lado maior das fotos nas etapas 2 a 5
VIS_MAX_DIM = 1600

COMPOSE_SCALE = 1.0       # 1.0 = resolucao total; 0.25 = previa
NUM_BANDS = 7

X4_DATASET = "A1"
X4_RESULTS = ROOT / "A1_results" / "x4_comparacao"
X4_STITCHER_DIM = 1500
