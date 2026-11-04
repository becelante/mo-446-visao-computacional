"""
Configuracao do pipeline: as secoes comuns do config.yaml com o bloco `videos.<video>`
(medidas de trena e camera de cada sala) aplicado por cima. O video vem de `video` no
config.yaml ou da variavel de ambiente VIDEO (p.ex. `VIDEO=A ./run.sh`).
"""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _merge(base, over):
    for k, v in over.items():
        base[k] = _merge(base.get(k) or {}, v) if isinstance(v, dict) else v
    return base

def load():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["video"] = os.environ.get("VIDEO", cfg["video"])
    return _merge(cfg, cfg.pop("videos").get(cfg["video"]) or {})

def video_path(cfg):
    return os.path.join(ROOT, "videos", f"{cfg['video']}.MTS")

def results_dir(cfg):
    return os.path.join(ROOT, "results", cfg["video"])
