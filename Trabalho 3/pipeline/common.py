"""Configuracao, GPU e semente."""
import os
import random

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = os.path.join(ROOT, cfg["data_dir"])
    cfg["results_dir"] = os.path.join(ROOT, cfg["results_dir"])
    return cfg


def get_device():
    """GPU se houver, senao CPU."""
    if not torch.cuda.is_available():
        print("[aviso] CUDA indisponivel, usando CPU")
        return torch.device("cpu")
    torch.backends.cudnn.benchmark = True          # escolhe o kernel mais rapido para cada tamanho de entrada
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 nas placas Ampere (RTX 30xx): mais rapido
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
