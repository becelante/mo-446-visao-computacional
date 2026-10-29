"""
Extracao de quadros do video (Etapa 1).

O ffmpeg decodifica o .MTS na GPU (NVDEC) e entrega os quadros crus por um pipe, sem gravar
o video inteiro em disco. O video e dividido em `n_frames` janelas de mesma duracao e, de cada
uma, fica o quadro mais nitido (maior variancia do laplaciano): com a camera na mao, quadros
vizinhos variam muito em borrao de movimento, e um quadro borrado rende poucos keypoints.
Sem desentrelacamento: o filtro idet do ffmpeg classifica os quadros deste video como
progressivos, e desentrelacar quadros progressivos so tiraria resolucao vertical.
"""
import json
import subprocess

import numpy as np
import cv2


def probe(video):
    """(largura, altura, fps, duracao em s) do primeiro fluxo de video."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    stream = info["streams"][0]
    num, den = map(int, stream["avg_frame_rate"].split("/"))
    return stream["width"], stream["height"], num / den, float(info["format"]["duration"])


def sharpness(frame):
    """Variancia do laplaciano em meia resolucao (borrao de movimento -> valor baixo)."""
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_32F).var()


def read_frames(video, width, height, hwaccel=True):
    """Gera os quadros BGR do video, decodificados pelo ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", *(["-hwaccel", "cuda"] if hwaccel else []), "-i", str(video),
           "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = width * height * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=size) as proc:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    if proc.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg falhou ({proc.returncode})")


def extract_frames(video, out_dir, n_frames=300, log=print):
    """
    Salva em `out_dir` o quadro mais nitido de cada uma das `n_frames` janelas do video, como
    `q<indice do quadro no video>.jpg` (o nome guarda a ordem da captura). Retorna a lista de
    arquivos e um dict com o resumo para o relatorio.
    """
    width, height, fps, duration = probe(video)
    total = int(round(duration * fps))
    window = max(1, total // n_frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("q*.jpg"):
        old.unlink()
    log(f"{video.name}: {width}x{height}, {fps:.0f} fps, {duration:.0f} s (~{total} quadros); "
        f"o mais nitido de cada {window} quadros")

    saved, scores = [], []
    best = (-1.0, None, -1)
    for k, frame in enumerate(read_frames(video, width, height)):
        s = sharpness(frame)
        if s > best[0]:
            best = (s, frame.copy(), k)
        if (k + 1) % window == 0:
            path = out_dir / f"q{best[2]:06d}.jpg"
            cv2.imwrite(str(path), best[1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(path)
            scores.append(best[0])
            best = (-1.0, None, -1)
    if best[1] is not None:
        path = out_dir / f"q{best[2]:06d}.jpg"
        cv2.imwrite(str(path), best[1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(path)
        scores.append(best[0])
    summary = {"video": video.name, "width": width, "height": height, "fps": fps, "duration_s": duration,
               "video_frames": k + 1, "window": window, "n_frames": len(saved),
               "sharpness_median": float(np.median(scores))}
    log(f"{len(saved)} quadros salvos em {out_dir}")
    return saved, summary


class FrameStore:
    """
    Lista de quadros lida do disco sob demanda (`store[i]` le o JPEG). Com ~2300 quadros 1080p, ter
    todos na memoria custaria ~14 GB; um cache pequeno evita reler o mesmo quadro em sequencia.
    """

    def __init__(self, paths, cache_size=16):
        self.paths = list(paths)
        self.cache, self.cache_size = {}, cache_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return FrameStore(self.paths[i], self.cache_size)
        if i not in self.cache:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = cv2.imread(str(self.paths[i]))
        return self.cache[i]

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class Subset:
    """Visao preguicosa de `images` nos indices dados: `Subset(images, idx)[n]` e `images[idx[n]]`."""

    def __init__(self, images, indices):
        self.images, self.indices = images, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, n):
        return self.images[self.indices[n]]

    def __iter__(self):
        return (self[n] for n in range(len(self)))


def load_frames(frames_dir, step=1):
    """Caminhos e quadros (`FrameStore`) em ordem de captura; `step` > 1 usa um a cada `step` (ablacao)."""
    paths = sorted(frames_dir.glob("q*.jpg"))[::step]
    if not paths:
        raise FileNotFoundError(f"Nenhum quadro em {frames_dir}; rode 1_extrair_quadros.py")
    return paths, FrameStore(paths)
