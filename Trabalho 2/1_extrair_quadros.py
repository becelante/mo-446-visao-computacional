"""Etapa 1: escolhe os quadros mais nitidos do .MTS em videos/ e salva em frames/<video>/."""
import json

from src import config
from src.step_0_frames import extract_frames

if __name__ == "__main__":
    video = config.video_path()
    _, summary = extract_frames(video, config.frames_dir(), config.N_FRAMES)
    config.video_results().mkdir(parents=True, exist_ok=True)
    (config.video_results() / "quadros.json").write_text(json.dumps(summary, indent=2))
