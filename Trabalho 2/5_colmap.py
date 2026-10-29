"""
Referencia: o mesmo video reconstruido com o COLMAP (pycolmap), em results/<video>/colmap/.
SIFT do COLMAP, uma camera so (SIMPLE_RADIAL: focal e distorcao radial estimadas no BA),
matching sequencial com sobreposicao quadratica e SfM incremental. Serve para comparar com o
SfM implementado aqui (cameras registradas, pontos, erro de reprojecao). Roda na CPU.
"""
import json
import shutil
import time

import pycolmap

from src import config

if __name__ == "__main__":
    frames = config.frames_dir()
    out = config.video_results() / "colmap"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    db = out / "database.db"
    n_frames = len(list(frames.glob("q*.jpg")))

    t0 = time.perf_counter()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = 8192
    pycolmap.extract_features(db, frames, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=extraction)
    t_extract = time.perf_counter() - t0

    t0 = time.perf_counter()
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 15
    pairing.quadratic_overlap = True
    pycolmap.match_sequential(db, pairing_options=pairing)
    t_match = time.perf_counter() - t0

    t0 = time.perf_counter()
    models = pycolmap.incremental_mapping(db, frames, out)
    t_map = time.perf_counter() - t0

    summary = {"n_frames": n_frames, "t_extract_s": t_extract, "t_match_s": t_match, "t_map_s": t_map, "models": []}
    for k, rec in sorted(models.items(), key=lambda kv: -kv[1].num_reg_images()):
        cam = next(iter(rec.cameras.values()))
        summary["models"].append({"id": k, "registered": rec.num_reg_images(), "points": rec.num_points3D(),
                                  "reproj_mean_px": rec.compute_mean_reprojection_error(),
                                  "mean_track_length": rec.compute_mean_track_length(),
                                  "camera_params": list(cam.params)})
    (out / "resumo.json").write_text(json.dumps(summary, indent=2))
    print(f"COLMAP {config.video_path().name}: {n_frames} quadros, {len(summary['models'])} modelo(s)")
    for m in summary["models"]:
        print(f"  modelo {m['id']}: {m['registered']} cameras, {m['points']} pontos, "
              f"reproj {m['reproj_mean_px']:.2f} px, camera {[round(p, 3) for p in m['camera_params']]}")
    print(f"  tempos: extracao {t_extract:.0f} s, matching {t_match:.0f} s, SfM {t_map:.0f} s")
