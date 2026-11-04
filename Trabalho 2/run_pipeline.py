"""
Pipeline completo da Variante A (ambiente interno) no video escolhido em config.yaml
(`video`, ou a variavel de ambiente VIDEO), com tudo em results/<video>/:

  Etapa 1   quadros-chave do video (sobreposicao controlada, sem borrados)
  Etapa 2-3 figuras/tabelas: keypoints, SIFT x ORB, matches, linhas epipolares,
            duas vistas, conectividade  (modulos step_1, step_2, step_3)
  Etapa 4   SfM (COLMAP) -> escala metrica (tabuleiro) -> alinhamento da sala ->
            limpeza -> planta baixa + trajetoria -> MVS (PatchMatch ou SGBM)
  Etapa 5   metricas (reprojecao, densidade, erro das dimensoes, deriva do laco) e
            ablacao (exaustivo x sequencial x laco; numero de quadros)
  Entrega   video de navegacao (nuvem esparsa + poses; e com a nuvem densa) e a cena em GLB

O que ja existe em results/<video>/ (quadros, SfM, ablacao, PatchMatch) e reaproveitado;
apague a pasta para refazer do zero.
"""
import json
import os
import time

import pycolmap

import visualize
from pipeline import config, step_0_video_frames, step_4_sfm, steps_1_3_report
from pipeline.room_pipeline import postprocess
from pipeline.step_5_ablation import run_ablation

class Logger:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

def extract_frames(cfg, image_dir, log):
    frames_json = os.path.join(image_dir, "frames.json")
    if os.path.exists(frames_json):
        log(f"[Etapa 1] quadros reaproveitados de {image_dir}")
        return json.load(open(frames_json))["names"]
    log("[Etapa 1] extraindo quadros-chave do video")
    fcfg = cfg["frames"]
    info = step_0_video_frames.extract_keyframes(
        config.video_path(cfg), image_dir, fcfg["target_overlap"], fcfg["min_frames"], fcfg["max_frames"],
        fcfg["max_dim"], fcfg["blur_rel_thresh"], fcfg["max_gap_s"])
    log(f"  {len(info['names'])} quadros de {info['n_video_frames']} ({info['duration_s']:.1f} s); "
        f"sobreposicao estimada media {100 * (info['mean_overlap_est'] or 0):.0f}% "
        f"(min {100 * (info['min_overlap_est'] or 0):.0f}%)")
    return info["names"]

def structure_from_motion(cfg, image_dir, names, sfm_dir, log):
    info_path = os.path.join(sfm_dir, "sfm_info.json")
    if os.path.exists(info_path):
        rec = pycolmap.Reconstruction(os.path.join(sfm_dir, "sparse"))
        log(f"[Etapa 4] SfM reaproveitado de {sfm_dir}")
        return rec, json.load(open(info_path))
    sfm_cfg, cam_cfg = cfg["sfm"], cfg["camera"]
    colmap_exe = step_4_sfm.find_colmap(sfm_cfg["colmap_exe"])
    log(f"[Etapa 4] Structure-from-Motion (COLMAP via {colmap_exe or 'pycolmap'})")
    rec, info = step_4_sfm.run_sfm(
        image_dir, sfm_dir, names, sfm_cfg["matching"], sfm_cfg["sequential_overlap"], sfm_cfg["loop_detection"],
        sfm_cfg["vocab_tree"], cam_cfg["model"], cam_cfg["focal_px"], sfm_cfg["max_image_size"],
        sfm_cfg["max_num_features"], sfm_cfg["use_gpu"], peak_threshold=sfm_cfg["sift_peak_threshold"],
        clahe=sfm_cfg["clahe"], mapper=sfm_cfg["mapper"], colmap_exe=colmap_exe, log=log)
    json.dump(info, open(info_path, "w"), indent=1)
    return rec, info

def main():
    cfg = config.load()
    out = config.results_dir(cfg)
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "pipeline.log"))
    t_start = time.time()
    log(f"=== Pipeline Variante A | {time.strftime('%Y-%m-%d %H:%M')} | video {cfg['video']}")

    image_dir = os.path.join(out, "quadros")
    names = extract_frames(cfg, image_dir, log)
    rec, sfm_info = structure_from_motion(cfg, image_dir, names, os.path.join(out, "sfm"), log)
    log(f"  registradas {sfm_info['n_reg_images']}/{len(names)} imagens, {sfm_info['n_points3d']} pontos, "
        f"reprojecao media {sfm_info['mean_reproj_colmap']:.3f} px, {sfm_info['n_models']} modelo(s)")
    if sfm_info["n_reg_images"] < 0.8 * len(names):
        log("  AVISO: menos de 80% das imagens registradas (pouca textura, borrao ou pouca sobreposicao)")

    log("[Etapas 2-3] keypoints, SIFT x ORB, matches, geometria epipolar, conectividade")
    colmap_sift = {"kps": step_4_sfm.keypoints_per_image(os.path.join(out, "sfm", "database.db")),
                   "t": sfm_info["time_features_s"] / len(names)}
    steps_1_3_report.run(image_dir, names, os.path.join(out, "etapas_2_3"), step_4_sfm.scene_arrays(rec)["K"],
                         colmap_sift, cfg["analysis"]["n_pairs"], cfg["analysis"]["pair_gap"], log=log)

    postprocess(rec, sfm_info, image_dir, names, out, cfg, log=log)
    log("\n" + open(os.path.join(out, "metricas.md")).read())

    if cfg["ablation"]["enabled"]:
        run_ablation(image_dir, names, os.path.join(out, "ablacao"), cfg, log=log)
        log("\n" + open(os.path.join(out, "ablacao", "ablacao.md")).read())

    log("[Entregavel] video de navegacao e cena em GLB")
    visualize.make_video(out, cfg["render"], sparse=True, log=log)
    visualize.export_glb(out, sparse=True, log=log)
    if os.path.exists(os.path.join(out, "nuvem_densa.ply")):
        visualize.make_video(out, cfg["render"], sparse=False, log=log)
        visualize.export_glb(out, sparse=False, log=log)
    log(f"=== concluido em {(time.time() - t_start) / 60:.1f} min. Resultados em {out}")

if __name__ == "__main__":
    main()
