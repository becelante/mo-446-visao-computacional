"""
Pipeline completo da Variante A (ambiente interno) a partir de um video (ou de
uma pasta de fotos):

  Etapa 1   quadros-chave do video (sobreposicao controlada, sem borrados)
  Etapa 2-3 figuras/tabelas: keypoints, SIFT x ORB, matches, linhas epipolares,
            duas vistas, conectividade  (modulos step_1, step_2, step_3)
  Etapa 4   SfM (COLMAP) -> escala metrica (ArUco/xadrez) -> alinhamento da sala ->
            limpeza -> planta baixa + trajetoria -> MVS (PatchMatch ou SGBM)
  Etapa 5   metricas (reprojecao, densidade, erro das dimensoes, deriva do laco),
            ablacao (exaustivo x sequencial x laco; numero de quadros) e video
            de navegacao.

Exemplos:
    python run_pipeline.py --video dados/sala.mp4 --out saida
    python run_pipeline.py --images dados/fotos --out saida --no-ablation
"""
import argparse
import json
import os
import sys
import time

import yaml

from pipeline import step_0_video_frames, step_4_sfm
from pipeline.room_pipeline import postprocess


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


class Logger:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video da sala (laco fechado)")
    src.add_argument("--images", help="pasta com fotos (alternativa ao video)")
    ap.add_argument("--out", default="saida")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-analysis", action="store_true", help="pula as figuras das Etapas 2-3")
    ap.add_argument("--no-dense", action="store_true", help="pula a densificacao")
    ap.add_argument("--no-ablation", action="store_true", help="pula o estudo de ablacao")
    ap.add_argument("--no-video", action="store_true", help="pula o video de navegacao")
    ap.add_argument("--reuse-sfm", action="store_true", help="reaproveita o SfM ja salvo em <out>/sfm")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    log = Logger(os.path.join(args.out, "pipeline.log"))
    t_start = time.time()
    log(f"=== Pipeline Variante A | {time.strftime('%Y-%m-%d %H:%M')} | config: {args.config}")

    # ---------------- Etapa 1: quadros -------------------------------------------------
    if args.video:
        image_dir = os.path.join(args.out, "quadros")
        vcfg = cfg.get("video", {})
        if os.path.exists(os.path.join(image_dir, "frames.json")) and args.reuse_sfm:
            names = json.load(open(os.path.join(image_dir, "frames.json")))["names"]
        else:
            log("[Etapa 1] extraindo quadros-chave do video")
            info = step_0_video_frames.extract_keyframes(
                args.video, image_dir, vcfg.get("target_overlap", 0.8), vcfg.get("min_frames", 100),
                vcfg.get("max_frames", 300), vcfg.get("max_dim", 1920), vcfg.get("blur_rel_thresh", 0.5),
                vcfg.get("max_gap_s", 2.0))
            names = info["names"]
            log(f"  {len(names)} quadros de {info['n_video_frames']} ({info['duration_s']:.1f} s); "
                f"sobreposicao estimada media {100 * (info['mean_overlap_est'] or 0):.0f}% "
                f"(min {100 * (info['min_overlap_est'] or 0):.0f}%)")
    else:
        image_dir = args.images
        names = step_0_video_frames.list_images(image_dir)
        log(f"[Etapa 1] {len(names)} imagens em {image_dir}")
    if len(names) < 3:
        sys.exit("Imagens insuficientes.")

    # ---------------- Etapa 4a: SfM ----------------------------------------------------
    sfm_cfg, cam_cfg = cfg.get("sfm", {}), cfg.get("camera", {})
    sfm_dir = os.path.join(args.out, "sfm")
    import pycolmap
    if args.reuse_sfm and os.path.exists(os.path.join(sfm_dir, "sparse")):
        rec = pycolmap.Reconstruction(os.path.join(sfm_dir, "sparse"))
        sfm_info = json.load(open(os.path.join(sfm_dir, "sfm_info.json")))
        log(f"[Etapa 4] SfM reaproveitado: {rec.num_reg_images()} imagens, {rec.num_points3D()} pontos")
    else:
        colmap_exe = step_4_sfm.find_colmap(sfm_cfg.get("colmap_exe", "auto"))
        log(f"[Etapa 4] Structure-from-Motion (COLMAP via {colmap_exe or 'pycolmap'})")
        rec, sfm_info = step_4_sfm.run_sfm(
            image_dir, sfm_dir, names, sfm_cfg.get("matching", "exhaustive"), sfm_cfg.get("sequential_overlap", 15),
            sfm_cfg.get("loop_detection", False), sfm_cfg.get("vocab_tree"), cam_cfg.get("model", "SIMPLE_RADIAL"),
            cam_cfg.get("focal_px"), sfm_cfg.get("max_image_size", 1600), sfm_cfg.get("max_num_features", 8192),
            sfm_cfg.get("use_gpu", "auto"), peak_threshold=sfm_cfg.get("sift_peak_threshold"),
            clahe=sfm_cfg.get("clahe", False), mapper=sfm_cfg.get("mapper", "incremental"),
            colmap_exe=colmap_exe, log=log)
        json.dump(sfm_info, open(os.path.join(sfm_dir, "sfm_info.json"), "w"), indent=1)
    log(f"  registradas {sfm_info['n_reg_images']}/{len(names)} imagens, {sfm_info['n_points3d']} pontos, "
        f"reprojecao media {sfm_info['mean_reproj_colmap']:.3f} px, {sfm_info['n_models']} modelo(s)")
    if sfm_info["n_reg_images"] < 0.8 * len(names):
        log("  AVISO: menos de 80% das imagens registradas (pouca textura, borrao ou pouca sobreposicao)")

    # ---------------- Etapas 2-3: figuras ----------------------------------------------
    if not args.no_analysis:
        log("[Etapas 2-3] keypoints, SIFT x ORB, matches, geometria epipolar, conectividade")
        from pipeline import steps_1_3_report
        sc = step_4_sfm.scene_arrays(rec)
        poses = {n: (R, t) for n, R, t in zip(sc["names"], sc["R"], sc["t"])}
        try:
            steps_1_3_report.run(image_dir, names, os.path.join(args.out, "etapas_2_3"), K=sc["K"],
                                 colmap_poses=poses, n_pairs=cfg.get("analysis", {}).get("n_pairs", 30),
                                 pair_gap=cfg.get("analysis", {}).get("pair_gap", 5), log=log)
        except Exception as e:
            log(f"  AVISO: analise das Etapas 2-3 falhou: {e}")

    # ---------------- Etapas 4-5 -------------------------------------------------------
    metrics = postprocess(rec, sfm_info, image_dir, names, args.out, cfg, do_dense=not args.no_dense, log=log)
    log("\n" + open(os.path.join(args.out, "metricas.md")).read())

    # ---------------- Ablacao ----------------------------------------------------------
    if not args.no_ablation and cfg.get("ablation", {}).get("enabled", True):
        from pipeline.step_5_ablation import run_ablation
        run_ablation(image_dir, names, os.path.join(args.out, "ablacao"), cfg, log=log)
        log("\n" + open(os.path.join(args.out, "ablacao", "ablacao.md")).read())

    # ---------------- Video de navegacao -----------------------------------------------
    if not args.no_video:
        log("[Entregavel] video de navegacao")
        import visualize
        visualize.make_video(args.out, cfg.get("render", {}), log=log)

    log(f"=== concluido em {(time.time() - t_start) / 60:.1f} min. Resultados em {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
