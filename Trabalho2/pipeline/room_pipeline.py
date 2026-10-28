"""
Encadeia as Etapas 4 e 5 da Variante A sobre um modelo do SfM:
escala metrica -> alinhamento -> limpeza -> dimensoes/planta/trajetoria ->
(densificacao) -> metricas. Usado pelo pipeline principal e pela ablacao.
"""
import json
import os

import numpy as np

from . import step_4_dense, step_4_room, step_4_scale_align, step_4_sfm, step_5_metrics
from .viz_utils import camera_frustums, trajectory_lines, write_lines_ply, write_ply


def postprocess(rec, sfm_info, image_dir, all_names, out_dir, cfg, do_dense=True, figures=True, log=print):
    os.makedirs(out_dir, exist_ok=True)

    # 1) escala metrica a partir da referencia
    log("[Etapa 4] escala metrica")
    scale, scale_info = step_4_scale_align.estimate_scale(rec, image_dir, cfg.get("scale", {}), log)
    if scale:
        step_4_sfm.apply_sim3(rec, scale)

    # 2) alinhamento: z para cima, chao em z = 0, paredes nos eixos
    log("[Etapa 4] alinhamento da sala (chao e paredes por RANSAC)")
    room_cfg = cfg.get("room", {})
    align = step_4_scale_align.align_room(rec, metric=scale is not None,
                                          plane_thresh=room_cfg.get("plane_thresh_m", 0.03), log=log)
    model_dir = os.path.join(out_dir, "modelo_metrico")
    os.makedirs(model_dir, exist_ok=True)
    rec.write(model_dir)
    os.makedirs(os.path.join(model_dir, "txt"), exist_ok=True)
    rec.write_text(os.path.join(model_dir, "txt"))

    sc = step_4_sfm.scene_arrays(rec)
    errors = step_4_sfm.reprojection_errors(rec)
    unit = 1.0 if scale else align["unit"]

    # 3) limpeza + medidas da sala
    log("[Etapa 4] limpeza da nuvem e medidas da sala")
    base = (sc["track_len"] >= 2)
    xyz0, rgb0, _ = step_4_room.clean_cloud(sc["xyz"], sc["rgb"], None, extra_mask=base)
    room = step_4_room.measure_room(xyz0, thresh=room_cfg.get("plane_thresh_m", 0.03) * unit)
    xyz, rgb, _ = step_4_room.clean_cloud(sc["xyz"], sc["rgb"], room, extra_mask=base,
                                          crop_margin=room_cfg.get("crop_margin_m", 0.3) * unit)
    if room_cfg.get("remeasure_after_crop", True):
        room = step_4_room.measure_room(xyz, thresh=room_cfg.get("plane_thresh_m", 0.03) * unit)
    log(f"  sala: {room['x_extent_m']:.3f} x {room['y_extent_m']:.3f} m"
        + (f", altura {room['height_m']:.3f} m" if room["height_m"] else ", teto nao detectado"))
    drift = step_5_metrics.loop_drift(sc["centers"], sc["names"], all_names)
    if drift:
        log(f"  deriva do laco: {drift['drift_cm']:.1f} cm ({drift['drift_pct']:.2f}% de {drift['trajectory_length_m']:.2f} m)")

    write_ply(os.path.join(out_dir, "nuvem_esparsa.ply"), xyz, rgb)
    size = (sc["camera"].width, sc["camera"].height)
    fr_depth = 0.2 * unit
    V, E = camera_frustums(sc["centers"], sc["R"], sc["K"], size, depth=fr_depth)
    write_lines_ply(os.path.join(out_dir, "cameras.ply"), V, E, np.tile([255, 120, 0], (len(E), 1)))
    V, E = trajectory_lines(sc["centers"])
    write_lines_ply(os.path.join(out_dir, "trajetoria.ply"), V, E, np.tile([255, 40, 40], (len(E), 1)))
    np.savetxt(os.path.join(out_dir, "trajetoria.csv"), np.column_stack([np.arange(len(sc["centers"])), sc["centers"]]),
               delimiter=",", header="idx,x,y,z", comments="", fmt=["%d", "%.5f", "%.5f", "%.5f"])

    # 4) densificacao
    dense_xyz = dense_rgb = None
    dense_method = None
    dcfg = cfg.get("dense", {})
    if do_dense and dcfg.get("method", "auto") != "none":
        log("[Etapa 4] densificacao (MVS)")
        try:
            dense_xyz, dense_rgb, dense_method = step_4_dense.densify(
                model_dir, image_dir, os.path.join(out_dir, "mvs"), dcfg.get("method", "auto"),
                dcfg.get("max_image_size", 1200), dcfg.get("sgbm_num_refs", 60), log,
                colmap_exe=step_4_sfm.find_colmap(cfg.get("sfm", {}).get("colmap_exe", "auto")))
            if dense_xyz is not None and len(dense_xyz):
                dense_xyz, dense_rgb, _ = step_4_room.clean_cloud(dense_xyz, dense_rgb, room,
                                                                  crop_margin=room_cfg.get("crop_margin_m", 0.3) * unit)
                write_ply(os.path.join(out_dir, "nuvem_densa.ply"), dense_xyz, dense_rgb)
                log(f"  nuvem densa: {len(dense_xyz)} pontos ({dense_method})")
        except Exception as e:  # a densa e recomendada, nao obrigatoria: nao derruba o pipeline
            log(f"  AVISO: densificacao falhou ({e}); seguindo so com a nuvem esparsa")
            dense_xyz = None
        # downsample (voxel) para abrir leve no MeshLab/Open3D; a nuvem completa continua salva
        voxel = dcfg.get("downsample_voxel_m", 0.01)
        if dense_xyz is not None and len(dense_xyz) and voxel:
            ds_xyz, ds_rgb = step_4_room.voxel_downsample(dense_xyz, dense_rgb, voxel * unit)
            write_ply(os.path.join(out_dir, "nuvem_densa_downsample.ply"), ds_xyz, ds_rgb)
            log(f"  downsample ({voxel * 100:.0f} cm): {len(ds_xyz)} pontos")

    # 5) metricas
    log("[Etapa 5] metricas")
    manual_path = os.path.join(out_dir, "medidas_manuais.json")
    manual = json.load(open(manual_path)) if os.path.exists(manual_path) else {}
    dims = step_5_metrics.dimension_errors(cfg.get("measurements", []), room, manual)
    metrics = step_5_metrics.compute_metrics(
        errors, len(sc["xyz"]), xyz, dense_xyz, room, drift, dims,
        extra={"sfm": sfm_info, "scale": {k: v for k, v in scale_info.items() if k not in ("corners_3d", "edges")},
               "alignment": align, "dense_method": dense_method, "metric": scale is not None})
    step_5_metrics.save_metrics(metrics, os.path.join(out_dir, "metricas.json"), os.path.join(out_dir, "metricas.md"))
    with open(os.path.join(out_dir, "sala.json"), "w") as f:
        json.dump(room, f, indent=1)
    with open(os.path.join(out_dir, "escala.json"), "w") as f:
        json.dump(scale_info, f, indent=1)

    if figures:
        log("[Etapa 4] planta baixa e trajetoria")
        cloud = (dense_xyz, dense_rgb) if dense_xyz is not None and len(dense_xyz) else (xyz, rgb)
        step_4_room.plot_floorplan(cloud[0], cloud[1], sc["centers"], sc["R"], room,
                                   os.path.join(out_dir, "planta_baixa.png"), drift=drift)
        step_4_room.plot_occupancy(cloud[0], room, os.path.join(out_dir, "planta_ocupacao.png"))
        from .render_scene import render_overview
        render_overview(xyz, rgb, sc, room, os.path.join(out_dir, "trajetoria_sobre_nuvem.png"))
        if dense_xyz is not None and len(dense_xyz):
            render_overview(dense_xyz, dense_rgb, sc, room, os.path.join(out_dir, "trajetoria_sobre_nuvem_densa.png"))
    return metrics
