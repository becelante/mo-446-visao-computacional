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

def manual_measurements(rec, points, floor_z, log=print):
    registered = set(rec.reg_image_ids())
    out = {}
    for name, spec in (points or {}).items():
        n = 1 if spec.get("to_floor") else 2
        obs = [[] for _ in range(n)]
        for frame, pts in spec["clicks"].items():
            img = rec.find_image_with_name(frame)
            if img is not None and img.image_id in registered:
                for k in range(n):
                    obs[k].append((img, np.array(pts[k], float)))
        P = [step_4_scale_align.triangulate_multiview(o, rec, max_error_px=8.0, min_views=2) for o in obs]
        if any(p is None for p in P):
            log(f"  AVISO: {name}: pontos marcados nao triangulados")
            continue
        dist = P[0][0][2] - floor_z if n == 1 else np.linalg.norm(P[0][0] - P[1][0])
        out[name] = {"distance_m": float(dist), "reproj_px": [p[1] for p in P], "n_views": [p[2] for p in P]}
        log(f"  {name}: {100 * dist:.1f} cm (reprojecao {', '.join(f'{p[1]:.2f}' for p in P)} px)")
    return out

def postprocess(rec, sfm_info, image_dir, all_names, out_dir, cfg, do_dense=True, figures=True, log=print):
    os.makedirs(out_dir, exist_ok=True)

    log("[Etapa 4] escala metrica")
    scale, scale_info = step_4_scale_align.estimate_scale(rec, image_dir, cfg["scale"], log)
    if scale:
        step_4_sfm.apply_sim3(rec, scale)

    log("[Etapa 4] alinhamento da sala (chao e paredes por RANSAC)")
    room_cfg = cfg["room"]
    align = step_4_scale_align.align_room(rec, metric=scale is not None, plane_thresh=room_cfg["plane_thresh_m"],
                                          log=log)
    model_dir = os.path.join(out_dir, "modelo_metrico")
    os.makedirs(model_dir, exist_ok=True)
    rec.write(model_dir)
    os.makedirs(os.path.join(model_dir, "txt"), exist_ok=True)
    rec.write_text(os.path.join(model_dir, "txt"))

    sc = step_4_sfm.scene_arrays(rec)
    errors = step_4_sfm.reprojection_errors(rec)
    unit = 1.0 if scale else align["unit"]

    log("[Etapa 4] limpeza da nuvem e medidas da sala")
    base = (sc["track_len"] >= 2)
    xyz0, rgb0, _ = step_4_room.clean_cloud(sc["xyz"], sc["rgb"], None, extra_mask=base)
    # teto: acima da altura tipica da camera na mao (mediana robusta a poses ruins)
    min_ceil = float(np.median(sc["centers"][:, 2])) + 0.7 * unit if len(sc["centers"]) else 1.5 * unit
    # as cameras ficam dentro da sala: separa as paredes opostas
    mid_xy = np.median(sc["centers"][:, :2], axis=0) if len(sc["centers"]) else None
    room = step_4_room.measure_room(xyz0, thresh=room_cfg["plane_thresh_m"] * unit, min_ceiling_z=min_ceil,
                                    center_xy=mid_xy)
    xyz, rgb, _ = step_4_room.clean_cloud(sc["xyz"], sc["rgb"], room, extra_mask=base,
                                          crop_margin=room_cfg["crop_margin_m"] * unit)
    room = step_4_room.measure_room(xyz, thresh=room_cfg["plane_thresh_m"] * unit, min_ceiling_z=min_ceil,
                                    center_xy=mid_xy)
    log(f"  sala: {room['x_extent_m']:.3f} x {room['y_extent_m']:.3f} m"
        + (f", altura {room['height_m']:.3f} m" if room["height_m"] else ", teto nao detectado"))
    # cameras com pose incoerente (fora da sala / saltos) ficam fora do trajeto, da deriva e das figuras
    cam_ok = step_4_room.valid_cameras(sc["centers"], room, unit)
    room["excluded_cameras"] = [n for n, keep in zip(sc["names"], cam_ok) if not keep]
    if room["excluded_cameras"]:
        log(f"  {len(room['excluded_cameras'])} de {len(cam_ok)} cameras descartadas do trajeto (fora da sala ou salto isolado)")
    sc = step_4_sfm.subset_cameras(sc, cam_ok)
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

    dense_xyz = dense_rgb = None
    dense_method = None
    dcfg = cfg["dense"]
    if do_dense and dcfg["method"] != "none":
        log("[Etapa 4] densificacao (MVS)")
        try:
            dense_xyz, dense_rgb, dense_method = step_4_dense.densify(
                model_dir, image_dir, os.path.join(out_dir, "mvs"), dcfg["method"], dcfg["max_image_size"],
                dcfg["sgbm_num_refs"], log, colmap_exe=step_4_sfm.find_colmap(cfg["sfm"]["colmap_exe"]),
                fusion={"min_num_pixels": dcfg["fusion_min_num_pixels"]}, num_src_images=dcfg["num_src_images"])
            if dense_xyz is not None and len(dense_xyz):
                dense_xyz, dense_rgb, _ = step_4_room.clean_cloud(dense_xyz, dense_rgb, room,
                                                                  crop_margin=room_cfg["crop_margin_m"] * unit)
                write_ply(os.path.join(out_dir, "nuvem_densa.ply"), dense_xyz, dense_rgb)
                log(f"  nuvem densa: {len(dense_xyz)} pontos ({dense_method})")
        except Exception as e:  # a densa e recomendada, nao obrigatoria: nao derruba o pipeline
            log(f"  AVISO: densificacao falhou ({e}); seguindo so com a nuvem esparsa")
            dense_xyz = None
        voxel = dcfg["downsample_voxel_m"]
        if dense_xyz is not None and len(dense_xyz) and voxel:
            ds_xyz, ds_rgb = step_4_room.voxel_downsample(dense_xyz, dense_rgb, voxel * unit)
            write_ply(os.path.join(out_dir, "nuvem_densa_downsample.ply"), ds_xyz, ds_rgb)
            log(f"  downsample ({voxel * 100:.0f} cm): {len(ds_xyz)} pontos")

    log("[Etapa 5] metricas")
    manual = manual_measurements(rec, cfg.get("manual_points"), room["floor_z"], log)
    dims = step_5_metrics.dimension_errors(cfg["measurements"], room, manual, metric=scale is not None)
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
        real = {d["source"][5:]: d["real_cm"] for d in dims if d["source"] in ("auto:width", "auto:length")}
        x_is_length = room["x_extent_m"] >= room["y_extent_m"]
        real_xy = {ax: real.get("length" if (ax == "x") == x_is_length else "width") for ax in "xy"}
        step_4_room.plot_floorplan(xyz, sc["centers"], sc["R"], room, os.path.join(out_dir, "planta_baixa.png"),
                                   drift=drift, real_cm={k: v for k, v in real_xy.items() if v is not None})
        step_4_room.plot_occupancy(cloud[0], room, os.path.join(out_dir, "planta_ocupacao.png"))
        step_5_metrics.plot_reprojection_histogram(errors, os.path.join(out_dir, "hist_reprojecao.png"))
        from .render_scene import render_overview
        render_overview(xyz, rgb, sc, room, os.path.join(out_dir, "trajetoria_sobre_nuvem.png"))
        if dense_xyz is not None and len(dense_xyz):
            render_overview(dense_xyz, dense_rgb, sc, room, os.path.join(out_dir, "trajetoria_sobre_nuvem_densa.png"))
    return metrics
