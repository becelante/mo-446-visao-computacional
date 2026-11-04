"""
Extras (opcionais) do enunciado da Variante A sobre results/<video>/ (o video de config.yaml),
cada um numa pasta de results/<video>/extras/:

  1_angulo_paredes/       planos do chao e das paredes por RANSAC e angulo entre paredes vizinhas
  2_casos_dificeis/       paredes lisas (baixa textura): limiar de contraste do SIFT e CLAHE
  3_patchmatch_vs_sgbm/   PatchMatch Stereo (COLMAP, GPU) x SGBM retificado (OpenCV, CPU)
  4_malha_poisson/        malha (Poisson) a partir da nuvem densa, com cores, figura e video

Os dois ultimos partem da nuvem do PatchMatch (results/<video>/mvs/fused.ply), que precisa de CUDA.
As poses das cameras em GLB (visualizacao web) ja saem do run_pipeline.py.
"""
import json
import os
import re
import shutil
import time

import cv2
import numpy as np

from pipeline import config, step_4_dense, step_4_room, step_4_sfm
from pipeline.room_pipeline import postprocess
from pipeline.render_scene import flythrough_video, render_overview
from pipeline.viz_utils import read_ply, write_ply

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _md_table(head, rows):
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    L += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(L)

def extra_casos_dificeis(out, d, cfg, log=print):
    image_dir = os.path.join(out, "quadros")
    names = json.load(open(os.path.join(image_dir, "frames.json")))["names"]
    sfm_cfg, cam_cfg = cfg.get("sfm", {}), cfg.get("camera", {})
    exe = step_4_sfm.find_colmap(sfm_cfg.get("colmap_exe", "auto"))
    runs = [("padrao_0.0067", None, False), ("limiar_0.002", sfm_cfg.get("sift_peak_threshold", 0.002), False),
            ("limiar_0.002+clahe", sfm_cfg.get("sift_peak_threshold", 0.002), True)]
    rows, kps = [], {}
    for name, peak, clahe in runs:
        run_dir = os.path.join(d, name)
        res_path = os.path.join(run_dir, "resultado.json")
        if os.path.exists(res_path):
            res = json.load(open(res_path))
        else:
            # o limiar 0.002 sem CLAHE e a configuracao do pipeline principal: reaproveita o SfM
            same_as_main = (peak == sfm_cfg.get("sift_peak_threshold") and clahe == sfm_cfg.get("clahe", False)
                            and os.path.exists(os.path.join(out, "sfm", "sparse")))
            log(f"[casos dificeis] {name}" + (" (SfM do pipeline principal)" if same_as_main else ""))
            if same_as_main:
                import pycolmap
                rec = pycolmap.Reconstruction(os.path.join(out, "sfm", "sparse"))
                info = json.load(open(os.path.join(out, "sfm", "sfm_info.json")))
                db = os.path.join(out, "sfm", "database.db")
            elif os.path.exists(os.path.join(run_dir, "sfm", "sfm_info.json")):
                import pycolmap
                rec = pycolmap.Reconstruction(os.path.join(run_dir, "sfm", "sparse"))
                info = json.load(open(os.path.join(run_dir, "sfm", "sfm_info.json")))
                db = os.path.join(run_dir, "sfm", "database.db")
            else:
                rec, info = step_4_sfm.run_sfm(
                    image_dir, os.path.join(run_dir, "sfm"), names, sfm_cfg.get("matching", "exhaustive"),
                    sfm_cfg.get("sequential_overlap", 15), sfm_cfg.get("loop_detection", False),
                    sfm_cfg.get("vocab_tree"), cam_cfg.get("model", "SIMPLE_RADIAL"), cam_cfg.get("focal_px"),
                    sfm_cfg.get("max_image_size", 1600), sfm_cfg.get("max_num_features", 8192),
                    sfm_cfg.get("use_gpu", "auto"), peak_threshold=peak, clahe=clahe,
                    mapper=sfm_cfg.get("mapper", "incremental"), colmap_exe=exe, log=log)
                db = os.path.join(run_dir, "sfm", "database.db")
                json.dump(info, open(os.path.join(run_dir, "sfm", "sfm_info.json"), "w"), indent=1)
            m = postprocess(rec, info, image_dir, names, os.path.join(run_dir, "pos"), cfg, do_dense=False,
                            figures=False, log=lambda *a: None)
            kp = step_4_sfm.keypoints_per_image(db)
            res = {"config": name, "peak_threshold": peak or 0.0067, "clahe": clahe,
                   "kps_mean": float(np.mean([len(v) for v in kp.values()])),
                   "kps_p10": float(np.percentile([len(v) for v in kp.values()], 10)),
                   "verified_pairs": info.get("n_verified_pairs"), "registered": info["n_reg_images"],
                   "frames": len(names), "points": info["n_points3d"], "track_len": info.get("mean_track_length"),
                   "reproj_px": m["reproj_mean_px"], "scale_found": m.get("metric", False),
                   "drift_cm": (m.get("loop_drift") or {}).get("drift_cm"),
                   "dims": {x["name"]: x.get("rel_err_pct") for x in m["dimensions"]}}
            os.makedirs(run_dir, exist_ok=True)
            json.dump(res, open(res_path, "w"), indent=1)
            np.savez_compressed(os.path.join(run_dir, "keypoints.npz"), **{k: v for k, v in kp.items()})
        kps[name] = dict(np.load(os.path.join(run_dir, "keypoints.npz")))
        rows.append(res)

    base = kps[runs[0][0]]
    frame = min(base, key=lambda n: len(base[n]))
    img = cv2.imread(os.path.join(image_dir, frame))
    panels = []
    for (name, _, _), res in zip(runs, rows):
        vis = img.copy()
        for x, y in kps[name][frame]:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{name}: {len(kps[name][frame])} keypoints", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                    (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(vis, f"{name}: {len(kps[name][frame])} keypoints", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                    (255, 255, 255), 3, cv2.LINE_AA)
        panels.append(cv2.resize(vis, None, fx=0.5, fy=0.5))
    cv2.imwrite(os.path.join(d, f"keypoints_{os.path.splitext(frame)[0]}.jpg"), np.hstack(panels))

    dim_names = list(rows[0]["dims"])
    fmt = lambda v, f="{:.1f}": "-" if v is None else f.format(v)
    table = _md_table(
        ["Config.", "Keypoints/img (media; p10)", "Pares verificados", "Registradas", "Pontos", "Trilha media",
         "Reproj. (px)", "Escala"] + [f"erro {n} (%)" for n in dim_names],
        [[r["config"], f"{r['kps_mean']:.0f}; {r['kps_p10']:.0f}", r["verified_pairs"],
          f"{r['registered']}/{r['frames']}", r["points"], fmt(r["track_len"], "{:.2f}"), fmt(r["reproj_px"], "{:.3f}"),
          "sim" if r["scale_found"] else "nao"] + [fmt(r["dims"].get(n)) for n in dim_names] for r in rows])
    with open(os.path.join(d, "casos_dificeis.md"), "w", encoding="utf-8") as f:
        f.write("# Extra: casos dificeis (paredes lisas, baixa textura)\n\n"
                "Mesmo video, mesmo emparelhamento exaustivo e mapeador global na GPU; muda so a extracao do SIFT:\n"
                "o limiar de contraste (`peak_threshold`, padrao do COLMAP 0.0067) e a equalizacao local de\n"
                "contraste (CLAHE) antes de extrair. Limiar menor aceita cantos de baixo contraste das paredes\n"
                "lisas; o CLAHE realca a textura fraca (cores e escala continuam vindo das imagens originais).\n\n"
                + table + "\n\n"
                f"Figura: `keypoints_{os.path.splitext(frame)[0]}.jpg` (quadro com menos keypoints no padrao).\n")
    log("\n" + table)

def _voxels(xyz, v=0.05):
    return set(map(tuple, np.floor(xyz / v).astype(np.int64)))

def _plane_noise(xyz, room, band=0.10):
    d = []
    for w in room["walls"]:
        ax = 0 if w["axis"] == "x" else 1
        other = 1 - ax
        sel = ((np.abs(xyz[:, ax] - w["offset"]) < band) & (xyz[:, other] > w["span"][0])
               & (xyz[:, other] < w["span"][1]) & (xyz[:, 2] > w["z_span"][0]) & (xyz[:, 2] < w["z_span"][1]))
        d.append(np.abs(xyz[sel, ax] - w["offset"]))
    sel = np.abs(xyz[:, 2] - room["floor_z"]) < band
    d_floor = np.abs(xyz[sel, 2] - room["floor_z"])
    d_wall = np.concatenate(d) if d else np.zeros(0)
    stat = lambda a: (100 * float(np.median(a)), 100 * float(np.sqrt(np.mean(a ** 2))), int(len(a))) if len(a) else (None, None, 0)
    return stat(d_wall), stat(d_floor)

def _colmap_times(log_path):
    t = [float(x) for x in re.findall(r"Elapsed time: ([\d.]+) \[minutes\]", open(log_path, errors="ignore").read())]
    return t

def extra_densa(out, d, cfg, log=print):
    room = json.load(open(os.path.join(out, "sala.json")))
    dcfg, rcfg = cfg.get("dense", {}), cfg.get("room", {})
    n_refs = 10000  # todas as imagens registradas como referencia, como no PatchMatch
    pm_xyz, pm_rgb = read_ply(os.path.join(out, "nuvem_densa.ply"))
    pm_raw = os.path.join(out, "mvs", "fused.ply")
    pm_n_raw = len(read_ply(pm_raw)[0]) if os.path.exists(pm_raw) else None
    times = _colmap_times(os.path.join(out, "mvs", "colmap.log")) if os.path.exists(os.path.join(out, "mvs", "colmap.log")) else []
    pm_time = sum(t for t in times if t > 0.5) * 60 if times else None  # patch_match_stereo + stereo_fusion

    sg_ply = os.path.join(d, "nuvem_sgbm.ply")
    info_path = os.path.join(d, "sgbm_info.json")
    if os.path.exists(sg_ply) and os.path.exists(info_path):
        sg_xyz, sg_rgb = read_ply(sg_ply)
        sg_info = json.load(open(info_path))
    else:
        work = os.path.join(d, "_mvs_sgbm")
        log(f"[densa] SGBM em CPU (todas as imagens como referencia, {dcfg.get('max_image_size', 1200)} px)")
        t0 = time.time()
        raw_xyz, raw_rgb = step_4_dense.dense_sgbm(os.path.join(out, "modelo_metrico"), os.path.join(out, "quadros"),
                                                   work, dcfg.get("max_image_size", 1200),
                                                   n_refs, log=log)
        n_pairs = len(step_4_dense._choose_pairs(__import__("pycolmap").Reconstruction(os.path.join(work, "sparse")), n_refs))
        sg_time = time.time() - t0
        sg_xyz, sg_rgb, _ = step_4_room.clean_cloud(raw_xyz, raw_rgb, room, crop_margin=rcfg.get("crop_margin_m", 0.3))
        write_ply(sg_ply, sg_xyz, sg_rgb)
        sg_info = {"time_s": sg_time, "n_raw": int(len(raw_xyz)), "n_pairs": n_pairs}
        json.dump(sg_info, open(info_path, "w"), indent=1)
        shutil.rmtree(work, ignore_errors=True)  # imagens retificadas/undistorcidas: so ocupam espaco

    sc = step_4_sfm.scene_arrays(__import__("pycolmap").Reconstruction(os.path.join(out, "modelo_metrico")))
    exc = set(room.get("excluded_cameras", []))
    sc = step_4_sfm.subset_cameras(sc, [n not in exc for n in sc["names"]])
    vp, vs = _voxels(pm_xyz), _voxels(sg_xyz)
    rows = []
    for name, xyz, n_raw, t, vox, other in [("PatchMatch (COLMAP, GPU)", pm_xyz, pm_n_raw, pm_time, vp, vs),
                                           ("SGBM retificado (OpenCV, CPU)", sg_xyz, sg_info["n_raw"], sg_info["time_s"], vs, vp)]:
        (wm, wr, wn), (fm, fr, fn) = _plane_noise(xyz, room)
        area = len(vox) * 0.05 ** 2
        rows.append([name, n_raw if n_raw is not None else "-", len(xyz), f"{area:.1f}",
                     f"{len(xyz) / max(area, 1e-9):.0f}", f"{100 * len(vox & other) / max(len(vox), 1):.0f}%",
                     "-" if wm is None else f"{wm:.2f} / {wr:.2f}", "-" if fm is None else f"{fm:.2f} / {fr:.2f}",
                     "-" if t is None else f"{t / 60:.1f}"])
    table = _md_table(["Metodo", "Pontos (bruto)", "Pontos (limpos)", "Superficie coberta (m2, voxels 5 cm)",
                       "Densidade (pts/m2 de superficie)", "Voxels tambem vistos pelo outro",
                       "Dist. as paredes mediana / RMS (cm)", "Dist. ao chao mediana / RMS (cm)", "Tempo (min)"], rows)

    imgs = []
    for name, xyz, rgb in [("PatchMatch", pm_xyz, pm_rgb), ("SGBM", sg_xyz, sg_rgb)]:
        p = os.path.join(d, f"vista_{name.lower()}.png")
        render_overview(xyz, rgb, sc, room, p)
        im = cv2.imread(p)
        cv2.putText(im, name, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
        imgs.append(im)
    cv2.imwrite(os.path.join(d, "patchmatch_vs_sgbm.png"), np.hstack(imgs))
    with open(os.path.join(d, "patchmatch_vs_sgbm.md"), "w", encoding="utf-8") as f:
        f.write("# Extra: PatchMatch Stereo x SGBM\n\n"
                "Os dois sobre o mesmo modelo metrico e alinhado e com a mesma limpeza (outliers + recorte da sala).\n"
                "- PatchMatch: COLMAP `patch_match_stereo` (consistencia geometrica, ~20 vistas por referencia) + "
                "`stereo_fusion`, na GPU, todas as imagens registradas.\n"
                f"- SGBM: cada imagem registrada como referencia, com o vizinho de base lateral adequada; "
                f"{sg_info.get('n_pairs', '?')} pares validos, retificados com as poses do SfM (`cv2.stereoRectify`), "
                "Semi-Global Block Matching e fusao por voxel confirmado por >= 2 pares, em CPU.\n\n"
                + table + "\n\n"
                "Distancia aos planos: pontos a ate 10 cm dos planos das paredes/chao encontrados por RANSAC; "
                "mede o ruido da superficie (menor = mais plana). Figura: `patchmatch_vs_sgbm.png`.\n")
    log("\n" + table)

def extra_angulos(out, d, cfg, log=print):
    room = json.load(open(os.path.join(out, "sala.json")))
    xyz, _ = read_ply(os.path.join(out, "nuvem_esparsa.ply"))
    walls = room["walls"]
    (x0, x1), (y0, y1) = room["bounds"][0], room["bounds"][1]
    ext = {(w["axis"], round(w["offset"], 2)) for w in walls if
           (w["axis"] == "x" and min(abs(w["offset"] - x0), abs(w["offset"] - x1)) < 1e-6) or
           (w["axis"] == "y" and min(abs(w["offset"] - y0), abs(w["offset"] - y1)) < 1e-6)}

    fig, ax = plt.subplots(figsize=(8, 8))
    band = (xyz[:, 2] > 0.3) & (xyz[:, 2] < 1.8)
    ax.scatter(xyz[band, 0], xyz[band, 1], s=1, c="0.75", label="nuvem esparsa (z 0,3-1,8 m)")
    rows = []
    for k, w in enumerate(walls):
        n = np.array(w["normal"])
        tilt = float(np.degrees(np.arcsin(min(1.0, abs(n[2])))))
        yaw = float(np.degrees(np.arctan2(n[1], n[0])))
        is_ext = (w["axis"], round(w["offset"], 2)) in ext
        a, b = w["span"]
        if w["axis"] == "x":
            P = np.array([[w["offset"], a], [w["offset"], b]])
        else:
            P = np.array([[a, w["offset"]], [b, w["offset"]]])
        ax.plot(P[:, 0], P[:, 1], "-", lw=3 if is_ext else 1.5, color="C3" if is_ext else "C0")
        ax.text(*P.mean(0), f"P{k}", fontsize=9, color="C3" if is_ext else "C0")
        rows.append([f"P{k}", w["axis"], f"{w['offset']:.3f}", w["n_inliers"], f"{yaw:.1f}", f"{tilt:.2f}",
                     "sim" if is_ext else "nao"])
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", lw=1, label="limites (paredes extremas)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Planos verticais (RANSAC); vermelho = paredes usadas nas medidas")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "paredes_ransac.png"), dpi=150)
    plt.close(fig)

    ang = room.get("wall_angles", [])
    errs = [a["error_deg"] for a in ang]
    with open(os.path.join(d, "angulo_paredes.md"), "w", encoding="utf-8") as f:
        f.write("# Extra: planos do chao/paredes (RANSAC) e angulo entre paredes vizinhas\n\n"
                "O chao e o primeiro plano horizontal (normal a ate 10 graus de z) e define z = 0; as paredes sao\n"
                "planos verticais detectados por RANSAC sequencial (limiar "
                f"{100 * cfg.get('room', {}).get('plane_thresh_m', 0.03):.0f} cm) na faixa de 15-85% da altura. O angulo entre duas paredes vizinhas e\n"
                "arccos(|n1 . n2|) e o esperado e 90 graus.\n\n"
                "## Angulo entre as paredes extremas (as que definem largura e comprimento)\n\n"
                + _md_table(["Paredes", "Angulo (graus)", "Erro vs 90 (graus)"],
                            [[a["walls"], f"{a['angle_deg']:.2f}", f"{a['error_deg']:.2f}"] for a in ang])
                + (f"\n\nErro medio: {np.mean(errs):.2f} graus; maximo: {np.max(errs):.2f} graus.\n" if errs else "\n")
                + "\n## Todos os planos verticais\n\n"
                + _md_table(["Plano", "Eixo", "Posicao (m)", "Inliers", "Direcao da normal (graus)",
                             "Desvio da vertical (graus)", "Extrema"], rows)
                + "\n\nFigura: `paredes_ransac.png`.\n")

def extra_malha(out, d, cfg, depth=10, density_q=0.08, log=print):
    import open3d as o3d
    room = json.load(open(os.path.join(out, "sala.json")))
    margin = cfg.get("room", {}).get("crop_margin_m", 0.3)
    t0 = time.time()
    pcd = o3d.io.read_point_cloud(os.path.join(out, "mvs", "fused.ply"))
    s, R, t, _ = step_4_dense._sim3_between_models(os.path.join(out, "mvs", "sparse"),
                                                   os.path.join(out, "modelo_metrico"))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t
    pcd.transform(T)
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) @ R.T)
    (x0, x1), (y0, y1), (z0, z1) = room["bounds"]
    box = o3d.geometry.AxisAlignedBoundingBox([x0 - margin, y0 - margin, z0 - margin], [x1 + margin, y1 + margin, z1 + margin])
    pcd = pcd.crop(box)
    pcd, _ = pcd.remove_statistical_outlier(16, 2.0)
    pcd = pcd.voxel_down_sample(0.01)
    n_in = len(pcd.points)
    log(f"[malha] Poisson (profundidade {depth}) sobre {n_in} pontos...")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_q))
    mesh = mesh.crop(box)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    t_mesh = time.time() - t0
    o3d.io.write_triangle_mesh(os.path.join(d, "malha_poisson.ply"), mesh)

    smp = mesh.sample_points_uniformly(1_000_000)
    xyz, rgb = np.asarray(smp.points), (np.asarray(smp.colors) * 255).astype(np.uint8)
    sc = step_4_sfm.scene_arrays(__import__("pycolmap").Reconstruction(os.path.join(out, "modelo_metrico")))
    exc = set(room.get("excluded_cameras", []))
    sc = step_4_sfm.subset_cameras(sc, [n not in exc for n in sc["names"]])
    render_overview(xyz, rgb, sc, room, os.path.join(d, "malha_vista_geral.png"))
    rcfg = cfg["render"]
    flythrough_video(xyz, rgb, sc, room, os.path.join(d, "video_malha.mp4"), seconds=rcfg["seconds"], fps=rcfg["fps"],
                     size=(rcfg["width"], rcfg["height"]), title=rcfg["title"] + " - malha (Poisson)", splat_m=0.012,
                     max_px=10, log=log)
    area = mesh.get_surface_area()
    with open(os.path.join(d, "malha_poisson.md"), "w", encoding="utf-8") as f:
        f.write("# Extra: malha por reconstrucao de Poisson\n\n"
                "Entrada: nuvem do PatchMatch (com normais do COLMAP) no referencial metrico, recortada na sala\n"
                f"(folga de {margin * 100:.0f} cm), sem outliers (16 vizinhos, 2 desvios) e reduzida a 1 cm: {n_in} pontos.\n"
                f"Poisson do Open3D com profundidade {depth} (celula de ~{max(x1 - x0, y1 - y0) / 2 ** depth * 100:.1f} cm); "
                f"vertices com densidade abaixo do quantil {density_q:.0%} removidos (superficie\n"
                "extrapolada longe de qualquer ponto). As cores vem dos pontos.\n\n"
                + _md_table(["Vertices", "Triangulos", "Area da malha (m2)", "Tempo (s)"],
                            [[len(mesh.vertices), len(mesh.triangles), f"{area:.1f}", f"{t_mesh:.0f}"]])
                + "\n\nA malha fecha buracos das paredes lisas e do piso por interpolacao: serve para visualizar, "
                "mas as medidas continuam vindo dos planos ajustados aos pontos.\n\n"
                "Arquivos: `malha_poisson.ply` (abre no MeshLab/Open3D/Blender), `malha_vista_geral.png`"
                + ", `video_malha.mp4`.\n")
    log(f"  malha: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangulos, {area:.1f} m2")

def main():
    cfg = config.load()
    out = config.results_dir(cfg)
    jobs = [("1_angulo_paredes", extra_angulos), ("2_casos_dificeis", extra_casos_dificeis)]
    if os.path.exists(os.path.join(out, "mvs", "fused.ply")):
        jobs += [("3_patchmatch_vs_sgbm", extra_densa), ("4_malha_poisson", extra_malha)]
    else:
        print("sem nuvem do PatchMatch (results/<video>/mvs/fused.ply): pulando PatchMatch x SGBM e malha")
    for name, fn in jobs:
        d = os.path.join(out, "extras", name)
        os.makedirs(d, exist_ok=True)
        print(f"=== extra: {name}", flush=True)
        fn(out, d, cfg)

if __name__ == "__main__":
    main()
