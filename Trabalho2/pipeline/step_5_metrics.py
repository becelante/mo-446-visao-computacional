"""
Etapa 5: metricas da Variante A.

  - Erro de reprojecao (media/mediana, px) de todas as observacoes do modelo final.
  - Numero de pontos esparsos/densos e densidade (pontos/m2): por area de planta
    (largura x comprimento) e por area de superficie observada (voxels de 5 cm ocupados).
  - Erro metrico de cada dimensao: |L^ - L| (cm) e e = |L^ - L| / L.
  - Deriva do laco: delta = ||C_N - C_1|| em cm e em % do comprimento do trajeto.
"""
import json

import numpy as np


def loop_drift(centers, names, all_names=None):
    """
    Deriva do laco entre a primeira e a ultima camera. Se o primeiro/ultimo quadro
    do video nao foi registrado, usa os extremos registrados e avisa.
    """
    if len(centers) < 2:
        return None
    length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    delta = float(np.linalg.norm(centers[-1] - centers[0]))
    info = {"first": names[0], "last": names[-1], "drift_m": delta, "drift_cm": 100 * delta,
            "trajectory_length_m": length, "drift_pct": 100 * delta / length if length > 0 else float("nan")}
    if all_names:
        info["first_is_first_frame"] = names[0] == all_names[0]
        info["last_is_last_frame"] = names[-1] == all_names[-1]
    return info


def surface_area_voxels(xyz, voxel=0.05):
    """Area aproximada da superficie observada: voxels ocupados x area de uma face."""
    if len(xyz) == 0:
        return 0.0
    n = len(np.unique(np.floor(xyz / voxel).astype(np.int64), axis=0))
    return n * voxel * voxel


def dimension_errors(measurements, room, manual=None):
    """
    measurements: lista de {name, real_cm, model} em que `model` e
      'auto:width' | 'auto:length' | 'auto:height' | 'auto:x' | 'auto:y'  (medido pelo pipeline)
      'manual:<nome>' (medido com measure.py, salvo em <saida>/medidas_manuais.json)
      um numero (cm) medido a mao no modelo (p.ex., no MeshLab).
    """
    manual = manual or {}
    auto = {"width": room.get("width_m"), "length": room.get("length_m"), "height": room.get("height_m"),
            "x": room.get("x_extent_m"), "y": room.get("y_extent_m")}
    rows = []
    for m in measurements or []:
        real = float(m["real_cm"])
        src = m.get("model", "")
        est = None
        if isinstance(src, (int, float)):
            est = float(src)
        elif str(src).startswith("auto:"):
            v = auto.get(src.split(":", 1)[1])
            est = 100 * v if v is not None else None
        elif str(src).startswith("manual:"):
            v = manual.get(src.split(":", 1)[1])
            est = 100 * v["distance_m"] if v is not None else None
        row = {"name": m["name"], "real_cm": real, "model_cm": est, "source": str(src)}
        if est is not None:
            row["abs_err_cm"] = abs(est - real)
            row["rel_err_pct"] = 100 * abs(est - real) / real
        rows.append(row)
    return rows


def compute_metrics(rec_errors, n_sparse_raw, sparse_clean, dense_xyz, room, drift, dims, extra=None):
    m = {
        "reproj_mean_px": float(np.mean(rec_errors)) if len(rec_errors) else None,
        "reproj_median_px": float(np.median(rec_errors)) if len(rec_errors) else None,
        "reproj_p90_px": float(np.percentile(rec_errors, 90)) if len(rec_errors) else None,
        "n_observations": int(len(rec_errors)),
        "n_sparse_points_raw": int(n_sparse_raw),
        "n_sparse_points_clean": int(len(sparse_clean)),
        "n_dense_points": int(len(dense_xyz)) if dense_xyz is not None else None,
        "floor_area_m2": room.get("floor_area_m2"),
        "surface_area_sparse_m2": surface_area_voxels(sparse_clean),
    }
    area = room.get("floor_area_m2") or float("nan")
    m["density_sparse_per_m2_floor"] = len(sparse_clean) / area
    m["density_sparse_per_m2_surface"] = len(sparse_clean) / max(m["surface_area_sparse_m2"], 1e-9)
    if dense_xyz is not None and len(dense_xyz):
        m["surface_area_dense_m2"] = surface_area_voxels(dense_xyz)
        m["density_dense_per_m2_floor"] = len(dense_xyz) / area
        m["density_dense_per_m2_surface"] = len(dense_xyz) / max(m["surface_area_dense_m2"], 1e-9)
    m["loop_drift"] = drift
    m["dimensions"] = dims
    m["room"] = {k: room[k] for k in ("width_m", "length_m", "height_m", "x_extent_m", "y_extent_m",
                                     "x_source", "y_source", "floor_area_m2", "wall_angles")}
    if extra:
        m.update(extra)
    return m


def _f(v, fmt="{:.2f}"):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def metrics_markdown(m):
    """Tabelas prontas para colar no relatorio."""
    L = ["# Metricas (Etapa 5, Variante A)", ""]
    sfm = m.get("sfm", {})
    L += ["## Reconstrucao", "", "| Metrica | Valor |", "|---|---|"]
    if sfm:
        L.append(f"| Imagens registradas | {sfm.get('n_reg_images')} / {sfm.get('n_images_input')} |")
        L.append(f"| Emparelhamento | {sfm.get('matching')}{' + deteccao de laco' if sfm.get('loop_detection') and sfm.get('matching') == 'sequential' else ''} |")
        L.append(f"| Comprimento medio das trilhas | {_f(sfm.get('mean_track_length'))} |")
    L.append(f"| Erro de reprojecao medio / mediano (px) | {_f(m['reproj_mean_px'], '{:.3f}')} / {_f(m['reproj_median_px'], '{:.3f}')} |")
    L.append(f"| Observacoes | {m['n_observations']} |")
    L.append(f"| Pontos esparsos (bruto / limpo) | {m['n_sparse_points_raw']} / {m['n_sparse_points_clean']} |")
    if m.get("n_dense_points") is not None:
        L.append(f"| Pontos densos ({m.get('dense_method', '')}) | {m['n_dense_points']} |")
    L.append(f"| Area da planta (m2) | {_f(m['floor_area_m2'])} |")
    L.append(f"| Densidade esparsa (pts/m2 de planta; pts/m2 de superficie) | {_f(m['density_sparse_per_m2_floor'], '{:.0f}')}; {_f(m['density_sparse_per_m2_surface'], '{:.0f}')} |")
    if m.get("density_dense_per_m2_floor") is not None:
        L.append(f"| Densidade densa (pts/m2 de planta; pts/m2 de superficie) | {_f(m['density_dense_per_m2_floor'], '{:.0f}')}; {_f(m['density_dense_per_m2_surface'], '{:.0f}')} |")
    sc = m.get("scale", {})
    if sc.get("scale"):
        L.append(f"| Escala ({sc['method']}) | {sc['scale']:.5f} m/unid. ({sc.get('n_edges_used')} arestas, desvio {100 * sc['scale_rel_std']:.2f}%) |")
    L.append("")

    L += ["## Erro metrico das dimensoes", "", "| Dimensao | Real L (cm) | Modelo L^ (cm) | abs(L^ - L) (cm) | e (%) |", "|---|---|---|---|---|"]
    for d in m["dimensions"]:
        L.append(f"| {d['name']} | {_f(d['real_cm'], '{:.1f}')} | {_f(d['model_cm'], '{:.1f}')} | "
                 f"{_f(d.get('abs_err_cm'), '{:.1f}')} | {_f(d.get('rel_err_pct'))} |")
    L.append("")
    dr = m.get("loop_drift")
    if dr:
        L += ["## Deriva do laco", "",
              f"- C1 = `{dr['first']}`, CN = `{dr['last']}`",
              f"- delta = ||CN - C1|| = **{dr['drift_cm']:.1f} cm**",
              f"- comprimento do trajeto = {dr['trajectory_length_m']:.2f} m -> **{dr['drift_pct']:.2f}%** do trajeto", ""]
        if not (dr.get("first_is_first_frame", True) and dr.get("last_is_last_frame", True)):
            L.append("> Atencao: o primeiro ou o ultimo quadro nao foi registrado; a deriva usa os extremos registrados.\n")
    if m["room"].get("wall_angles"):
        L += ["## Extra: angulo entre paredes vizinhas", "", "| Paredes | Angulo (graus) | Erro vs 90 (graus) |", "|---|---|---|"]
        for a in m["room"]["wall_angles"]:
            L.append(f"| {a['walls']} | {a['angle_deg']:.2f} | {a['error_deg']:.2f} |")
        L.append("")
    return "\n".join(L)


def save_metrics(m, json_path, md_path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(json_path, "w") as f:
        json.dump(m, f, indent=2, default=conv, ensure_ascii=False)
    with open(md_path, "w") as f:
        f.write(metrics_markdown(m))
