"""
Pipeline da Variante A (ambiente interno), das features a planta baixa, com cache em disco.

O cache guarda, por detector, as features de cada quadro e a geometria de cada par ja casado
(indexados pelo nome do quadro), e as deteccoes do tabuleiro: as variantes da ablacao
(sequencial, com laco, menos quadros) reaproveitam tudo e so casam os pares que faltam.
"""
import json
import pickle
import time

import numpy as np
import cv2

from . import config
from .step_0_frames import Subset
from .step_1_keypoints_detector import array_to_keypoints, detect_features_all, keypoints_to_array
from .step_2_matching import Matcher, select_pairs
from .step_3_epipolar import build_connectivity_matrix, build_intrinsics
from .step_4_postprocessing import (align_to_floor, camera_centers, clean_point_cloud, draw_floor_plan,
                                    loop_drift, plane_measurements, point_density, room_dimensions,
                                    save_cameras, to_open3d)
from .step_4_sfm import (checkerboard_scale, detect_checkerboards, reconstruction_stats, run_sfm,
                         scale_reconstruction, transform_reconstruction)


def _load_pickle(path, params):
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("params") == params:
            return data
    return {"params": params}


def _save_pickle(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_features(paths, images, method, log=print):
    """Features de cada quadro (cache por nome do quadro e parametros do detector)."""
    params = {"method": method, "max_kp": config.MAX_KEYPOINTS, "contrast": config.SIFT_CONTRAST,
              "equalize": config.EQUALIZE, "sift_uint8": True}
    path = config.cache_dir() / f"features_{method}.pkl"
    cache = _load_pickle(path, params)
    missing = [k for k, p in enumerate(paths) if p.name not in cache]
    if missing:
        t0 = time.perf_counter()
        kps, descs = detect_features_all(Subset(images, missing), method, config.EQUALIZE,
                                         config.MAX_KEYPOINTS, config.SIFT_CONTRAST, config.device())
        log(f"{method}: {len(missing)} quadros em {time.perf_counter() - t0:.1f} s")
        for k, kp, d in zip(missing, kps, descs):  # float16 no disco: metade do tamanho
            cache[paths[k].name] = (keypoints_to_array(kp), d.astype(np.float16) if d is not None and d.dtype == np.float32 else d)
        _save_pickle(path, cache)
    all_kps = [array_to_keypoints(cache[p.name][0]) for p in paths]
    descs = [cache[p.name][1] for p in paths]
    return all_kps, [d.astype(np.float32) if d is not None and d.dtype == np.float16 else d for d in descs]


def _pack(g):
    """Matches como array N x 3 (consulta, treino, distancia): milhoes de cv2.DMatch nao cabem na RAM."""
    g = dict(g)
    g["good"] = np.float32([(m.queryIdx, m.trainIdx, m.distance) for m in g["good"]]).reshape(-1, 3)
    return g


def unpack_matches(good):
    """Array N x 3 de `_pack` -> lista de cv2.DMatch (para desenhar um par)."""
    return [cv2.DMatch(int(q), int(t), float(d)) for q, t, d in good]


def load_geometries(paths, all_kps, all_descs, method, pairs, K, image_size, log=print):
    """Geometria de duas vistas de cada par (cache por nomes dos quadros); casa so os que faltam."""
    # os matches guardam indices de keypoints: dependem tambem dos parametros da deteccao
    params = {"method": method, "ransac": config.RANSAC_THRESH, "min_matches": config.MIN_MATCHES,
              "ratio": config.RATIO_THRESH, "K": K.round(6).tolist(), "max_kp": config.MAX_KEYPOINTS,
              "contrast": config.SIFT_CONTRAST, "equalize": config.EQUALIZE, "cross_check": config.CROSS_CHECK}
    path = config.cache_dir() / f"geometrias_{method}.pkl"
    cache = _load_pickle(path, params)
    names = [p.name for p in paths]
    todo = [(i, j) for i, j in pairs if (names[i], names[j]) not in cache]
    if todo:
        log(f"Casando {len(todo)} pares ({len(pairs) - len(todo)} no cache)...")
        t0 = time.perf_counter()
        matcher = Matcher(all_kps, all_descs, method, image_size, config.RATIO_THRESH, config.CROSS_CHECK,
                          device=config.device())
        _, geometries = build_connectivity_matrix(all_kps, matcher, todo, K, config.RANSAC_THRESH,
                                                  config.MIN_MATCHES, log)
        cache["time_per_pair"] = (time.perf_counter() - t0) / len(todo)
        for (i, j), g in geometries.items():
            cache[(names[i], names[j])] = _pack(g)
        _save_pickle(path, cache)
    return {(i, j): cache[(names[i], names[j])] for i, j in pairs}, cache.get("time_per_pair")


def load_checkerboards(paths, images, pattern, log=print):
    """Cantos do tabuleiro em cada quadro (cache por nome do quadro)."""
    path = config.cache_dir() / "tabuleiros.pkl"
    cache = _load_pickle(path, {"pattern": pattern})
    missing = [k for k, p in enumerate(paths) if p.name not in cache]
    if missing:
        found = detect_checkerboards(images, missing, pattern)
        for k in missing:
            cache[paths[k].name] = found.get(k)
        _save_pickle(path, cache)
        log(f"Tabuleiro procurado em {len(missing)} quadros: achado em {len(found)}")
    return {k: cache[p.name] for k, p in enumerate(paths) if cache[p.name] is not None}


def load_sfm(paths, all_kps, geometries, K, method, pairing, log=print):
    """SfM incremental (cache pelos quadros, detector, pares e parametros); devolve uma copia nova."""
    params = {"frames": [p.name for p in paths], "method": method, "pairing": pairing,
              "window": config.SEQ_WINDOW, "top_k": config.LOOP_TOP_K, "pnp": config.MIN_INLIERS_PNP,
              "reproj": config.MAX_REPROJ_ERROR, "ransac": config.RANSAC_THRESH, "K": K.round(6).tolist(),
              "max_kp": config.MAX_KEYPOINTS, "contrast": config.SIFT_CONTRAST, "equalize": config.EQUALIZE,
              "cross_check": config.CROSS_CHECK}
    path = config.cache_dir() / f"sfm_{method}_{pairing}_{len(paths)}.pkl"
    cache = _load_pickle(path, params)
    if "recon" not in cache:
        # checkpoint ao fim da fase incremental: se o refinamento final cair, recomeca dali
        partial_path = path.with_name(path.stem + "_incremental.pkl")
        partial = _load_pickle(partial_path, params)
        if "recon" in partial:
            log(f"Retomando o SfM do fim da fase incremental ({len(partial['recon'].poses)} cameras)")

        def checkpoint(recon):
            _save_pickle(partial_path, {"params": params, "recon": recon})

        t0 = time.perf_counter()
        cache["recon"] = run_sfm(all_kps, geometries, K, min_inliers_pnp=config.MIN_INLIERS_PNP,
                                 max_reproj_error=config.MAX_REPROJ_ERROR, device=config.device(), verbose=True,
                                 resume=partial.get("recon"), checkpoint=checkpoint)
        cache["time"] = time.perf_counter() - t0
        _save_pickle(path, cache)
    return pickle.loads(pickle.dumps(cache["recon"])), cache["time"]


def choose_keyframes(n, inliers, window, min_inliers):
    """
    Selecao gulosa: a partir de cada quadro-chave i, o proximo e o quadro mais distante (ate
    `window` adiante) que ainda tem >= `min_inliers` inliers de E com i: o maior baseline que ainda
    mantem sobreposicao. Se nenhum chega a isso (giro brusco, borrao), fica o quadro seguinte.
    """
    keys = [0]
    while keys[-1] < n - 1:
        i = keys[-1]
        ok = [j for j in range(i + 1, min(n, i + window + 1)) if inliers.get((i, j), 0) >= min_inliers]
        keys.append(max(ok) if ok else i + 1)
    return keys


def select_keyframes(paths, images, method, all_kps, all_descs, K, image_size, threshold, log=print):
    """
    Quadros-chave a partir dos quadros densos: casa os pares sequenciais (janela DENSE_WINDOW) e
    aplica a selecao gulosa (`choose_keyframes`) com `threshold` inliers. Retorna (indices, info);
    info traz tambem quantos quadros-chave sairiam com outros limiares (para o relatorio).
    """
    pairs = select_pairs(len(paths), "sequential", window=config.DENSE_WINDOW)
    geometries, _ = load_geometries(paths, all_kps, all_descs, method, pairs, K, image_size, log)
    inliers = {p: g["metrics"]["inliers_E"] for p, g in geometries.items()}
    options = {t: choose_keyframes(len(paths), inliers, config.DENSE_WINDOW, t) for t in (40, 100, 200, 400)}
    keys = choose_keyframes(len(paths), inliers, config.DENSE_WINDOW, threshold)
    weak = sum(inliers.get((a, b), 0) < threshold for a, b in zip(keys, keys[1:]))
    log(f"Quadros-chave: {len(keys)} de {len(paths)} (limiar de {threshold} inliers; "
        f"{weak} passos abaixo do limiar)")
    return keys, {"n_dense": len(paths), "n_keyframes": len(keys), "min_inliers": threshold,
                  "weak_steps": weak, "counts_by_threshold": {t: len(k) for t, k in options.items()}}


def reconstruct(paths, images, method=None, pairing=None, keyframes=None, log=print):
    """
    Etapas 2 a 4 e as metricas da Etapa 5: features -> pares -> geometria epipolar -> SfM ->
    escala pelo tabuleiro -> limpeza -> alinhamento ao chao -> planta baixa. Retorna um dict com
    a reconstrucao (metrica e alinhada), a nuvem limpa, as geometrias e as metricas.
    """
    method, pairing = method or config.DETECTOR, pairing or config.PAIRING
    height, width = images[0].shape[:2]
    K = build_intrinsics(width, height)
    times = {}

    all_kps, all_descs = load_features(paths, images, method, log)
    keyframe_info = None
    if config.KEYFRAMES if keyframes is None else keyframes:
        keys, keyframe_info = select_keyframes(paths, images, method, all_kps, all_descs, K, (width, height),
                                               config.KEYFRAME_MIN_INLIERS, log)
        paths, images = [paths[k] for k in keys], Subset(images, keys)
        all_kps, all_descs = [all_kps[k] for k in keys], [all_descs[k] for k in keys]
    t0 = time.perf_counter()
    pairs = select_pairs(len(paths), pairing, all_descs, method, config.SEQ_WINDOW, config.LOOP_TOP_K,
                         config.device())
    times["pair_selection_s"] = time.perf_counter() - t0
    geometries, time_per_pair = load_geometries(paths, all_kps, all_descs, method, pairs, K, (width, height), log)
    times["matching_s_estimated"] = (time_per_pair or 0) * len(pairs)

    recon, times["sfm_s"] = load_sfm(paths, all_kps, geometries, K, method, pairing, log)
    log(f"SfM: {len(recon.poses)}/{len(paths)} cameras, {len(recon.points)} pontos, f = {recon.K[0, 0]:.0f} px")

    pattern, square = config.checkerboard_spec()
    detections = load_checkerboards(paths, images, pattern, log)
    try:
        scale = checkerboard_scale(recon, images, pattern, square, detections=detections)
        scale_reconstruction(recon, scale["scale"])
        log(f"Escala: {scale['scale']:.4f} m/unidade, {len(scale['boards'])} tabuleiro(s), "
            f"erro relativo medio das distancias {100 * scale['pair_rel_error_mean']:.2f}%")
    except ValueError as exc:  # p.ex. o tabuleiro nao aparece nas cameras registradas
        log(f"AVISO: sem escala metrica ({exc}); medidas em unidades do SfM")
        scale = {"scale": float("nan"), "boards": [], "corners3d": None, "pair_rel_error_mean": float("nan"),
                 "pair_rel_error_max": float("nan"), "board_scale_spread": float("nan"), "error": str(exc)}

    pcd = to_open3d(recon, all_kps, images)
    pcd_clean, clean_counts = clean_point_cloud(pcd, **config.CLEAN)
    T, align_info = align_to_floor(pcd_clean, recon)
    transform_reconstruction(recon, T)
    pcd.transform(T)
    pcd_clean.transform(T)
    points = np.asarray(pcd_clean.points)
    dims = room_dimensions(points)
    planes = plane_measurements(points)
    ceilings = [h["height"] for h in planes["horizontal"] if h["height"] > 2.0]
    ceiling = max(ceilings) if ceilings else None  # planos abaixo de 2 m sao mesas, nao o teto

    stats = reconstruction_stats(recon, all_kps)
    metrics = {
        "detector": method, "pairing": pairing, "n_frames": len(paths), "n_pairs": len(pairs),
        "keyframes": keyframe_info,
        "n_pairs_verified": sum(g["mask_E"] is not None and g["metrics"]["inliers_E"] >= config.MIN_MATCHES
                                for g in geometries.values()),
        **stats, "focal_px": float(recon.K[0, 0]), "focal_initial_px": float(K[0, 0]),
        "registered_fraction": len(recon.poses) / len(paths),
        "cleaning": clean_counts, "floor_found": align_info["floor_found"],
        "scale": {k: v for k, v in scale.items() if k not in ("boards", "corners3d")},
        "boards": [{k: v for k, v in b.items() if k != "corners3d"} for b in scale["boards"]],
        "dimensions": {"largura": dims["width"], "comprimento": dims["length"],
                       "x_extent": dims["x_extent"], "y_extent": dims["y_extent"]},
        "planes": planes, "drift": loop_drift(recon),
        "density_sparse": point_density(len(points), dims, ceiling), "times": times,
    }
    # paths/images: os quadros efetivamente usados (os quadros-chave), na indexacao da reconstrucao
    return {"recon": recon, "paths": paths, "images": images, "all_kps": all_kps, "geometries": geometries,
            "pcd": pcd, "pcd_clean": pcd_clean,
            "dims": dims, "metrics": metrics, "K_initial": K}


def metric_errors(model, real):
    """|L^ - L| (cm) e e = |L^ - L| / L para as medidas reais preenchidas em `config.real_measurements()`."""
    rows = []
    for name, L in real.items():
        if L is None or name not in model:
            continue
        L_hat = model[name]
        rows.append({"medida": name, "real_m": L, "modelo_m": L_hat, "erro_cm": 100 * abs(L_hat - L),
                     "erro_relativo": abs(L_hat - L) / L})
    return rows


def model_measurements(metrics):
    """Medidas do modelo: largura/comprimento automaticos e as marcadas a mao (`visualizar.py --medir`)."""
    out = {"largura": metrics["dimensions"]["largura"], "comprimento": metrics["dimensions"]["comprimento"]}
    manual = config.results_dir(metrics["detector"]) / "medidas_modelo.json"
    if manual.exists():
        out.update(json.loads(manual.read_text()))
    return out


def save_outputs(result, images, out_dir, width, height):
    """Salva nuvens, cameras, planta baixa, reconstrucao (pickle) e metricas."""
    import open3d as o3d
    out_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_dir / "esparsa.ply"), result["pcd"])
    o3d.io.write_point_cloud(str(out_dir / "esparsa_limpa.ply"), result["pcd_clean"])
    save_cameras(result["recon"], width, height, out_dir)
    draw_floor_plan(np.asarray(result["pcd_clean"].points), camera_centers(result["recon"]), result["dims"],
                    out_dir / "planta_baixa.png", max_height=_below_ceiling(result["metrics"]))
    with open(out_dir / "reconstrucao.pkl", "wb") as f:
        pickle.dump({"recon": result["recon"], "kps": [keypoints_to_array(k) for k in result["all_kps"]]}, f)
    write_metrics(result["metrics"], out_dir)


def _below_ceiling(metrics):
    """Altura de corte da planta baixa: um pouco abaixo do teto, que cobriria tudo visto de cima."""
    horizontal = metrics["planes"]["horizontal"]
    # so conta como teto um plano acima de 2 m: mais baixo e tampo de mesa ou balcao
    ceilings = [h["height"] for h in horizontal if h["height"] > 2.0]
    return max(ceilings) - 0.15 if ceilings else None


def write_metrics(metrics, out_dir):
    """metricas.json completo e metricas.md com as tabelas para o relatorio."""
    measures = model_measurements(metrics)
    metrics["metric_errors"] = metric_errors(measures, config.real_measurements())
    (out_dir / "metricas.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    d, dr = metrics["drift"], metrics["density_sparse"]
    lines = ["| metrica | valor |", "|---|---|",
             f"| cameras registradas | {metrics['n_cameras']}/{metrics['n_frames']} |",
             f"| pontos esparsos (limpos) | {metrics['n_points']} ({metrics['cleaning']}) |",
             f"| erro de reprojecao media / mediana (px) | {metrics['reproj_mean']:.2f} / {metrics['reproj_median']:.2f} |",
             f"| tamanho medio dos tracks | {metrics['mean_track_length']:.2f} |",
             f"| focal inicial -> refinada (px) | {metrics['focal_initial_px']:.0f} -> {metrics['focal_px']:.0f} |",
             f"| densidade (pontos/m2 de planta) | {dr['points_per_m2_floor']:.0f} |",
             *([f"| densidade (pontos/m2 de superficie) | {dr['points_per_m2_surface']:.0f} |"] if "points_per_m2_surface" in dr else []),
             f"| deriva do laco | {100 * d['delta_m']:.1f} cm ({d['delta_percent']:.2f}% de {d['path_length_m']:.1f} m) |",
             f"| escala (m/unidade), tabuleiros | {metrics['scale']['scale']:.4f}, {len(metrics['boards'])} (dispersao {100 * metrics['scale']['board_scale_spread']:.1f}%) |",
             "", "| medida | real (m) | modelo (m) | erro (cm) | erro relativo |", "|---|---|---|---|---|"]
    for m in measures:
        row = next((r for r in metrics["metric_errors"] if r["medida"] == m), None)
        if row:
            lines.append(f"| {m} | {row['real_m']:.3f} | {row['modelo_m']:.3f} | {row['erro_cm']:.1f} | {100 * row['erro_relativo']:.1f}% |")
        else:
            lines.append(f"| {m} | (preencher em config.real_measurements()) | {measures[m]:.3f} | | |")
    planes = metrics["planes"]
    lines += ["", "Planos horizontais (altura em m): " + ", ".join(f"{h['height']:.2f} ({h['n_points']} pts)" for h in planes["horizontal"]),
              "Angulos entre paredes (graus): " + ", ".join(f"{a['angle']:.1f}" for a in planes["wall_angles"])]
    (out_dir / "metricas.md").write_text("\n".join(lines) + "\n")


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def load_reconstruction(out_dir):
    with open(out_dir / "reconstrucao.pkl", "rb") as f:
        data = pickle.load(f)
    return data["recon"], [array_to_keypoints(k) for k in data["kps"]]

