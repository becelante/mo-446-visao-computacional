"""
Etapa 4a: Structure-from-Motion com COLMAP (via pycolmap ou via colmap.exe com CUDA).

extracao SIFT -> emparelhamento (exaustivo ou sequencial, com ou sem deteccao
de laco por vocabulary tree) -> verificacao geometrica (E/F/H com RANSAC) ->
SfM incremental com bundle adjustment. Todos os quadros compartilham a mesma
camera (mesmo celular, mesmo zoom), o que estabiliza a calibracao.

Tambem converte a reconstrucao em arrays numpy (poses, centros, pontos) usados
pelas etapas seguintes.

GPU: o pycolmap do PyPI para Windows vem sem CUDA. Com `sfm.colmap_exe` (build oficial
do COLMAP com CUDA, mesma versao do pycolmap), extracao SIFT, emparelhamento e
mapeamento rodam pelo executavel, na GPU; o resto do pipeline continua no pycolmap.
"""
import glob
import hashlib
import os
import shutil
import subprocess
import time
import urllib.request

import cv2
import numpy as np
import pycolmap

# arvore de vocabulario oficial do COLMAP. Baixada pelo Python: o curl do pycolmap do PyPI nao acha os
# certificados do sistema no Linux e aborta o processo
VOCAB_TREE_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin"
VOCAB_TREE_SHA256 = "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc"

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLED_COLMAP = os.path.join(PROJECT_DIR, "ferramentas", "colmap", "bin", "colmap.exe")

def find_colmap(setting="auto"):
    if not setting:
        return None
    if setting == "auto":
        if os.path.exists(BUNDLED_COLMAP):
            return BUNDLED_COLMAP
        return shutil.which("colmap")
    path = setting if os.path.isabs(setting) else os.path.join(PROJECT_DIR, setting)
    if not os.path.exists(path):
        raise FileNotFoundError(f"sfm.colmap_exe nao encontrado: {path}")
    return path

def run_colmap(exe, command, opts, log_path=None):
    args = [exe, command]
    for k, v in opts.items():
        if v is None:
            continue
        args += [f"--{k}", str(int(v) if isinstance(v, bool) else v)]
    r = subprocess.run(args, capture_output=True, text=True, errors="replace")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"$ {' '.join(args)}\n{r.stdout}{r.stderr}\n")
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"colmap {command} falhou (codigo {r.returncode}):\n{tail}")

def _write_image_list(path, image_names):
    with open(path, "w") as f:
        f.write("\n".join(image_names) + "\n")
    return path

def _load_models(out_dir):
    recs = {}
    for d in sorted(glob.glob(os.path.join(out_dir, "[0-9]*"))):
        if os.path.exists(os.path.join(d, "images.bin")):
            recs[int(os.path.basename(d))] = pycolmap.Reconstruction(d)
    return recs

def _gpu_flag(use_gpu):
    return use_gpu in (None, "auto") or bool(use_gpu)

def _device(use_gpu):
    if use_gpu in (None, "auto"):
        return pycolmap.Device.auto
    return pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

def clahe_images(image_dir, out_dir, image_names, clip=3.0, tiles=8):
    os.makedirs(out_dir, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    for name in image_names:
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst):
            continue
        lab = cv2.cvtColor(cv2.imread(os.path.join(image_dir, name)), cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        cv2.imwrite(dst, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out_dir

def extract_features(database, image_dir, image_names, camera_model="SIMPLE_RADIAL",
                     focal_px=None, max_image_size=1600, max_num_features=8192, use_gpu="auto",
                     peak_threshold=None, clahe=False, colmap_exe=None):
    if clahe:
        image_dir = clahe_images(image_dir, os.path.join(os.path.dirname(os.path.abspath(database)),
                                                         "imagens_clahe"), image_names)
    camera_params = None
    if focal_px:
        h, w = cv2.imread(os.path.join(image_dir, image_names[0])).shape[:2]
        n_params = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5, "OPENCV": 8}[camera_model]
        if camera_model in ("PINHOLE", "OPENCV"):
            params = [focal_px, focal_px, w / 2, h / 2]
        else:
            params = [focal_px, w / 2, h / 2]
        params += [0.0] * (n_params - len(params))
        camera_params = ",".join(str(p) for p in params)
    if colmap_exe:
        work = os.path.dirname(os.path.abspath(database))
        run_colmap(colmap_exe, "feature_extractor", {
            "database_path": database, "image_path": image_dir,
            "image_list_path": _write_image_list(os.path.join(work, "lista_imagens.txt"), image_names),
            "ImageReader.single_camera": True, "ImageReader.camera_model": camera_model,
            "ImageReader.camera_params": camera_params,
            "FeatureExtraction.use_gpu": _gpu_flag(use_gpu), "FeatureExtraction.max_image_size": max_image_size,
            "SiftExtraction.max_num_features": max_num_features, "SiftExtraction.peak_threshold": peak_threshold,
        }, os.path.join(work, "colmap.log"))
        return
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    if camera_params:
        reader.camera_params = camera_params
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = max_image_size
    ext.sift.max_num_features = max_num_features
    if peak_threshold:
        ext.sift.peak_threshold = float(peak_threshold)
    pycolmap.extract_features(database, image_dir, image_names=image_names,
                              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext, device=_device(use_gpu))

def vocab_tree_path(path=None):
    if path:
        return path
    local = os.path.join(os.path.expanduser("~"), ".cache", "colmap", os.path.basename(VOCAB_TREE_URL))
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        urllib.request.urlretrieve(VOCAB_TREE_URL, local + ".part")
        digest = hashlib.sha256()
        with open(local + ".part", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != VOCAB_TREE_SHA256:
            os.remove(local + ".part")
            raise RuntimeError(f"vocabulary tree corrompida: {VOCAB_TREE_URL}")
        os.replace(local + ".part", local)
    return local

def match(database, mode="exhaustive", overlap=15, loop_detection=False, vocab_tree=None, use_gpu="auto",
          colmap_exe=None):
    if colmap_exe:
        _match_exe(colmap_exe, database, mode, overlap, loop_detection, vocab_tree, use_gpu)
        return
    mopts = pycolmap.FeatureMatchingOptions()
    if mode == "exhaustive":
        pycolmap.match_exhaustive(database, matching_options=mopts, device=_device(use_gpu))
    elif mode == "sequential":
        popts = pycolmap.SequentialPairingOptions()
        popts.overlap = overlap
        popts.quadratic_overlap = True
        popts.loop_detection = bool(loop_detection)
        if loop_detection:
            popts.vocab_tree_path = vocab_tree_path(vocab_tree)
            popts.loop_detection_period = 5
            popts.loop_detection_num_images = 30
        pycolmap.match_sequential(database, matching_options=mopts, pairing_options=popts, device=_device(use_gpu))
    else:
        raise ValueError(f"Modo de emparelhamento desconhecido: {mode}")

def _match_exe(exe, database, mode, overlap, loop_detection, vocab_tree, use_gpu):
    opts = {"database_path": database, "FeatureMatching.use_gpu": _gpu_flag(use_gpu)}
    if mode == "exhaustive":
        command = "exhaustive_matcher"
    elif mode == "sequential":
        command = "sequential_matcher"
        opts.update({"SequentialMatching.overlap": overlap, "SequentialMatching.quadratic_overlap": True,
                     "SequentialMatching.loop_detection": bool(loop_detection)})
        if loop_detection:
            opts.update({"SequentialMatching.vocab_tree_path": vocab_tree_path(vocab_tree),
                         "SequentialMatching.loop_detection_period": 5,
                         "SequentialMatching.loop_detection_num_images": 30})
    else:
        raise ValueError(f"Modo de emparelhamento desconhecido: {mode}")
    run_colmap(exe, command, opts, os.path.join(os.path.dirname(os.path.abspath(database)), "colmap.log"))

def _map_exe(exe, command, list_option, database, image_dir, out_dir, image_names, seed):
    shutil.rmtree(out_dir, ignore_errors=True)  # modelos de uma execucao anterior
    os.makedirs(out_dir)
    work = os.path.dirname(os.path.abspath(database))
    opts = {"database_path": database, "image_path": image_dir, "output_path": out_dir,
            "default_random_seed": seed}
    if image_names:
        opts[list_option] = _write_image_list(os.path.join(work, "lista_imagens.txt"), image_names)
    run_colmap(exe, command, opts, os.path.join(work, "colmap.log"))
    return _load_models(out_dir)

def map_incremental(database, image_dir, out_dir, image_names=None, seed=0, colmap_exe=None):
    if colmap_exe:
        recs = _map_exe(colmap_exe, "mapper", "Mapper.image_list_path", database, image_dir, out_dir,
                        image_names, seed)
        if not recs:
            raise RuntimeError("O COLMAP nao conseguiu inicializar nenhum modelo (pouca textura/sobreposicao?)")
        best = max(recs.values(), key=lambda r: r.num_reg_images())
        return best, len(recs)
    os.makedirs(out_dir, exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.random_seed = seed
    opts.mapper.random_seed = seed
    if image_names:
        opts.image_names = list(image_names)
    recs = pycolmap.incremental_mapping(database, image_dir, out_dir, options=opts)
    if not recs:
        raise RuntimeError("O COLMAP nao conseguiu inicializar nenhum modelo (pouca textura/sobreposicao?)")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    return best, len(recs)

def map_global(database, image_dir, out_dir, image_names=None, seed=0, colmap_exe=None):
    if colmap_exe:
        recs = _map_exe(colmap_exe, "global_mapper", "GlobalMapper.image_list_path", database, image_dir,
                        out_dir, image_names, seed)
        if not recs:
            raise RuntimeError("O mapeador global nao conseguiu reconstruir nenhum modelo")
        best = max(recs.values(), key=lambda r: r.num_reg_images())
        return best, len(recs)
    os.makedirs(out_dir, exist_ok=True)
    opts = pycolmap.GlobalPipelineOptions()
    opts.random_seed = seed
    if image_names:
        opts.image_names = list(image_names)
    recs = pycolmap.global_mapping(database, image_dir, out_dir, options=opts)
    if not recs:
        raise RuntimeError("O mapeador global nao conseguiu reconstruir nenhum modelo")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    return best, len(recs)

def keypoints_per_image(db):
    import sqlite3
    con = sqlite3.connect(db)
    try:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        out = {}
        for image_id, rows, cols, data in con.execute("SELECT image_id, rows, cols, data FROM keypoints"):
            kp = np.frombuffer(data, np.float32).reshape(rows, cols) if rows else np.zeros((0, 2), np.float32)
            out[names[image_id]] = kp[:, :2]
        return out
    finally:
        con.close()

def count_verified_pairs(db):
    import sqlite3
    con = sqlite3.connect(db)
    try:
        return int(con.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0").fetchone()[0])
    finally:
        con.close()

def run_sfm(image_dir, work_dir, image_names, matching="exhaustive", overlap=15,
            loop_detection=False, vocab_tree=None, camera_model="SIMPLE_RADIAL", focal_px=None,
            max_image_size=1600, max_num_features=8192, use_gpu="auto", features_db=None,
            peak_threshold=None, clahe=False, mapper="incremental", colmap_exe=None, log=print):
    os.makedirs(work_dir, exist_ok=True)
    db = os.path.join(work_dir, "database.db")
    if os.path.exists(db):
        os.remove(db)
    info = {"matching": matching, "loop_detection": bool(loop_detection), "n_images_input": len(image_names),
            "backend": "colmap.exe" if colmap_exe else "pycolmap"}
    t0 = time.time()
    if features_db:
        shutil.copy(features_db, db)
    else:
        log(f"  extraindo SIFT de {len(image_names)} imagens{' (com CLAHE)' if clahe else ''}...")
        extract_features(db, image_dir, image_names, camera_model, focal_px, max_image_size,
                         max_num_features, use_gpu, peak_threshold, clahe, colmap_exe)
    info["time_features_s"] = time.time() - t0

    t0 = time.time()
    log(f"  emparelhando ({matching}{' + laco' if loop_detection and matching == 'sequential' else ''})...")
    match(db, matching, overlap, loop_detection, vocab_tree, use_gpu, colmap_exe)
    info["time_matching_s"] = time.time() - t0
    info["n_verified_pairs"] = count_verified_pairs(db)

    t0 = time.time()
    log(f"  SfM {mapper} (com bundle adjustment)...")
    mapping = map_global if mapper == "global" else map_incremental
    rec, n_models = mapping(db, image_dir, os.path.join(work_dir, "models"), image_names, colmap_exe=colmap_exe)
    info["time_mapping_s"] = time.time() - t0
    info["mapper"] = mapper
    info["n_models"] = n_models
    sparse = os.path.join(work_dir, "sparse")
    os.makedirs(sparse, exist_ok=True)
    rec.write(sparse)
    info.update(n_reg_images=rec.num_reg_images(), n_points3d=rec.num_points3D(),
                mean_reproj_colmap=rec.compute_mean_reprojection_error(),
                mean_track_length=rec.compute_mean_track_length())
    return rec, info

def image_pose(img):
    cfw = img.cam_from_world
    if callable(cfw):
        cfw = cfw()
    M = np.asarray(cfw.matrix())
    return M[:, :3].copy(), M[:, 3].copy()

def registered_images(rec):
    imgs = [rec.image(i) for i in rec.reg_image_ids()]
    return sorted(imgs, key=lambda im: im.name)

def scene_arrays(rec):
    imgs = registered_images(rec)
    Rs, ts = zip(*[image_pose(im) for im in imgs]) if imgs else ([], [])
    Rs, ts = np.array(Rs), np.array(ts)
    centers = -np.einsum("nji,nj->ni", Rs, ts) if len(imgs) else np.zeros((0, 3))
    ids = list(rec.point3D_ids())
    pts = [rec.point3D(i) for i in ids]
    cam = rec.camera(imgs[0].camera_id) if imgs else None
    return {
        "names": [im.name for im in imgs],
        "image_ids": [im.image_id for im in imgs],
        "R": Rs, "t": ts, "centers": centers,
        "point_ids": np.array(ids, dtype=np.int64),
        "xyz": np.array([p.xyz for p in pts]).reshape(-1, 3),
        "rgb": np.array([p.color for p in pts], dtype=np.uint8).reshape(-1, 3),
        "error": np.array([p.error for p in pts]),
        "track_len": np.array([p.track.length() for p in pts]),
        "camera": cam,
        "K": np.asarray(cam.calibration_matrix()) if cam is not None else None,
    }

def subset_cameras(sc, mask):
    mask = np.asarray(mask, bool)
    out = dict(sc)
    for k in ("R", "t", "centers"):
        out[k] = sc[k][mask]
    for k in ("names", "image_ids"):
        out[k] = [v for v, keep in zip(sc[k], mask) if keep]
    return out

def reprojection_errors(rec):
    ids = np.array(list(rec.point3D_ids()), dtype=np.int64)
    if len(ids) == 0:
        return np.zeros(0)
    xyz = np.array([rec.point3D(i).xyz for i in ids])
    order = np.argsort(ids)
    ids_sorted = ids[order]
    errors = []
    for img in registered_images(rec):
        obs = [(p.xy, p.point3D_id) for p in img.points2D if p.has_point3D()]
        if not obs:
            continue
        uv = np.array([o[0] for o in obs])
        pid = np.array([o[1] for o in obs], dtype=np.int64)
        X = xyz[order[np.searchsorted(ids_sorted, pid)]]
        R, t = image_pose(img)
        Xc = X @ R.T + t
        cam = rec.camera(img.camera_id)
        proj = np.asarray(cam.img_from_cam(Xc, False)).reshape(-1, 2)
        errors.append(np.linalg.norm(proj - uv, axis=1))
    return np.concatenate(errors) if errors else np.zeros(0)

def apply_sim3(rec, scale=1.0, R=np.eye(3), t=np.zeros(3)):
    rec.transform(pycolmap.Sim3d(float(scale), pycolmap.Rotation3d(np.asarray(R, float)),
                                 np.asarray(t, float).reshape(3)))
