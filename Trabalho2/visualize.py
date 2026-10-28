"""
Visualizacao dos resultados de run_pipeline.py.

    python visualize.py saida              # janela 3D (Open3D): nuvem + cameras + trajetoria
    python visualize.py saida --sparse     # usa a nuvem esparsa em vez da densa
    python visualize.py saida --video      # (re)gera o video de navegacao (30-60 s)
    python visualize.py saida --glb        # Extra: exporta cena.glb para visualizacao web

Teclas no Open3D: mouse gira/translada, scroll aproxima, '+'/'-' tamanho do ponto,
'H' ajuda. O GLB abre em https://gltf-viewer.donmccurdy.com ou no <model-viewer>.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess

import numpy as np

from pipeline import step_4_sfm
from pipeline.step_4_room import voxel_downsample
from pipeline.viz_utils import camera_frustums, read_ply, trajectory_lines


def load_results(out_dir, sparse=False):
    import pycolmap
    dense = os.path.join(out_dir, "nuvem_densa.ply")
    path = dense if (os.path.exists(dense) and not sparse) else os.path.join(out_dir, "nuvem_esparsa.ply")
    xyz, rgb = read_ply(path)
    rec = pycolmap.Reconstruction(os.path.join(out_dir, "modelo_metrico"))
    sc = step_4_sfm.scene_arrays(rec)
    room = json.load(open(os.path.join(out_dir, "sala.json")))
    return xyz, (rgb if rgb is not None else np.full((len(xyz), 3), 200, np.uint8)), sc, room, path


def show_open3d(out_dir, sparse=False):
    import open3d as o3d
    xyz, rgb, sc, room, path = load_results(out_dir, sparse)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    geoms = [pcd]
    size = (sc["camera"].width, sc["camera"].height)
    V, E = camera_frustums(sc["centers"], sc["R"], sc["K"], size, depth=0.15,
                           every=max(1, len(sc["centers"]) // 80))
    cams = o3d.geometry.LineSet(o3d.utility.Vector3dVector(V), o3d.utility.Vector2iVector(E))
    cams.paint_uniform_color([1.0, 0.5, 0.0])
    V, E = trajectory_lines(sc["centers"])
    traj = o3d.geometry.LineSet(o3d.utility.Vector3dVector(V), o3d.utility.Vector2iVector(E))
    traj.paint_uniform_color([0.9, 0.1, 0.1])
    geoms += [cams, traj, o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)]
    print(f"{os.path.basename(path)}: {len(xyz)} pontos, {len(sc['centers'])} cameras")
    o3d.visualization.draw_geometries(geoms, window_name="Reconstrucao - Variante A", width=1400, height=900)


def make_video(out_dir, rcfg=None, sparse=False, log=print):
    from pipeline.render_scene import flythrough_video
    rcfg = rcfg or {}
    xyz, rgb, sc, room, path = load_results(out_dir, sparse)
    max_pts = rcfg.get("max_points", 600000)
    voxel = 0.005
    while len(xyz) > max_pts:
        voxel *= 1.4
        xyz, rgb = voxel_downsample(xyz, rgb, voxel)
    raw = os.path.join(out_dir, "video_navegacao_mp4v.mp4")
    final = os.path.join(out_dir, "video_navegacao.mp4")
    w, h = rcfg.get("width", 1280), rcfg.get("height", 720)
    flythrough_video(xyz, rgb, sc, room, raw, seconds=rcfg.get("seconds", 45), fps=rcfg.get("fps", 30),
                     size=(w, h), title=rcfg.get("title", "Reconstrucao 3D - Variante A"), log=log)
    if shutil.which("ffmpeg"):  # H.264 abre em qualquer player/PowerPoint
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "20", "-pix_fmt", "yuv420p", final], check=False)
        if os.path.exists(final):
            os.remove(raw)
    else:
        os.replace(raw, final)
    log(f"  video salvo em {final}")
    return final


def export_glb(out_dir, sparse=False, max_points=1_000_000):
    """Extra: cena em glTF binario (pontos coloridos + piramides das cameras + trajetoria)."""
    xyz, rgb, sc, room, _ = load_results(out_dir, sparse)
    voxel = 0.004
    while len(xyz) > max_points:
        voxel *= 1.4
        xyz, rgb = voxel_downsample(xyz, rgb, voxel)
    size = (sc["camera"].width, sc["camera"].height)
    Vc, Ec = camera_frustums(sc["centers"], sc["R"], sc["K"], size, depth=0.15, every=max(1, len(sc["centers"]) // 80))
    Vt, Et = trajectory_lines(sc["centers"])
    # glTF usa y para cima: (x, y, z) -> (x, z, -y)
    conv = lambda P: np.column_stack([P[:, 0], P[:, 2], -P[:, 1]]).astype(np.float32)
    prims = [(conv(xyz), np.column_stack([rgb, np.full(len(rgb), 255)]).astype(np.uint8), None, 0),
             (conv(Vc), np.tile([255, 128, 0, 255], (len(Vc), 1)).astype(np.uint8), Ec.astype(np.uint32), 1),
             (conv(Vt), np.tile([230, 30, 30, 255], (len(Vt), 1)).astype(np.uint8), Et.astype(np.uint32), 1)]
    blob, views, accessors, meshes = b"", [], [], []

    def add_view(data, target=None):
        nonlocal blob
        pad = (-len(blob)) % 4
        blob += b"\0" * pad
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), **({"target": target} if target else {})})
        blob += data
        return len(views) - 1

    for pos, col, idx, mode in prims:
        a_pos = len(accessors)
        accessors.append({"bufferView": add_view(pos.tobytes(), 34962), "componentType": 5126, "count": len(pos),
                          "type": "VEC3", "min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
        a_col = len(accessors)
        accessors.append({"bufferView": add_view(col.tobytes(), 34962), "componentType": 5121, "normalized": True,
                          "count": len(col), "type": "VEC4"})
        prim = {"attributes": {"POSITION": a_pos, "COLOR_0": a_col}, "mode": mode}
        if idx is not None:
            prim["indices"] = len(accessors)
            accessors.append({"bufferView": add_view(idx.ravel().tobytes(), 34963), "componentType": 5125,
                              "count": int(idx.size), "type": "SCALAR"})
        meshes.append({"primitives": [prim]})
    gltf = {"asset": {"version": "2.0", "generator": "t2-variante-a"}, "scene": 0,
            "scenes": [{"nodes": list(range(len(meshes)))}],
            "nodes": [{"mesh": i, "name": n} for i, n in enumerate(["nuvem", "cameras", "trajetoria"])],
            "meshes": meshes, "accessors": accessors, "bufferViews": views,
            "buffers": [{"byteLength": len(blob) + (-len(blob)) % 4}]}
    js = json.dumps(gltf).encode()
    js += b" " * ((-len(js)) % 4)
    blob += b"\0" * ((-len(blob)) % 4)
    path = os.path.join(out_dir, "cena.glb")
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)
    print(f"GLB salvo em {path} ({len(xyz)} pontos)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", nargs="?", default="saida")
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--glb", action="store_true")
    args = ap.parse_args()
    if args.video:
        rcfg = {}
        if os.path.exists("config.yaml"):
            import yaml
            rcfg = (yaml.safe_load(open("config.yaml")) or {}).get("render", {})
        if args.seconds:
            rcfg["seconds"] = args.seconds
        make_video(args.out_dir, rcfg, args.sparse)
    elif args.glb:
        export_glb(args.out_dir, args.sparse)
    else:
        show_open3d(args.out_dir, args.sparse)


if __name__ == "__main__":
    main()
