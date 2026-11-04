"""
Visualizacao do modelo de results/<detector>/ (rode 3_reconstrucao.py antes; --detector ORB
para o modelo do ORB):
  uv run python visualizar.py                     # janela interativa: nuvem, cameras e trajetoria
  uv run python visualizar.py --densa             # o mesmo com a nuvem densa
  uv run python visualizar.py --malha             # a malha do extra (6_malha.py)
  uv run python visualizar.py --video navegacao.mp4 [--duracao 45]
        # video de navegacao: orbita sobre a sala e depois percorre a trajetoria da camera
  uv run python visualizar.py --medir altura_porta
        # shift + clique em 2 pontos; a distancia vai para results/<detector>/medidas_modelo.json e entra
        # na tabela de erro metrico se a chave estiver em config.real_measurements()
"""
import argparse
import json

import numpy as np
import cv2
import open3d as o3d
from scipy.ndimage import gaussian_filter1d

from src import config
from src.pipeline import load_reconstruction
from src.step_4_postprocessing import camera_centers, camera_frustums, measure_interactive, trajectory


def look_at(eye, target, up=(0, 0, 1)):
    """Extrinseca 4 x 4 (convencao OpenCV: x direita, y para baixo, z para frente) olhando de eye para target."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, forward])
    T[:3, 3] = -T[:3, :3] @ eye
    return T


def flight_path(recon, points, n_frames, orbit_fraction=0.35):
    """
    Extrinsecas do video: uma orbita sobre a sala (vista de 3/4, mostrando a trajetoria toda) e
    depois o percurso das cameras, um pouco atras e acima de cada uma, com centros e direcoes
    suavizados para nao tremer como a camera na mao.
    """
    lo, hi = np.percentile(points, 2, axis=0), np.percentile(points, 98, axis=0)
    center = (lo + hi) / 2
    radius = 0.6 * np.linalg.norm(hi[:2] - lo[:2])  # a sala ocupa a maior parte do quadro
    n_orbit = int(n_frames * orbit_fraction)
    views = []
    for k in range(n_orbit):
        a = 2 * np.pi * k / n_orbit
        eye = center + np.array([radius * np.cos(a), radius * np.sin(a), 0.7 * radius])
        views.append(look_at(eye, center))

    imgs = sorted(recon.poses)
    C = camera_centers(recon)
    forward = np.float64([recon.poses[i][0][2] for i in imgs])
    C = gaussian_filter1d(C, 4, axis=0, mode="nearest")
    forward = gaussian_filter1d(forward, 8, axis=0, mode="nearest")
    forward[:, 2] *= 0.3  # olhar mais na horizontal
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    s = np.linspace(0, len(C) - 1, n_frames - n_orbit)
    for x in s:
        k, w = int(np.floor(x)), x - np.floor(x)
        k1 = min(k + 1, len(C) - 1)
        c = (1 - w) * C[k] + w * C[k1]
        f = (1 - w) * forward[k] + w * forward[k1]
        f /= np.linalg.norm(f)
        eye = c - 0.6 * f + np.array([0, 0, 0.3])
        views.append(look_at(eye, c + 1.5 * f))
    return views


def record_video(geometries, views, out_path, width=1280, height=720, fps=30):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.mesh_show_back_face = True
    opt.background_color = np.array([0.08, 0.08, 0.1])
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.intrinsic.set_intrinsics(width, height, 0.6 * width, 0.6 * width, width / 2 - 0.5, height / 2 - 0.5)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for T in views:
        params.extrinsic = T
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        writer.write(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--densa", action="store_true")
    parser.add_argument("--malha", action="store_true", help="mostra a malha do extra 6_malha.py")
    parser.add_argument("--detector", default=config.DETECTOR, help="modelo de results/<detector>/")
    parser.add_argument("--video", help="grava o video de navegacao neste arquivo (.mp4)")
    parser.add_argument("--duracao", type=float, default=45, help="segundos de video")
    parser.add_argument("--medir", metavar="CHAVE", help="mede uma distancia com 2 cliques")
    args = parser.parse_args()

    out = config.results_dir(args.detector)
    model_file = out / ("malha.ply" if args.malha else "densa.ply" if args.densa else "esparsa_limpa.ply")
    if args.malha:
        model = o3d.io.read_triangle_mesh(str(model_file))
        model.compute_vertex_normals()
        points = np.asarray(model.vertices)
    else:
        model = o3d.io.read_point_cloud(str(model_file))
        points = np.asarray(model.points)
    if args.medir:
        pcd = model if not args.malha else o3d.geometry.PointCloud(model.vertices)
        dist = measure_interactive(pcd)
        if dist is None:
            raise SystemExit("Marque 2 pontos com shift + clique")
        path = out / "medidas_modelo.json"
        measures = json.loads(path.read_text()) if path.exists() else {}
        measures[args.medir] = dist
        path.write_text(json.dumps(measures, indent=2))
        print(f"{args.medir} = {dist:.3f} m (salvo em {path}); rode 3_reconstrucao.py para atualizar as metricas")
        raise SystemExit

    recon, _ = load_reconstruction(out)
    width, height = int(round(2 * recon.K[0, 2])), int(round(2 * recon.K[1, 2]))
    # ~60 piramides no total: com todas as cameras (milhares, num video) elas cobrem a cena
    frustums = camera_frustums(recon, width, height, step=max(1, len(recon.poses) // 60))
    geometries = [model, frustums, trajectory(recon)]
    if args.video:
        n = int(args.duracao * 30)
        record_video(geometries, flight_path(recon, points, n), args.video)
        print(f"Video salvo em {args.video} ({args.duracao:.0f} s)")
    else:
        o3d.visualization.draw_geometries([*geometries, o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)],
                                          window_name=model_file.name, mesh_show_back_face=True)
