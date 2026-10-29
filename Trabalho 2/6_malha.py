"""
Extra (opcional): malha colorida a partir da nuvem densa, em results/<video>/<detector>/:
  - malha.ply (vertices coloridos) e malha.glb (para visualizacao web, p.ex. no three.js ou no
    https://gltf-viewer.donmccurdy.com), e vistas_malha.jpg.
Rode depois de `3_reconstrucao.py --densa`. Passos:
  1. reduz a nuvem densa por voxel e estima as normais pelos vizinhos;
  2. orienta cada normal para a camera registrada mais proxima: todo ponto foi visto de alguma
     camera, entao a superficie "olha" para ela (a orientacao por consistencia entre vizinhos e
     lenta demais para milhoes de pontos e se perde nas quinas);
  3. reconstrucao de Poisson (superficie implicita cujo gradiente segue as normais);
  4. corta os triangulos de baixa densidade de pontos: o Poisson fecha a superficie e "inventa"
     membranas onde nao ha dados (janelas, partes nao filmadas), que ficam com densidade baixa.
"""
import argparse

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from src import config
from src.pipeline import load_reconstruction
from src.step_4_postprocessing import camera_centers


def build_mesh(pcd, centers, voxel=0.02, depth=10, density_quantile=0.08):
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
    pts, normals = np.asarray(pcd.points), np.asarray(pcd.normals)
    _, nearest = cKDTree(centers).query(pts)
    flip = np.sum(normals * (centers[nearest] - pts), axis=1) < 0
    normals[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.quantile(density, density_quantile))
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, len(pcd.points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default=config.DETECTOR)
    parser.add_argument("--voxel", type=float, default=0.02, help="lado do voxel antes do Poisson (m)")
    parser.add_argument("--profundidade", type=int, default=10, help="profundidade da octree do Poisson")
    args = parser.parse_args()

    out = config.results_dir(args.detector)
    dense = out / "densa.ply"
    if not dense.exists():
        raise SystemExit(f"{dense} nao existe: rode `3_reconstrucao.py --densa` antes")
    recon, _ = load_reconstruction(out)
    mesh, n_points = build_mesh(o3d.io.read_point_cloud(str(dense)), camera_centers(recon), args.voxel,
                                args.profundidade)
    o3d.io.write_triangle_mesh(str(out / "malha.ply"), mesh)
    o3d.io.write_triangle_mesh(str(out / "malha.glb"), mesh)
    print(f"Malha: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangulos (de {n_points} pontos), "
          f"em {out / 'malha.ply'} e {out / 'malha.glb'}")
