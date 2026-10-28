"""
Medidas manuais no modelo metrico (p.ex. altura da porta ou da mesa), para
comparar com as medidas de trena (Etapa 5). O resultado vai para
<saida>/medidas_manuais.json e e usado em config.yaml com model: "manual:<nome>".
Depois de medir, rode `python run_pipeline.py ... --reuse-sfm` ou
`python measure.py saida --update-metrics` para atualizar as metricas.

Modo 3D (padrao): abre a nuvem no Open3D; Shift + clique esquerdo em 2 pontos e feche a janela.
    python measure.py saida --name altura_porta

Modo 2D (mais preciso): clique nos MESMOS 2 pontos (p.ex. topo e base da porta) em
2 ou mais quadros; cada ponto e triangulado com as poses do SfM.
    python measure.py saida --name altura_porta --frames frame_00012.jpg frame_00020.jpg

Modo 2D sem janela: um JSON {"frame_00012.jpg": [[x1, y1], [x2, y2]], ...}
    python measure.py saida --name altura_porta --clicks cliques.json
"""
import argparse
import json
import os

import cv2
import numpy as np


def save(out_dir, name, dist, details):
    path = os.path.join(out_dir, "medidas_manuais.json")
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[name] = {"distance_m": float(dist), **details}
    json.dump(data, open(path, "w"), indent=1)
    print(f"{name}: {dist * 100:.1f} cm  (salvo em {path})")


def measure_3d(out_dir, sparse=False):
    import open3d as o3d
    from visualize import load_results
    xyz, rgb, _, _, _ = load_results(out_dir, sparse)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("Shift + clique em 2 pontos, depois feche a janela", 1400, 900)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    picked = vis.get_picked_points()
    if len(picked) < 2:
        raise SystemExit("Selecione 2 pontos (Shift + clique).")
    a, b = xyz[picked[-2]], xyz[picked[-1]]
    return float(np.linalg.norm(a - b)), {"mode": "3d", "p1": a.tolist(), "p2": b.tolist()}


def click_points(img_path, n=2):
    img = cv2.imread(img_path)
    s = min(1.0, 1400 / img.shape[1])
    disp = cv2.resize(img, None, fx=s, fy=s)
    pts = []

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x / s, y / s))
            cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(disp, str(len(pts)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(win, disp)

    win = f"{os.path.basename(img_path)}: clique ponto 1 e ponto 2 (mesma ordem em todos), Enter p/ seguir"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    cv2.imshow(win, disp)
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == 27 or (len(pts) >= n and k in (13, 10, 32)):
            break
    cv2.destroyWindow(win)
    if len(pts) < n:
        raise SystemExit("Cancelado.")
    return pts


def measure_2d(out_dir, image_dir, clicks):
    import pycolmap
    from pipeline.step_4_scale_align import triangulate_multiview
    rec = pycolmap.Reconstruction(os.path.join(out_dir, "modelo_metrico"))
    obs = [[], []]
    registered = set(rec.reg_image_ids())
    for name, pts in clicks.items():
        img = rec.find_image_with_name(name)
        if img is None or img.image_id not in registered:
            print(f"aviso: {name} nao registrado no modelo; ignorado")
            continue
        for k in range(2):
            obs[k].append((img, np.array(pts[k], float)))
    P = []
    for k in range(2):
        r = triangulate_multiview(obs[k], rec, max_error_px=8.0, min_views=2)
        if r is None:
            raise SystemExit(f"Ponto {k + 1}: triangulacao falhou (use quadros com mais paralaxe).")
        P.append(r)
        print(f"ponto {k + 1}: {np.round(r[0], 3)}  (reprojecao media {r[1]:.2f} px em {r[2]} vistas)")
    return float(np.linalg.norm(P[0][0] - P[1][0])), {
        "mode": "2d", "p1": P[0][0].tolist(), "p2": P[1][0].tolist(),
        "reproj_px": [P[0][1], P[1][1]], "frames": list(clicks)}


def update_metrics(out_dir, config="config.yaml"):
    import yaml
    from pipeline import step_5_metrics
    cfg = yaml.safe_load(open(config)) or {}
    m = json.load(open(os.path.join(out_dir, "metricas.json")))
    room = json.load(open(os.path.join(out_dir, "sala.json")))
    manual = json.load(open(os.path.join(out_dir, "medidas_manuais.json")))
    m["dimensions"] = step_5_metrics.dimension_errors(cfg.get("measurements", []), room, manual)
    step_5_metrics.save_metrics(m, os.path.join(out_dir, "metricas.json"), os.path.join(out_dir, "metricas.md"))
    print(open(os.path.join(out_dir, "metricas.md")).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--name", help="nome da medida (o mesmo de config.yaml, manual:<nome>)")
    ap.add_argument("--frames", nargs="+", help="quadros para o modo 2D")
    ap.add_argument("--clicks", help="JSON com os cliques (modo 2D sem janela)")
    ap.add_argument("--images", help="pasta das imagens (padrao: <saida>/quadros)")
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--update-metrics", action="store_true", help="recalcula a tabela de erros com as medidas salvas")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    if args.update_metrics:
        update_metrics(args.out_dir, args.config)
        return
    if not args.name:
        ap.error("--name e obrigatorio")
    image_dir = args.images or os.path.join(args.out_dir, "quadros")
    if args.clicks:
        dist, det = measure_2d(args.out_dir, image_dir, json.load(open(args.clicks)))
    elif args.frames:
        clicks = {f: click_points(os.path.join(image_dir, f)) for f in args.frames}
        dist, det = measure_2d(args.out_dir, image_dir, clicks)
    else:
        dist, det = measure_3d(args.out_dir, args.sparse)
    save(args.out_dir, args.name, dist, det)


if __name__ == "__main__":
    main()
