# T2 – Reconstrução 3D, Variante A (ambiente interno)

Pipeline completo a partir de um vídeo da sala (laço fechado): quadros-chave →
SIFT/ORB → matching → geometria epipolar → SfM (COLMAP) → escala métrica (ArUco ou
xadrez) → alinhamento da sala → limpeza → planta baixa + trajetória → MVS →
métricas → ablação → vídeo de navegação.

## Instalação

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # com GPU NVIDIA: troque pycolmap por pycolmap-cuda12
sudo apt install ffmpeg                  # opcional: vídeo final em H.264
```

### GPU no Windows (COLMAP com CUDA)

O `pycolmap` do PyPI para Windows vem sem CUDA. Esta pasta traz o build oficial
`colmap-x64-windows-cuda` 4.2.1 (mesma versao do pycolmap) em `ferramentas/colmap/`;
com `sfm.colmap_exe: auto` no `config.yaml`, extracao SIFT, emparelhamento, mapeamento
e PatchMatch Stereo rodam por ele, na GPU. Para baixar de novo:
https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip
(descompacte em `ferramentas/colmap/`). `colmap_exe: null` volta para o pycolmap (CPU).

## Antes de rodar: `config.yaml`

1. `scale.marker_size_m`: lado do quadrado **preto** do ArUco impresso (meça com régua).
   Gere o marcador com `DICT_4X4_50` (ou mude `scale.method` para `checkerboard`).
2. `measurements`: as medidas de trena (cm): largura, comprimento e pé-direito.
3. `camera.model`: `OPENCV` se gravou com a lente grande-angular (0,5x).

## Rodar

```bash
chmod +x run.sh
./run.sh dados/sala.mp4 saida                 # tudo (Etapas 1–5 + ablação + vídeo)
./run.sh dados/sala.mp4 saida --no-ablation   # mais rápido (a ablação roda o SfM 5 vezes)
./run.sh pasta_de_fotos/ saida                # fotos em vez de vídeo
./run.sh dados/sala.mp4 saida --reuse-sfm     # refaz Etapas 4–5 sem recalcular o SfM
./run.sh dados/sala.mp4 saida --reuse-sfm --reuse-dense   # idem, reaproveitando a nuvem do PatchMatch
./run.sh demo                                 # sala sintética 5,0 x 4,0 x 2,7 m (sanidade)
```

Outras opções: `--no-dense`, `--no-video`, `--no-analysis`, `--config outro.yaml`.

## Visualizar e medir

```bash
python visualize.py saida             # Open3D: nuvem + câmeras + trajetória
python visualize.py saida --sparse    # nuvem esparsa
python visualize.py saida --video --sparse                            # vídeo obrigatório (esparsa + poses)
python visualize.py saida --video --name video_navegacao_densa.mp4    # com a nuvem densa
python visualize.py saida --glb       # cena.glb para a web
python extras.py saida                # todos os extras em extras/ (abaixo)

# medida manual (ex.: porta), depois some em config.yaml como model: "manual:altura_porta"
python measure.py saida --name altura_porta --frames frame_00012.jpg frame_00020.jpg
python measure.py saida --update-metrics
```

## Saídas (`saida/`)

| Arquivo | Conteúdo |
|---|---|
| `metricas.md` / `.json` | reprojeção, nº de pontos, densidade, erro das dimensões, deriva do laço, ângulo entre paredes |
| `planta_baixa.png`, `planta_ocupacao.png` | planta vista de cima com cotas e trajetória |
| `trajetoria_sobre_nuvem*.png` | trajetória + poses das câmeras sobre a nuvem |
| `nuvem_esparsa.ply`, `nuvem_densa.ply`, `nuvem_densa_downsample.ply` | nuvens limpas, em metros, z para cima |
| `cameras.ply`, `trajetoria.ply`, `trajetoria.csv` | poses (pirâmides) e trajeto |
| `video_navegacao.mp4` | vídeo de 30–60 s: nuvem esparsa + poses das câmeras (obrigatório) |
| `video_navegacao_densa.mp4` | o mesmo percurso com a nuvem densa |
| `etapas_2_3/` | keypoints, SIFT x ORB, matches antes/depois, linhas epipolares, conectividade |
| `ablacao/ablacao.md`, `.csv`, `.png` | exaustivo x sequencial (± laço) x nº de quadros |
| `modelo_metrico/` | modelo COLMAP escalado e alinhado (abre no GUI do COLMAP) |
| `pipeline.log` | log completo |

Câmeras com pose incoerente (fora da sala ou salto isolado na trajetória; o mapeador
global às vezes deixa algumas mal ancoradas) ficam fora do trajeto, da deriva do laço,
das figuras e dos vídeos; a lista sai em `sala.json` (`excluded_cameras`).

## Extras (`python extras.py saida`)

| Pasta | Conteúdo |
|---|---|
| `extras/1_poses_glb/` | nuvem + poses das câmeras + trajetória em GLB (esparsa e densa) |
| `extras/2_casos_dificeis/` | paredes lisas: limiar de contraste do SIFT (0,0067 × 0,002) e CLAHE |
| `extras/3_patchmatch_vs_sgbm/` | PatchMatch (GPU) × SGBM retificado (CPU): pontos, cobertura, ruído, tempo |
| `extras/4_angulo_paredes/` | planos do chão/paredes por RANSAC e ângulo entre paredes vizinhas |
| `extras/5_malha_poisson/` | malha (Poisson, Open3D) a partir da nuvem densa, com figura e vídeo |

`dense.fusion_min_num_pixels` (config) controla quantas vistas confirmam cada ponto na fusão
do PatchMatch (COLMAP: 5); com `--reuse-dense`, mudar esse valor refaz só a fusão (~2 min).
