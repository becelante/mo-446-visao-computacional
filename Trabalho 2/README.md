# Trabalho 2: Reconstrução 3D de ambiente interno (MO446, Variante A)

Coloque o vídeo `.MTS` em `videos/`, escolha-o em `VIDEO` no config (ou rode `VIDEO=B.mts ./run.sh`) e rode `./run.sh`. Quadros, cache e resultados de cada vídeo ficam separados, em `frames/<vídeo>/` e `results/<vídeo>/`. Os parâmetros ficam em [src/config.py](src/config.py).

```bash
uv run python 1_extrair_quadros.py        # quadros mais nítidos do vídeo -> frames/<vídeo>/
uv run python 2_comparar_detectores.py    # Etapa 2 -> results/deteccao/
uv run python 3_reconstrucao.py --detector SIFT --densa   # Etapas 3 a 5 -> results/sift/
uv run python 3_reconstrucao.py --detector ORB            # mesmo pipeline com ORB -> results/orb/
uv run python 4_ablacao.py --detector SIFT                # ablação -> results/sift/ablacao/
uv run python visualizar.py [--detector ORB]              # janela interativa (--densa, --video, --medir)
```

Extras opcionais (fora do `run.sh` principal):

```bash
uv run python 5_colmap.py      # referência: o mesmo vídeo reconstruído pelo COLMAP -> results/<vídeo>/colmap/
uv run python 6_malha.py       # malha colorida (Poisson) da nuvem densa -> malha.ply, malha.glb
uv run python visualizar.py --malha [--video navegacao_malha.mp4]
```

A malha é feita assim: a nuvem densa é reduzida por voxel, as normais são orientadas para a câmera mais próxima e a superfície sai pela reconstrução de Poisson. Os triângulos de baixa densidade são cortados, porque o Poisson "inventa" membranas onde não há pontos. O `.glb` abre em visualizadores web de glTF.

## Antes de rodar

- **Tabuleiro**: o PDF fica em `images/`. Os cantos internos são lidos do nome do arquivo (`...-8x6...`: 9 × 7 quadrados, ou seja, 8 × 6 cantos internos). O lado do quadrado é o medido na folha impressa, `CHECKERBOARD_SQUARE_M = 0.022` (o PDF diz 25 mm, mas a impressão saiu reduzida para 88%). Um só tabuleiro foi impresso e mudado de lugar durante a gravação. As detecções são agrupadas por posição, cada posição dá uma escala, e a dispersão entre elas entra nas métricas: ela mostra se a escala deriva ao longo do trajeto.
- **Medidas reais**: preencha `REAL_MEASUREMENTS_BY_VIDEO` em `config.py`, com as medidas da trena em metros, separadas por vídeo (cada vídeo é uma sala). Largura e comprimento são medidos no modelo automaticamente. Outras medidas, como a altura da porta, são marcadas à mão com `uv run python visualizar.py --medir altura_porta` (shift + clique em 2 pontos). Depois, rode `3_reconstrucao.py` de novo para atualizar a tabela de erro. Ele usa o cache, então leva poucos minutos.

## Pipeline

| Etapa | Onde | Roda em |
|---|---|---|
| Quadros: o mais nítido (variância do laplaciano) de cada janela de `duração / N_FRAMES` | `step_0_frames.py` | ffmpeg com NVDEC |
| Detecção: SIFT (padrão) ou ORB, do OpenCV | `step_1_keypoints_detector.py` | CPU com threads |
| Matching: força bruta exata (como o `BFMatcher`) e ratio test de Lowe | `step_2_matching.py` | GPU |
| Pares: sequencial (`SEQ_WINDOW`) + laço por bag of visual words (k-means e tf-idf) | `step_2_matching.py` | GPU |
| F e E com RANSAC, pose, triangulação | `step_3_epipolar.py` | CPU com threads, em paralelo ao matching |
| SfM incremental (PnP + triangulação) | `step_4_sfm.py` | CPU, buscas vetorizadas |
| Bundle adjustment (LM + complemento de Schur, refina a focal) | `step_4_sfm.py` | GPU |
| Escala pelo(s) tabuleiro(s), limpeza, alinhamento ao chão, planta baixa, planos | `step_4_sfm.py`, `step_4_postprocessing.py` | CPU |
| Nuvem densa: plane-sweep com NCC e consistência geométrica | `step_4_dense.py` | GPU |

O SIFT continua no OpenCV: no kornia (GPU) ele foi cerca de 10× mais lento nos quadros 1080p. RANSAC (F e E) e PnP continuam no OpenCV, na CPU. Com `DEVICE = "cpu"` no config, todo o código torch roda na CPU, só que mais devagar. Como extra, o código também aceita SuperPoint, DISK e ALIKED com LightGlue (`DETECTOR` / `DETECTORS_COMPARED`).

## Saídas (`results/<vídeo>/<detector>/`; a comparação de detectores fica em `results/<vídeo>/deteccao/`)

- `deteccao/`: keypoints por detector e `comparacao.md`.
- `epipolar/`: matches antes e depois do RANSAC e linhas epipolares de um par sequencial e do fechamento de laço.
- `esparsa.ply`, `esparsa_limpa.ply`, `densa.ply`, `cameras.ply` (frustums), `trajetoria.ply`, `planta_baixa.png`.
- `metricas.md` / `metricas.json`: erro de reprojeção, número de pontos, densidade, deriva do laço, erro métrico, planos horizontais e ângulos entre paredes.
- `ablacao/ablacao.md`: sequencial × sequencial com laço × exaustivo, e número de quadros.
