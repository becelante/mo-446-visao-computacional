# Trabalho 1: Panoramas

Panoramas gerados automaticamente a partir de fotos RAW fora de ordem (Sony A7 II, FE 28-70 mm), sem uso de EXIF:

| Conjunto | Cena | Fotos | Captura | Resultado |
|---|---|---|---|---|
| **A1** | auditório | 61 + 1 intrusa (`DSC03778.ARW`, um gato de outra cena) | 28 mm | 19986 x 3505 px, 212° |
| **B1** | pátio | 48 | 28 mm, em colunas (várias alturas por direção), objetos próximos | 30331 x 5300 px, 360° (Extra X2) |
| **D1** | arquibancada ao ar livre | 9 | 30 mm | 12999 x 3340 px, 128° |
| **C1** | parque | 19 | 37 mm, objetos próximos, folhagem ao vento | 11132 x 4645 px |

## Como replicar

As fotos não são versionadas (o GitHub recusa arquivos acima de 100 MB). Antes de rodar, coloque os `.ARW` de cada conjunto em `A1/`, `B1/`, `C1/` e `D1/` (no `A1/`, as 61 fotos do auditório e a intrusa `DSC03778.ARW`). Requisitos: Python 3.12, [uv](https://docs.astral.sh/uv/) e, para o passo 3, uma GPU NVIDIA.

```bash
uv sync                               # dependências (inclui PyTorch e LightGlue, usados no passo 3)
uv run python 1_revelar_raw.py        # RAW -> PNG 16 bits com nomes embaralhados, em *_png/ (~2 min)
uv run python 2_panoramas.py          # etapas 1 a 6 para A1, B1, D1 e C1, em *_results/ (~16 min, pico de ~13 GB de RAM)
uv run python 3_comparacao_x4.py      # Extra X4 no A1: SuperPoint + LightGlue na GPU e cv2.Stitcher (~3 min)
```

Os scripts não recebem argumentos: todos os parâmetros ficam em `src/config.py` (por exemplo, `COMPOSE_SCALE = 0.25` gera uma prévia bem mais rápida). O passo 2 roda na CPU, porque o OpenCV do PyPI não tem CUDA; só o passo 3 usa a GPU. A ordem de leitura é embaralhada e o FLANN (que é aleatório) roda com semente fixa, então o resultado é idêntico entre execuções. Isso importa no B1: com muita paralaxe, pequenas variações nos casamentos mudavam de que lado de um objeto próximo a costura passava.

## Estrutura

```
1_revelar_raw.py, 2_panoramas.py, 3_comparacao_x4.py   passos, em ordem de execução
src/config.py        todos os parâmetros, inclusive os ajustes por conjunto
src/raw_convert.py   passo 1: revelação do RAW
src/features.py      etapa 2: detecção (SIFT, ORB) e comparação de detectores
src/matching.py      etapa 3: emparelhamento e ratio test
src/ordering.py      etapa 4: matriz de conectividade, intrusas e ordem
src/homography.py    etapa 5: homografias, bundle adjustment, correção de onda
src/blending.py      etapa 6: projeção, fotometria, costuras, alinhamento local, multibanda
src/pipeline.py      encadeia as etapas 1 a 6 para um conjunto
src/comparacao.py    passo 3: Extra X4
```

As saídas em `*_results/` levam o número da etapa no nome, na ordem em que são produzidas:

| Arquivo | Conteúdo | Etapa |
|---|---|---|
| `0_mapeamento_nomes.csv` | nome embaralhado → nome original, só para conferir a ordem | 4.1 |
| `2_keypoints/` | keypoints de cada foto (escala e orientação) e `comparacao_detectores.csv` (SIFT x ORB) | 2.2 a 2.4 |
| `3_matches/` | casamentos de cada par consecutivo antes (`_antes`) e depois (`_depois`) da filtragem; verde: inlier, vermelho: outlier | 3.3 |
| `4_matriz_conectividade.png` / `.csv` | inliers por par, na ordem inferida, com as intrusas em vermelho no fim | 4.2, 4.5 |
| `5_metricas_alinhamento.csv` | por par consecutivo: matches, inliers, taxa de inliers, erro de reprojeção e erro após o bundle adjustment | 5.3 |
| `5_mosaico_progressivo/` | mosaico com 2, 4, 8, ... fotos | 5.4 |
| `6_panorama_final.jpg` / `.png` | panorama final (8 bits / 16 bits, este não versionado) | 6.1 a 6.3 |
| `6_panorama_ingenuo.png`, `6_costuras.png` | média simples (com fantasmas) e costuras escolhidas, a 1/4 da resolução | 6.4 |
| `6_comparacao_fantasmas.png` | recorte ampliado da região com mais fantasma: média simples x resultado final | 6.4 |
| `6_metricas_composicao.csv` | métricas do resultado (erro do alinhamento, descontinuidade nas costuras, ganhos, vinheta) | 6.5 |
| `x4_comparacao/` (só A1) | SIFT x SuperPoint + LightGlue por par e resumo; cv2.Stitcher e comparação visual | X4 |

## Pipeline

Para cada etapa, a escolha feita e por que ela foi preferida às alternativas.

**Passo 1. Revelação do RAW** (`raw_convert.py`). Demosaicagem DHT, saída sRGB de 16 bits em resolução total, balanço de branco e brilho únicos por conjunto, nomes aleatórios.
- RAW em vez do JPG da câmera: os 14 bits do sensor recuperam as sombras (no A1 o brilho é multiplicado por ~4,7) e a distorção da lente não vem "corrigida", então pode ser modelada junto com o alinhamento.
- DHT em vez do AHD (padrão do LibRaw): menos artefatos em bordas finas; o AMaZE não está disponível no rawpy.
- Balanço de branco e brilho únicos (mediana do conjunto) em vez de automáticos por foto: a câmera estava em AWB, e cada foto viria com cor e exposição diferentes, criando degraus entre vizinhas.
- 16 bits em vez de 8: clarear as sombras em 8 bits gera bandas.
- Nomes aleatórios (Etapa 4.1): a ordem de captura não pode ser deduzida dos arquivos.

**Etapa 1. Carregamento.** As imagens são lidas em ordem embaralhada; as etapas 2 a 5 usam uma cópia com lado maior de 3000 px.
- Cópia reduzida em vez da resolução total: 4 vezes menos pixels nas etapas mais caras (1891 pares no A1), sem perda real, pois o erro final é dominado pela paralaxe, não pela localização dos pontos.

**Etapa 2. Detecção.** SIFT, com CLAHE antes da detecção.
- SIFT em vez de ORB (`2_keypoints/comparacao_detectores.csv`): no par central do A1 os dois funcionam (1890 x 2063 inliers, erro de ~1 px), mas nos outros conjuntos o SIFT acha de 3 a 6 vezes mais inliers (B1: 731 x 159; D1: 6760 x 1115; C1: 97 x 30). O descritor de 128 floats do SIFT é mais distintivo em texturas repetitivas; o ORB é rápido, mas limitado a 4000 pontos e sensível a escala. O AKAZE saiu do módulo principal no OpenCV 5.
- CLAHE: nas fotos escuras das pontas do A1, multiplica os keypoints (~700 para ~5000 numa foto de canto).

**Etapa 3. Emparelhamento.** FLANN (k-d tree) com k = 2 e ratio test de Lowe a 0,75.
- FLANN em vez de força bruta: muito mais rápido para milhares de descritores float, com o mesmo resultado prático.
- Limiar 0,75 em vez de 0,8 (sugestão de Lowe): mais estrito, reduz casamentos falsos nos painéis do quadro e no forro.

**Etapa 4. Ordenação.** Homografia RANSAC para todos os pares (1891 no A1); o par só conta como vizinho se `inliers > 8 + β·matches` (Brown & Lowe), se a homografia não é degenerada (escala local entre 0,2 e 5) e se tem pelo menos 20 inliers. Imagem sem vizinhos é intrusa. A ordem sai da seriação espectral (vetor de Fiedler do Laplaciano) com trocas locais.
- Verificação de Brown & Lowe e teste de degeneração em vez de só um limiar de inliers: os painéis repetidos do A1 geram dezenas de "inliers" entre fotos sem sobreposição, e o cascalho da foto do gato (~50 mil keypoints) gerava 93 "inliers" colapsando centenas de pontos num só. Com os testes, o gato fica sem vizinhos e é rejeitado.
- β = 0,3 (valor de Brown & Lowe) em A1, B1 e D1; β = 0,05 no C1: na folhagem ao vento só 10 a 30% dos matches de um par real são inliers, e com 0,3 o C1 ficava com 2 das 19 fotos.
- Seriação espectral em vez de caminhada gulosa: a gulosa pulava imagens (no A1, a 38 tem mais inliers com a 36 que com a 37); o espectral usa todos os pares de uma vez, e as trocas locais corrigem inversões causadas por ruído. Numa volta de 360° o vetor de Fiedler "dobra" o anel, então a ordem sai do ângulo no plano dos dois primeiros autovetores.

**Etapa 5. Homografia e alinhamento** (Extra X1). As homografias RANSAC dos pares consecutivos dão as métricas da Etapa 5.3. O alinhamento final é um *bundle adjustment*: cada foto é uma câmera que só gira, com focal, ponto principal e distorção radial (k1 a k3) comuns, ajustados com perda robusta soft-L1 sobre todos os pares; no fim, a correção de onda deixa o horizonte reto.
- Bundle adjustment em vez de encadear homografias: o encadeamento acumula erro ao longo de 60 elos, e um plano não representa mais de 180° (pontos a 90° da referência vão para o infinito).
- Câmera em rotação pura com lente comum em vez de homografias livres: menos parâmetros e uma única distorção para a lente inteira, que o RAW não corrige (barril a 28 mm).
- Erro entre raios (como o `BundleAdjusterRay` do OpenCV) com perda robusta: casamentos em pessoas em movimento e com paralaxe não puxam a solução. Com pouca paralaxe (A1, D1), pares com erro mediano acima de 4 px saem numa segunda passada. Com muita paralaxe (B1, C1), nenhum par sai e a perda é mais tolerante, porque cortar os inconsistentes deixava câmeras presas a poucos pares e desalinhadas em centenas de px.

**Etapa 6. Composição** (`blending.py`). Projeção cilíndrica (A1, D1, C1) ou esférica (B1); compensação de ganho e vinheta; costura ótima; alinhamento local nas costuras; blending multibanda; recorte.
- Cilindro em vez de plano: o plano não comporta 212°, e o cilindro mantém as verticais retas. Esfera no B1, que olha bastante para cima e para baixo, e porque a projeção equirretangular é o formato padrão de 360°.
- Um único remap Lanczos4 por foto (rotação, distorção e projeção juntas) em vez de warps sucessivos: interpolar uma vez só preserva a nitidez.
- Ganho por foto e vinheta da lente (Extra X3), estimados nas sobreposições no espaço linear com mínimos quadrados robustos: a vinheta medida é de cerca de −0,8 EV nos cantos (f/3,5), e no B1 o ganho varia de 0,90 a 1,20 entre fotos.
- Costura ótima por programação dinâmica em vez de mediana ou média: as pessoas ficam paradas em várias fotos seguidas, então a mediana não as remove; a costura escolhe uma única foto por pixel, desvia das regiões em que as vistas discordam (fantasmas) e é atraída para o meio entre fotos vizinhas, onde distorção, vinheta e paralaxe são menores. Também foram testados corte em grafo (`cv2.detail_GraphCutSeamFinder`) e costuras horizontais entre fotos da mesma coluna no B1; ambos cortavam objetos próximos e duplicavam estruturas, e foram descartados.
- Alinhamento local por fluxo óptico (DIS) em cada costura: sem cabeça panorâmica há paralaxe, e objetos próximos ficam desalinhados mesmo após o bundle adjustment. Cada foto é deslocada metade do fluxo, só perto da costura, para o ajuste não se acumular; no B1 e no C1 o deslocamento máximo e a suavização são maiores (`src/config.py`).
- Multibanda (7 níveis) em vez de feathering: um feathering largo cria fantasma e um estreito deixa degrau; o multibanda troca os detalhes finos na costura e mistura a iluminação numa faixa larga. É feito em resolução total, uma foto por vez, só na área do recorte.
- Recorte: maior retângulo sem bordas pretas. No 360°, exatamente uma volta, cortada na coluna em que o início e o fim do canvas mais concordam, com uma transição curta para as bordas se emendarem.

## Resultados

| | A1 | B1 | D1 | C1 |
|---|---|---|---|---|
| Panorama final | 19986 x 3505 px, 212° | 30331 x 5300 px, 360° | 12999 x 3340 px, 128° | 11132 x 4645 px |
| Fotos usadas | 61 de 62 (intrusa rejeitada) | 48 de 48 | 9 de 9 | 19 de 19 |
| Ordem inferida | correta | ver limitações | correta (1) | correta |
| Erro do bundle adjustment (pares consecutivos, mediana) | 1,4 px | 12 px | 3,6 px | 15 px |
| Diferença média nas costuras, antes → depois do alinhamento local (0 a 255) | 3,8 → 3,5 | 8,6 → 6,5 | 4,3 → 3,2 | 17,3 → 17,2 |

(1) As fotos 737 e 738 do D1 são quase a mesma vista (22020 inliers entre elas, contra ~7000 com as vizinhas), e a ordem inferida as troca, sem efeito no panorama.

**Extra X4** (A1, `A1_results/x4_comparacao/`):

| Método | Keypoints | Matches | Inliers | Taxa de inliers | Erro de reprojeção | Tempo por par |
|---|---|---|---|---|---|---|
| SIFT + ratio test (CPU) | 7451 | 1754 | 1516 | 85% | 2,0 px | 0,51 s |
| SuperPoint + LightGlue (GPU) | 6358 | 3396 | 2691 | 78% | 3,5 px | 0,49 s |

Médias sobre os 60 pares consecutivos, com a mesma entrada (3000 px, CLAHE) e o mesmo RANSAC. O LightGlue encontra 1,8 vez mais inliers, mas localiza os pontos com menos precisão; como o erro final é dominado pela paralaxe, o SIFT basta para o pipeline. O `cv2.Stitcher` usou 61 das 62 fotos (também rejeitou a intrusa) e levou 36 s com as fotos reduzidas a 1500 px (`stitcher_opencv.jpg`, `comparacao_stitcher.jpg`).

## Limitações

- **Paralaxe no B1 e no C1.** A câmera se desloca entre as fotos (não gira em torno do ponto nodal da lente), e objetos próximos se movem em relação ao fundo de um jeito que nenhuma rotação explica. O alinhamento local corrige a maior parte, mas restam pequenos degraus onde estruturas próximas encontram o fundo: no B1, as bordas das marquises e corrimãos a 1 ou 2 m da câmera; no C1, os postes e frades a poucos metros. Eliminar isso exigiria refotografar com cabeça panorâmica (girando no ponto nodal da lente).
- **B1 foi fotografado em colunas** (3 ou 4 fotos em alturas diferentes para cada direção), então a ordem inferida não reproduz a ordem dos arquivos: as fotos não formam uma sequência única. A composição usa os ângulos do bundle adjustment e não depende dessa ordem.
- **C1:** com folhagem ao vento e paralaxe, a focal estimada fica acima da real (13052 px, contra ~7300 esperados para 37 mm), e o panorama sai com ângulo menor que o real. É o conjunto de pior qualidade: um dos postes de luz aparece duplicado (a costura passa entre as duas posições dele, que diferem pela paralaxe) e há uma faixa horizontal na copa à direita.
- As fotos de origem já são um pouco macias (f/3,5, ISO 640) e não há redução de ruído: o panorama mantém a nitidez das fotos, mas não a melhora.
