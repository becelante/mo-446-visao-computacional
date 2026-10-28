# Panoramas — Trabalho 1 (Visão Computacional, 2º sem. 2026)

Pipeline completo, passo a passo, para o Trabalho 1 (Prof. Anderson Rocha):
detecção de características → emparelhamento → ordenação automática sem
EXIF → homografia com RANSAC → composição com remoção de fantasmas.

## Estrutura

```
panorama_pipeline/
├── requirements.txt
├── src/
│   ├── raw_convert.py   # Etapa 0 (auxiliar): converte .ARW -> .PNG
│   ├── features.py      # Etapa 2: detecção de características (SIFT/ORB/AKAZE)
│   ├── matching.py      # Etapa 3: emparelhamento + ratio test de Lowe
│   ├── ordering.py       # Etapa 4: matriz de conectividade, grafo, ordenação, intrusa
│   ├── homography.py    # Etapa 5: homografia RANSAC + composição de transformações
│   ├── blending.py      # Etapa 6: composição, costura ótima, feathering
│   └── pipeline.py      # Script principal que executa tudo em sequência
└── README.md
```

Cada módulo corresponde a uma etapa do enunciado e pode ser importado e
testado separadamente (é assim que o pipeline foi validado, com imagens
sintéticas, antes de ser entregue a você).

## 1. Instalação

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Suas fotos estão em .ARW — converta primeiro

Como vocês já fotografaram em RAW (Sony `.ARW`), o primeiro passo é
converter para PNG (o restante do pipeline usa OpenCV, que não lê `.ARW`
diretamente):

```bash
python -m src.raw_convert --input_dir fotos_arw --output_dir fotos_png
```

Isso faz a demosaicagem + balanço de branco da câmera e redimensiona o
lado maior para no máximo 1600 px (ajustável com `--max_dim`), o que já
acelera bastante o pipeline sem perder qualidade perceptível para o
relatório. Use `--anonymize` se quiser garantir que os nomes de saída não
carreguem nenhuma pista da ordem de captura (por padrão os nomes originais
da câmera já não seguem uma ordem alfabética óbvia).

**Lembrete do enunciado (Etapa 1.2):** garanta que pelo menos uma cena
capturada tenha um elemento móvel (pessoa, carro, folhagem ao vento)
presente em posições diferentes em pelo menos duas fotos — é isso que gera
o fantasma que a Etapa 6 precisa remover. Se as fotos que vocês já
tiraram não tiverem isso, vale bater mais uma leva rápida só com esse
requisito.

## 3. Rodando o pipeline completo

```bash
python -m src.pipeline --input_dir fotos_png --output_dir resultados \
    --detector SIFT --ratio 0.75 --min_inliers 20
```

Isso roda as Etapas 2 a 6 automaticamente e gera em `resultados/`:

- `panorama_ingenuo.png` — composição simples, evidenciando os fantasmas
  (útil para o "antes" da comparação pedida na Etapa 6.4)
- `panorama_final.png` — panorama final, com costura ótima + feathering
- `matriz_conectividade.csv` — matriz de inliers por par (Etapa 4.5)
- `etapa2_keypoints/` — keypoints desenhados por imagem e por detector
  (para a comparação da Etapa 2.3)
- `etapa3_matches/` — visualização dos matches para cada par consecutivo
  da ordem inferida (Etapa 3.3)

O terminal também imprime, em texto, tudo que o relatório (E1) pede para
citar: contagem de keypoints por detector, número de matches antes/depois
do ratio test, a ordem inferida, quais imagens foram rejeitadas como
intrusas, e a taxa de inliers + erro de reprojeção médio de cada par
(Etapa 5.3).

### Parâmetros úteis

| Parâmetro | Para quê serve | Padrão |
|---|---|---|
| `--detector` | `SIFT`, `ORB` ou `AKAZE` (Etapa 2.3 pede comparar pelo menos 2) | `SIFT` |
| `--ratio` | limiar do ratio test de Lowe (Etapa 3.2) | `0.75` |
| `--ransac_thresh` | tolerância (px) do RANSAC na homografia | `4.0` |
| `--min_inliers` | mínimo de inliers para considerar duas imagens vizinhas no grafo (Etapa 4.2) | `20` |
| `--feather_width` | largura (px) da faixa de suavização ao redor da costura | `15` |

Se a imagem intrusa (Etapa 4.4) não estiver sendo rejeitada, ou se imagens
de fato vizinhas estiverem sendo descartadas por engano, ajuste
`--min_inliers` (suba para ser mais rígido, desça para ser mais permissivo).

## 4. Como cada etapa do enunciado foi implementada

- **Etapa 2** (`features.py`): `cv2.SIFT_create`, `cv2.ORB_create` e
  `cv2.AKAZE_create`. `compare_detectors()` roda os três na mesma imagem e
  conta keypoints — é o ponto de partida para a justificativa pedida em
  2.3 (na prática, SIFT costuma achar menos pontos, porém mais estáveis;
  ORB acha muitos, rápidos, porém menos robustos a mudança de escala).
- **Etapa 3** (`matching.py`): FLANN para descritores SIFT (float) e
  `BFMatcher` com distância de Hamming para ORB/AKAZE (binários), sempre
  com `knnMatch(k=2)` + ratio test de Lowe.
- **Etapa 4** (`ordering.py`): para cada par de imagens, conta quantos
  matches sobrevivem como *inliers* de uma homografia RANSAC — isso monta
  a matriz de conectividade da Figura 4. Pares com poucos inliers viram
  "não vizinhos" no grafo; imagens sem nenhum vizinho válido são
  descartadas como intrusas. A ordem é inferida caminhando pelo grafo
  sempre em direção ao vizinho não visitado com mais inliers (maior
  confiança), partindo da imagem de menor grau (tende a ser uma ponta do
  panorama).
- **Etapa 5** (`homography.py`): homografia par a par com
  `cv2.findHomography(..., cv2.RANSAC, ...)`, com taxa de inliers e erro
  de reprojeção médio calculados explicitamente. As homografias dos pares
  consecutivos são compostas (multiplicação de matrizes, propagando a
  partir da imagem central da sequência) para levar todas as imagens a um
  referencial comum — é uma simplificação do *bundle adjustment* global
  citado no Extra X1 (erros pequenos se acumulam ao longo da cadeia; para
  poucas imagens isso costuma ser imperceptível).
- **Etapa 6** (`blending.py`): a composição ingênua faz média simples nas
  sobreposições (evidenciando fantasma). A remoção de fantasmas busca,
  via programação dinâmica (mesma ideia de *seam carving*), o caminho de
  custo mínimo (menor diferença de cor entre as duas imagens) atravessando
  a região de sobreposição — esse caminho tende a contornar o objeto
  móvel, já que ali a diferença entre as duas fotos é grande. Um
  feathering linear numa faixa estreita ao redor da costura suaviza a
  transição residual.

## 5. Sobre o código executável (Entregável E3)

`src/pipeline.py` já é exatamente esse entregável: recebe a pasta de
imagens fora de ordem e produz o panorama final sem intervenção manual.
Para a entrega, basta apontar `--input_dir` para a pasta final de fotos
(depois de convertidas de `.ARW`) — não precisa editar nada no código.

## 6. Validação feita antes da entrega

Antes de te passar o código, rodei o pipeline inteiro com imagens
sintéticas (uma cena gerada proceduralmente, recortada em 5 janelas
sobrepostas + 1 imagem intrusa aleatória, com um "objeto móvel" colocado
deliberadamente na faixa de sobreposição entre duas fotos) para confirmar
que: a intrusa é rejeitada, a ordem é inferida corretamente a partir do
grafo, e a composição final realmente remove o fantasma que aparece na
composição ingênua. Com fotos reais (que têm textura rica) a costura
tende a ficar ainda mais estável do que no teste sintético.

## Sugestões para os itens Extras (X1–X4)

Se sobrar tempo, os pontos mais fáceis de encaixar no que já existe:

- **X3** (compensação de exposição): normalizar o brilho médio de cada
  imagem antes da Etapa 6, usando a razão entre as médias na região de
  sobreposição.
- **X4** (comparação com `cv2.Stitcher`): rodar `cv2.Stitcher_create()`
  nas mesmas fotos e comparar visualmente/qualitativamente com
  `panorama_final.png` — dá para incluir isso como uma função extra sem
  mexer no pipeline principal.
