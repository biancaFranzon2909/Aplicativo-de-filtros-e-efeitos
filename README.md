# App de Filtros - Trabalho Grau B (Processamento Gráfico)

## Equipe

Bianca Franzon
Bruno Haefliger
Tobias Klein

## Descrição do Projeto

Protótipo de aplicativo de edição de imagem e vídeo inspirado nos *stories* do Instagram,
desenvolvido em Python com OpenCV. Permite aplicar filtros, operações aritméticas e stickers
sobre imagens estáticas ou frames de webcam/vídeo.

## Estrutura do Projeto

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Ponto de entrada; loop principal; Modo Foto e Modo Vídeo |
| `filtros.py` | 16 filtros com descrições e trackbars de parâmetros |
| `operacoes.py` | 4 operações aritméticas entre duas imagens |
| `stickers.py` | Overlay com alpha, stickers estáticos por clique e sticker animado (spritesheet) |
| `face.py` | Detecção de faces com Haar Cascades; óculos automáticos e blur de privacidade |
| `assets/stickers/` | PNGs RGBA dos stickers (gerados automaticamente + eyeglasses.png) |
| `assets/spritesheet/` | Spritesheet `pipo-nekonin022.png` para sticker animado |
| `assets/haarcascades/` | XMLs do Haar Cascade para face e olhos |
| `assets/imagens/` | Imagens de entrada padrão (sample.jpg, sample2.jpg) |
| `outputs/` | Imagens salvas pelo app |
| `requirements.txt` | Dependências Python |

## Informações Técnicas

- **Linguagem:** Python 3
- **Bibliotecas:** `opencv-contrib-python`, `numpy`
- **Interface:** OpenCV HighGUI (`imshow`, `waitKey`, `createTrackbar`, `setMouseCallback`)
- **Sem** framework de GUI externo

> **IMPORTANTE:** O pacote necessário é `opencv-contrib-python`, **não** `opencv-python`.
> Nas versões mais recentes do OpenCV (5+), `CascadeClassifier` foi movido para o módulo `contrib`.

## Instalação e Execução

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar
python main.py
```

## Controles de Teclado

| Tecla | Ação |
|-------|------|
| `F` / `f` | Próximo / Anterior filtro (pré-visualização em tempo real) |
| `A` | Aplicar filtro na imagem de trabalho |
| `U` | Desfazer último filtro |
| `C` | Trocar canal de exibição (Todos → R → G → B) |
| `1` `2` `3` `4` | Operações aritméticas (Adição / Subtração / Blending / Diferença) |
| `S` / `s` | Próximo / Anterior sticker |
| Clique esquerdo | Colocar sticker na posição do mouse (Modo Foto) |
| `D` | Detectar faces - toggle |
| `E` | Trocar modo de face (óculos ↔ blur de privacidade) |
| `Z` | Sticker animado - toggle |
| `V` | Trocar modo (Foto ↔ Vídeo) |
| `G` | Gravar / Salvar imagem atual |
| `R` | Resetar (volta à imagem original) |
| `M` | Mostrar menu de controles |
| `Q` / `ESC` | Sair |

## Checklist de Requisitos

### Requisitos Base
- [x] Modo Foto (imagem do disco)
- [x] Modo Vídeo (webcam com fallback para arquivo)
- [x] ≥ 10 filtros de processamento de imagem (implementados: 16)
  - Suavização: Box Blur, Gaussian Blur, Mediana
  - Realce: Nitidez (Sharpen), Laplaciano
  - Bordas: Sobel, Canny
  - Outros: Emboss, Negativo, Escala de Cinza, Sépia, Brilho/Contraste, Limiarização, Dilatação, Erosão, Posterização
- [x] Descrição do filtro exibida ao selecionar (console)
- [x] Parâmetros configuráveis via trackbars (Controles)
- [x] Seleção de canais individuais R / G / B
- [x] ≥ 3 operações aritméticas entre duas imagens (implementadas: 4)
- [x] ≥ 5 stickers PNG com canal alpha (7 disponíveis)
- [x] Posicionamento de sticker por clique do mouse
- [x] Salvar resultado
- [x] Resetar para imagem original

### Extras (+ 1,0 ponto)
- [x] **Combinação de filtros** - filtros aplicados em sequência sobre `img_trabalho` + desfazer
- [x] **Detecção de face** - Haar Cascade com óculos automáticos (modo sticker) e blur de privacidade
- [x] **Stickers animados** - spritesheet `pipo-nekonin022.png` (3×4 frames) animado em tempo real
- [ ] **Interface Gráfica** - Implementar a interface gráfica usando uma biblioteca externa (ex: Qt, Dear ImGui, Tkinter, etc.), indo além do console ou HighGUI básico.

## Referências

- Spritesheet: Pipoya Free RPG Character Sprites - https://pipoya.itch.io/pipoya-free-rpg-character-sprites-nekonin
- Haar Cascades: OpenCV (Intel Corporation) - https://github.com/opencv/opencv/tree/master/data/haarcascades
- Blending alpha: https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
- Exemplos base: repositório da Prof. Rossana Baptista Queiroz (PG2026-2)
