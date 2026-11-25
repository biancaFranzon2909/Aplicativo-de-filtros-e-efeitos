"""
stickers.py — Stickers com canal alpha e sticker animado (spritesheet).

Padrão de blending alpha retirado de MouseSticker.py da professora:
  roi * (1 - alpha_3c) + sticker_bgr * alpha_3c

Sticker animado usa spritesheet pipo-nekonin022.png (3 colunas × 4 linhas = 12 frames).
"""

import cv2 as cv
import numpy as np
import os
import time

DIR = os.path.dirname(os.path.abspath(__file__))
DIR_STICKERS = os.path.join(DIR, 'assets', 'stickers')
DIR_SPRITE   = os.path.join(DIR, 'assets', 'spritesheet')

CAMINHO_SPRITESHEET = os.path.join(DIR_SPRITE, 'pipo-nekonin022.png')

# ─── Geração de stickers em código (caso PNGs não existam) ───────────────────

def _criar_imagem_rgba(w, h):
    """Cria uma imagem BGRA transparente de tamanho w×h."""
    return np.zeros((h, w, 4), dtype=np.uint8)

def _gerar_coracao():
    """Gera um sticker coração vermelho com alpha."""
    img = _criar_imagem_rgba(100, 90)
    # Dois círculos + triângulo formam um coração
    cv.circle(img, (30, 30), 28, (50, 50, 220, 255), -1)
    cv.circle(img, (70, 30), 28, (50, 50, 220, 255), -1)
    pts = np.array([[2, 42], [98, 42], [50, 88]], np.int32)
    cv.fillPoly(img, [pts], (50, 50, 220, 255))
    return img

def _gerar_estrela():
    """Gera um sticker estrela de 5 pontas dourada com alpha."""
    img = _criar_imagem_rgba(100, 100)
    cx, cy, r_ext, r_int = 50, 50, 48, 20
    pts = []
    for i in range(10):
        ang = np.radians(i * 36 - 90)
        r = r_ext if i % 2 == 0 else r_int
        pts.append([int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))])
    pts_np = np.array(pts, np.int32)
    cv.fillPoly(img, [pts_np], (0, 215, 255, 255))   # dourado (BGR)
    cv.polylines(img, [pts_np], True, (0, 150, 180, 255), 2)
    return img

def _gerar_bigode():
    """Gera um sticker bigode preto com alpha."""
    img = _criar_imagem_rgba(130, 60)
    # Dois arcos formam um bigode
    cv.ellipse(img, (40, 20), (38, 22), 0, 0, 180, (10, 10, 10, 255), -1)
    cv.ellipse(img, (90, 20), (38, 22), 0, 0, 180, (10, 10, 10, 255), -1)
    cv.ellipse(img, (40, 20), (38, 22), 0, 0, 180, (60, 60, 60, 255), 2)
    cv.ellipse(img, (90, 20), (38, 22), 0, 0, 180, (60, 60, 60, 255), 2)
    return img

def _gerar_chapeu():
    """Gera um sticker chapéu-coco preto com alpha."""
    img = _criar_imagem_rgba(110, 100)
    # Aba
    cv.ellipse(img, (55, 80), (55, 18), 0, 0, 360, (20, 20, 20, 255), -1)
    # Copa
    pts = np.array([[20, 80], [90, 80], [82, 10], [28, 10]], np.int32)
    cv.fillPoly(img, [pts], (20, 20, 20, 255))
    # Fita
    cv.rectangle(img, (20, 65), (90, 75), (30, 10, 80, 255), -1)
    return img

def _gerar_balao():
    """Gera um sticker balão de fala branco com alpha."""
    img = _criar_imagem_rgba(120, 100)
    # Bolha principal
    cv.ellipse(img, (60, 42), (58, 40), 0, 0, 360, (255, 255, 255, 230), -1)
    cv.ellipse(img, (60, 42), (58, 40), 0, 0, 360, (180, 180, 180, 255), 2)
    # Rabo do balão
    pts = np.array([[40, 72], [60, 82], [80, 72]], np.int32)
    cv.fillPoly(img, [pts], (255, 255, 255, 230))
    # Pontinhos "..."
    for dx in [-16, 0, 16]:
        cv.circle(img, (60 + dx, 42), 6, (120, 120, 120, 255), -1)
    return img

def _gerar_coroa():
    """Gera um sticker coroa dourada com alpha."""
    img = _criar_imagem_rgba(120, 80)
    pts_base = np.array([[5, 70], [115, 70], [115, 35], [90, 55],
                          [60, 10], [30, 55], [5, 35]], np.int32)
    cv.fillPoly(img, [pts_base], (0, 200, 255, 255))   # dourado
    cv.polylines(img, [pts_base], True, (0, 140, 200, 255), 2)
    # Gemas
    for cx, cy, cor in [(30, 60, (255, 50, 50, 255)),
                         (60, 60, (50, 255, 50, 255)),
                         (90, 60, (50, 50, 255, 255))]:
        cv.circle(img, (cx, cy), 7, cor, -1)
    return img


GERADORES = [
    ('coracao.png',  _gerar_coracao),
    ('estrela.png',  _gerar_estrela),
    ('bigode.png',   _gerar_bigode),
    ('chapeu.png',   _gerar_chapeu),
    ('balao.png',    _gerar_balao),
    ('coroa.png',    _gerar_coroa),
]

def garantir_stickers():
    """
    Gera os PNGs de sticker que não existirem.
    Mantém eyeglasses.png e sticker_original.png (copiados da professora) intactos.
    """
    os.makedirs(DIR_STICKERS, exist_ok=True)
    for nome_arquivo, gerador in GERADORES:
        caminho = os.path.join(DIR_STICKERS, nome_arquivo)
        if not os.path.exists(caminho):
            img = gerador()
            cv.imwrite(caminho, img)
            print(f"[Sticker gerado]: {nome_arquivo}")


# ─── Carregamento dos stickers ────────────────────────────────────────────────

def carregar_stickers(tamanho_max=90):
    """Carrega todos os PNGs com alpha da pasta assets/stickers/ e padroniza o tamanho."""
    garantir_stickers()
    stickers = []
    for arquivo in sorted(os.listdir(DIR_STICKERS)):
        if arquivo.lower().endswith('.png'):
            caminho = os.path.join(DIR_STICKERS, arquivo)
            img = cv.imread(caminho, cv.IMREAD_UNCHANGED)
            if img is not None:
                # Garante 4 canais (BGRA)
                if len(img.shape) == 2:
                    img = cv.cvtColor(img, cv.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3:
                    img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

                # Redimensiona proporcionalmente para caber em tamanho_max
                h, w = img.shape[:2]
                escala = tamanho_max / max(h, w)
                novo_w = max(1, int(w * escala))
                novo_h = max(1, int(h * escala))
                img = cv.resize(img, (novo_w, novo_h), interpolation=cv.INTER_AREA)

                nome = os.path.splitext(arquivo)[0].replace('_', ' ').title()
                stickers.append({'nome': nome, 'imagem': img})
    if not stickers:
        print("[AVISO] Nenhum sticker encontrado em assets/stickers/")
    return stickers


# ─── Blending alpha (padrão MouseSticker.py da professora) ───────────────────

def aplicar_sticker_alpha(fundo, sticker_bgra, x, y):
    """
    Sobrepõe sticker_bgra sobre fundo na posição (x, y) usando blending alpha.
    Padrão direto de MouseSticker.py:
        resultado = fundo*(1-alpha) + sticker*alpha
    """
    h_fundo, w_fundo = fundo.shape[:2]
    h_s, w_s = sticker_bgra.shape[:2]

    # Recorta se o sticker ultrapassa a borda direita/inferior
    x2 = min(x + w_s, w_fundo)
    y2 = min(y + h_s, h_fundo)
    if x >= w_fundo or y >= h_fundo or x2 <= 0 or y2 <= 0:
        return fundo

    # Ajusta coordenadas do sticker caso comece antes da borda esquerda/superior
    x_s = max(0, -x)
    y_s = max(0, -y)
    x = max(0, x)
    y = max(0, y)

    sticker_crop = sticker_bgra[y_s:y_s+(y2-y), x_s:x_s+(x2-x)]

    b, g, r, a = cv.split(sticker_crop)
    sticker_bgr = cv.merge([b, g, r])

    # Normaliza alpha: 0-255 → 0.0-1.0
    alpha = a.astype(np.float32) / 255.0
    alpha_3c = cv.merge([alpha, alpha, alpha])

    roi = fundo[y:y2, x:x2].astype(np.float32)

    fundo_suavizado   = roi * (1.0 - alpha_3c)
    sticker_suavizado = sticker_bgr.astype(np.float32) * alpha_3c
    resultado = cv.add(fundo_suavizado, sticker_suavizado).astype(np.uint8)

    fundo_copia = fundo.copy()
    fundo_copia[y:y2, x:x2] = resultado
    return fundo_copia


def aplicar_lista_stickers(fundo, posicoes, stickers_lista, idx_sticker):
    """
    Aplica todos os stickers da lista de posições sobre fundo.
    Cada entrada em posicoes é um dict {'x', 'y', 'idx'}.
    """
    resultado = fundo.copy()
    for pos in posicoes:
        s = stickers_lista[pos['idx']]
        resultado = aplicar_sticker_alpha(resultado, s['imagem'], pos['x'], pos['y'])
    return resultado


# ─── Sticker animado (spritesheet) ───────────────────────────────────────────

class StickerAnimado:
    """
    Carrega um spritesheet BGRA e anima os frames ao longo do tempo.
    pipo-nekonin022.png: grade 3 colunas × 4 linhas = 12 frames.
    """

    def __init__(self, caminho=CAMINHO_SPRITESHEET, n_colunas=3, n_linhas=4, fps=8):
        self.frames = []
        self.idx_frame = 0
        self.fps = fps
        self._ultimo_tempo = time.time()
        self.posicao = None       # (x, y) definido pelo clique do mouse
        self.ativo = False
        self.escala = 1.5         # amplia o sprite para ficar visível

        if os.path.exists(caminho):
            self._carregar_spritesheet(caminho, n_colunas, n_linhas)
        else:
            print(f"[AVISO] Spritesheet não encontrado: {caminho}")

    def _carregar_spritesheet(self, caminho, n_colunas, n_linhas):
        """Fatia o spritesheet em frames individuais."""
        sheet = cv.imread(caminho, cv.IMREAD_UNCHANGED)
        if sheet is None:
            print(f"[ERRO] Não foi possível carregar: {caminho}")
            return
        # Garante 4 canais
        if sheet.shape[2] == 3:
            sheet = cv.cvtColor(sheet, cv.COLOR_BGR2BGRA)
        h_sheet, w_sheet = sheet.shape[:2]
        h_frame = h_sheet // n_linhas
        w_frame = w_sheet // n_colunas
        self.frames = []
        for linha in range(n_linhas):
            for col in range(n_colunas):
                y0 = linha * h_frame
                x0 = col * w_frame
                frame = sheet[y0:y0+h_frame, x0:x0+w_frame].copy()
                # Redimensiona conforme escala
                novo_w = int(w_frame * self.escala)
                novo_h = int(h_frame * self.escala)
                frame = cv.resize(frame, (novo_w, novo_h), interpolation=cv.INTER_NEAREST)
                self.frames.append(frame)
        print(f"[Spritesheet carregado]: {len(self.frames)} frames ({w_frame}×{h_frame} px cada)")

    def avancar_frame(self):
        """Avança o frame com base no tempo decorrido."""
        if not self.frames:
            return
        agora = time.time()
        if agora - self._ultimo_tempo >= 1.0 / self.fps:
            self.idx_frame = (self.idx_frame + 1) % len(self.frames)
            self._ultimo_tempo = agora

    def desenhar(self, fundo):
        """Aplica o frame atual sobre fundo na posição definida."""
        if not self.ativo or self.posicao is None or not self.frames:
            return fundo
        self.avancar_frame()
        frame = self.frames[self.idx_frame]
        x, y = self.posicao
        # Centraliza o sticker no clique
        h_f, w_f = frame.shape[:2]
        x -= w_f // 2
        y -= h_f // 2
        return aplicar_sticker_alpha(fundo, frame, x, y)

    def definir_posicao(self, x, y):
        self.posicao = (x, y)
        self.ativo = True

    def toggle(self):
        """Liga/desliga o sticker animado (mantém posição)."""
        self.ativo = not self.ativo
        estado = "ATIVO" if self.ativo else "INATIVO"
        print(f"[Sticker animado]: {estado}")
