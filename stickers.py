"""stickers.py — Stickers com canal alpha e sticker animado por spritesheet."""

import cv2 as cv
import numpy as np
import os
import time

DIR = os.path.dirname(os.path.abspath(__file__))
DIR_STICKERS = os.path.join(DIR, 'assets', 'stickers')
DIR_SPRITE   = os.path.join(DIR, 'assets', 'spritesheet')

CAMINHO_SPRITESHEET = os.path.join(DIR_SPRITE, 'pipo-nekonin022.png')


def _criar_imagem_rgba(w, h):
    return np.zeros((h, w, 4), dtype=np.uint8)

def _gerar_coracao():
    img = _criar_imagem_rgba(100, 90)
    # dois círculos + triângulo formam um coração
    cv.circle(img, (30, 30), 28, (50, 50, 220, 255), -1)
    cv.circle(img, (70, 30), 28, (50, 50, 220, 255), -1)
    pts = np.array([[2, 42], [98, 42], [50, 88]], np.int32)
    cv.fillPoly(img, [pts], (50, 50, 220, 255))
    return img

def _gerar_estrela():
    img = _criar_imagem_rgba(100, 100)
    cx, cy, r_ext, r_int = 50, 50, 48, 20
    pts = []
    for i in range(10):
        ang = np.radians(i * 36 - 90)
        r = r_ext if i % 2 == 0 else r_int
        pts.append([int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))])
    pts_np = np.array(pts, np.int32)
    cv.fillPoly(img, [pts_np], (0, 215, 255, 255))
    cv.polylines(img, [pts_np], True, (0, 150, 180, 255), 2)
    return img

def _gerar_bigode():
    img = _criar_imagem_rgba(130, 60)
    cv.ellipse(img, (40, 20), (38, 22), 0, 0, 180, (10, 10, 10, 255), -1)
    cv.ellipse(img, (90, 20), (38, 22), 0, 0, 180, (10, 10, 10, 255), -1)
    cv.ellipse(img, (40, 20), (38, 22), 0, 0, 180, (60, 60, 60, 255), 2)
    cv.ellipse(img, (90, 20), (38, 22), 0, 0, 180, (60, 60, 60, 255), 2)
    return img

def _gerar_chapeu():
    img = _criar_imagem_rgba(110, 100)
    cv.ellipse(img, (55, 80), (55, 18), 0, 0, 360, (20, 20, 20, 255), -1)
    pts = np.array([[20, 80], [90, 80], [82, 10], [28, 10]], np.int32)
    cv.fillPoly(img, [pts], (20, 20, 20, 255))
    cv.rectangle(img, (20, 65), (90, 75), (30, 10, 80, 255), -1)
    return img

def _gerar_balao():
    img = _criar_imagem_rgba(120, 100)
    cv.ellipse(img, (60, 42), (58, 40), 0, 0, 360, (255, 255, 255, 230), -1)
    cv.ellipse(img, (60, 42), (58, 40), 0, 0, 360, (180, 180, 180, 255), 2)
    pts = np.array([[40, 72], [60, 82], [80, 72]], np.int32)
    cv.fillPoly(img, [pts], (255, 255, 255, 230))
    for dx in [-16, 0, 16]:
        cv.circle(img, (60 + dx, 42), 6, (120, 120, 120, 255), -1)
    return img

def _gerar_coroa():
    img = _criar_imagem_rgba(120, 80)
    pts_base = np.array([[5, 70], [115, 70], [115, 35], [90, 55],
                          [60, 10], [30, 55], [5, 35]], np.int32)
    cv.fillPoly(img, [pts_base], (0, 200, 255, 255))
    cv.polylines(img, [pts_base], True, (0, 140, 200, 255), 2)
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
    """Gera os PNGs de sticker que não existirem em assets/stickers/."""
    os.makedirs(DIR_STICKERS, exist_ok=True)
    for nome_arquivo, gerador in GERADORES:
        caminho = os.path.join(DIR_STICKERS, nome_arquivo)
        if not os.path.exists(caminho):
            img = gerador()
            cv.imwrite(caminho, img)
            print(f"[Sticker gerado]: {nome_arquivo}")


_EXTS_STICKER   = ('.png', '.jpg', '.jpeg')
_TAMANHO_MAX_PX = 200   # lado maior em pixels; imagens maiores são redimensionadas


def _normalizar_sticker(img):
    """Converte qualquer imagem carregada para BGRA redimensionada se necessário."""
    if img is None:
        return None
    # garante 3 dimensões (grayscale vem como 2D)
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGRA)
    elif img.shape[2] == 1:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
    # redimensiona se muito grande
    h, w = img.shape[:2]
    if max(h, w) > _TAMANHO_MAX_PX:
        escala = _TAMANHO_MAX_PX / max(h, w)
        img = cv.resize(img, (max(1, int(w * escala)), max(1, int(h * escala))),
                        interpolation=cv.INTER_AREA)
    return img


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


def aplicar_sticker_alpha(fundo, sticker_bgra, x, y):
    """Sobrepõe sticker_bgra sobre fundo na posição (x, y) usando blending alpha."""
    h_fundo, w_fundo = fundo.shape[:2]
    h_s, w_s = sticker_bgra.shape[:2]

    x2 = min(x + w_s, w_fundo)
    y2 = min(y + h_s, h_fundo)
    if x >= w_fundo or y >= h_fundo or x2 <= 0 or y2 <= 0:
        return fundo

    x_s = max(0, -x)
    y_s = max(0, -y)
    x = max(0, x)
    y = max(0, y)

    sticker_crop = sticker_bgra[y_s:y_s+(y2-y), x_s:x_s+(x2-x)]

    b, g, r, a = cv.split(sticker_crop)
    sticker_bgr = cv.merge([b, g, r])

    alpha = a.astype(np.float32) / 255.0
    alpha_3c = cv.merge([alpha, alpha, alpha])

    roi = fundo[y:y2, x:x2].astype(np.float32)
    resultado = cv.add(roi * (1.0 - alpha_3c),
                       sticker_bgr.astype(np.float32) * alpha_3c).astype(np.uint8)

    fundo_copia = fundo.copy()
    fundo_copia[y:y2, x:x2] = resultado
    return fundo_copia


def aplicar_lista_stickers(fundo, posicoes, stickers_lista, idx_sticker):
    """Aplica todos os stickers da lista de posições sobre fundo."""
    resultado = fundo.copy()
    for pos in posicoes:
        s = stickers_lista[pos['idx']]
        resultado = aplicar_sticker_alpha(resultado, s['imagem'], pos['x'], pos['y'])
    return resultado


class StickerAnimado:
    """Spritesheet animado — grade 3x4 = 12 frames, avança pelo tempo."""

    def __init__(self, caminho=CAMINHO_SPRITESHEET, n_colunas=3, n_linhas=4, fps=8):
        self.frames = []
        self.idx_frame = 0
        self.fps = fps
        self._ultimo_tempo = time.time()
        self.posicao = None
        self.ativo = False
        self.escala = 1.5

        if os.path.exists(caminho):
            self._carregar_spritesheet(caminho, n_colunas, n_linhas)
        else:
            print(f"[AVISO] Spritesheet não encontrado: {caminho}")

    def _carregar_spritesheet(self, caminho, n_colunas, n_linhas):
        sheet = cv.imread(caminho, cv.IMREAD_UNCHANGED)
        if sheet is None:
            print(f"[ERRO] Não foi possível carregar: {caminho}")
            return
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
                novo_w = int(w_frame * self.escala)
                novo_h = int(h_frame * self.escala)
                frame = cv.resize(frame, (novo_w, novo_h), interpolation=cv.INTER_NEAREST)
                self.frames.append(frame)
        print(f"[Spritesheet carregado]: {len(self.frames)} frames ({w_frame}×{h_frame} px cada)")

    def avancar_frame(self):
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
        h_f, w_f = frame.shape[:2]
        # centraliza no ponto de clique
        x -= w_f // 2
        y -= h_f // 2
        return aplicar_sticker_alpha(fundo, frame, x, y)

    def definir_posicao(self, x, y):
        self.posicao = (x, y)
        self.ativo = True

    def toggle(self):
        self.ativo = not self.ativo
        print(f"[Sticker animado]: {'ATIVO' if self.ativo else 'INATIVO'}")


# ── múltiplos spritesheets posicionáveis ─────────────────────────────────────

_GRADE_CONHECIDA = {
    'pipo-nekonin022.png': (3, 4),
    'estrelapiscante.png': (5, 4),
    'foguinho.png':        (5, 3),
}

_ESCALA_CONHECIDA = {
    'estrelapiscante.png': 0.525,  # 35% do tamanho padrão (redução de 65%)
    'foguinho.png':        0.525,
}

def _detectar_grade(h, w):
    """Escolhe (n_colunas, n_linhas) cujo frame fique mais próximo de quadrado."""
    candidatos = [
        (3, 4), (4, 4), (3, 3), (5, 4), (5, 3), (6, 4), (4, 6),
        (2, 4), (4, 3), (4, 2), (2, 3), (3, 2), (1, 4), (4, 1),
    ]
    melhor, melhor_score = (3, 4), float('inf')
    for n_col, n_lin in candidatos:
        fw, fh = w / n_col, h / n_lin
        if fw < 16 or fh < 16:
            continue
        score = abs(1.0 - fw / fh)
        if score < melhor_score:
            melhor_score, melhor = score, (n_col, n_lin)
    return melhor


def _limpar_artefatos(frame):
    """Remove blobs isolados pequenos via componentes conectados no canal alpha."""
    if frame.shape[2] < 4:
        return frame
    alpha = frame[:, :, 3]
    _, mask = cv.threshold(alpha, 10, 255, cv.THRESH_BINARY)
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask)
    if n_labels <= 2:
        return frame
    area_max = stats[1:, cv.CC_STAT_AREA].max()
    result = frame.copy()
    for label in range(1, n_labels):
        if stats[label, cv.CC_STAT_AREA] < area_max * 0.1:
            result[labels == label, 3] = 0
    return result



def _bbox_conteudo(frame):
    """(c0, r0, c1, r1) do conteúdo não-transparente, ou None se frame vazio."""
    if frame.shape[2] < 4:
        h, w = frame.shape[:2]
        return 0, 0, w, h
    rows = np.any(frame[:, :, 3] > 10, axis=1)
    cols = np.any(frame[:, :, 3] > 10, axis=0)
    if not rows.any():
        return None
    r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return c0, r0, c1 + 1, r1 + 1


class GerenciadorSpritesAnimados:
    """Múltiplos spritesheets animados com instâncias posicionáveis (como stickers normais)."""

    def __init__(self, fps=8, escala=1.5):
        self._fps    = fps
        self._escala = escala
        self._sprites  = []   # [{nome, frames, idx_frame, ultimo_tempo}]
        self._idx      = 0
        self._posicoes = []   # [{x, y, idx}]
        self._carregar_todos()

    # ── carregamento ──────────────────────────────────────────────────────────

    def _carregar_todos(self):
        exts = ('.png', '.jpg', '.jpeg', '.bmp')
        if not os.path.isdir(DIR_SPRITE):
            return
        for arq in sorted(os.listdir(DIR_SPRITE)):
            if not arq.lower().endswith(exts):
                continue
            frames = self._carregar_spritesheet(os.path.join(DIR_SPRITE, arq))
            if not frames:
                continue
            nome = os.path.splitext(arq)[0].replace('_', ' ').replace('-', ' ').title()
            self._sprites.append({'nome': nome, 'frames': frames,
                                   'idx_frame': 0, 'ultimo_tempo': time.time()})
        if self._sprites:
            print(f"[Spritesheets]: {len(self._sprites)} carregados")
            for i, s in enumerate(self._sprites):
                print(f"  [{i+1}] {s['nome']} ({len(s['frames'])} frames)")
        else:
            print("[AVISO] Nenhum spritesheet encontrado em assets/spritesheet/")

    def _carregar_spritesheet(self, caminho):
        sheet = cv.imread(caminho, cv.IMREAD_UNCHANGED)
        if sheet is None:
            return []
        if len(sheet.shape) == 2:
            sheet = cv.cvtColor(sheet, cv.COLOR_GRAY2BGRA)
        elif sheet.shape[2] == 3:
            sheet = cv.cvtColor(sheet, cv.COLOR_BGR2BGRA)
        h, w = sheet.shape[:2]
        nome_base = os.path.basename(caminho)
        if nome_base in _GRADE_CONHECIDA:
            n_col, n_lin = _GRADE_CONHECIDA[nome_base]
        else:
            n_col, n_lin = _detectar_grade(h, w)
        escala = _ESCALA_CONHECIDA.get(nome_base, self._escala)
        interp = cv.INTER_AREA if escala < 1.0 else cv.INTER_NEAREST
        INS = 2
        x_bounds = [round(c * w / n_col) for c in range(n_col + 1)]
        y_bounds = [round(r * h / n_lin) for r in range(n_lin + 1)]

        # 1ª passagem: extrair frames e remover artefatos isolados
        frames_brutos = []
        for lin in range(n_lin):
            y0 = y_bounds[lin] + INS
            y1 = max(y0 + 1, y_bounds[lin + 1] - INS)
            for col in range(n_col):
                x0 = x_bounds[col] + INS
                x1 = max(x0 + 1, x_bounds[col + 1] - INS)
                frames_brutos.append(_limpar_artefatos(sheet[y0:y1, x0:x1].copy()))

        # 2ª passagem: centralizar pelo conteúdo (union bbox de todos os frames)
        bboxes = [_bbox_conteudo(f) for f in frames_brutos]
        bboxes_v = [b for b in bboxes if b is not None]
        if bboxes_v:
            PAD = 2
            ux0 = max(0, min(b[0] for b in bboxes_v) - PAD)
            uy0 = max(0, min(b[1] for b in bboxes_v) - PAD)
            ux1 = max(b[2] for b in bboxes_v) + PAD
            uy1 = max(b[3] for b in bboxes_v) + PAD
        else:
            ux0, uy0 = 0, 0
            ux1 = frames_brutos[0].shape[1] if frames_brutos else 1
            uy1 = frames_brutos[0].shape[0] if frames_brutos else 1

        nw_base = max(1, int((ux1 - ux0) * escala))
        nh_base = max(1, int((uy1 - uy0) * escala))

        frames = []
        for f in frames_brutos:
            fh, fw = f.shape[:2]
            cropped = f[uy0:min(uy1, fh), ux0:min(ux1, fw)]
            if cropped.size == 0:
                frames.append(np.zeros((nh_base, nw_base, 4), dtype=np.uint8))
            else:
                frames.append(cv.resize(cropped, (nw_base, nh_base), interpolation=interp))
        return frames

    # ── propriedades ──────────────────────────────────────────────────────────

    @property
    def n_sprites(self):
        return len(self._sprites)

    @property
    def idx_atual(self):
        return self._idx

    @property
    def nome_atual(self):
        return self._sprites[self._idx]['nome'] if self._sprites else '(nenhum)'

    @property
    def posicoes(self):
        return self._posicoes

    # ── controle ──────────────────────────────────────────────────────────────

    def proximo(self):
        if self._sprites:
            self._idx = (self._idx + 1) % self.n_sprites
            print(f"[Anim]: {self.nome_atual} [{self._idx+1}/{self.n_sprites}]")

    def anterior(self):
        if self._sprites:
            self._idx = (self._idx - 1) % self.n_sprites
            print(f"[Anim]: {self.nome_atual} [{self._idx+1}/{self.n_sprites}]")

    def colocar(self, x, y):
        if self._sprites:
            self._posicoes.append({'x': x, 'y': y, 'idx': self._idx})
            print(f"[Anim colocado]: {self.nome_atual} em ({x}, {y})")

    def resetar(self):
        self._posicoes.clear()
        self._idx = 0
        for s in self._sprites:
            s['idx_frame'] = 0

    def get_posicoes_snapshot(self):
        return list(self._posicoes)

    def set_posicoes_snapshot(self, snap):
        self._posicoes.clear()
        self._posicoes.extend(snap)

    # ── renderização ──────────────────────────────────────────────────────────

    def desenhar(self, fundo):
        if not self._posicoes:
            return fundo
        agora = time.time()
        ativos = {pos['idx'] for pos in self._posicoes}
        for i, spr in enumerate(self._sprites):
            if i not in ativos:
                continue
            if agora - spr['ultimo_tempo'] >= 1.0 / self._fps:
                spr['idx_frame'] = (spr['idx_frame'] + 1) % len(spr['frames'])
                spr['ultimo_tempo'] = agora
        resultado = fundo
        for pos in self._posicoes:
            spr   = self._sprites[pos['idx']]
            frame = spr['frames'][spr['idx_frame']]
            h_f, w_f = frame.shape[:2]
            resultado = aplicar_sticker_alpha(resultado, frame,
                                              pos['x'] - w_f // 2,
                                              pos['y'] - h_f // 2)
        return resultado
