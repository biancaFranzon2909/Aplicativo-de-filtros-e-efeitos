"""
face.py — Detecção de faces com Haar Cascades e efeitos sobrepostos.

Padrão de detecção retirado de facedetect.py da professora.

Modos de efeito (ciclar com [E]):
  sticker      → óculos posicionados sobre os olhos
  blur         → blur gaussiano (privacidade)
  pixel        → pixelização em blocos (censura)
  cachorro     → orelhas floppy, nariz e língua de cachorro
  gato         → orelhas pontudas, nariz e bigodes de gato
  coroa_flores → coroa de flores ao longo da testa

Anti-flicker:
  - Detecção roda a cada DETECT_EVERY frames (não todo frame).
  - Resultado mantido em cache por até MAX_FRAMES_SEM_FACE frames sem detecção.
  - Suavização EMA nas coordenadas da face para evitar saltos bruscos.

NOTA: Requer opencv-contrib-python (não opencv-python padrão).
      pip install opencv-contrib-python
"""

import cv2 as cv
import numpy as np
import os

DIR        = os.path.dirname(os.path.abspath(__file__))
DIR_HAAR   = os.path.join(DIR, 'assets', 'haarcascades')
CAMINHO_FACE   = os.path.join(DIR_HAAR, 'haarcascade_frontalface_default.xml')
CAMINHO_OLHO   = os.path.join(DIR_HAAR, 'haarcascade_eye.xml')
CAMINHO_OCULOS = os.path.join(DIR, 'assets', 'stickers', 'eyeglasses.png')

_MODOS = ['sticker', 'blur', 'pixel', 'cachorro', 'gato', 'coroa_flores']

_DESC_MODOS = {
    'sticker':      'oculos sobre os olhos',
    'blur':         'blur gaussiano (privacidade)',
    'pixel':        'pixelizacao (censura)',
    'cachorro':     'filtro cachorro (orelhas + nariz + lingua)',
    'gato':         'filtro gato (orelhas + nariz + bigodes)',
    'coroa_flores': 'coroa de flores na testa',
}

_NOME_CURTO = {
    'sticker': 'oculos', 'blur': 'blur', 'pixel': 'pixel',
    'cachorro': 'cao',   'gato': 'gato', 'coroa_flores': 'flores',
}


class DetectorFace:
    """Detecta faces com Haar Cascades e aplica 6 efeitos diferentes."""

    # ── Anti-flicker: parâmetros de suavização temporal ───────────────────────
    _DETECT_EVERY        = 3    # roda detecção bruta a cada N frames
    _MAX_FRAMES_SEM_FACE = 15   # após N frames sem detecção, limpa o cache
    _ALPHA_SUAVE         = 0.10 # peso EMA: 0=completamente suave, 1=bruto
    _DEADBAND            = 8    # ignora movimentos menores que N pixels

    def __init__(self):
        self.ativo = False
        self.modo  = 'sticker'
        self._cascade_face = None
        self._cascade_olho = None
        self._oculos       = None
        # Estado interno do anti-flicker
        self._faces_cache       = []
        self._frames_sem_face   = 0
        self._frame_count       = 0
        self._inicializar()

    def _inicializar(self):
        """Carrega classificadores Haar e o sticker de óculos."""
        if not os.path.exists(CAMINHO_FACE):
            print(f"[ERRO] Haar cascade não encontrado: {CAMINHO_FACE}")
            return
        self._cascade_face = cv.CascadeClassifier(CAMINHO_FACE)
        if os.path.exists(CAMINHO_OLHO):
            self._cascade_olho = cv.CascadeClassifier(CAMINHO_OLHO)
        if os.path.exists(CAMINHO_OCULOS):
            self._oculos = cv.imread(CAMINHO_OCULOS, cv.IMREAD_UNCHANGED)
            if self._oculos is not None and self._oculos.shape[2] == 3:
                self._oculos = cv.cvtColor(self._oculos, cv.COLOR_BGR2BGRA)
        else:
            print(f"[AVISO] eyeglasses.png não encontrado: {CAMINHO_OCULOS}")

    # ── Detecção com suavização temporal ──────────────────────────────────────

    def _detectar_bruta(self, gray):
        """
        Roda detectMultiScale uma vez.
        Parâmetros ajustados para estabilidade:
          scaleFactor=1.2  → mais preciso (1.3 é muito grosso)
          minNeighbors=6   → mais vizinhos = menos falsos positivos e menos flicker
        """
        if self._cascade_face is None or self._cascade_face.empty():
            return []
        gray_eq = cv.equalizeHist(gray)
        faces = self._cascade_face.detectMultiScale(
            gray_eq,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(50, 50),
            flags=cv.CASCADE_SCALE_IMAGE
        )
        return list(faces) if len(faces) > 0 else []

    def _detectar_suave(self, gray):
        """
        Dois mecanismos anti-flicker:
          1. Caching: se não detectou, usa o último resultado por N frames.
          2. EMA: suaviza as coordenadas da bounding box entre detecções.
        Resultado: a face não some e não treme entre frames.
        """
        self._frame_count += 1

        # Pula frames intermediários — usa cache sem recalcular
        if self._frame_count % self._DETECT_EVERY != 0:
            return self._faces_cache

        novas = self._detectar_bruta(gray)

        if len(novas) == 0:
            self._frames_sem_face += 1
            if self._frames_sem_face > self._MAX_FRAMES_SEM_FACE:
                self._faces_cache = []   # sumiu de vez: limpa cache
            return self._faces_cache

        # Detectou faces
        self._frames_sem_face = 0

        if len(self._faces_cache) == len(novas):
            # Mesmo número → suavização EMA + deadband para evitar saltos bruscos
            a = self._ALPHA_SUAVE
            db = self._DEADBAND
            suaves = []
            for nova, cache in zip(novas, self._faces_cache):
                # Só atualiza se o deslocamento for maior que o deadband
                if max(abs(nova[0]-cache[0]), abs(nova[1]-cache[1]),
                       abs(nova[2]-cache[2]), abs(nova[3]-cache[3])) < db:
                    suaves.append(cache)   # movimento pequeno demais: congela
                else:
                    x = int(a * nova[0] + (1-a) * cache[0])
                    y = int(a * nova[1] + (1-a) * cache[1])
                    w = int(a * nova[2] + (1-a) * cache[2])
                    h = int(a * nova[3] + (1-a) * cache[3])
                    suaves.append((x, y, w, h))
            self._faces_cache = suaves
        else:
            # Número de faces mudou → aceita bruta
            self._faces_cache = [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in novas]

        return self._faces_cache

    # ── Efeito: óculos ────────────────────────────────────────────────────────

    def _aplicar_oculos(self, img, x, y, w, h):
        from stickers import aplicar_sticker_alpha
        if self._oculos is None:
            return img
        # Posição fixa em 1/3 da face — estável e precisa o suficiente.
        # Detecção de olhos por cascade dentro do ROI era fonte de jitter vertical.
        pos_y = h // 3
        h_oc = int(w * self._oculos.shape[0] / self._oculos.shape[1])
        return aplicar_sticker_alpha(img,
                                     cv.resize(self._oculos, (w, h_oc), interpolation=cv.INTER_LINEAR),
                                     x, y + pos_y)

    # ── Efeito: blur ──────────────────────────────────────────────────────────

    def _aplicar_blur_face(self, img, x, y, w, h):
        roi = img[y:y+h, x:x+w]
        k = max(15, (w // 10) | 1)
        img[y:y+h, x:x+w] = cv.GaussianBlur(roi, (k, k), 0)
        return img

    # ── Efeito: pixelização ───────────────────────────────────────────────────

    def _aplicar_pixelacao(self, img, x, y, w, h):
        roi = img[y:y+h, x:x+w]
        t = max(5, min(w, h) // 15)
        img[y:y+h, x:x+w] = cv.resize(
            cv.resize(roi, (t, t), interpolation=cv.INTER_LINEAR),
            (w, h), interpolation=cv.INTER_NEAREST
        )
        return img

    # ── Efeito: cachorro ──────────────────────────────────────────────────────

    def _aplicar_cachorro(self, img, x, y, w, h):
        """Orelhas floppy marrons nos lados, nariz preto oval, língua rosa."""
        res = img.copy()
        MARROM     = ( 40,  90, 150)   # BGR
        MARROM_INT = ( 90, 130, 190)
        PRETO      = ( 15,  15,  15)
        ROSA       = (110,  85, 230)
        ROSA_ESC   = ( 75,  60, 185)

        ew = max(18, w // 4)
        eh = max(28, h // 2)
        iv = max(4, ew // 4)
        cy_ear = y + h // 3

        # Orelha esquerda: meia-elipse pendendo para a esquerda da face
        cv.ellipse(res, (x, cy_ear), (ew, eh), 0, -90, 90, MARROM, -1)
        cv.ellipse(res, (x, cy_ear), (ew - iv, eh - iv), 0, -90, 90, MARROM_INT, -1)

        # Orelha direita: meia-elipse pendendo para a direita
        cv.ellipse(res, (x + w, cy_ear), (ew, eh), 0, 90, 270, MARROM, -1)
        cv.ellipse(res, (x + w, cy_ear), (ew - iv, eh - iv), 0, 90, 270, MARROM_INT, -1)

        # Nariz oval com pequeno reflexo de luz
        nx, ny = x + w // 2, y + int(h * 0.60)
        cv.ellipse(res, (nx, ny), (max(6, w // 7), max(4, h // 10)), 0, 0, 360, PRETO, -1)
        cv.circle(res, (nx - max(2, w // 15), ny - max(2, h // 18)), max(2, w // 22), (200, 200, 200), -1)

        # Língua com linha central
        lx, ly = nx, y + int(h * 0.78)
        cv.ellipse(res, (lx, ly), (max(6, w // 5), max(5, h // 8)), 0, 0, 360, ROSA, -1)
        cv.line(res, (lx, y + int(h * 0.70)), (lx, y + int(h * 0.88)), ROSA_ESC, max(1, w // 35))
        return res

    # ── Efeito: gato ──────────────────────────────────────────────────────────

    def _aplicar_gato(self, img, x, y, w, h):
        """Orelhas pontudas cinza/rosa, nariz triangular, bigodes brancos."""
        res = img.copy()
        CINZA    = (160, 160, 160)
        ROSA_INT = (160, 100, 220)
        ROSA_N   = (140, 100, 215)
        BIGODE   = (230, 230, 230)

        ew   = max(16, w // 3)
        alt  = max(10, h // 3)

        # Orelha esquerda: triângulo cinza com interior rosa
        b_e = np.array([[x,         y],
                         [x + ew,    y],
                         [x + ew//2, y - alt]], np.int32)
        cv.fillPoly(res, [b_e], CINZA)
        i_e = np.array([[x + ew//5,     y],
                         [x + ew*4//5,  y],
                         [x + ew//2,    y - max(6, h//5)]], np.int32)
        cv.fillPoly(res, [i_e], ROSA_INT)

        # Orelha direita (espelhada)
        ox = x + w - ew
        b_d = np.array([[ox,          y],
                         [ox + ew,    y],
                         [ox + ew//2, y - alt]], np.int32)
        cv.fillPoly(res, [b_d], CINZA)
        i_d = np.array([[ox + ew//5,    y],
                         [ox + ew*4//5, y],
                         [ox + ew//2,   y - max(6, h//5)]], np.int32)
        cv.fillPoly(res, [i_d], ROSA_INT)

        # Nariz triangular
        nx, ny = x + w // 2, y + int(h * 0.57)
        ns = max(5, w // 10)
        pts_n = np.array([[nx - ns, ny], [nx + ns, ny], [nx, ny + ns]], np.int32)
        cv.fillPoly(res, [pts_n], ROSA_N)

        # Bigodes — 3 linhas de cada lado, levemente inclinadas
        wy    = y + int(h * 0.60)
        wlen  = max(20, w // 2)
        esp   = max(3, h // 10)
        lw    = max(1, w // 50)
        for dy in [-esp, 0, esp]:
            cv.line(res, (nx - 4, wy + dy), (x  - wlen // 3, wy + dy - dy // 3), BIGODE, lw)
            cv.line(res, (nx + 4, wy + dy), (x + w + wlen // 3, wy + dy - dy // 3), BIGODE, lw)
        return res

    # ── Efeito: coroa de flores ───────────────────────────────────────────────

    def _aplicar_coroa_flores(self, img, x, y, w, h):
        """Coroa de flores coloridas ao longo da testa (acima da face)."""
        res = img.copy()
        PALETA = [
            (100, 180, 255),  # amarelo
            (120, 100, 220),  # rosa
            ( 60, 200, 255),  # laranja
            (200, 100, 100),  # azul-violeta
            (180, 220, 100),  # verde-amarelo
        ]
        VERDE  = (60, 150,  60)
        CENTRO = ( 0, 220, 255)   # dourado

        y_crown  = y - max(8, h // 10)
        n_flores = max(3, w // 28)
        espaco   = w // n_flores
        r_petal  = max(6, espaco // 3)
        r_ctr    = max(3, r_petal // 2)

        for i in range(n_flores):
            fx  = x + espaco // 2 + i * espaco
            fy  = y_crown
            cor = PALETA[i % len(PALETA)]

            # Folhinhas
            cv.circle(res, (fx - r_petal, fy + r_petal // 2), max(2, r_petal // 2), VERDE, -1)
            cv.circle(res, (fx + r_petal, fy + r_petal // 2), max(2, r_petal // 2), VERDE, -1)

            # 5 pétalas dispostas em círculo
            for ang in range(0, 360, 72):
                rad = np.radians(ang)
                px  = fx + int(r_petal * np.cos(rad))
                py  = fy + int(r_petal * np.sin(rad))
                cv.circle(res, (px, py), r_petal, cor, -1)

            # Centro dourado
            cv.circle(res, (fx, fy), r_ctr, CENTRO, -1)
        return res

    # ── API pública ───────────────────────────────────────────────────────────

    def processar(self, img):
        """Detecta faces e aplica o efeito ativo."""
        if not self.ativo or self._cascade_face is None:
            return img

        gray  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self._detectar_suave(gray)

        resultado = img.copy()
        handlers = {
            'sticker':      self._aplicar_oculos,
            'blur':         self._aplicar_blur_face,
            'pixel':        self._aplicar_pixelacao,
            'cachorro':     self._aplicar_cachorro,
            'gato':         self._aplicar_gato,
            'coroa_flores': self._aplicar_coroa_flores,
        }
        fn = handlers.get(self.modo)
        for (x, y, w, h) in faces:
            if fn:
                resultado = fn(resultado, x, y, w, h)
            if self.modo in ('blur', 'pixel'):
                cor = (0, 200, 0) if self.modo == 'blur' else (0, 150, 255)
                cv.rectangle(resultado, (x, y), (x+w, y+h), cor, 2)

        if len(faces) > 0:
            cv.putText(resultado, f'{len(faces)} face(s)', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return resultado

    def toggle_ativo(self):
        self.ativo = not self.ativo
        if self.ativo:
            print(f"[Detecção de face]: ATIVA  |  Modo: {_DESC_MODOS[self.modo]}")
        else:
            print("[Detecção de face]: INATIVA")

    def trocar_modo(self):
        """Cicla pelos 6 modos: sticker → blur → pixel → cachorro → gato → flores → …"""
        idx = _MODOS.index(self.modo)
        self.modo = _MODOS[(idx + 1) % len(_MODOS)]
        print(f"[Detecção de face] Modo: {_DESC_MODOS[self.modo]}")
