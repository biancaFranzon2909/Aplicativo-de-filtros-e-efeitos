"""filtros.py — Filtros de processamento de imagem."""

import cv2 as cv
import numpy as np

NOME_JANELA_CONTROLES = 'Controles'


def _nada(x):
    pass

def _get_tb(nome, default=5):
    """Lê trackbar com fallback seguro caso a janela ainda não exista."""
    try:
        return cv.getTrackbarPos(nome, NOME_JANELA_CONTROLES)
    except Exception:
        return default

def _kernel_impar(val, minimo=3):
    """Garante que o valor seja ímpar e >= minimo."""
    k = max(minimo, val)
    return k if k % 2 == 1 else k + 1


def criar_janela_controles():
    cv.namedWindow(NOME_JANELA_CONTROLES)
    # Kernel: 1-20 → kernel real = val*2+1 (3 a 41)
    cv.createTrackbar('Kernel',    NOME_JANELA_CONTROLES,  1, 20, _nada)
    # Sigma: 0-30 → sigma real = val/10 (0.0 a 3.0)
    cv.createTrackbar('Sigma',     NOME_JANELA_CONTROLES, 10, 30, _nada)
    cv.createTrackbar('Limiar1',   NOME_JANELA_CONTROLES, 50, 255, _nada)
    cv.createTrackbar('Limiar2',   NOME_JANELA_CONTROLES, 150, 255, _nada)
    # Brilho: 50 = neutro → beta = (val-50)*2
    cv.createTrackbar('Brilho',    NOME_JANELA_CONTROLES, 50, 100, _nada)
    # Contraste: val/10 → 0.1 a 3.0
    cv.createTrackbar('Contraste', NOME_JANELA_CONTROLES, 10, 30, _nada)


def aplicar_blur(img):
    k = _kernel_impar(_get_tb('Kernel', 1) * 2 + 1)
    return cv.blur(img, (k, k))

def aplicar_gaussian(img):
    k = _kernel_impar(_get_tb('Kernel', 1) * 2 + 1)
    sigma = _get_tb('Sigma', 10) / 10.0
    return cv.GaussianBlur(img, (k, k), sigma)

def aplicar_mediana(img):
    k = _kernel_impar(_get_tb('Kernel', 1) * 2 + 1)
    return cv.medianBlur(img, k)

def aplicar_sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv.filter2D(img, -1, kernel)

def aplicar_laplaciano(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    lap = cv.Laplacian(gray, cv.CV_64F)
    resultado = cv.convertScaleAbs(lap)
    return cv.cvtColor(resultado, cv.COLOR_GRAY2BGR) if len(img.shape) == 3 else resultado

def aplicar_sobel(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    magnitude = cv.convertScaleAbs(np.sqrt(sx**2 + sy**2))
    return cv.cvtColor(magnitude, cv.COLOR_GRAY2BGR) if len(img.shape) == 3 else magnitude

def aplicar_canny(img):
    t1 = _get_tb('Limiar1', 50)
    t2 = _get_tb('Limiar2', 150)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    bordas = cv.Canny(gray, t1, t2)
    return cv.cvtColor(bordas, cv.COLOR_GRAY2BGR) if len(img.shape) == 3 else bordas

def aplicar_emboss(img):
    # kernel direcional de relevo
    kernel = np.array([[-2, -1,  0],
                       [-1,  1,  1],
                       [ 0,  1,  2]], dtype=np.float32)
    resultado = cv.filter2D(img.astype(np.float32), -1, kernel)
    return cv.convertScaleAbs(resultado)

def aplicar_negativo(img):
    return cv.bitwise_not(img)

def aplicar_cinza(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

def aplicar_sepia(img):
    img_float = img.astype(np.float32)
    b = img_float[:, :, 0]
    g = img_float[:, :, 1]
    r = img_float[:, :, 2]
    novo_r = np.clip(r * 0.393 + g * 0.769 + b * 0.189, 0, 255)
    novo_g = np.clip(r * 0.349 + g * 0.686 + b * 0.168, 0, 255)
    novo_b = np.clip(r * 0.272 + g * 0.534 + b * 0.131, 0, 255)
    return cv.merge([novo_b, novo_g, novo_r]).astype(np.uint8)

def aplicar_brilho_contraste(img):
    brilho = (_get_tb('Brilho', 50) - 50) * 2    # -100 a +100
    contraste = _get_tb('Contraste', 10) / 10.0   # 0.1 a 3.0
    return cv.convertScaleAbs(img, alpha=contraste, beta=brilho)

def aplicar_limiarizacao(img):
    t1 = _get_tb('Limiar1', 127)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv.threshold(gray, t1, 255, cv.THRESH_BINARY)
    return cv.cvtColor(thresh, cv.COLOR_GRAY2BGR) if len(img.shape) == 3 else thresh

def aplicar_dilatacao(img):
    k = _kernel_impar(_get_tb('Kernel', 1) * 2 + 1)
    elemento = np.ones((k, k), np.uint8)
    return cv.dilate(img, elemento, iterations=1)

def aplicar_erosao(img):
    k = _kernel_impar(_get_tb('Kernel', 1) * 2 + 1)
    elemento = np.ones((k, k), np.uint8)
    return cv.erode(img, elemento, iterations=1)

def aplicar_posterizacao(img):
    niveis = max(2, _get_tb('Limiar1', 64) // 32 + 2)  # 2-9 níveis
    fator = 256.0 / niveis
    return np.clip((img // fator) * fator, 0, 255).astype(np.uint8)


FILTROS = [
    {
        'nome':       'Sem filtro',
        'descricao':  'Nenhum efeito aplicado. A imagem é exibida como está.',
        'parametros': 'Nenhum',
        'funcao':     lambda img: img.copy(),
    },
    {
        'nome':       'Box Blur',
        'descricao':  'Suaviza a imagem com média aritmética dos pixels vizinhos. Reduz ruído e detalhes de alta frequência.',
        'parametros': 'Trackbar [Kernel]: tamanho da janela de suavização (ímpar)',
        'funcao':     aplicar_blur,
    },
    {
        'nome':       'Gaussian Blur',
        'descricao':  'Suavização com peso gaussiano. O pixel central tem mais influência que os da borda. Mais natural que Box Blur.',
        'parametros': 'Trackbars [Kernel] (tamanho) e [Sigma] (desvio padrão da gaussiana)',
        'funcao':     aplicar_gaussian,
    },
    {
        'nome':       'Mediana',
        'descricao':  'Suavização pela mediana dos vizinhos. Excelente para remover ruído "sal e pimenta" sem borrar bordas.',
        'parametros': 'Trackbar [Kernel]: tamanho da janela',
        'funcao':     aplicar_mediana,
    },
    {
        'nome':       'Nitidez (Sharpen)',
        'descricao':  'Realça bordas e detalhes com kernel de aguçamento. Aumenta o contraste local — efeito oposto ao blur.',
        'parametros': 'Nenhum (kernel fixo: [[0,-1,0],[-1,5,-1],[0,-1,0]])',
        'funcao':     aplicar_sharpen,
    },
    {
        'nome':       'Laplaciano',
        'descricao':  'Detector de bordas pela segunda derivada. Destaca regiões de variação rápida de intensidade.',
        'parametros': 'Nenhum',
        'funcao':     aplicar_laplaciano,
    },
    {
        'nome':       'Sobel',
        'descricao':  'Detector de bordas pela derivada direcional (H + V). Calcula a magnitude do gradiente da imagem.',
        'parametros': 'Nenhum (ksize=3)',
        'funcao':     aplicar_sobel,
    },
    {
        'nome':       'Canny',
        'descricao':  'Detector de bordas robusto com histerese. Usa dois limiares para binarizar bordas de forma precisa.',
        'parametros': 'Trackbars [Limiar1] (mínimo) e [Limiar2] (máximo) do Canny',
        'funcao':     aplicar_canny,
    },
    {
        'nome':       'Emboss (Relevo)',
        'descricao':  'Cria efeito de relevo 3D/escultura com kernel direcional. Dá sensação de profundidade.',
        'parametros': 'Nenhum (kernel direcional fixo)',
        'funcao':     aplicar_emboss,
    },
    {
        'nome':       'Negativo',
        'descricao':  'Inverte todos os valores de pixel: I_novo = 255 - I. Efeito de negativo fotográfico analógico.',
        'parametros': 'Nenhum',
        'funcao':     aplicar_negativo,
    },
    {
        'nome':       'Escala de Cinza',
        'descricao':  'Remove a informação de cor, exibindo apenas a luminância (brilho) da imagem.',
        'parametros': 'Nenhum',
        'funcao':     aplicar_cinza,
    },
    {
        'nome':       'Sépia',
        'descricao':  'Aplica tonalidade quente envelhecida (sépia). Transformação linear dos canais BGR para tons amarronzados.',
        'parametros': 'Nenhum',
        'funcao':     aplicar_sepia,
    },
    {
        'nome':       'Brilho e Contraste',
        'descricao':  'Ajusta brilho (deslocamento aditivo nos pixels) e contraste (escala multiplicativa).',
        'parametros': 'Trackbars [Brilho] (0=escuro / 50=neutro / 100=claro) e [Contraste] (0.1 a 3.0)',
        'funcao':     aplicar_brilho_contraste,
    },
    {
        'nome':       'Limiarização',
        'descricao':  'Binariza a imagem: pixels acima do limiar ficam brancos, abaixo ficam pretos.',
        'parametros': 'Trackbar [Limiar1]: valor do limiar (0-255)',
        'funcao':     aplicar_limiarizacao,
    },
    {
        'nome':       'Dilatação',
        'descricao':  'Operação morfológica que expande regiões claras. Útil para preencher buracos e conectar componentes.',
        'parametros': 'Trackbar [Kernel]: tamanho do elemento estruturante',
        'funcao':     aplicar_dilatacao,
    },
    {
        'nome':       'Erosão',
        'descricao':  'Operação morfológica que encolhe regiões claras. Útil para remover ruído de pequenas partículas.',
        'parametros': 'Trackbar [Kernel]: tamanho do elemento estruturante',
        'funcao':     aplicar_erosao,
    },
    {
        'nome':       'Posterização',
        'descricao':  'Reduz o número de níveis de cor, criando efeito visual estilo poster/BD.',
        'parametros': 'Trackbar [Limiar1]: controla o número de níveis de cor (2-9)',
        'funcao':     aplicar_posterizacao,
    },
]

N_FILTROS = len(FILTROS)


def preview_filtro(img, idx):
    """Aplica o filtro e retorna o resultado sem modificar o estado."""
    try:
        return FILTROS[idx]['funcao'](img.copy())
    except Exception as e:
        print(f"[ERRO ao aplicar filtro '{FILTROS[idx]['nome']}']: {e}")
        return img.copy()


def descrever_filtro(idx):
    f = FILTROS[idx]
    print(f"\n{'='*55}")
    print(f"  Filtro [{idx+1}/{N_FILTROS}]: {f['nome']}")
    print(f"  {f['descricao']}")
    print(f"  Parâmetros: {f['parametros']}")
    print(f"{'='*55}")


def exibir_canal(img, canal):
    """Retorna imagem com apenas o canal selecionado não-zero."""
    if canal == 'todos':
        return img
    b, g, r = cv.split(img)
    zeros = np.zeros_like(b)
    if canal == 'B':
        return cv.merge([b, zeros, zeros])
    elif canal == 'G':
        return cv.merge([zeros, g, zeros])
    elif canal == 'R':
        return cv.merge([zeros, zeros, r])
    return img
