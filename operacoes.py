"""operacoes.py — Operações aritméticas entre duas imagens."""

import cv2 as cv
import numpy as np


def _redimensionar_para(img, referencia):
    h, w = referencia.shape[:2]
    return cv.resize(img, (w, h), interpolation=cv.INTER_LINEAR)


def op_adicao(img1, img2):
    """Soma saturada pixel a pixel: min(A + B, 255)."""
    img2r = _redimensionar_para(img2, img1)
    return cv.add(img1, img2r)


def op_subtracao(img1, img2):
    """Subtração saturada pixel a pixel: max(A - B, 0)."""
    img2r = _redimensionar_para(img2, img1)
    return cv.subtract(img1, img2r)


def op_blending(img1, img2, alpha=0.5):
    """Mistura ponderada: alpha*A + (1-alpha)*B."""
    img2r = _redimensionar_para(img2, img1)
    return cv.addWeighted(img1, alpha, img2r, 1.0 - alpha, 0)


def op_diferenca(img1, img2):
    """Diferença absoluta: |A - B|. Realça onde as imagens diferem."""
    img2r = _redimensionar_para(img2, img1)
    return cv.absdiff(img1, img2r)


OPERACOES = [
    {
        'nome':      'Adição',
        'descricao': 'Soma saturada pixel a pixel: min(A + B, 255). Clareia a imagem.',
        'funcao':    op_adicao,
        'tecla':     '1',
    },
    {
        'nome':      'Subtração',
        'descricao': 'Subtração saturada pixel a pixel: max(A - B, 0). Escurece e destaca diferenças.',
        'funcao':    op_subtracao,
        'tecla':     '2',
    },
    {
        'nome':      'Blending (50/50)',
        'descricao': 'Mistura ponderada: 0.5 * A + 0.5 * B. Transparência entre as duas imagens.',
        'funcao':    op_blending,
        'tecla':     '3',
    },
    {
        'nome':      'Diferença Absoluta',
        'descricao': 'Diferença absoluta: |A - B|. Realça onde as imagens diferem.',
        'funcao':    op_diferenca,
        'tecla':     '4',
    },
]

N_OPERACOES = len(OPERACOES)


def descrever_operacao(idx):
    op = OPERACOES[idx]
    print(f"\n{'='*55}")
    print(f"  Operação [{idx+1}/{N_OPERACOES}]: {op['nome']}  (tecla [{op['tecla']}])")
    print(f"  {op['descricao']}")
    print(f"{'='*55}")


def aplicar_operacao(img_trabalho, img_secundaria, idx):
    """Aplica a operação aritmética sobre img_trabalho."""
    if img_secundaria is None:
        print("[AVISO] Imagem secundária não carregada. Operação cancelada.")
        return img_trabalho
    op = OPERACOES[idx]
    print(f"\n[Operação aplicada]: {op['nome']}")
    return op['funcao'](img_trabalho, img_secundaria)
