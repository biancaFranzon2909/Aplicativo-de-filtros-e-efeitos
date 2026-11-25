"""
main.py — Aplicativo de edição de imagem/vídeo estilo stories do Instagram.

Trabalho Grau B — Processamento Gráfico: Fundamentos (Unisinos 2026/2)

Controles:
  [F / f]   Próximo / Anterior filtro (pré-visualização em tempo real)
  [A]        Aplicar filtro (grava em img_trabalho)
  [U]        Desfazer (filtros e stickers)
  [Y]        Refazer
  [C]        Trocar canal de exibição (Todos → R → G → B → Todos)
  [1-4]      Aplicar operação aritmética com imagem secundária
  [S / s]    Próximo / Anterior sticker
  [clique]   Colocar sticker na posição do mouse
  [P]        Capturar frame atual e ir para Modo Foto (Modo Vídeo)
  [D]        Detectar faces — toggle
  [E]        Trocar modo de face (óculos → blur → pixel → …)
  [Z]        Sticker animado — toggle
  [clique+Z] Reposicionar sticker animado (com [Z] ativo)
  [V]        Trocar modo (Foto ↔ Vídeo)
  [G]        Gravar/Salvar imagem atual
  [R]        Resetar (volta ao original e limpa tudo — Foto e Vídeo)
  [M]        Mostrar menu de controles
  [Q / ESC]  Sair
"""

import cv2 as cv
import numpy as np
import os
import datetime
import threading

import filtros
import operacoes
import stickers
import face as face_mod

DIR        = os.path.dirname(os.path.abspath(__file__))
DIR_ASSETS = os.path.join(DIR, 'assets', 'imagens')
DIR_SAIDA  = os.path.join(DIR, 'outputs')
SAMPLE_2   = os.path.join(DIR_ASSETS, 'sample2.jpg')
VIDEO_FALLBACK = os.path.join(DIR, 'assets', 'video_fallback.mp4')

NOME_JANELA = 'Processamento Grafico - Grau B'
LARGURA_MAX = 900
ALTURA_MAX  = 700


def _redimensionar_se_grande(img):
    h, w = img.shape[:2]
    if w <= LARGURA_MAX and h <= ALTURA_MAX:
        return img
    escala = min(LARGURA_MAX / w, ALTURA_MAX / h)
    return cv.resize(img, (int(w * escala), int(h * escala)), interpolation=cv.INTER_AREA)

def _carregar_imagem(caminho):
    img = cv.imread(caminho)
    if img is None:
        return None
    return _redimensionar_se_grande(img)

def _gerar_imagem_secundaria(referencia):
    """Cria imagem de teste colorida (gradiente + quadrado central)."""
    h, w = referencia.shape[:2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(255, 0, w, dtype=np.uint8), (h, 1))
    img[:, :, 1] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    cx, cy, rw, rh = w // 2, h // 2, w // 4, h // 4
    cv.rectangle(img, (cx-rw, cy-rh), (cx+rw, cy+rh), (0, 0, 200), -1)
    cv.putText(img, 'Imagem 2', (cx - 50, cy + 10),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img

def _salvar(img):
    os.makedirs(DIR_SAIDA, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    caminho = os.path.join(DIR_SAIDA, f'resultado_{ts}.png')
    cv.imwrite(caminho, img)
    print(f"[Salvo]: {caminho}")


def _salvar_historia(estado):
    """Salva snapshot de img_trabalho + stickers antes de cada ação destrutiva."""
    snapshot = {
        'img':      estado['img_trabalho'].copy(),
        'stickers': list(estado['posicoes_stickers']),
        'anim':     estado['gerenciador_anim'].get_posicoes_snapshot(),
    }
    estado['historico'].append(snapshot)
    estado['historico_redo'].clear()

def _desfazer(estado):
    if not estado['historico']:
        print("[Desfazer]: nada para desfazer")
        return
    estado['historico_redo'].append({
        'img':      estado['img_trabalho'].copy(),
        'stickers': list(estado['posicoes_stickers']),
        'anim':     estado['gerenciador_anim'].get_posicoes_snapshot(),
    })
    snapshot = estado['historico'].pop()
    estado['img_trabalho'] = snapshot['img'].copy()
    estado['posicoes_stickers'].clear()
    estado['posicoes_stickers'].extend(snapshot['stickers'])
    estado['gerenciador_anim'].set_posicoes_snapshot(snapshot['anim'])
    print(f"[Desfazer]: OK  ({len(estado['historico'])} passo(s) no histórico)")

def _refazer(estado):
    if not estado['historico_redo']:
        print("[Refazer]: nada para refazer")
        return
    estado['historico'].append({
        'img':      estado['img_trabalho'].copy(),
        'stickers': list(estado['posicoes_stickers']),
        'anim':     estado['gerenciador_anim'].get_posicoes_snapshot(),
    })
    snapshot = estado['historico_redo'].pop()
    estado['img_trabalho'] = snapshot['img'].copy()
    estado['posicoes_stickers'].clear()
    estado['posicoes_stickers'].extend(snapshot['stickers'])
    estado['gerenciador_anim'].set_posicoes_snapshot(snapshot['anim'])
    print("[Refazer]: OK")

def _resetar(estado, img_original):
    """Volta ao estado inicial: restaura imagem, limpa stickers, desativa efeitos."""
    if img_original is not None:
        estado['img_trabalho'] = img_original.copy()
    estado['posicoes_stickers'].clear()
    estado['historico'].clear()
    estado['historico_redo'].clear()
    estado['canal']      = 'todos'
    estado['idx_filtro'] = 0
    estado['gerenciador_anim'].resetar()
    estado['modo_sprite'] = False
    estado['detector'].ativo = False
    print("[Resetar]: filtros, stickers e efeitos removidos")


def _gerar_splash(lista_imagens=None):
    """Gera tela de boas-vindas com gradiente roxo→laranja."""
    if lista_imagens is None:
        lista_imagens = []
    h, w = 500, 700
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # gradiente vertical roxo → laranja (BGR)
    y_arr = np.arange(h, dtype=np.float32).reshape(-1, 1)
    t = y_arr / h
    img[:, :, 0] = np.clip(200 * (1-t) + 50 * t,  0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(0   * (1-t) + 120 * t, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(130 * (1-t) + 255 * t, 0, 255).astype(np.uint8)

    cv.putText(img, 'Stories Filter App', (55, 78),
               cv.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv.LINE_AA)
    cv.putText(img, 'Processamento Grafico: Fundamentos  |  Unisinos 2026/2', (22, 112),
               cv.FONT_HERSHEY_SIMPLEX, 0.47, (230, 220, 255), 1, cv.LINE_AA)
    cv.line(img, (40, 128), (w-40, 128), (255, 255, 255), 1)

    cv.putText(img, 'Pressione uma tecla nesta janela:', (58, 165),
               cv.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 220), 1, cv.LINE_AA)

    y_base = 205
    if lista_imagens:
        for i, nome in enumerate(lista_imagens[:6]):
            cv.putText(img, f'[{i+1}]  {nome}', (78, y_base + i * 32),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        y_base += min(len(lista_imagens), 6) * 32 + 10
    else:
        cv.putText(img, '(Sem fotos em assets/imagens/)', (78, y_base),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv.LINE_AA)
        y_base += 38

    cv.putText(img, '[0]  Digitar o caminho de uma foto', (78, y_base),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv.LINE_AA)
    cv.putText(img, '[V]  Abrir a camera (Modo Video)', (78, y_base + 32),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv.LINE_AA)
    cv.putText(img, '[Q / ESC]  Sair', (78, y_base + 64),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1, cv.LINE_AA)

    return img


def _menu_inicial():
    """
    Seleção via janela OpenCV E via terminal em paralelo.
    Main thread roda cv.waitKey (janela responsiva); daemon thread roda input().
    Retorna: (img_principal, img_secundaria, modo: 'foto'|'video'|'sair')
    """
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    lista_imagens = []
    if os.path.isdir(DIR_ASSETS):
        for f in sorted(os.listdir(DIR_ASSETS)):
            if f.lower().endswith(exts) and f != 'sample2.jpg':
                lista_imagens.append(f)

    print("\n" + "=" * 52)
    if lista_imagens:
        print("Fotos disponíveis em assets/imagens/:")
        for i, nome in enumerate(lista_imagens[:9]):
            print(f"  [{i+1}] {nome}")
    else:
        print("(Nenhuma foto encontrada em assets/imagens/)")
    print("  [0] Digitar o caminho de uma foto")
    print("  [V] Abrir a câmera (Modo Vídeo)")
    print("  [Q / ESC] Sair")
    print("=" * 52)
    print("→ Pressione na janela  OU  digite aqui e pressione Enter")

    resultado = [None]
    evento = threading.Event()

    def _ler_terminal():
        while not evento.is_set():
            try:
                escolha = input("\nEscolha: ").strip()
            except EOFError:
                return
            if evento.is_set():
                return

            lc = escolha.lower()
            if lc in ('q', 'esc'):
                resultado[0] = 'q'
                evento.set()
                return
            if lc == 'v':
                resultado[0] = 'v'
                evento.set()
                return
            if escolha == '0':
                try:
                    caminho = input("Caminho da foto: ").strip().strip('"')
                except EOFError:
                    return
                if not evento.is_set():
                    resultado[0] = ('path', caminho)
                    evento.set()
                return
            if escolha.isdigit() and 1 <= int(escolha) <= len(lista_imagens):
                resultado[0] = escolha
                evento.set()
                return
            if os.path.isfile(escolha.strip('"')):
                resultado[0] = ('path', escolha.strip('"'))
                evento.set()
                return
            print("  Opção inválida. Digite um número da lista, [0], [V], [Q] ou um caminho.")

    t = threading.Thread(target=_ler_terminal, daemon=True)
    t.start()

    splash = _gerar_splash(lista_imagens)
    while not evento.is_set():
        cv.imshow(NOME_JANELA, splash)
        filtros.tick_controles()
        tecla = cv.waitKey(30) & 0xFF

        if tecla in (ord('q'), ord('Q'), 27):
            resultado[0] = 'q'
            evento.set()
        elif tecla in (ord('v'), ord('V')):
            resultado[0] = 'v'
            evento.set()
        elif ord('1') <= tecla <= ord('9'):
            resultado[0] = chr(tecla)
            evento.set()

    r = resultado[0]

    if r is None or r == 'q':
        return None, None, 'sair'

    if r == 'v':
        return None, None, 'video'

    if isinstance(r, tuple) and r[0] == 'path':
        img = _carregar_imagem(r[1])
        if img is not None:
            img2_tmp = _carregar_imagem(SAMPLE_2)
            img2 = img2_tmp if img2_tmp is not None else _gerar_imagem_secundaria(img)
            return img, img2, 'foto'
        print(f"[ERRO] Não foi possível abrir: {r[1]}")
        return None, None, 'sair'

    if isinstance(r, str) and r.isdigit():
        idx = int(r) - 1
        if 0 <= idx < len(lista_imagens):
            caminho = os.path.join(DIR_ASSETS, lista_imagens[idx])
            img = _carregar_imagem(caminho)
            if img is not None:
                img2_tmp = _carregar_imagem(SAMPLE_2)
                img2 = img2_tmp if img2_tmp is not None else _gerar_imagem_secundaria(img)
                return img, img2, 'foto'

    return None, None, 'sair'


def _desenhar_hud(img, estado):
    """Desenha informações de estado no topo da imagem."""
    copia = img.copy()
    modo   = estado['modo'].upper()
    canal  = estado['canal']
    idx_f  = estado['idx_filtro']
    n_f    = filtros.N_FILTROS
    nome_f = filtros.FILTROS[idx_f]['nome']
    stks   = estado['stickers_lista']
    idx_s  = estado['idx_sticker']
    nome_s = stks[idx_s]['nome'] if stks else '(nenhum)'
    det    = estado['detector']
    face_info = face_mod._NOME_CURTO.get(det.modo, det.modo) if det.ativo else 'OFF'
    n_hist = len(estado['historico'])
    ger    = estado['gerenciador_anim']
    modo_sprite = estado['modo_sprite']

    seta_stat = '  << clique' if not modo_sprite else ''
    seta_anim = '  << clique' if modo_sprite else ''

    linhas = [
        f"[{modo}]  Canal:{canal}  Face:{face_info}  Hist:{n_hist}",
        f"Filtro [{idx_f+1}/{n_f}]: {nome_f}",
        f"Sticker [{idx_s+1}/{len(stks) if stks else 0}]: {nome_s}{seta_stat}",
        f"Anim [{ger.idx_atual+1}/{ger.n_sprites if ger.n_sprites else 1}]: {ger.nome_atual}{seta_anim}",
    ]

    h_hud = len(linhas) * 24 + 8
    overlay = copia.copy()
    cv.rectangle(overlay, (0, 0), (copia.shape[1], h_hud), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.55, copia, 0.45, 0, copia)

    for i, linha in enumerate(linhas):
        cv.putText(copia, linha, (8, 20 + i * 24),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv.LINE_AA)
    return copia


def imprimir_menu():
    print("""
╔══════════════════════════════════════════════════════╗
║          CONTROLES — Processamento Gráfico           ║
╠══════════════════════════════════════════════════════╣
║  [F / f]   Próximo / Anterior filtro  ([0]=Sem filtro) ║
║  [A]        Aplicar filtro na imagem de trabalho     ║
║  [U]        Desfazer  (filtros e stickers)           ║
║  [Y]        Refazer                                  ║
║  [C]        Trocar canal (Todos/R/G/B)               ║
╠══════════════════════════════════════════════════════╣
║  [1]  Operação: Adição           [2] Subtração       ║
║  [3]  Operação: Blending 50/50   [4] Diferença Abs.  ║
╠══════════════════════════════════════════════════════╣
║  [S / s]   Próximo / Anterior sticker                ║
║  [Z / z]   Próximo / Anterior sticker animado        ║
║  [clique]  Colocar sticker ou animado (modo atual)   ║
║  [D]        Detecção de face — toggle                ║
║  [E]        Modo face (ciclar 6 opções):             ║
║             oculos/blur/pixel/cachorro/gato/flores   ║
╠══════════════════════════════════════════════════════╣
║  [V]  Trocar modo (Foto ↔ Vídeo)                     ║
║  [P]  Capturar frame → Modo Foto  (só no Vídeo)      ║
║  [G]  Gravar / Salvar imagem                         ║
║  [R]  Resetar (remove tudo — Foto e Vídeo)           ║
║  [M]  Mostrar este menu                              ║
║  [Q / ESC]  Sair                                     ║
╚══════════════════════════════════════════════════════╝
""")


_ultimo_clique = None
_posicao_mouse = (0, 0)

def _mouse_callback(event, x, y, flags, param):
    global _ultimo_clique, _posicao_mouse
    _posicao_mouse = (x, y)
    if event == cv.EVENT_LBUTTONDOWN:
        _ultimo_clique = (x, y)


def loop_foto(estado):
    """Loop principal do Modo Foto. Retorna: 'sair' | 'video'"""
    global _ultimo_clique

    img_original      = estado['img_original']
    img_trabalho      = estado['img_trabalho']
    img_secundaria    = estado['img_secundaria']
    stickers_lista    = estado['stickers_lista']
    posicoes_stickers = estado['posicoes_stickers']
    detector          = estado['detector']
    ger               = estado['gerenciador_anim']

    cv.namedWindow(NOME_JANELA)
    cv.setMouseCallback(NOME_JANELA, _mouse_callback)
    print("\n[Modo FOTO] Pronto. Pressione [M] para ver os controles.")

    _HUD_H = 4 * 24 + 8  # 4 linhas de HUD

    while True:
        idx_f = estado['idx_filtro']

        # pré-visualização em tempo real sem alterar img_trabalho
        preview = filtros.preview_filtro(img_trabalho, idx_f)
        display = filtros.exibir_canal(preview, estado['canal'])
        display = stickers.aplicar_lista_stickers(display, posicoes_stickers,
                                                  stickers_lista, estado['idx_sticker'])
        display = ger.desenhar(display)
        display = detector.processar(display)
        display = _desenhar_hud(display, estado)
        cv.imshow(NOME_JANELA, display)
        filtros.tick_controles()

        if _ultimo_clique is not None:
            cx, cy = _ultimo_clique
            _ultimo_clique = None
            cy_real = max(0, cy - _HUD_H)
            if estado['modo_sprite'] and ger.n_sprites > 0:
                _salvar_historia(estado)
                ger.colocar(cx, cy_real)
            elif stickers_lista:
                _salvar_historia(estado)
                posicoes_stickers.append({'x': cx, 'y': cy_real, 'idx': estado['idx_sticker']})
                nome_s = stickers_lista[estado['idx_sticker']]['nome']
                print(f"[Sticker colocado]: {nome_s} em ({cx}, {cy_real})")

        tecla = cv.waitKey(30) & 0xFF

        if tecla in (ord('q'), 27):
            return 'sair'

        elif tecla == ord('v'):
            print("\n[Modo VÍDEO ativado]")
            return 'video'

        elif tecla == ord('m'):
            imprimir_menu()

        elif tecla == ord('F'):
            estado['idx_filtro'] = (idx_f + 1) % filtros.N_FILTROS
            filtros.descrever_filtro(estado['idx_filtro'])

        elif tecla == ord('f'):
            estado['idx_filtro'] = (idx_f - 1) % filtros.N_FILTROS
            filtros.descrever_filtro(estado['idx_filtro'])

        elif tecla == ord('a'):
            _salvar_historia(estado)
            img_trabalho = filtros.preview_filtro(img_trabalho, idx_f)
            estado['img_trabalho'] = img_trabalho
            print(f"[Filtro aplicado]: {filtros.FILTROS[idx_f]['nome']}")

        elif tecla == ord('u'):
            _desfazer(estado)
            img_trabalho = estado['img_trabalho']

        elif tecla == ord('y'):
            _refazer(estado)
            img_trabalho = estado['img_trabalho']

        elif tecla == ord('c'):
            canais = ['todos', 'R', 'G', 'B']
            estado['canal'] = canais[(canais.index(estado['canal']) + 1) % len(canais)]
            print(f"[Canal]: {estado['canal']}")

        elif tecla in (ord('1'), ord('2'), ord('3'), ord('4')):
            idx_op = int(chr(tecla)) - 1
            operacoes.descrever_operacao(idx_op)
            _salvar_historia(estado)
            img_trabalho = operacoes.aplicar_operacao(img_trabalho, img_secundaria, idx_op)
            estado['img_trabalho'] = img_trabalho

        elif tecla == ord('S'):
            if stickers_lista:
                estado['idx_sticker'] = (estado['idx_sticker'] + 1) % len(stickers_lista)
                estado['modo_sprite'] = False
                print(f"[Sticker]: {stickers_lista[estado['idx_sticker']]['nome']}")

        elif tecla == ord('s'):
            if stickers_lista:
                estado['idx_sticker'] = (estado['idx_sticker'] - 1) % len(stickers_lista)
                estado['modo_sprite'] = False
                print(f"[Sticker]: {stickers_lista[estado['idx_sticker']]['nome']}")

        elif tecla == ord('d'):
            detector.toggle_ativo()

        elif tecla == ord('e'):
            detector.trocar_modo()

        elif tecla == ord('Z'):
            ger.proximo()
            estado['modo_sprite'] = True

        elif tecla == ord('z'):
            ger.anterior()
            estado['modo_sprite'] = True

        elif tecla == ord('g'):
            _salvar(display)

        elif tecla == ord('r'):
            _resetar(estado, img_original)
            img_trabalho = estado['img_trabalho']
            print("[Resetar]: imagem original restaurada")

    return 'sair'


def loop_video(estado):
    """Loop principal do Modo Vídeo. Retorna: 'sair' | 'foto'"""
    global _ultimo_clique
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("[AVISO] Webcam não encontrada. Tentando vídeo de fallback...")
        cap = cv.VideoCapture(VIDEO_FALLBACK)
    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir câmera nem vídeo de fallback.")
        print("       Coloque 'assets/video_fallback.mp4' e tente novamente.")
        return 'foto'

    cv.namedWindow(NOME_JANELA)
    cv.setMouseCallback(NOME_JANELA, _mouse_callback)
    print("\n[Modo VÍDEO] Pronto. Pressione [M] para ver os controles.")

    detector = estado['detector']
    ger      = estado['gerenciador_anim']

    _HUD_H = 4 * 24 + 8  # 4 linhas de HUD

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        frame  = cv.flip(_redimensionar_se_grande(frame), 1)
        idx_f  = estado['idx_filtro']

        display = filtros.exibir_canal(filtros.preview_filtro(frame, idx_f), estado['canal'])
        display = stickers.aplicar_lista_stickers(display, estado['posicoes_stickers'],
                                                  estado['stickers_lista'], estado['idx_sticker'])
        display = ger.desenhar(display)
        display = detector.processar(display)
        display = _desenhar_hud(display, estado)
        cv.imshow(NOME_JANELA, display)
        filtros.tick_controles()

        if _ultimo_clique is not None:
            cx, cy = _ultimo_clique
            _ultimo_clique = None
            cy_real = max(0, cy - _HUD_H)
            stickers_lista = estado['stickers_lista']
            if estado['modo_sprite'] and ger.n_sprites > 0:
                ger.colocar(cx, cy_real)
            elif stickers_lista:
                estado['posicoes_stickers'].append({'x': cx, 'y': cy_real, 'idx': estado['idx_sticker']})
                nome_s = stickers_lista[estado['idx_sticker']]['nome']
                print(f"[Sticker colocado]: {nome_s} em ({cx}, {cy_real})")

        tecla = cv.waitKey(1) & 0xFF

        if tecla in (ord('q'), 27):
            cap.release()
            return 'sair'

        elif tecla == ord('v'):
            print("\n[Modo FOTO ativado]")
            cap.release()
            return 'foto'

        elif tecla == ord('m'):
            imprimir_menu()

        elif tecla == ord('F'):
            estado['idx_filtro'] = (idx_f + 1) % filtros.N_FILTROS
            filtros.descrever_filtro(estado['idx_filtro'])

        elif tecla == ord('f'):
            estado['idx_filtro'] = (idx_f - 1) % filtros.N_FILTROS
            filtros.descrever_filtro(estado['idx_filtro'])

        elif tecla == ord('c'):
            canais = ['todos', 'R', 'G', 'B']
            estado['canal'] = canais[(canais.index(estado['canal']) + 1) % len(canais)]
            print(f"[Canal]: {estado['canal']}")

        elif tecla == ord('S'):
            stickers_lista = estado['stickers_lista']
            if stickers_lista:
                estado['idx_sticker'] = (estado['idx_sticker'] + 1) % len(stickers_lista)
                estado['modo_sprite'] = False
                print(f"[Sticker]: {stickers_lista[estado['idx_sticker']]['nome']}")

        elif tecla == ord('s'):
            stickers_lista = estado['stickers_lista']
            if stickers_lista:
                estado['idx_sticker'] = (estado['idx_sticker'] - 1) % len(stickers_lista)
                estado['modo_sprite'] = False
                print(f"[Sticker]: {stickers_lista[estado['idx_sticker']]['nome']}")

        elif tecla == ord('d'):
            detector.toggle_ativo()

        elif tecla == ord('e'):
            detector.trocar_modo()

        elif tecla == ord('Z'):
            ger.proximo()
            estado['modo_sprite'] = True

        elif tecla == ord('z'):
            ger.anterior()
            estado['modo_sprite'] = True

        elif tecla == ord('g'):
            _salvar(display)

        elif tecla == ord('r'):
            _resetar(estado, img_original=None)

        elif tecla == ord('p'):
            captura = filtros.exibir_canal(filtros.preview_filtro(frame, idx_f), estado['canal'])
            captura = detector.processar(captura)
            estado['img_original']   = captura
            estado['img_trabalho']   = captura.copy()
            estado['img_secundaria'] = _gerar_imagem_secundaria(captura)
            estado['idx_filtro']     = 0
            estado['historico'].clear()
            estado['historico_redo'].clear()
            estado['detector'].ativo = False
            print("[Captura]: frame salvo — Modo Foto")
            cap.release()
            return 'foto'

    cap.release()
    return 'sair'


def _carregar_estado_inicial():
    stickers_lista = stickers.carregar_stickers()
    print(f"[Stickers carregados]: {len(stickers_lista)}")
    for i, s in enumerate(stickers_lista):
        print(f"  [{i+1}] {s['nome']}")

    detector = face_mod.DetectorFace()
    gerenciador_anim = stickers.GerenciadorSpritesAnimados()

    img, img2, modo_inicial = _menu_inicial()

    if img is not None and img2 is None:
        img2 = _gerar_imagem_secundaria(img)

    return {
        'modo':              modo_inicial,
        'img_original':      img.copy() if img is not None else None,
        'img_trabalho':      img.copy() if img is not None else None,
        'img_secundaria':    img2,
        'canal':             'todos',
        'idx_filtro':        0,
        'idx_sticker':       0,
        'stickers_lista':    stickers_lista,
        'posicoes_stickers': [],
        'historico':         [],
        'historico_redo':    [],
        'detector':          detector,
        'gerenciador_anim':  gerenciador_anim,
        'modo_sprite':       False,
    }


def main():
    print("=" * 56)
    print("  Processamento Gráfico: Fundamentos — Grau B")
    print("  Stories Filter App — Python + OpenCV")
    print("=" * 56)

    cv.namedWindow(NOME_JANELA)
    filtros.criar_janela_controles()
    cv.moveWindow(NOME_JANELA, 0, 0)
    cv.moveWindow(filtros.NOME_JANELA_CONTROLES, LARGURA_MAX + 10, 0)

    estado = _carregar_estado_inicial()
    if estado['modo'] == 'sair':
        cv.destroyAllWindows()
        print("\n[App encerrado]")
        return
    imprimir_menu()

    modo = estado['modo']
    while True:
        if modo == 'foto':
            if estado['img_original'] is None:
                img, img2, novo_modo = _menu_inicial()
                if novo_modo == 'sair':
                    break
                elif novo_modo == 'video' or img is None:
                    modo = 'video'
                    estado['modo'] = 'video'
                    continue
                estado['img_original']   = img
                estado['img_trabalho']   = img.copy()
                estado['img_secundaria'] = img2 or _gerar_imagem_secundaria(img)
                estado['posicoes_stickers'].clear()
                estado['historico'].clear()
                estado['historico_redo'].clear()
            resultado = loop_foto(estado)
        else:
            resultado = loop_video(estado)

        if resultado == 'sair':
            break
        elif resultado == 'video':
            modo = 'video'
            estado['modo'] = 'video'
        elif resultado == 'foto':
            modo = 'foto'
            estado['modo'] = 'foto'

    cv.destroyAllWindows()
    print("\n[App encerrado]")


if __name__ == '__main__':
    main()
