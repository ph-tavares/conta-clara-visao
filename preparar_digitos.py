"""Monta a base do classificador de dígitos: recortes reais (das caixas anotadas) + dígitos sintéticos.

A divisão treino/val/teste é herdada de preparar_reais.py (por cena), então nenhum recorte de teste
vem de uma foto usada no treino. Os sintéticos entram só no treino.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from conta_clara import MAX_TILT, angulo_texto, recorte_quadrado  # mesmo recorte no treino e na leitura
from gerar_dados import DIGITS, SEGMENTS

PASTAS_FONTES = ('/System/Library/Fonts', '/System/Library/Fonts/Supplemental', '/Library/Fonts',
                 '/usr/share/fonts', 'C:/Windows/Fonts')

@lru_cache
def fontes() -> tuple[str, ...]:
    achadas = []
    for pasta in PASTAS_FONTES:
        for arquivo in sorted(Path(pasta).glob('*')) if Path(pasta).is_dir() else []:
            if arquivo.suffix.lower() in {'.ttf', '.otf'} and desenha_digitos(str(arquivo)):
                achadas.append(str(arquivo))
    return tuple(achadas)

def desenha_digitos(caminho: str) -> bool:
    """Fonte útil: os 10 dígitos saem diferentes entre si e diferentes do glifo vazio (quadradinho)."""
    try:
        fonte = ImageFont.truetype(caminho, 32)
    except Exception:
        return False
    def mapa(texto):
        imagem = Image.new('L', (48, 48), 0)
        ImageDraw.Draw(imagem).text((8, 4), texto, font=fonte, fill=255)
        return np.asarray(imagem).tobytes()
    digitos = [mapa(str(k)) for k in range(10)]
    vazio = mapa('\U000F0000')
    return len(set(digitos)) == 10 and vazio not in digitos and all(any(d) for d in digitos)

def digito_sintetico(digito: int, semente: int, lado: int = 64) -> np.ndarray:
    """Um dígito de mostrador desenhado: 7 segmentos (LCD) ou fonte de rolete, com cor, giro, borrão e ruído."""
    rng = np.random.default_rng(semente * 10 + digito)
    escuro_no_claro = rng.random() < .5
    fundo = tuple(int(v) for v in (rng.integers(150, 235, 3) if escuro_no_claro else rng.integers(5, 70, 3)))
    if not escuro_no_claro and rng.random() < .15:
        fundo = (int(rng.integers(150, 220)), int(rng.integers(10, 50)), int(rng.integers(10, 50)))  # rolete decimal vermelho
    tinta = tuple(int(v) for v in (rng.integers(0, 60, 3) if escuro_no_claro else rng.integers(200, 256, 3)))
    grande = lado * 3
    imagem = Image.new('RGB', (grande, grande), fundo)
    draw = ImageDraw.Draw(imagem)
    altura = int(grande * rng.uniform(.62, .8))
    largura = int(altura * rng.uniform(.5, .66))
    x0, y0 = (grande - largura) // 2, (grande - altura) // 2
    lista = fontes()
    if rng.random() < .4 or not lista:
        apagado = tuple(int(f * .85 + t * .15) for f, t in zip(fundo, tinta))
        for i, poligono in enumerate(SEGMENTS):
            cor = tinta if DIGITS[digito] & (1 << i) else apagado
            draw.polygon([(x0 + px * largura / 58, y0 + py * altura / 83) for px, py in poligono], fill=cor)
    else:
        fonte = ImageFont.truetype(lista[int(rng.integers(len(lista)))], altura)
        a, b, c, d = draw.textbbox((0, 0), str(digito), font=fonte)
        draw.text(((grande - (c - a)) / 2 - a, (grande - (d - b)) / 2 - b), str(digito), font=fonte, fill=tinta)
    array = np.asarray(imagem)
    matriz = cv2.getRotationMatrix2D((grande / 2, grande / 2), float(rng.uniform(-8, 8)), float(rng.uniform(.9, 1.1)))
    matriz[:, 2] += rng.uniform(-grande * .05, grande * .05, 2)
    array = cv2.warpAffine(array, matriz, (grande, grande), borderMode=cv2.BORDER_REPLICATE)
    array = cv2.resize(array, (lado, lado), interpolation=cv2.INTER_AREA)
    if rng.random() < .5:
        array = cv2.GaussianBlur(array, (3, 3), float(rng.uniform(.3, 1.2)))
    ruido = rng.normal(0, rng.uniform(2, 12), array.shape)
    return np.clip(array.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', type=Path, required=True, help='Pasta criada por preparar_reais.py')
    parser.add_argument('--destino', type=Path, default=Path('dados_digitos'))
    parser.add_argument('--sinteticos-por-classe', type=int, default=600)
    parser.add_argument('--lado', type=int, default=64)
    args = parser.parse_args()
    if args.destino.exists():
        shutil.rmtree(args.destino)
    contagem = {}
    for divisao in ('train', 'val', 'test'):
        for k in range(10):
            (args.destino / divisao / str(k)).mkdir(parents=True, exist_ok=True)
        for foto in sorted((args.dados / 'images' / divisao).glob('*')):
            imagem = cv2.imread(str(foto))
            h, w = imagem.shape[:2]
            rotulo = args.dados / 'labels' / divisao / (foto.stem + '.txt')
            linhas = [l.split() for l in rotulo.read_text().splitlines() if l.split() and int(l.split()[0]) <= 9]
            centros = [(float(l[1]) * w, float(l[2]) * h) for l in linhas]
            angulo = angulo_texto(centros) if len(centros) > 1 else 0.0
            if abs(angulo) > MAX_TILT:  # foto girada: 6 e 9 ficam ambiguos; o leitor pede para girar a foto
                contagem[divisao + '_fotos_giradas_fora'] = contagem.get(divisao + '_fotos_giradas_fora', 0) + 1
                continue
            for n, partes in enumerate(linhas):
                k, x, y, bw, bh = int(partes[0]), *map(float, partes[1:5])
                caixa = ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
                cv2.imwrite(str(args.destino / divisao / str(k) / f'{foto.stem}_{n}.png'),
                            recorte_quadrado(imagem, caixa, args.lado, angulo=angulo))
                contagem[divisao] = contagem.get(divisao, 0) + 1
    for k in range(10):
        for i in range(args.sinteticos_por_classe):
            cv2.imwrite(str(args.destino / 'train' / str(k) / f'sint_{k}_{i:04d}.png'),
                        cv2.cvtColor(digito_sintetico(k, i, args.lado), cv2.COLOR_RGB2BGR))
    contagem['train_sinteticos'] = 10 * args.sinteticos_por_classe
    print({**contagem, 'fontes': len(fontes())})

if __name__ == '__main__':
    main()
