"""Prepara as fotos reais (exportação YOLO do Roboflow) com divisão por grupo, sem vazamento entre treino e teste.

Fotos que são cópias da mesma cena (mesma foto girada, recortada ou repetida) ficam sempre na mesma divisão.
Dois critérios juntam fotos num grupo: o mesmo conjunto de dígitos anotados e a imagem quase igual.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import yaml

from conta_clara import Digit, text_axis

FRACOES = (('train', .70), ('val', .15), ('test', .15))

def construir_grupos(digitos: dict[str, list[str]], hashes: dict[str, np.ndarray], limite: int = 12) -> dict[str, str]:
    """Junta no mesmo grupo as fotos com o mesmo conjunto de dígitos OU com imagem quase igual (hash perceptual).

    hashes[foto] tem as 4 rotações de 90 graus (4 x 256 bits); duas fotos são quase iguais quando alguma
    rotação de uma fica a no máximo `limite` bits da outra. Devolve foto -> identificador do grupo.
    """
    fotos = sorted(digitos)
    uniao = Grupos(fotos)
    por_digitos = {}
    for foto in fotos:
        por_digitos.setdefault(''.join(sorted(digitos[foto])), []).append(foto)
    for membros in por_digitos.values():
        for outro in membros[1:]:
            uniao.unir(outro, membros[0])
    todas = np.array([hashes[f] for f in fotos]).astype(np.int16)        # (n, 4, 256)
    base = todas[:, 0, :]
    for i, foto in enumerate(fotos):
        distancia = np.abs(todas[i][:, None, :] - base[None]).sum(2).min(0)
        for j in np.nonzero(distancia <= limite)[0]:
            uniao.unir(foto, fotos[j])
    return {foto: uniao.raiz(foto) for foto in fotos}

def dividir_por_grupo(grupos: dict[str, str], fracoes=FRACOES) -> dict[str, str]:
    """Sorteio determinístico por grupo (hash do nome do grupo): um grupo inteiro cai numa única divisão."""
    def divisao(grupo):
        posicao = int(hashlib.sha256(grupo.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        acumulado = 0.0
        for nome, fracao in fracoes:
            acumulado += fracao
            if posicao <= acumulado:
                return nome
        return fracoes[-1][0]
    return {arquivo: divisao(grupo) for arquivo, grupo in grupos.items()}

def assinatura(caminho: Path) -> np.ndarray:
    """Hash perceptual 16x16, o menor entre as 4 rotações de 90 graus (pega cópias giradas)."""
    with Image.open(caminho) as imagem:
        base = np.asarray(imagem.convert('L').resize((16, 16)), dtype=np.float32)
    versoes = [np.rot90(base, k) for k in range(4)]
    return np.array([(v > v.mean()).flatten() for v in versoes])

def ler_rotulos(arquivo: Path) -> list[tuple[int, float, float, float, float]]:
    linhas = [l.split() for l in arquivo.read_text().splitlines() if l.strip()]
    return [(int(k), float(x), float(y), float(w), float(h)) for k, x, y, w, h in linhas]

def leitura(rotulos) -> str:
    """Leitura de referência: dígitos anotados ordenados no sentido do texto."""
    digitos = [Digit(str(k), 1.0, (x - w / 2, y - h / 2, x + w / 2, y + h / 2)) for k, x, y, w, h in rotulos]
    ux, uy = text_axis([d.center for d in digitos])
    return ''.join(d.text for d in sorted(digitos, key=lambda d: d.center[0] * ux + d.center[1] * uy))

class Grupos:
    def __init__(self, itens):
        self.pai = {i: i for i in itens}
    def raiz(self, i):
        while self.pai[i] != i:
            self.pai[i] = self.pai[self.pai[i]]
            i = self.pai[i]
        return i
    def unir(self, a, b):
        self.pai[self.raiz(a)] = self.raiz(b)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origem', type=Path, required=True, help='Pasta da exportação YOLO (train/valid/test)')
    parser.add_argument('--destino', type=Path, default=Path('dados_reais'))
    parser.add_argument('--sinteticos', type=Path, help='Pasta de gerar_dados.py: soma o treino sintetico ao treino real')
    parser.add_argument('--max-por-grupo', type=int, default=0, help='No treino, copias da mesma cena mantidas (0 = todas)')
    parser.add_argument('--max-sinteticos', type=int, default=0, help='Imagens sinteticas somadas ao treino (0 = todas)')
    parser.add_argument('--limite-hash', type=int, default=12, help='Bits diferentes para considerar duas fotos quase iguais')
    args = parser.parse_args()

    fotos = sorted(p for p in args.origem.glob('*/images/*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    rotulos = {p: ler_rotulos(p.parent.parent / 'labels' / (p.stem + '.txt')) for p in fotos}
    fotos = [p for p in fotos if rotulos[p]]
    hash_de = {str(p): assinatura(p) for p in fotos}
    grupos = construir_grupos({str(p): [str(r[0]) for r in rotulos[p]] for p in fotos}, hash_de, args.limite_hash)
    divisao = dividir_por_grupo(grupos)
    if args.max_por_grupo:
        mantidas = {}
        for p in sorted(fotos, key=lambda p: hashlib.sha256(p.name.encode()).hexdigest()):
            if divisao[str(p)] == 'train':
                mantidas.setdefault(grupos[str(p)], []).append(p)
        descartar = {str(p) for ps in mantidas.values() for p in ps[args.max_por_grupo:]}
        fotos = [p for p in fotos if str(p) not in descartar]

    if args.destino.exists():
        shutil.rmtree(args.destino)
    destino = args.destino.resolve()
    destino.mkdir(parents=True)
    with (destino / 'manifesto.csv').open('w', newline='', encoding='utf-8') as arquivo:
        writer = csv.writer(arquivo, lineterminator='\n')
        writer.writerow(['divisao', 'arquivo', 'leitura', 'grupo'])
        for p in fotos:
            parte = divisao[str(p)]
            for pasta in ('images', 'labels'):
                (destino / pasta / parte).mkdir(parents=True, exist_ok=True)
            nome = f'real_{hashlib.sha256(str(p.name).encode()).hexdigest()[:12]}'
            shutil.copyfile(p, destino / 'images' / parte / (nome + p.suffix.lower()))
            shutil.copyfile(p.parent.parent / 'labels' / (p.stem + '.txt'), destino / 'labels' / parte / (nome + '.txt'))
            writer.writerow([parte, nome + p.suffix.lower(), leitura(rotulos[p]), hashlib.sha256(grupos[str(p)].encode()).hexdigest()[:10]])

    contagem = {n: sum(divisao[str(p)] == n for p in fotos) for n, _ in FRACOES}
    grupos_por_divisao = {}
    for p, g in grupos.items():  # todas as fotos, inclusive copias descartadas
        grupos_por_divisao.setdefault(g, set()).add(divisao[p])
    cruzados = sum(len(s) > 1 for s in grupos_por_divisao.values())
    base_treino = np.array([hash_de[str(p)][0] for p in fotos if divisao[str(p)] == 'train']).astype(np.int16)
    quase = sum(bool((np.abs(hash_de[str(p)][:, None, :].astype(np.int16) - base_treino[None]).sum(2).min() <= args.limite_hash))
                for p in fotos if divisao[str(p)] == 'test')
    treino_yaml = [str(destino / 'images/train')]
    if args.sinteticos:
        sinteticas = sorted((args.sinteticos.resolve() / 'images/train').glob('*.jpg'))
        if args.max_sinteticos:
            sinteticas = sinteticas[:args.max_sinteticos]
        (destino / 'treino_sintetico.txt').write_text(''.join(f'{f}\n' for f in sinteticas), encoding='utf-8')
        treino_yaml.append(str(destino / 'treino_sintetico.txt'))
    config = {'path': str(destino), 'train': treino_yaml, 'val': 'images/val', 'test': 'images/test',
              'names': {i: str(i) for i in range(10)}}
    (destino / 'dados.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print({'fotos': len(fotos), 'grupos': len(grupos_por_divisao), 'divisoes': contagem,
           'grupos_cruzados': cruzados, 'teste_com_quase_copia_no_treino': quase})

if __name__ == '__main__':
    main()
