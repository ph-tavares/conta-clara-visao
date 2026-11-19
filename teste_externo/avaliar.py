"""Avalia o leitor no teste externo: fotos reais de outra fonte (Wikimedia Commons), nunca usadas no treino.

O recorte (ROI) é o enquadramento do mostrador; a leitura de referência foi conferida manualmente na foto.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parent))
from conta_clara import DEFAULT_CLASSIFIER, DEFAULT_MODEL, MIN_SCORE, Reader, load_image, render  # noqa: E402

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pesos', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--classificador',type=Path,default=DEFAULT_CLASSIFIER)
    parser.add_argument('--limiar', type=float, default=MIN_SCORE)
    parser.add_argument('--saida', type=Path, default=PASTA.parent / 'evidencias')
    args = parser.parse_args()
    reader = Reader(args.pesos, threshold=args.limiar, classificador=args.classificador)
    anotadas = args.saida / 'teste_externo'
    anotadas.mkdir(parents=True, exist_ok=True)
    registros = []
    for linha in csv.DictReader((PASTA / 'manifesto.csv').open(encoding='utf-8')):
        imagem = load_image(PASTA / 'fotos' / linha['arquivo_commons'])
        escala = imagem.shape[1] / 1280
        roi = tuple(round(int(linha[k]) * escala) for k in ('roi_x', 'roi_y', 'roi_largura', 'roi_altura'))
        resultado = reader.predict(imagem, roi)
        x, y, largura, altura = roi
        margem = round(altura * .6)
        recorte = render(imagem, resultado)[max(0, y - margem):y + altura + margem, max(0, x - margem):x + largura + margem]
        cv2.imwrite(str(anotadas / f"{linha['id']}.jpg"), recorte)
        registros.append({'id': linha['id'], 'referencia': linha['leitura'], 'candidata': resultado['leitura_candidata_kwh'],
                          'status': resultado['status'], 'motivo': resultado['motivo'],
                          'confianca_minima': resultado['confianca_minima'],
                          'correta': resultado['leitura_candidata_kwh'] == linha['leitura']})
    sugeridas = [r for r in registros if r['status'] == 'leitura_sugerida']
    resumo = {'fotos': len(registros), 'exatas': sum(r['correta'] for r in registros),
              'sugeridas': len(sugeridas), 'sugeridas_corretas': sum(r['correta'] for r in sugeridas),
              'revisao': len(registros) - len(sugeridas), 'registros': registros}
    (args.saida / 'teste_externo.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in resumo.items() if k != 'registros'}, ensure_ascii=False))
    for r in registros:
        print(f"{r['id']:7} ref {r['referencia']:>8}  lido {str(r['candidata']):>8}  {r['status']}  {r['motivo'] or ''}")

if __name__ == '__main__':
    main()
