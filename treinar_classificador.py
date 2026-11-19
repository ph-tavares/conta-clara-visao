"""Treina o classificador de dígitos (YOLO11n-cls) sobre os recortes de preparar_digitos.py e mede na validação.

O teste não é usado aqui: ele é medido uma vez, com o modelo congelado, por `--somente-avaliar --divisao test`.
"""
import argparse
import json
from pathlib import Path
import shutil

from ultralytics import YOLO

def medir(pesos, args, divisao, saida):
    metrics = YOLO(str(pesos)).val(data=str(args.dados.resolve()), split=divisao, imgsz=args.imgsz, batch=64,
        device=args.device, workers=0, project=str(Path('runs').resolve()), name=f'classificador_{divisao}',
        exist_ok=True, plots=True, verbose=False)
    saida.mkdir(parents=True, exist_ok=True)
    resultado = {'divisao': divisao, 'acuracia_top1': float(metrics.top1), 'imgsz': args.imgsz}
    (saida / 'classificador.json').write_text(json.dumps(resultado, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(Path(metrics.save_dir) / 'confusion_matrix_normalized.png', saida / 'classificador_confusao.png')
    return resultado

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', type=Path, default=Path('dados_digitos'))
    parser.add_argument('--base', default='yolo11n-cls.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=64)
    parser.add_argument('--somente-avaliar', action='store_true', help='Nao treina: mede --base na divisao escolhida')
    parser.add_argument('--divisao', default='val', choices=('val', 'test'))
    parser.add_argument('--saida', type=Path, default=Path('evidencias'))
    args = parser.parse_args()
    if args.somente_avaliar:
        print(medir(args.base, args, args.divisao, args.saida))
        return
    treino = YOLO(args.base).train(data=str(args.dados.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=64,
        device=args.device, workers=0, seed=97877, fliplr=0, flipud=0,  # dígito espelhado vira outro
        project=str(Path('runs').resolve()), name='classificador', exist_ok=True, plots=True, verbose=False)
    Path('modelos').mkdir(exist_ok=True)
    shutil.copyfile(Path(treino.save_dir) / 'weights' / 'best.pt', 'modelos/digitos.pt')
    print(medir('modelos/digitos.pt', args, 'val', args.saida / 'validacao_classificador'))

if __name__ == '__main__':
    main()
