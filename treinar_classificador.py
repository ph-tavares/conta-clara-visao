"""Treina o classificador de dígitos (YOLO11n-cls) sobre os recortes de preparar_digitos.py e mede no teste."""
import argparse
import json
from pathlib import Path
import shutil

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', type=Path, default=Path('dados_digitos'))
    parser.add_argument('--base', default='yolo11n-cls.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=64)
    args = parser.parse_args()
    treino = YOLO(args.base).train(data=str(args.dados.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=64,
        device=args.device, workers=0, seed=97877, fliplr=0, flipud=0,  # dígito espelhado vira outro
        project=str(Path('runs').resolve()), name='classificador', exist_ok=True, plots=True, verbose=False)
    Path('modelos').mkdir(exist_ok=True)
    shutil.copyfile(Path(treino.save_dir) / 'weights' / 'best.pt', 'modelos/digitos.pt')
    teste = YOLO('modelos/digitos.pt').val(data=str(args.dados.resolve()), split='test', imgsz=args.imgsz, batch=64,
        device=args.device, workers=0, project=str(Path('runs').resolve()), name='classificador_teste', exist_ok=True,
        plots=True, verbose=False)
    saida = Path('evidencias')
    saida.mkdir(exist_ok=True)
    resultado = {'acuracia_top1_teste': float(teste.top1), 'epocas': args.epochs, 'imgsz': args.imgsz}
    (saida / 'classificador.json').write_text(json.dumps(resultado, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(Path(teste.save_dir) / 'confusion_matrix_normalized.png', saida / 'classificador_confusao.png')
    print(resultado)

if __name__ == '__main__':
    main()
