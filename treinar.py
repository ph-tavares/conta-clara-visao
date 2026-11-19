"""Treina o detector YOLO11n e mede as caixas na validação.

O teste não é usado aqui: ele é medido uma vez, com o modelo congelado, por `--somente-avaliar --divisao test`.
"""
import argparse
import json
from pathlib import Path
import shutil
from ultralytics import YOLO

def medir(pesos, args, divisao, saida):
    metrics=YOLO(str(pesos)).val(data=str(args.dados.resolve()),split=divisao,imgsz=args.imgsz,batch=16,
        workers=0,device=args.device,project=str(Path('runs').resolve()),name=f'deteccao_{divisao}',exist_ok=True,
        plots=True,verbose=False,single_cls=args.classe_unica)
    box=metrics.box
    saida.mkdir(parents=True,exist_ok=True)
    result={'divisao':divisao,'map50':float(box.map50),'map50_95':float(box.map),'precisao_macro':float(box.mp),
            'recall_macro':float(box.mr),'ap50_por_classe':{str(int(k)):float(v) for k,v in zip(box.ap_class_index,box.ap50)},
            'imgsz':args.imgsz,'classe_unica':args.classe_unica}
    (saida/'deteccao.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for name in ('BoxPR_curve.png','confusion_matrix_normalized.png'):
        shutil.copyfile(Path(metrics.save_dir)/name,saida/name)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados',type=Path,default=Path('dados/dados.yaml'))
    parser.add_argument('--base',default='yolo11n.pt')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--imgsz',type=int,default=416)
    parser.add_argument('--classe-unica',action='store_true',help='Detector so localiza "digito"; quem le o numero e o classificador')
    parser.add_argument('--somente-avaliar',action='store_true',help='Nao treina: mede --base na divisao escolhida')
    parser.add_argument('--divisao',default='val',choices=('val','test'))
    parser.add_argument('--saida',type=Path,default=Path('evidencias'))
    args=parser.parse_args()
    if args.somente_avaliar:
        print(medir(args.base,args,args.divisao,args.saida))
        return
    train=YOLO(args.base).train(data=str(args.dados.resolve()),epochs=args.epochs,imgsz=args.imgsz,batch=16,
        device=args.device,workers=0,seed=97877,optimizer='auto',mosaic=1.0,close_mosaic=5,
        fliplr=0,flipud=0,degrees=0,scale=.5,translate=.1,hsv_s=.5,hsv_v=.4,  # fotos reais: sem espelhar (digito espelhado vira outro)
        project=str(Path('runs').resolve()),name='conta_clara',exist_ok=True,plots=True,verbose=False,
        single_cls=args.classe_unica)
    Path('modelos').mkdir(exist_ok=True)
    shutil.copyfile(Path(train.save_dir)/'weights'/'best.pt','modelos/best.pt')
    args.saida.mkdir(parents=True,exist_ok=True)
    for name in ('results.csv','results.png'):
        shutil.copyfile(Path(train.save_dir)/name,args.saida/name)
    print(medir('modelos/best.pt',args,'val',args.saida/'validacao'))

if __name__=='__main__':
    main()
