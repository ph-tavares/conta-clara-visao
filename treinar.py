"""Executa o ajuste supervisionado do YOLO11n e avalia caixas no teste."""
import argparse
import json
from pathlib import Path
import shutil
from ultralytics import YOLO

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados',type=Path,default=Path('dados/dados.yaml'))
    parser.add_argument('--base',default='yolo11n.pt')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--epochs',type=int,default=12)
    args=parser.parse_args()
    model=YOLO(args.base)
    train=model.train(data=str(args.dados.resolve()),epochs=args.epochs,imgsz=320,batch=16,
        device=args.device,workers=0,seed=97877,optimizer='AdamW',lr0=.002,mosaic=0,
        fliplr=0,flipud=0,degrees=0,scale=.12,translate=.07,hsv_s=.12,
        project=str(Path('runs').resolve()),name='conta_clara',exist_ok=True,plots=True,verbose=False)
    Path('modelos').mkdir(exist_ok=True)
    checkpoint=Path(train.save_dir)/'weights'/'best.pt'
    shutil.copyfile(checkpoint,'modelos/best.pt')
    metrics=YOLO('modelos/best.pt').val(data=str(args.dados.resolve()),split='test',imgsz=320,batch=16,
        workers=0,device=args.device,project=str(Path('runs').resolve()),name='teste',exist_ok=True,plots=True,verbose=False)
    box=metrics.box
    out=Path('evidencias')
    out.mkdir(exist_ok=True)
    result={'map50':float(box.map50),'map50_95':float(box.map),'precisao_macro':float(box.mp),'recall_macro':float(box.mr),
            'ap50_por_classe':{str(int(k)):float(v) for k,v in zip(box.ap_class_index,box.ap50)},
            'epocas':args.epochs,'imgsz':320}
    (out/'deteccao.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for name in ('results.csv','results.png','args.yaml'):
        shutil.copyfile(Path(train.save_dir)/name,out/name)
    for name in ('BoxPR_curve.png','confusion_matrix_normalized.png'):
        shutil.copyfile(Path(metrics.save_dir)/name,out/name)

if __name__=='__main__':
    main()
