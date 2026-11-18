"""Detecção e transcrição assistida de cinco dígitos de medidores."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
from PIL import Image, ImageOps
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'modelos' / 'best.pt'
MIN_SCORE = 0.65
EXPECTED_DIGITS = 5

@dataclass(frozen=True)
class Digit:
    text: str
    confidence: float
    bounds: tuple[int, int, int, int]

    @property
    def center(self):
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def height(self):
        return self.bounds[3] - self.bounds[1]

def read_digits(digits: list[Digit], threshold: float = MIN_SCORE) -> dict:
    digits = sorted(digits, key=lambda digit: digit.center[0])
    confidences = [d.confidence for d in digits]
    reading = ''.join(d.text for d in digits) if len(digits) == EXPECTED_DIGITS else None
    reason = None
    if reading is None:
        reason = 'quantidade_de_digitos'
    else:
        heights = [d.height for d in digits]
        centers_x, centers_y = zip(*(d.center for d in digits))
        gaps = [b - a for a, b in zip(centers_x, centers_x[1:])]
        reference_height = median(heights)
        if (min(heights) <= 0 or (max(centers_y) - min(centers_y)) > 0.85 * reference_height
                or min(gaps) <= 0 or max(gaps) > 2.1 * min(gaps)):
            reason = 'disposicao_inconsistente'
        elif min(confidences) < threshold:
            reason = 'baixa_confianca'
    return {
        'leitura_candidata_kwh': reading,
        'confianca_minima': round(min(confidences), 4) if reading is not None else None,
        'status': 'confirmacao_necessaria' if reason else 'leitura_sugerida',
        'motivo': reason,
        'digitos': [{'valor': d.text, 'confianca': round(d.confidence, 4), 'caixa_xyxy': list(d.bounds)}
                    for d in digits],
    }

def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

class Reader:
    def __init__(self, weights: Path = DEFAULT_MODEL, device: str = 'cpu', threshold: float = MIN_SCORE):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device
        self.threshold = threshold

    def predict(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> dict:
        h, w = frame.shape[:2]
        x, y, width, height = roi if roi else (0, 0, w, h)
        if x < 0 or y < 0 or width < 16 or height < 16 or x + width > w or y + height > h:
            raise ValueError('O recorte deve estar dentro da imagem e medir ao menos 16 x 16 pixels.')
        started = perf_counter()
        prediction = self.model.predict(frame[y:y + height, x:x + width], imgsz=320,
            device=self.device, conf=.15, iou=.40, max_det=20, agnostic_nms=True, verbose=False)[0]
        boxes = prediction.boxes.cpu()
        digits = [
            Digit(str(int(k)), float(score), (round(x1)+x, round(y1)+y, round(x2)+x, round(y2)+y))
            for k, score, (x1, y1, x2, y2) in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())
        ]
        report = read_digits(digits, self.threshold)
        report['inferencia_ms'] = round((perf_counter() - started) * 1000, 2)
        return report

def render(frame, result):
    output = frame.copy()
    valid = result['status'] == 'leitura_sugerida'
    color = (92, 235, 112) if valid else (49, 199, 250)
    for d in result['digitos']:
        x1,y1,x2,y2=d['caixa_xyxy']
        cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
        label=f"{d['valor']}  {d['confianca']:.2f}"
        y=max(18,y1-5)
        cv2.rectangle(output,(x1,y-15),(x1+64,y+3),(19,31,35),-1)
        cv2.putText(output,label,(x1+2,y),cv2.FONT_HERSHEY_SIMPLEX,.40,color,1,cv2.LINE_AA)
    return output

def process_video(args, reader):
    source = int(args.fonte) if args.fonte.isdecimal() else args.fonte
    if args.saida and args.saida.suffix.lower() != '.mp4':
        raise ValueError('A saída do vídeo deve ter extensão .mp4.')
    if args.saida and isinstance(source,str) and Path(source).resolve() == args.saida.resolve():
        raise ValueError('Use caminhos diferentes para entrada e saída.')
    capture = cv2.VideoCapture(source)
    writer = None
    try:
        if not capture.isOpened():
            raise OSError('Não foi possível abrir a fonte de vídeo.')
        fps = capture.get(cv2.CAP_PROP_FPS) or 24
        if not np.isfinite(fps) or fps <= 0:
            fps = 24
        frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            result = reader.predict(frame, args.roi)
            rendered = render(frame, result)
            if args.saida:
                args.saida.parent.mkdir(parents=True, exist_ok=True)
                if writer is None:
                    h, w = rendered.shape[:2]
                    writer = cv2.VideoWriter(str(args.saida), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
                    if not writer.isOpened():
                        raise OSError('Não foi possível gravar a saída.')
                writer.write(rendered)
            if args.exibir:
                cv2.imshow('Conta Clara CV', rendered)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
            if frames % 15 == 1:
                print(json.dumps({'quadro': frames, **result}, ensure_ascii=False),flush=True)
        if not frames:
            raise OSError('A fonte não retornou quadros.')
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.exibir:
            cv2.destroyAllWindows()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fonte',required=True, help='Imagem, vídeo local ou índice da webcam')
    parser.add_argument('--pesos',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--saida',type=Path,help='Imagem anotada ou vídeo .mp4')
    parser.add_argument('--roi',type=int,nargs=4,metavar=('X','Y','LARGURA','ALTURA'))
    parser.add_argument('--limiar',type=float,default=MIN_SCORE)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--exibir',action='store_true')
    args=parser.parse_args()
    if not .15 <= args.limiar <= 1:
        parser.error('--limiar deve ficar entre 0,15 e 1')
    if not args.pesos.is_file():
        parser.error(f'Pesos não encontrados em {args.pesos}')
    reader=Reader(args.pesos,args.device,args.limiar)
    source=Path(args.fonte)
    try:
        if source.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'}:
            if args.saida and args.saida.resolve() == source.resolve():
                raise ValueError('Use uma saída diferente da foto original.')
            image=load_image(source)
            result=reader.predict(image,args.roi)
            if args.saida:
                args.saida.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(args.saida),render(image,result)):
                    raise OSError('Não foi possível gravar a imagem anotada.')
            if args.exibir:
                cv2.imshow('Conta Clara CV',render(image,result))
                cv2.waitKey()
                cv2.destroyAllWindows()
            print(json.dumps(result,ensure_ascii=False,indent=2))
        else:
            process_video(args,reader)
    except (ValueError, OSError, cv2.error) as error:
        parser.error(str(error))

if __name__=='__main__':
    main()
