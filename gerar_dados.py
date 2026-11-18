"""Gera medidores sintéticos, rótulos YOLO e leituras de referência."""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

WIDTH, HEIGHT = 512, 288
SPLITS = {'train': (650, 97877), 'val': (120, 47881), 'test': (160, 86821)}
DIGITS = (0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f)
SEGMENTS = (
    ((7,4), (47,4), (54,10), (47,16), (7,16), (0,10)),
    ((51,10), (57,17), (57,36), (51,42), (45,36), (45,17)),
    ((51,44), (57,50), (57,69), (51,76), (45,69), (45,50)),
    ((7,70), (47,70), (54,76), (47,82), (7,82), (0,76)),
    ((5,44), (11,50), (11,69), (5,76), (0,69), (0,50)),
    ((5,10), (11,17), (11,36), (5,42), (0,36), (0,17)),
    ((8,37), (46,37), (53,43), (46,49), (8,49), (1,43)),
)

def jitter(color, offset):
    return tuple(max(0, min(255, c + offset)) for c in color)

@lru_cache
def font(size):
    return ImageFont.load_default(size=size)

def draw_digit(draw, x, y, w, h, digit, ink, shadow, mechanical):
    if mechanical:
        draw.rectangle((x-4,y-3,x+w+4,y+h+3), fill=shadow)
        f=font(int(h*0.98))
        value=str(digit)
        a,b,c,d=draw.textbbox((0,0),value,font=f)
        draw.text((x+(w-c+a)//2-a,y+(h-d+b)//2-b),value,font=f,fill=ink)
    else:
        for i, poly in enumerate(SEGMENTS):
            fill = ink if DIGITS[digit] & (1 << i) else shadow
            draw.polygon([(x+round(px*w/58),y+round(py*h/83)) for px,py in poly], fill=fill)

def scene(rng, digits, style):
    mechanical = style == 'mecanico'
    reflect = style == 'reflexo'
    background = jitter((99,105,108),int(rng.integers(-38,39)))
    body = jitter((179,184,179),int(rng.integers(-34,29)))
    glass = jitter((49,54,51),int(rng.integers(-13,37)))
    lcd = jitter((173,188,151) if not mechanical else (50,57,55),int(rng.integers(-22,29)))
    ink = jitter((28,37,28) if not mechanical else (227,230,214),int(rng.integers(-14,14)))
    shadow = jitter(lcd, -12 if not mechanical else -15)
    image = Image.new('RGB',(WIDTH,HEIGHT),background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((29,18,481,276),radius=27,fill=body,outline=jitter(body,-42),width=5)
    draw.rounded_rectangle((44,32,468,244),radius=17,fill=jitter(body,-19),outline=jitter(body,-32),width=2)
    draw.text((62,47),'MEDIDOR DIGITAL' if not mechanical else 'MEDIDOR DE ENERGIA',font=font(16),fill=jitter(body,-94))
    draw.text((62,218),'CONTA CLARA  /  AMOSTRA SINTETICA',font=font(11),fill=jitter(body,-77))
    draw.ellipse((426,219,437,230),fill=(138,68,48))
    h=int(rng.integers(66,101)); w=int(h*rng.uniform(.49,.66)); gap=int(rng.integers(8,17))
    dw=5*w+4*gap
    x=(WIDTH-dw)//2+int(rng.integers(-36,37))
    y=95+int(rng.integers(-18,18))
    draw.rounded_rectangle((x-23,y-19,x+dw+30,y+h+22),radius=8,fill=glass,outline=jitter(glass,-19),width=2)
    draw.rounded_rectangle((x-13,y-11,x+dw+13,y+h+11),radius=3,fill=lcd)
    boxes=[]
    for i, digit in enumerate(digits):
        dx=x+i*(w+gap); dy=y+int(rng.integers(-2,3))
        draw_digit(draw,dx,dy,w,h,digit,ink,shadow,mechanical)
        boxes.append((digit,dx,dy,dx+w,dy+h))
    draw.text((x+dw-21,y+h+9),'kWh',font=font(9),fill=jitter(body,28))
    array=np.asarray(image)
    yaw=float(rng.uniform(-10,10)); zoom=float(rng.uniform(.73,1.07))
    matrix=cv2.getRotationMatrix2D((WIDTH/2, HEIGHT/2),yaw,zoom)
    matrix[:,2]+=rng.uniform(-12,12,2)
    array=cv2.warpAffine(array,matrix,(WIDTH,HEIGHT),borderMode=cv2.BORDER_REPLICATE)
    if reflect or rng.random()<.19:
        x_coord,y_coord=np.meshgrid(np.arange(WIDTH),np.arange(HEIGHT))
        stripe=np.exp(-((x_coord+.43*y_coord-rng.uniform(140,390))/rng.uniform(14,48))**2)
        array=np.clip(array.astype(np.float32)+stripe[:,:,None]*rng.uniform(23,74),0,255).astype(np.uint8)
    if rng.random()<.45:
        array=cv2.GaussianBlur(array,(3,3),float(rng.uniform(.4,1.4)))
    array=np.clip(array.astype(np.float32)+rng.normal(0,rng.uniform(1.5,7),(HEIGHT,WIDTH,1)),0,255).astype(np.uint8)
    adjusted=[]
    for digit,x1,y1,x2,y2 in boxes:
        corners=np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]],dtype=np.float32)
        points=cv2.transform(corners[None],matrix)[0]
        lo=points.min(axis=0); hi=points.max(axis=0)
        adjusted.append((digit,*(lo+hi)/2,*(hi-lo)))
    return cv2.cvtColor(array,cv2.COLOR_RGB2BGR), adjusted

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destino',type=Path,default=Path('dados'))
    root=parser.parse_args().destino.resolve()
    root.mkdir(parents=True,exist_ok=True)
    styles=('lcd','mecanico','reflexo')
    used=set()
    with (root/'manifesto.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file,lineterminator='\n')
        writer.writerow(['divisao','arquivo','leitura','cenario'])
        for split,(count,seed) in SPLITS.items():
            rng=np.random.default_rng(seed)
            images=root/'images'/split
            labels=root/'labels'/split
            images.mkdir(parents=True,exist_ok=True)
            labels.mkdir(parents=True,exist_ok=True)
            for i in range(count):
                if i % 7 == 0:
                    reading=f'{int(rng.integers(0,10000)):05}'
                else:
                    reading=f'{int(rng.integers(0,100000)):05}'
                while reading in used:
                    reading=f'{int(rng.integers(0,100000)):05}'
                used.add(reading)
                style=styles[i%len(styles)]
                image,boxes=scene(rng,[int(c) for c in reading],style)
                name=f'{split}_{i:04}'
                if not cv2.imwrite(str(images/f'{name}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(76,98))]):
                    raise OSError(f'Falha ao criar {name}')
                lines=[f'{k} {x/WIDTH:.7f} {y/HEIGHT:.7f} {w/WIDTH:.7f} {h/HEIGHT:.7f}' for k,x,y,w,h in boxes]
                (labels/f'{name}.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
                writer.writerow((split,f'{name}.jpg',reading,style))
    config={'path':str(root),'train':'images/train','val':'images/val','test':'images/test','names':{i:str(i) for i in range(10)}}
    (root/'dados.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')

if __name__ == '__main__':
    main()
