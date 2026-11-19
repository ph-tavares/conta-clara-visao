"""Avalia a leitura completa, a triagem por confiança e a latência em CPU."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
from conta_clara import DEFAULT_CLASSIFIER, DEFAULT_MODEL, MIN_SCORE, Reader, load_image

def measure(reader, data, rows):
    paths = [data/'images'/r['divisao']/r['arquivo'] for r in rows]
    images = [load_image(path) for path in paths]
    for frame in images[:10]:
        reader.predict(frame)
    records, times = [], []
    for row, image in zip(rows, images):
        start = perf_counter()
        found = reader.predict(image)
        times.append(1000*(perf_counter()-start))
        candidate = found['leitura_candidata_kwh']
        records.append({
            'arquivo': row['arquivo'], 'cenario': row.get('cenario', 'real'), 'referencia': row['leitura'],
            'candidata': candidate, 'confianca_minima': found['confianca_minima'],
            'status': found['status'], 'motivo': found['motivo'], 'correta': candidate==row['leitura'],
        })
    return records, times

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados',type=Path,default=Path('dados'))
    parser.add_argument('--pesos',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--classificador',type=Path,default=DEFAULT_CLASSIFIER)
    parser.add_argument('--saida',type=Path,default=Path('evidencias'))
    parser.add_argument('--divisao',default='test',choices=('val','test'),help='val para calibrar; test so na medicao final')
    parser.add_argument('--limiar',type=float,default=MIN_SCORE)
    args=parser.parse_args()
    with (args.dados/'manifesto.csv').open(encoding='utf-8') as file:
        rows=[r for r in csv.DictReader(file) if r['divisao']==args.divisao]
    if len(rows) < 10:
        parser.error('O manifesto deve incluir ao menos 10 amostras de teste.')
    records,times=measure(Reader(args.pesos,threshold=args.limiar,classificador=args.classificador),args.dados,rows)
    eligible=[r for r in records if r['status']=='leitura_sugerida']
    exact=sum(r['correta'] for r in records)
    exact_eligible=sum(r['correta'] for r in eligible)
    result={'imagens':len(records),'exatas':exact,'sugeridas':len(eligible),'sugeridas_corretas':exact_eligible,
            'revisao':len(records)-len(eligible),'limiar':args.limiar,'divisao':args.divisao,'latencia_media_ms':mean(times),
            'latencia_p95_ms':float(np.percentile(times,95)),'fps':1000/mean(times),
            'por_cenario':{case:{'amostras':sum(r['cenario']==case for r in records),
                                'exatas':sum(r['cenario']==case and r['correta'] for r in records)}
                          for case in sorted({r['cenario'] for r in records})}}
    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida/'leituras.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (args.saida/'predicoes_teste.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)

if __name__=='__main__':
    main()
