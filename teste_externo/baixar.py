"""Baixa do Wikimedia Commons as fotos do teste externo (miniatura de 1280 px, a base das coordenadas do recorte).

As fotos não são redistribuídas neste repositório: o manifesto guarda título, autor, licença e página de cada uma.
"""
import csv
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

PASTA = Path(__file__).resolve().parent
AGENTE = {'User-Agent': 'conta-clara-visao/1.0 (projeto academico)'}

def baixar(url):
    for tentativa in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=60).read()
        except urllib.error.HTTPError as erro:
            if erro.code != 429:
                raise
            time.sleep(10 * (tentativa + 1))
    raise RuntimeError(f'Commons recusou {url}')

def main():
    destino = PASTA / 'fotos'
    destino.mkdir(exist_ok=True)
    titulos = sorted({linha['arquivo_commons'] for linha in csv.DictReader((PASTA / 'manifesto.csv').open(encoding='utf-8'))})
    for titulo in titulos:
        arquivo = destino / titulo
        if arquivo.exists():
            continue
        consulta = urllib.parse.urlencode({'action': 'query', 'titles': f'File:{titulo}', 'prop': 'imageinfo',
                                           'iiprop': 'url', 'iiurlwidth': 1280, 'format': 'json'})
        paginas = json.loads(baixar(f'https://commons.wikimedia.org/w/api.php?{consulta}'))['query']['pages']
        info = next(iter(paginas.values()))['imageinfo'][0]
        arquivo.write_bytes(baixar(info.get('thumburl', info['url'])))
        time.sleep(1.5)
    print(f'{len(titulos)} fotos em {destino}')

if __name__ == '__main__':
    main()
