# Conta Clara | Applied Computer Vision

**Pedro Henrique Pedrosa Tavares · RM97877**
Global Solution FIAP 2026.2 · CitizenTech · Engenharia de Software 4ES

O Conta Clara ajuda famílias de baixa renda a acompanhar a conta de luz. Este repositório é o componente de visão computacional: a família fotografa o mostrador do medidor e o sistema lê o número de kWh, que ela confere antes de salvar.

## Como funciona

A leitura acontece em duas etapas, a arquitetura usada na literatura de leitura automática de medidores (Laroca et al., 2019):

1. **Detector (YOLO11n, treino customizado):** encontra cada dígito do mostrador. Treinado com 1.568 fotos reais de medidores e 300 imagens sintéticas.
2. **Classificador (YOLO11n-cls, treino customizado):** lê cada dígito recortado. Antes de ler, o recorte é endireitado pelo ângulo da linha de dígitos.

Depois, o leitor ordena os dígitos no sentido do texto, preserva os zeros à esquerda e decide:

- **`leitura_sugerida`**: de 5 a 8 dígitos alinhados e confiança mínima de 0,65;
- **`confirmacao_necessaria`**: qualquer outro caso, com o motivo (`quantidade_de_digitos`, `disposicao_inconsistente`, `baixa_confianca` ou `foto_girada`, quando a foto está girada mais de 45°, porque aí um 6 pode ser um 9 de cabeça para baixo).

Em todos os casos a família confirma o número antes de ele entrar no histórico. O sistema nunca grava sozinho.

## Resultados

Medidos com o modelo final, em dados fora do treino. A leitura completa e o teste externo foram rodados uma única vez; o mAP de detecção no teste também foi consultado nas etapas intermediárias (ver [Reproduzir](#reproduzir)).

| Teste | O que é | Resultado |
|---|---|---|
| Detecção de dígitos | 196 fotos reais reservadas (31 medidores fora do treino) | mAP50 **0,933**, precisão 0,898, recall 0,904 |
| Leitura completa, fotos reais | mesmas 196 fotos | **74 leituras exatas (38%)**; 74% dos dígitos certos quando a quantidade bate |
| Teste externo | 17 leituras em 14 fotos do Wikimedia Commons, outra fonte, conferidas manualmente | **10 de 17 exatas (59%)** |
| Bancada sintética | 160 imagens do gerador original | **160 de 160** |
| Velocidade | CPU de notebook (Apple M5), mesmas 196 fotos | 21,9 ms por foto em média (~46 quadros/s) |

Arquivos: [`evidencias/deteccao_teste/`](evidencias/deteccao_teste), [`evidencias/classificador_teste/`](evidencias/classificador_teste) (acerto por dígito 86,2%), [`evidencias/leitura_reais/`](evidencias/leitura_reais), [`evidencias/leitura_sinteticas/`](evidencias/leitura_sinteticas), [`evidencias/teste_externo.json`](evidencias/teste_externo.json) e as imagens anotadas em [`evidencias/teste_externo/`](evidencias/teste_externo).

**Sem vazamento entre treino e teste.** A base pública traz várias cópias da mesma foto (giradas e recortadas). Na divisão original, 155 das 188 fotos de teste tinham uma cópia no treino. `preparar_reais.py` agrupa as cópias por cena (mesmos dígitos ou imagem quase igual) e coloca cada cena inteira numa só divisão: 0 cenas cruzadas, 0 quase-cópias entre teste e treino.

## Executar

Python 3.12, CPU.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest -v                # 18 testes
```

```bash
# foto real (mostrador enquadrado): imprime JSON com a leitura, as caixas, a confiança e o status
python conta_clara.py --fonte demonstracao/medidor_aclara_74590.jpg --saida saidas/aclara.jpg      # 74590, leitura_sugerida
python conta_clara.py --fonte demonstracao/medidor_landis_903123.jpg --saida saidas/landis.jpg     # 903123, confirmação pedida
python conta_clara.py --fonte demonstracao/medidor_publico_cc0.jpg --saida saidas/reflexo.jpg      # reflexo apaga 2 dígitos: confirmação pedida
# só o mostrador de uma foto aberta (x y largura altura) e quantidade de dígitos conhecida
python conta_clara.py --fonte foto.jpg --roi 320 360 510 90 --digitos 5
# vídeo quadro a quadro e webcam ao vivo (q ou Esc para sair)
python conta_clara.py --fonte demonstracao/entrada_sintetica.mp4 --saida saidas/video.mp4
python conta_clara.py --fonte 0 --exibir
```

Os pesos estão em `modelos/best.pt` (detector) e `modelos/digitos.pt` (classificador). A foto é processada localmente e só é salva quando há `--saida`. Enquadre só a janela do contador (na foto inteira, use `--roi`): com folga, os números pequenos da escala impressa embaixo dos roletes entram como dígitos a mais.

## Reproduzir

1. Baixe a base [meter digits v2](https://universe.roboflow.com/meter-i5vud/meter-digits-frvqn/dataset/2) (Roboflow Universe, CC BY 4.0) no formato YOLOv11 e descompacte numa pasta, por exemplo `fotos_reais/`.
2. Prepare os dados e treine. Os pesos publicados vieram desta sequência, cada etapa partindo da anterior (CPU, 416 px):

```bash
python gerar_dados.py                                                    # 930 medidores sintéticos
# etapa 1: 30 épocas, até 3 cópias por cena, aumento de dados da versão sintética (commit 5e13817)
python preparar_reais.py --origem fotos_reais --sinteticos dados --max-sinteticos 300 --max-por-grupo 3
python treinar.py --dados dados_reais/dados.yaml --base yolo11n.pt --epochs 30
# etapa 2: 25 épocas com todas as fotos reais e aumento de dados para foto real
python preparar_reais.py --origem fotos_reais --sinteticos dados --max-sinteticos 300
python treinar.py --dados dados_reais/dados.yaml --base modelos/best.pt --epochs 25
# etapa 3: 20 épocas como detector de uma classe ("dígito")
python treinar.py --dados dados_reais/dados.yaml --base modelos/best.pt --epochs 20 --classe-unica
# classificador
python preparar_digitos.py --dados dados_reais
python treinar_classificador.py --dados dados_digitos --epochs 20
```

A divisão treino/validação/teste é a mesma nas três etapas: ela depende só das cenas, não do limite de cópias.

3. Avalie:

```bash
python treinar.py --somente-avaliar --base modelos/best.pt --dados dados_reais/dados.yaml --classe-unica --divisao test --saida evidencias/deteccao_teste
python treinar_classificador.py --somente-avaliar --base modelos/digitos.pt --divisao test --saida evidencias/classificador_teste
python avaliar.py --dados dados_reais --saida evidencias/leitura_reais          # leitura completa, fotos reais reservadas
python avaliar.py --dados dados --saida evidencias/leitura_sinteticas           # bancada sintética
python teste_externo/baixar.py && python teste_externo/avaliar.py
```

O limiar de sugestão (0,65) foi conferido na validação (`avaliar.py --divisao val --limiar ...`). Os scripts de treino medem só na validação. Durante o desenvolvimento, a versão anterior desses scripts também imprimia o mAP de teste do detector ao fim das etapas 1 e 2; esses números (0,64 e 0,77) mostraram que o detector de dez classes não bastava e motivaram a arquitetura em duas etapas. O teste da leitura completa e o teste externo foram rodados uma vez, com o modelo final.

## Estrutura

| Arquivo | Função |
|---|---|
| `conta_clara.py` | leitor: detecção, classificação, ordenação e decisão (foto, vídeo e webcam) |
| `preparar_reais.py` | fotos reais com divisão por cena, sem vazamento |
| `preparar_digitos.py` | recortes de dígitos (reais endireitados + sintéticos com 94 fontes e 7 segmentos) |
| `gerar_dados.py` | medidores sintéticos (LCD, mecânico e com reflexo) |
| `treinar.py`, `treinar_classificador.py` | treino do detector e do classificador |
| `avaliar.py`, `teste_externo/` | medições |
| `test_*.py` | testes automáticos |

## Créditos e licenças

Detalhes em [`documentacao/FONTES_E_LICENCAS.md`](documentacao/FONTES_E_LICENCAS.md). Fotos reais de treino: *meter digits*, Roboflow Universe (meter-i5vud), CC BY 4.0. Fotos do teste externo: Wikimedia Commons, com autor e licença de cada uma em [`teste_externo/manifesto.csv`](teste_externo/manifesto.csv); as fotos originais não são redistribuídas (`teste_externo/baixar.py`), e os recortes anotados em `evidencias/teste_externo/` seguem a licença de cada foto. Detector e classificador: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), AGPL-3.0, a mesma licença deste repositório (`LICENSE`).
