# Conta Clara | Applied Computer Vision

**Pedro Henrique Pedrosa Tavares · RM97877**  
Global Solution FIAP 2026.2 · CitizenTech · 4ES

O Conta Clara propõe auxiliar famílias a acompanhar o consumo de energia. Este protótipo de visão computacional localiza dígitos em mostradores de medidor, ordena os números e sinaliza quando a leitura precisa ser revisada.

| Entregável | Caminho |
|---|---|
| Relatório da disciplina | [PDF técnico](documentacao/Conta_Clara_Relatorio_Applied_CV.pdf) |
| Demonstração com inferência real | [Vídeo MP4](demonstracao/Conta_Clara_Demonstracao.mp4) |
| Checkpoint YOLO11n ajustado | [modelos/best.pt](modelos/best.pt) |
| Métricas e resultados | [Detecção](evidencias/deteccao.json), [sequência e latência](evidencias/leituras.json), [predições do teste](evidencias/predicoes_teste.csv) |
| Guia de entrega | [ENTREGA.md](ENTREGA.md) |

## Executar localmente

Use Python 3.12, a versão testada. A inferência funciona em CPU com os pesos incluídos.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Para instalação em CPU no Linux ou Windows:
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest -v
```

### Foto ou sequência em vídeo

```bash
python conta_clara.py --fonte demonstracao/amostra_zeros.jpg --saida saidas/zeros.jpg
python conta_clara.py --fonte demonstracao/amostra_oclusao.jpg --saida saidas/revisao.jpg
python conta_clara.py --fonte demonstracao/entrada_sintetica.mp4 --saida saidas/video_anotado.mp4
python conta_clara.py --fonte 0 --exibir
```

No modo de imagem, o comando imprime JSON com candidato, caixas `[x1, y1, x2, y2]`, confiança por dígito, menor confiança e motivo da revisão. O candidato é texto, preservando zeros iniciais. Em vídeo, a saída é anotada quadro a quadro e há registro amostrado em JSON no terminal. A webcam e a exibição exigem ambiente com interface gráfica. Use `q` ou Esc para encerrar a janela ao vivo.

Por padrão, o protótipo espera cinco dígitos alinhados e confiança mínima de 0,65. Quantidade, geometria ou confiança fora dos limites produzem `confirmacao_necessaria`. Uma `leitura_sugerida` é candidata à confirmação do cidadão. Em fotos mais abertas, enquadre o mostrador ou informe `--roi X Y LARGURA ALTURA` em pixels. Os dados são processados localmente. O código salva imagens somente quando há um caminho informado em `--saida`.

## Reproduzir o experimento

```bash
python gerar_dados.py
python treinar.py --base yolo11n.pt --device cpu
python avaliar.py
```

O Ultralytics obtém os pesos YOLO11n de partida se o arquivo local estiver ausente. A geração é determinística com as versões testadas e disponibiliza um manifesto com a leitura verdadeira e condições simuladas. A amostra sintética tem 650 imagens de treino, 120 de validação e 160 de teste, com leituras distintas entre as divisões. As variações cobrem mostradores LCD, mecânicos e com reflexo, além de rotação, ruído, contraste e desfoque. O treino usa 12 épocas, entrada de 320 px, lote 16, AdamW e semente 97877. A avaliação de caixas usa o checkpoint escolhido na validação. A avaliação da leitura exige que a sequência inteira coincida com a referência. A latência exclui abertura da foto e renderização. A medição foi feita em Linux x86_64, AMD EPYC 9V74, PyTorch 2.8.0+cpu, com OMP_NUM_THREADS e MKL_NUM_THREADS em 4.

O piloto de campo e a avaliação no dataset acadêmico UFPR-AMR são trabalhos futuros. A foto pública CC0 no ZIP é uma verificação qualitativa externa. Ela não entra no treino nem nas métricas. Confira [créditos e licenças](documentacao/FONTES_E_LICENCAS.md).

## Limites

O valor transcrito é um **registro acumulado em kWh**. Consumo no período exige duas leituras datadas e compatíveis do mesmo medidor. Histórico de famílias, estimativa tarifária, contestação, API de produção e integração com distribuidora são componentes previstos no projeto geral. O detector atual é um protótipo local de leitura do mostrador.
