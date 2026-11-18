# Fontes e licenças

- **Bancada sintética.** Imagens, anotações e leituras geradas por `gerar_dados.py`, com sementes registradas no script. Sem fotos de famílias, faturas, endereços ou números de série.
- **Foto externa.** [Electricity meter.JPG](https://commons.wikimedia.org/wiki/File:Electricity_meter.JPG), Editor abcdef, Wikimedia Commons, [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). `demonstracao/medidor_publico_cc0.jpg` é um recorte do mostrador, com bordas adicionadas e metadados removidos. Os arquivos anotados acrescentam caixas e a saída do detector.
- **Referência acadêmica.** Laroca et al. *Convolutional Neural Networks for Automatic Meter Reading*. Journal of Electronic Imaging, 28(1), 013023, 2019. [DOI](https://doi.org/10.1117/1.JEI.28.1.013023) · [UFPR-AMR e procedimento de acesso](https://github.com/raysonlaroca/ufpr-amr-dataset). Imagens e rótulos da UFPR-AMR ficam fora do treinamento e do pacote.
- **Detector.** [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) e pesos COCO como inicialização. Licença AGPL-3.0 conforme `LICENSE`.
- **Ferramentas.** PyTorch, OpenCV, Pillow e NumPy. Dependências registradas em `requirements.txt`.
