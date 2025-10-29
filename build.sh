#!/bin/bash

# Ativa venv
source .venv/bin/activate

# Instala PyInstaller
pip install pyinstaller

# Build direto (onedir - mais rápido para testar)
pyinstaller --clean \
    --name="VideoInterpolation" \
    --windowed \
    --add-data="assets:assets" \
    --add-data="adicionar_it.py:." \
    --add-data="model:model" \
    --add-data="dataset:dataset" \
    --add-data="trainer:trainer" \
    --hidden-import=adicionar_it \
    --hidden-import=model.layers \
    --hidden-import=dataset.frame_dataset \
    --hidden-import=trainer.trainer \
    --collect-all torch \
    --collect-all torchvision \
    --exclude-module matplotlib \
    --exclude-module pandas \
    interface.py

echo "Build concluído! Executável em: dist/VideoInterpolation/"