#!/bin/bash

DATADIR=./datasets/grid_prepare/train
VAL_DATADIR=./datasets/grid_prepare/test
MODEL_NAME=${MODEL_NAME:-VballNetGridV1c}

uv run src/train_grid.py \
  --data "$DATADIR" \
  --val_data "$VAL_DATADIR" \
  --model_name "$MODEL_NAME" \
  --seq 9 \
  --grayscale \
  --epochs 40 \
  --batch 4 \
  --optimizer AdamW \
  --lr 0.001 \
  --workers 8
