#!/bin/bash

DATADIR=./datasets/grid_prepare_balanced/train/ 
VAL_DATADIR=./datasets/grid_prepare_balanced/test/ 
MODEL_NAME=${MODEL_NAME:-VballNetGridV1b}
RESUME=${RESUME:-}
EPOCHS=${EPOCHS:-60}

uv run src/train_grid.py \
  --data "$DATADIR" \
  --val_data "$VAL_DATADIR" \
  --model_name "$MODEL_NAME" \
  --seq 9 \
  --grayscale \
  --no-amp \
  ${RESUME:+--resume "$RESUME"} \
  --epochs "$EPOCHS" \
  --batch 8 \
  --optimizer AdamW \
  --lr 0.001 \
  --workers 8
