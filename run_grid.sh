#!/bin/bash

DATADIR=./datasets/grid_prepare/train
VAL_DATADIR=./datasets/grid_prepare/test

uv run src/train_grid.py \
  --data "$DATADIR" \
  --val_data "$VAL_DATADIR" \
  --epochs 40 \
  --batch 4 \
  --optimizer AdamW \
  --lr 0.001 \
  --workers 8
