#!/bin/bash

# Training script for VballNetV1c (GRU model) with 15 grayscale frames

# Set data paths
DATADIR=datasets/mix-vb/train_preprocessed
VAL_DATADIR=datasets/mix-vb/test_preprocessed

# Run training
uv run  src/train_gru.py \
  --data "$DATADIR" \
  --val_data "$VAL_DATADIR" \
  --model_name VballNetV1c \
  --seq 15 \
  --grayscale \
  --optimizer AdamW \
  --lr 0.001 \
  --epochs 200 \
  --batch 4  \
  --scheduler ReduceLROnPlateau \
  --workers 8 \
  --resume ./outputs/exp_VballNetV1c_seq15_grayscale_20251216_143819/checkpoints/VballNetV1c_seq15_grayscale_best.pth

echo "Training completed!"
