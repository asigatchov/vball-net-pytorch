#!/bin/bash

# Training script for VballNetV1c (GRU model) with 15 grayscale frames

# Set data paths
DATADIR=datasets/mix-vb/train_preprocessed
VAL_DATADIR=datasets/mix-vb/test_preprocessed

DATADIR=../TrackNetV4-PyTorch/datasets/mix_volleyball_preprocessed
VAL_DATADIR=../TrackNetV4-PyTorch/datasets/mix_volleyball_test_preprocessed

# Run training
uv run  src/train_gru.py \
  --data "$DATADIR" \
  --val_data "$VAL_DATADIR" \
  --model_name VballNetV1b \
  --seq 15 \
  --grayscale \
  --optimizer AdamW \
  --lr 0.001 \
  --epochs 200 \
  --batch 8  \
  --scheduler ReduceLROnPlateau \
  --workers 12 \
  --resume  outputs/VballNetV1b_seq15_grayscale_20251218_181221/checkpoints/VballNetV1b_seq15_grayscale_best.pth

echo "Training completed!"
