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
  --model_name VballNetFastV1 \
  --seq 15 \
  --grayscale \
  --optimizer AdamW \
  --lr 0.001 \
  --epochs 200 \
  --batch 20 \
  --scheduler ReduceLROnPlateau \
  --workers 16 \
  --resume  outputs/VballNetFastV1_seq15_grayscale_20251219_151249/checkpoints/VballNetFastV1_seq15_grayscale_best.pth

echo "Training completed!"
