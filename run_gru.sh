#!/bin/bash

# Training script for VballNetV1c (GRU model) with 15 grayscale frames

# Set data paths
#DATADIR=datasets/mix-vb/train_preprocessed
#VAL_DATADIR=datasets/mix-vb/test_preprocessed
#
#DATADIR=../TrackNetV4-PyTorch/datasets/mix_volleyball_preprocessed
#VAL_DATADIR=../TrackNetV4-PyTorch/datasets/mix_volleyball_test_preprocessed
#
DATADIR=./datasets/dji/train
VAL_DATADIR=./datasets/dji/val
# Run training
uv run  src/train_gru.py \
  --data "$DATADIR" \
  --val_data "$VAL_DATADIR" \
  --model_name VballNetV1c \
  --seq 9 \
  --grayscale \
  --optimizer AdamW \
  --lr 0.001 \
  --epochs 1 \
  --batch 16  \
  --scheduler ReduceLROnPlateau \
  --workers 8 \
#  --resume ./outputs/VballNetV1c_seq15_grayscale_badminton/checkpoints/VballNetV1c_seq15_grayscale_best.pth

echo "Training completed!"
