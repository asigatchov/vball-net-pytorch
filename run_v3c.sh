#!/bin/bash

# Training script for VballNetV1c (GRU model) with 15 grayscale frames

# Set data paths
DATADIR=datasets/mix-vb/train_preprocessed
VAL_DATADIR=datasets/mix-vb/test_preprocessed

#DATADIR=../TrackNetV4-PyTorch/datasets/mix_volleyball_preprocessed
#VAL_DATADIR=../TrackNetV4-PyTorch/datasets/mix_volleyball_test_preprocessed
#
DATADIR=./datasets/mix_video_vb/train_preprocessed/
VAL_DATADIR=./datasets/mix_video_vb/test_preprocessed/

# Run training
uv run  src/train_gru.py \
  --data "$DATADIR" \
  --val_data "$VAL_DATADIR" \
  --model_name VballNetV2 \
  --seq 9 \
  --grayscale \
  --optimizer AdamW \
  --lr 0.001 \
  --epochs 1 \
  --batch 10  \
  --scheduler ReduceLROnPlateau \
  --workers 11 \
  --alpha 0.5 

echo "Training completed!"
