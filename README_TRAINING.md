# VballNet Training with TensorBoard Monitoring

## Overview

This document explains how to train VballNet models with enhanced metric tracking using TensorBoard.

## New Metrics Tracked

The enhanced training script now tracks the following metrics:

1. **Validation Loss** - Standard loss metric
2. **Validation F1 Score** - F1 score calculated at threshold=0.5 for the central frame
3. **Validation Accuracy** - Accuracy measured as distance ≤ 10 pixels for the central frame
4. **Learning Rate** - Current learning rate during training

## Running Training

To run training with the enhanced metrics:

```bash
# Example for VballNetV1c with 15 grayscale frames
uv run src/train_v2.py \
  --data datasets/your_train_data \
  --val_data datasets/your_val_data \
  --model_name VballNetV1c \
  --seq 15 \
  --grayscale \
  --optimizer AdamW \
  --lr 0.001 \
  --epochs 100 \
  --batch 4
```

## Viewing TensorBoard Logs

During or after training, you can monitor the metrics using TensorBoard:

```bash
# Navigate to your project directory
cd /path/to/vball-net-pytorch

# Launch TensorBoard (logs are saved in the outputs directory)
tensorboard --logdir outputs/
```

Then open your browser to `http://localhost:6006` to view the metrics in real-time.

## Metrics Calculation Details

### Central Frame Focus

All metrics are calculated specifically for the central frame in the sequence to provide meaningful evaluation:
- For 15 input frames, the 8th frame (index 7) is considered the central frame
- This approach ensures consistent evaluation regardless of sequence length

### F1 Score Calculation

F1 score is calculated at a fixed threshold of 0.5:
1. Predictions and targets are converted to binary values using threshold=0.5
2. True positives, false positives, and false negatives are computed
3. Precision and recall are calculated
4. F1 score is derived using the standard formula: 2 * (precision * recall) / (precision + recall)

### Accuracy Calculation

Accuracy is measured as the percentage of predictions where the distance between predicted and ground truth positions is ≤ 10 pixels:
1. Position of maximum activation is found in both prediction and target heatmaps
2. Euclidean distance between these positions is calculated
3. Accuracy is the percentage of samples with distance ≤ 10 pixels