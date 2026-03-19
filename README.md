# vball-net-pytorch
Pytorch Ball tracking. Volleyball tracking - VballNet is a specialized deep learning framework designed for volleyball tracking, built upon the foundation of TrackNetV4. This repository includes two primary models, VballNetV1 and VballNetFastV1

## Grid Version

### 1. Prepare data for the grid model

The grid version uses a separate data format:
- input frames `768x432`
- sequence of `5` RGB frames
- annotations in the form of `grid confidence + x_offset + y_offset`

Prepare train:

```bash
uv run src/video_to_heatmap.py \
  --source datasets/train \
  --output datasets/grid_prepare/train \
  --mode grid \
  --force
```

Prepare test/val:

```bash
uv run src/video_to_heatmap.py \
  --source datasets/test \
  --output datasets/grid_prepare/test \
  --mode grid \
  --force
```

After that, the data will be located in:
- `datasets/grid_prepare/train`
- `datasets/grid_prepare/test`

### 2. Start training the grid model

Basic run:

```bash
uv run src/train_grid.py \
  --data datasets/grid_prepare/train \
  --val_data datasets/grid_prepare/test \
  --epochs 30 \
  --batch 2 \
  --optimizer AdamW \
  --lr 0.001 \
  --workers 4
```

Short example using a shell script:

```bash
bash run_grid.sh
```

Training results are saved in `outputs/VballNetGridV1a_seq5_<timestamp>/`:
- `config.json`
- `checkpoints/latest.pth`
- `checkpoints/best.pth`

### 3. Resume training from a checkpoint

`train_grid.py` supports `--resume`.

Example of resuming from the best checkpoint:

```bash
uv run src/train_grid.py \
  --data datasets/grid_prepare/train \
  --val_data datasets/grid_prepare/test \
  --resume outputs/VballNetGridV1a_seq5_20260316_103433/checkpoints/best.pth \
  --epochs 60 \
  --batch 2 \
  --optimizer AdamW \
  --lr 0.001 \
  --workers 4
```

How it works:
- model weights are loaded from `--resume`
- optimizer state is loaded if available
- training continues from the next epoch
- new outputs are saved into a new folder in `outputs/`

### 4. Grid model inference

Example prediction run:

```bash
uv run src/predict_grid.py \
  --model_path outputs/VballNetGridV1a_seq5_20260316_103433/checkpoints/best.pth \
  --output_dir ./demo-grid/ \
  --video_path datasets/test/match5/video/pobead_4m-2w_1place_00004.mp4
```

### 5. Export the grid model to ONNX

```bash
uv run src/model/vballnet_grid_v1a.py \
  --model_path outputs/VballNetGridV1a_seq5_20260316_103433/checkpoints/best.pth \
  --export_onnx
```
