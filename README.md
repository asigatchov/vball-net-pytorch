# vball-net-pytorch
Pytorch Ball tracking. Volleyball tracking - VballNet is a specialized deep learning framework designed for volleyball tracking, built upon the foundation of TrackNetV4. This repository includes two primary models, VballNetV1 and VballNetFastV1

## Grid Version

### 1. Подготовка данных для grid-модели

Grid-версия использует отдельный формат данных:
- входные кадры `768x432`
- последовательность `5` RGB-кадров
- разметка в виде `grid confidence + x_offset + y_offset`

Подготовка train:

```bash
uv run src/video_to_heatmap.py \
  --source datasets/train \
  --output datasets/grid_prepare/train \
  --mode grid \
  --force
```

Подготовка test/val:

```bash
uv run src/video_to_heatmap.py \
  --source datasets/test \
  --output datasets/grid_prepare/test \
  --mode grid \
  --force
```

После этого данные будут лежать в:
- `datasets/grid_prepare/train`
- `datasets/grid_prepare/test`

### 2. Запуск обучения grid-модели

Базовый запуск:

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

Короткий пример через shell-скрипт:

```bash
bash run_grid.sh
```

Результаты обучения сохраняются в `outputs/VballNetGridV1a_seq5_<timestamp>/`:
- `config.json`
- `checkpoints/latest.pth`
- `checkpoints/best.pth`

### 3. Продолжить обучение с чекпоинта

`train_grid.py` поддерживает `--resume`.

Пример продолжения с лучшего чекпоинта:

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

Как это работает:
- из `--resume` загружаются веса модели
- если есть, загружается состояние optimizer
- обучение продолжается со следующей эпохи
- новое сохранение идёт в новую папку в `outputs/`

### 4. Инференс grid-модели

Пример запуска предсказания:

```bash
uv run src/predict_grid.py \
  --model_path outputs/VballNetGridV1a_seq5_20260316_103433/checkpoints/best.pth \
  --output_dir ./demo-grid/ \
  --video_path datasets/test/match5/video/pobead_4m-2w_1place_00004.mp4
```

### 5. Экспорт grid-модели в ONNX

```bash
uv run src/model/vballnet_grid_v1a.py \
  --model_path outputs/VballNetGridV1a_seq5_20260316_103433/checkpoints/best.pth \
  --export_onnx
```
