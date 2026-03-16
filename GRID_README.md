# Grid Model

## Что это за модель

`VballNetGridV1a` в [src/model/vballnet_grid_v1a.py](/home/ubuntu/projetcs/vball-net-pytorch/src/model/vballnet_grid_v1a.py) это PyTorch-порт `GridTrackNet`.

Модель работает не с heatmap-выходом полного разрешения, а с сеткой детекции:
- вход: последовательность из `5` RGB-кадров
- выход: для каждого кадра набор grid-карт `confidence + x_offset + y_offset`

То есть модель решает задачу в стиле:
- в какой ячейке сетки находится мяч
- где именно внутри ячейки он расположен

## Что получает на вход

Ожидаемый входной тензор:

```text
[B, 15, 432, 768]
```

Где:
- `B` — batch size
- `15` = `5 кадров * 3 RGB-канала`
- `432 x 768` — размер каждого кадра после подготовки данных

Порядок каналов:
- сначала RGB первого кадра
- потом RGB второго
- ...
- потом RGB пятого

То есть один сэмпл это:

```text
[R1,G1,B1,R2,G2,B2,R3,G3,B3,R4,G4,B4,R5,G5,B5]
```

## Что выдаёт на выход

Выходной тензор:

```text
[B, 15, 27, 48]
```

Где:
- `15` = `5 кадров * 3 карты на кадр`
- `27 x 48` — размер grid-сетки

Для каждого из 5 кадров модель предсказывает 3 карты:
- `conf` — вероятность наличия мяча в ячейке
- `x_offset` — смещение по X внутри ячейки
- `y_offset` — смещение по Y внутри ячейки

После reshaping:

```text
[B, 5, 3, 27, 48]
```

Логика декодирования:
- выбирается ячейка с максимальным `conf`
- если `conf < threshold`, считаем мяч невидимым
- если `conf >= threshold`, координаты считаются так:

```text
x = (col + x_offset) * (768 / 48)
y = (row + y_offset) * (432 / 27)
```

Затем координаты масштабируются обратно в размер исходного видео.

## Что готовит пайплайн данных

Подготовка для grid-модели делается через:

```bash
uv run src/video_to_heatmap.py --mode grid ...
```

Результат сохраняется в:
- `datasets/grid_prepare/train`
- `datasets/grid_prepare/test`

Формат:
- `inputs/<sequence>/<frame>.png` — RGB-кадры `768x432`
- `annotations/<sequence>.csv` — координаты мяча уже в той же системе координат `768x432`

Во время обучения dataset в [src/utils/grid_dataset.py](/home/ubuntu/projetcs/vball-net-pytorch/src/utils/grid_dataset.py):
- собирает окна длиной `5`
- преобразует координаты в grid-таргеты `conf/x_offset/y_offset`
- возвращает:
  - `x`: `[15, 432, 768]`
  - `y`: `[15, 27, 48]`

## Принцип работы на инференсе

Скрипт [src/predict_grid.py](/home/ubuntu/projetcs/vball-net-pytorch/src/predict_grid.py):
- читает видео
- держит буфер из 5 последних кадров
- каждый кадр ресайзит в `768x432`
- собирает вход `[1, 15, 432, 768]`
- получает выход `[1, 15, 27, 48]`
- берёт предсказание для последнего кадра окна
- декодирует `conf/x_offset/y_offset` в `(X, Y)`
- пишет результат в CSV и при необходимости рисует трек на видео

Это значит, что предсказание для кадра делается с использованием короткого временного контекста из 5 кадров.

## ONNX экспорт

Экспорт добавлен прямо в модель:

```bash
uv run src/model/vballnet_grid_v1a.py \
  --model_path outputs/VballNetGridV1a_seq5_20260316_103433/checkpoints/best.pth \
  --export_onnx
```

Пример результата:
- `outputs/VballNetGridV1a_seq5_20260316_103433/checkpoints/best.onnx`

## Ожидаемый FPS

Ниже цифры для batch `1`, то есть для одного окна из 5 кадров.

Локально измерено в этой среде:
- GPU: `NVIDIA GeForce RTX 3060`
- ONNX Runtime providers: только `CPUExecutionProvider`

Замеры:
- `ONNX CPU`: около `219 ms` на окно, то есть примерно `4.6 FPS`
- `PyTorch CUDA`: около `40.9 ms` на окно, то есть примерно `24.5 FPS`

Ожидаемый FPS для ONNX:
- `ONNX CPU`: ожидайте примерно `4-5 FPS`
- `ONNX GPU`: ожидайте примерно `20-30 FPS` на RTX 3060 уровня, если использовать `onnxruntime-gpu` с `CUDAExecutionProvider`

Важно:
- значение `ONNX GPU` здесь не измерено напрямую, а оценено по локальному `PyTorch CUDA` замеру
- реальный FPS зависит от:
  - версии `onnxruntime-gpu`
  - CUDA/cuDNN
  - batch size
  - того, считается ли FPS по окнам или по финальным кадрам
  - накладных расходов на декодирование видео и запись результата

Если считать полный pipeline `read + preprocess + infer + postprocess + write`, реальный FPS обычно будет ниже, чем голый FPS модели.
