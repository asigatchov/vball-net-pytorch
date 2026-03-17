#!/usr/bin/env python3

import argparse
import csv
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


INPUT_WIDTH = 768
INPUT_HEIGHT = 432
GRID_COLS = 48
GRID_ROWS = 27
SEQ = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection with GridTrackNet ONNX Runtime"
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model_path", type=str, default=None, help="Path to .onnx model")
    parser.add_argument("--track_length", type=int, default=8, help="Track length")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--only_csv", action="store_true", help="Save only CSV")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="ONNX Runtime execution device",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of 5-frame clips to infer per ONNX call; CPU often works best with 1",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    return parser.parse_args()


def resolve_model_path(model_path):
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        return path

    candidates = sorted(
        Path("outputs").glob("VballNetGridV1a_seq5_*/checkpoints/best.onnx"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No default VballNetGridV1a ONNX model found in outputs/")
    return candidates[0]


def get_providers(device):
    available = ort.get_available_providers()
    if device == "gpu":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in this onnxruntime installation. "
                f"Available providers: {available}. "
                "Install a CUDA-enabled runtime such as `onnxruntime-gpu` and make sure the "
                "matching CUDA/cuDNN libraries are available, or rerun with `--device cpu` "
                "or `--device auto`."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_session(model_path, device):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = get_providers(device)
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=providers,
    )
    input_name = session.get_inputs()[0].name
    return session, input_name


def initialize_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_width, frame_height, fps, total_frames


def setup_output_writer(video_basename, output_dir, frame_width, frame_height, fps, only_csv):
    if output_dir is None or only_csv:
        return None, None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_basename}_predict.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    return writer, output_path


def setup_csv_file(video_basename, output_dir):
    if output_dir is None:
        return None, None
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{video_basename}_predict_ball.csv"
    csv_file = open(csv_path, "w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Visibility", "X", "Y"])
    return csv_file, csv_writer


def append_to_csv(csv_writer, frame_index, visibility, x, y):
    if csv_writer is not None:
        csv_writer.writerow([frame_index, visibility, x, y])


def is_headless():
    return not any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY"))


def preprocess_frames(frames):
    batches = []
    for start in range(0, len(frames), SEQ):
        batch = frames[start:start + SEQ]
        if len(batch) == SEQ:
            batches.append(batch)

    units = []
    for batch in batches:
        unit = []
        for frame in batch:
            frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.moveaxis(frame, -1, 0)
            unit.append(frame[0])
            unit.append(frame[1])
            unit.append(frame[2])
        units.append(unit)

    if not units:
        return np.empty((0, SEQ * 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32), 0

    batch = np.asarray(units, dtype=np.float32)
    batch /= 255.0
    return batch, len(units) * SEQ


def decode_clip_predictions(output, threshold, frame_width, frame_height):
    output = np.split(output, SEQ, axis=1)
    output = np.stack(output, axis=2)
    output = np.moveaxis(output, 1, -1)

    conf_grid, x_offset_grid, y_offset_grid = np.split(output, 3, axis=-1)
    conf_grid = np.squeeze(conf_grid, axis=-1)
    x_offset_grid = np.squeeze(x_offset_grid, axis=-1)
    y_offset_grid = np.squeeze(y_offset_grid, axis=-1)

    results = []
    for batch_index in range(conf_grid.shape[0]):
        for frame_index in range(conf_grid.shape[1]):
            curr_conf_grid = conf_grid[batch_index][frame_index]
            curr_x_offset_grid = x_offset_grid[batch_index][frame_index]
            curr_y_offset_grid = y_offset_grid[batch_index][frame_index]

            max_conf_val = float(np.max(curr_conf_grid))
            pred_row, pred_col = np.unravel_index(np.argmax(curr_conf_grid), curr_conf_grid.shape)
            if max_conf_val < threshold:
                results.append((0, -1, -1))
                continue

            x_offset = curr_x_offset_grid[pred_row][pred_col]
            y_offset = curr_y_offset_grid[pred_row][pred_col]
            x_pred = int((x_offset + pred_col) * (INPUT_WIDTH / GRID_COLS))
            y_pred = int((y_offset + pred_row) * (INPUT_HEIGHT / GRID_ROWS))
            x_orig = int((x_pred / INPUT_WIDTH) * frame_width)
            y_orig = int((y_pred / INPUT_HEIGHT) * frame_height)
            results.append((1, x_orig, y_orig))

    return results


def draw_track(frame, track_points):
    points = list(track_points)
    for point in points[:-1]:
        if point is not None:
            cv2.circle(frame, point, 5, (255, 0, 0), -1)
    if points and points[-1] is not None:
        cv2.circle(frame, points[-1], 8, (0, 0, 255), -1)
    return frame


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    model_path = resolve_model_path(args.model_path)
    visualize = args.visualize and not is_headless()
    if args.visualize and not visualize:
        print("Visualization disabled: headless environment detected")
    batch_size = max(1, args.batch_size) * SEQ

    session, input_name = load_session(model_path, args.device)
    cap, frame_width, frame_height, fps, total_frames = initialize_video(video_path)
    basename = video_path.stem
    writer, _ = setup_output_writer(basename, output_dir, frame_width, frame_height, fps, args.only_csv)
    csv_file, csv_writer = setup_csv_file(basename, output_dir)

    track = deque(maxlen=args.track_length)
    frame_index = 0

    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    try:
        while cap.isOpened():
            chunk_frames = []
            while len(chunk_frames) < batch_size:
                ret, frame = cap.read()
                if not ret:
                    break
                chunk_frames.append(frame)
            if not chunk_frames:
                break

            processed_chunk, usable_frames = preprocess_frames(chunk_frames)
            predictions = []
            if usable_frames:
                output = session.run(None, {input_name: processed_chunk})[0]
                predictions = decode_clip_predictions(
                    output,
                    args.threshold,
                    frame_width,
                    frame_height,
                )
                if len(predictions) != usable_frames:
                    raise RuntimeError(
                        f"Prediction/frame mismatch: got {len(predictions)} predictions for "
                        f"{usable_frames} frames"
                    )

            for offset, frame in enumerate(chunk_frames):
                if offset < usable_frames:
                    visibility, x_orig, y_orig = predictions[offset]
                    if visibility:
                        track.append((x_orig, y_orig))
                    else:
                        if track:
                            track.popleft()
                else:
                    visibility = 0
                    x_orig, y_orig = -1, -1
                    if track:
                        track.popleft()

                append_to_csv(csv_writer, frame_index, visibility, x_orig, y_orig)

                if writer or visualize:
                    vis_frame = draw_track(frame.copy(), track)
                    if visibility:
                        cv2.circle(vis_frame, (x_orig, y_orig), 12, (0, 255, 0), 2)
                    if writer:
                        writer.write(vis_frame)
                    if visualize:
                        cv2.imshow("Grid Prediction ONNX", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return

                frame_index += 1
                pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        if visualize:
            cv2.destroyAllWindows()

    print(f"Model: {model_path}")
    print(f"Providers: {session.get_providers()}")
    if output_dir:
        print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
