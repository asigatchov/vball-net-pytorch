#!/usr/bin/env python3

import argparse
import csv
import os
import time
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
        help="Number of sliding windows to infer per ONNX call; CPU often works best with 1",
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


def infer_model_params(model_path, session):
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_shape = input_meta.shape
    output_shape = output_meta.shape
    in_dim = input_shape[1]
    out_dim = output_shape[1]

    if not isinstance(in_dim, int) or not isinstance(out_dim, int):
        raise ValueError(
            f"Static channel dimensions are required, got input={input_shape}, output={output_shape}"
        )

    model_name = model_path.name.lower()
    grayscale = "grayscale" in model_name
    if not grayscale and in_dim % 3 != 0:
        grayscale = True

    seq = in_dim if grayscale else in_dim // 3
    expected_out_dim = seq * 3
    if out_dim != expected_out_dim:
        raise ValueError(
            f"Unexpected output channels: got {out_dim}, expected {expected_out_dim} for seq={seq}"
        )

    return {
        "seq": seq,
        "grayscale": grayscale,
        "input_height": input_shape[2],
        "input_width": input_shape[3],
    }


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


def preprocess_frame(frame, input_width, input_height, grayscale):
    resized = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
    if grayscale:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32) / 255.0
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def make_clip_batch(clips, grayscale):
    batch = []
    for clip_frames in clips:
        if grayscale:
            clip = np.stack(clip_frames, axis=0)
        else:
            clip = np.concatenate([np.transpose(frame, (2, 0, 1)) for frame in clip_frames], axis=0)
        batch.append(clip.astype(np.float32, copy=False))
    if not batch:
        raise ValueError("Expected at least one clip for batching")
    return np.asarray(batch, dtype=np.float32)


def flush_pending(
    pending_frames,
    session,
    input_name,
    seq,
    grayscale,
    threshold,
    frame_width,
    frame_height,
    input_width,
    input_height,
    csv_writer,
    writer,
    visualize,
    track,
):
    if not pending_frames:
        return 0.0, 0

    clips = make_clip_batch([item["clip"] for item in pending_frames], grayscale=grayscale)
    start_infer = time.perf_counter()
    output = session.run(None, {input_name: clips})[0]
    infer_elapsed = time.perf_counter() - start_infer

    predictions = decode_last_predictions(
        output,
        seq=seq,
        threshold=threshold,
        frame_width=frame_width,
        frame_height=frame_height,
        input_width=input_width,
        input_height=input_height,
    )

    for item, prediction in zip(pending_frames, predictions, strict=True):
        visibility, x_orig, y_orig, _ = prediction
        if visibility:
            track.append((x_orig, y_orig))
        else:
            if track:
                track.popleft()

        append_to_csv(csv_writer, item["frame_index"], visibility, x_orig, y_orig)

        if writer or visualize:
            vis_frame = draw_track(item["frame"].copy(), track)
            if visibility:
                cv2.circle(vis_frame, (x_orig, y_orig), 12, (0, 255, 0), 2)
            if writer:
                writer.write(vis_frame)
            if visualize:
                cv2.imshow("Grid Prediction ONNX", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

    pending_frames.clear()
    return infer_elapsed, len(predictions)


def decode_last_predictions(output, seq, threshold, frame_width, frame_height, input_width, input_height):
    output = output.reshape(output.shape[0], seq, 3, GRID_ROWS, GRID_COLS)
    output = output[:, -1]

    results = []
    for batch_index in range(output.shape[0]):
        conf_grid = output[batch_index, 0]
        x_offset_grid = output[batch_index, 1]
        y_offset_grid = output[batch_index, 2]

        max_conf_val = float(np.max(conf_grid))
        pred_row, pred_col = np.unravel_index(np.argmax(conf_grid), conf_grid.shape)
        if max_conf_val < threshold:
            results.append((0, -1, -1, max_conf_val))
            continue

        x_offset = float(x_offset_grid[pred_row, pred_col])
        y_offset = float(y_offset_grid[pred_row, pred_col])
        x_pred = (x_offset + pred_col) * (input_width / GRID_COLS)
        y_pred = (y_offset + pred_row) * (input_height / GRID_ROWS)
        x_pred = int(np.clip(x_pred, 0, input_width - 1))
        y_pred = int(np.clip(y_pred, 0, input_height - 1))
        x_orig = int(x_pred * frame_width / input_width)
        y_orig = int(y_pred * frame_height / input_height)
        results.append((1, x_orig, y_orig, max_conf_val))

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
    model_path = resolve_model_path(args.model_path)
    visualize = args.visualize and not is_headless()
    if args.visualize and not visualize:
        print("Visualization disabled: headless environment detected")

    session, input_name = load_session(model_path, args.device)
    model_params = infer_model_params(model_path, session)
    seq = model_params["seq"]
    grayscale = model_params["grayscale"]
    input_height = model_params["input_height"]
    input_width = model_params["input_width"]
    output_dir = Path(args.output_dir) if args.output_dir else (video_path.parent if args.only_csv else None)
    batch_size = max(1, args.batch_size)

    cap, frame_width, frame_height, fps, total_frames = initialize_video(video_path)
    basename = video_path.stem
    writer, _ = setup_output_writer(basename, output_dir, frame_width, frame_height, fps, args.only_csv)
    csv_file, csv_writer = setup_csv_file(basename, output_dir)

    processed_buffer = deque(maxlen=seq)
    pending_frames = []
    track = deque(maxlen=args.track_length)
    frame_index = 0
    total_start = time.perf_counter()
    inference_time = 0.0
    inference_windows = 0

    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = preprocess_frame(frame, input_width, input_height, grayscale)
            processed_buffer.append(processed)

            if len(processed_buffer) < seq:
                visibility = 0
                x_orig, y_orig = -1, -1
                if track:
                    track.popleft()
            else:
                pending_frames.append(
                    {
                        "frame_index": frame_index,
                        "frame": frame.copy(),
                        "clip": list(processed_buffer),
                    }
                )
                if len(pending_frames) >= batch_size:
                    infer_elapsed, num_predictions = flush_pending(
                        pending_frames=pending_frames,
                        session=session,
                        input_name=input_name,
                        seq=seq,
                        grayscale=grayscale,
                        threshold=args.threshold,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        input_width=input_width,
                        input_height=input_height,
                        csv_writer=csv_writer,
                        writer=writer,
                        visualize=visualize,
                        track=track,
                    )
                    inference_time += infer_elapsed
                    inference_windows += num_predictions
                visibility = None
                x_orig = None
                y_orig = None

            if visibility is not None:
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

        if pending_frames:
            infer_elapsed, num_predictions = flush_pending(
                pending_frames=pending_frames,
                session=session,
                input_name=input_name,
                seq=seq,
                grayscale=grayscale,
                threshold=args.threshold,
                frame_width=frame_width,
                frame_height=frame_height,
                input_width=input_width,
                input_height=input_height,
                csv_writer=csv_writer,
                writer=writer,
                visualize=visualize,
                track=track,
            )
            inference_time += infer_elapsed
            inference_windows += num_predictions
    finally:
        pbar.close()
        cap.release()
        if writer:
            writer.release()
        if csv_file:
            csv_file.close()
        if visualize:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - total_start
    pipeline_fps_frames = frame_index / total_time if total_time > 0 else 0.0
    pipeline_fps_windows = inference_windows / total_time if total_time > 0 else 0.0
    infer_fps_windows = inference_windows / inference_time if inference_time > 0 else 0.0
    infer_ms_per_window = (inference_time / inference_windows * 1000.0) if inference_windows else 0.0

    print(f"Model: {model_path}")
    print(f"Providers: {session.get_providers()}")
    print(
        f"Input: seq={seq}, grayscale={grayscale}, size={input_width}x{input_height}, batch_size={batch_size}"
    )
    print(f"Processed frames: {frame_index}")
    print(f"Inference windows: {inference_windows}")
    print(f"Total time, s: {total_time:.2f}")
    print(f"Pipeline FPS, frames/s: {pipeline_fps_frames:.2f}")
    print(f"Pipeline FPS, windows/s: {pipeline_fps_windows:.2f}")
    print(f"Inference time, s: {inference_time:.2f}")
    print(f"Inference FPS, windows/s: {infer_fps_windows:.2f}")
    print(f"Inference latency, ms/window: {infer_ms_per_window:.2f}")
    if output_dir:
        print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
