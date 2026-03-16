#!/usr/bin/env python3

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.vballnet_grid_v1a import VballNetGridV1a


INPUT_WIDTH = 768
INPUT_HEIGHT = 432
GRID_COLS = 48
GRID_ROWS = 27
SEQ = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection with GridTrackNet PyTorch"
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model_path", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--track_length", type=int, default=8, help="Track length")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--only_csv", action="store_true", help="Save only CSV")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, mps, auto")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    return parser.parse_args()


def resolve_model_path(model_path):
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = sorted(
        Path("outputs").glob("VballNetGridV1a_seq5_*/checkpoints/best.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No default VballNetGridV1a checkpoint found in outputs/")
    return candidates[0]


def get_device(name):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def load_model(model_path, device):
    model = VballNetGridV1a(
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        in_dim=SEQ * 3,
        out_dim=SEQ * 3,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


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
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{video_basename}_predict_ball.csv"
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y"]).to_csv(csv_path, index=False)
    return csv_path


def append_to_csv(result, csv_path):
    if csv_path is not None:
        pd.DataFrame([result]).to_csv(csv_path, mode="a", header=False, index=False)


def preprocess_frame(frame):
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def make_input_tensor(frame_buffer, device):
    stacked = np.concatenate([np.transpose(frame, (2, 0, 1)) for frame in frame_buffer], axis=0)
    tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)
    return tensor


def decode_predictions(output, threshold):
    output = output.reshape(SEQ, 3, GRID_ROWS, GRID_COLS)
    results = []
    for frame_idx in range(SEQ):
        conf = output[frame_idx, 0]
        x_offset = output[frame_idx, 1]
        y_offset = output[frame_idx, 2]
        max_index = int(np.argmax(conf))
        row = max_index // GRID_COLS
        col = max_index % GRID_COLS
        conf_score = float(conf[row, col])
        if conf_score < threshold:
            results.append((0, -1, -1, conf_score))
            continue
        x = (col + float(x_offset[row, col])) * (INPUT_WIDTH / GRID_COLS)
        y = (row + float(y_offset[row, col])) * (INPUT_HEIGHT / GRID_ROWS)
        x = int(np.clip(x, 0, INPUT_WIDTH - 1))
        y = int(np.clip(y, 0, INPUT_HEIGHT - 1))
        results.append((1, x, y, conf_score))
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
    device = get_device(args.device)

    model = load_model(model_path, device)
    cap, frame_width, frame_height, fps, total_frames = initialize_video(video_path)
    basename = video_path.stem
    writer, _ = setup_output_writer(basename, output_dir, frame_width, frame_height, fps, args.only_csv)
    csv_path = setup_csv_file(basename, output_dir)

    processed_buffer = deque(maxlen=SEQ)
    raw_buffer = deque(maxlen=SEQ)
    track = deque(maxlen=args.track_length)
    frame_index = 0

    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        raw_buffer.append(frame.copy())
        processed_buffer.append(preprocess_frame(frame))

        if len(processed_buffer) < SEQ:
            result = {"Frame": frame_index, "Visibility": 0, "X": -1, "Y": -1}
            append_to_csv(result, csv_path)
            if writer or args.visualize:
                vis_frame = draw_track(frame.copy(), track)
                if writer:
                    writer.write(vis_frame)
                if args.visualize:
                    cv2.imshow("Grid Prediction", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            frame_index += 1
            pbar.update(1)
            continue

        input_tensor = make_input_tensor(processed_buffer, device)
        with torch.no_grad():
            output = model(input_tensor).squeeze(0).detach().cpu().numpy()
        predictions = decode_predictions(output, args.threshold)
        visibility, x_resized, y_resized, _ = predictions[-1]

        if visibility:
            x_orig = int(x_resized * frame_width / INPUT_WIDTH)
            y_orig = int(y_resized * frame_height / INPUT_HEIGHT)
            track.append((x_orig, y_orig))
        else:
            x_orig, y_orig = -1, -1
            if track:
                track.popleft()

        result = {"Frame": frame_index, "Visibility": visibility, "X": x_orig, "Y": y_orig}
        append_to_csv(result, csv_path)

        if writer or args.visualize:
            vis_frame = frame.copy()
            vis_frame = draw_track(vis_frame, track)
            if visibility:
                cv2.circle(vis_frame, (x_orig, y_orig), 12, (0, 255, 0), 2)
            if writer:
                writer.write(vis_frame)
            if args.visualize:
                cv2.imshow("Grid Prediction", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_index += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()
    if args.visualize:
        cv2.destroyAllWindows()

    print(f"Model: {model_path}")
    if output_dir:
        print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
