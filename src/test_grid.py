#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.vballnet_grid_v1a import VballNetGridV1a
from model.vballnet_grid_v1b import VballNetGridV1b
from model.vballnet_grid_v1c import VballNetGridV1c
from train_grid import GridTrackNetLoss, compute_metrics, get_device
from utils.grid_dataset import GridSequenceDataset


INPUT_WIDTH = 768
INPUT_HEIGHT = 432
GRID_COLS = 48
GRID_ROWS = 27


def parse_args():
    parser = argparse.ArgumentParser(description="GridTrackNet test-only evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to prepared test split")
    parser.add_argument(
        "--model_name",
        choices=["auto", "VballNetGridV1a", "VballNetGridV1b", "VballNetGridV1c"],
        default="auto",
        help="Model architecture to instantiate",
    )
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, mps, auto")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--tol", type=int, default=4, help="Distance tolerance in pixels")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length override")
    parser.add_argument("--grayscale", action="store_true", help="Force grayscale mode")
    parser.add_argument("--stride", type=int, default=2, help="Stride for dataset windowing")
    parser.add_argument("--height", type=int, default=432, help="Input height")
    parser.add_argument("--width", type=int, default=768, help="Input width")
    parser.add_argument("--grid_rows", type=int, default=27, help="Grid rows")
    parser.add_argument("--grid_cols", type=int, default=48, help="Grid cols")
    parser.add_argument("--out", type=str, default="outputs", help="Output root for metrics")
    return parser.parse_args()


def parse_optional_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def infer_model_name(model_path, requested_model_name):
    if requested_model_name != "auto":
        return requested_model_name
    model_name = Path(model_path).parents[1].name.split("_seq", 1)[0]
    if model_name not in {"VballNetGridV1a", "VballNetGridV1b", "VballNetGridV1c"}:
        raise ValueError(
            "Could not infer model_name from checkpoint path. Pass --model_name explicitly."
        )
    return model_name


def infer_dims_from_state_dict(state_dict):
    conv_weights = [
        (key, value)
        for key, value in state_dict.items()
        if key.endswith("weight") and getattr(value, "ndim", 0) == 4
    ]
    if not conv_weights:
        raise KeyError("Could not infer input/output channels from checkpoint state_dict")

    first_key, first_weight = conv_weights[0]
    if "depthwise" in first_key:
        in_dim = int(first_weight.shape[0])
    else:
        in_dim = int(first_weight.shape[1])

    _, last_weight = conv_weights[-1]
    out_dim = int(last_weight.shape[0])
    return in_dim, out_dim


def infer_params_from_path(model_path):
    run_dir_name = model_path.parents[1].name
    seq_match = re.search(r"_seq(\d+)", run_dir_name)
    if not seq_match:
        return None

    seq = int(seq_match.group(1))
    grayscale = "_grayscale" in run_dir_name.lower()
    return {
        "seq": seq,
        "grayscale": grayscale,
        "input_height": INPUT_HEIGHT,
        "input_width": INPUT_WIDTH,
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
    }


def infer_model_params(model_path, checkpoint, cli_args):
    config_path = model_path.parents[1] / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
        seq = int(config["seq"])
        grayscale = parse_optional_bool(config.get("grayscale"), default=False)
        height = int(config.get("height", INPUT_HEIGHT))
        width = int(config.get("width", INPUT_WIDTH))
        grid_rows = int(config.get("grid_rows", GRID_ROWS))
        grid_cols = int(config.get("grid_cols", GRID_COLS))
        return {
            "seq": seq,
            "grayscale": grayscale,
            "input_height": height,
            "input_width": width,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
        }

    path_params = infer_params_from_path(model_path)
    if path_params is not None:
        return path_params

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    inferred_in_dim, inferred_out_dim = infer_dims_from_state_dict(state_dict)
    inferred_grayscale = inferred_in_dim % 3 != 0
    inferred_seq = inferred_in_dim if inferred_grayscale else inferred_in_dim // 3
    if inferred_out_dim != inferred_seq * 3:
        raise ValueError(
            f"Could not infer model parameters from checkpoint: in_dim={inferred_in_dim}, out_dim={inferred_out_dim}"
        )

    grayscale = cli_args.grayscale or inferred_grayscale
    seq = cli_args.seq or inferred_seq
    return {
        "seq": seq,
        "grayscale": grayscale,
        "input_height": cli_args.height if hasattr(cli_args, "height") else INPUT_HEIGHT,
        "input_width": cli_args.width if hasattr(cli_args, "width") else INPUT_WIDTH,
        "grid_rows": cli_args.grid_rows if hasattr(cli_args, "grid_rows") else GRID_ROWS,
        "grid_cols": cli_args.grid_cols if hasattr(cli_args, "grid_cols") else GRID_COLS,
    }


def build_model(model_name, height, width, seq, grayscale):
    model_cls = {
        "VballNetGridV1a": VballNetGridV1a,
        "VballNetGridV1b": VballNetGridV1b,
        "VballNetGridV1c": VballNetGridV1c,
    }[model_name]
    in_dim = seq if grayscale else seq * 3
    return model_cls(
        input_height=height,
        input_width=width,
        in_dim=in_dim,
        out_dim=seq * 3,
    )


def load_model(model_path, model_name, model_params, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(
        model_name,
        model_params["input_height"],
        model_params["input_width"],
        model_params["seq"],
        model_params["grayscale"],
    ).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = get_device(args.device)
    checkpoint = torch.load(model_path, map_location=device)
    model_name = infer_model_name(model_path, args.model_name)
    model_params = infer_model_params(model_path, checkpoint, args)
    if args.seq is not None:
        model_params["seq"] = args.seq
    if args.grayscale:
        model_params["grayscale"] = True

    model = load_model(model_path, model_name, model_params, device)
    criterion = GridTrackNetLoss(seq=model_params["seq"])

    test_ds = GridSequenceDataset(
        args.test_data,
        seq=model_params["seq"],
        stride=args.stride,
        height=model_params["input_height"],
        width=model_params["input_width"],
        grid_rows=model_params["grid_rows"],
        grid_cols=model_params["grid_cols"],
        augment=False,
        grayscale=model_params["grayscale"],
    )
    if len(test_ds) == 0:
        raise RuntimeError(f"Empty test dataset: {args.test_data}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    total_loss = 0.0
    metrics_sum = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    pbar = tqdm(test_loader, desc="test", ncols=100)
    with torch.inference_mode():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_metrics = compute_metrics(
                outputs.detach().float(),
                targets.detach().float(),
                seq=model_params["seq"],
                tol=args.tol,
                grid_cols=model_params["grid_cols"],
                grid_rows=model_params["grid_rows"],
                width=model_params["input_width"],
                height=model_params["input_height"],
            )

            total_loss += loss.item()
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "f1": f"{batch_metrics['f1']:.4f}"})

    count = max(len(test_loader), 1)
    avg_loss = total_loss / count
    avg_metrics = {key: value / count for key, value in metrics_sum.items()}

    output_dir = Path(args.out) / f"{model_path.parents[1].name}_test_{Path(args.test_data).name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "loss": avg_loss,
        "metrics": avg_metrics,
        "checkpoint": str(model_path),
        "model_name": model_name,
        "test_data": args.test_data,
        "samples": len(test_ds),
        "params": model_params,
    }
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        "Test summary "
        f"loss={avg_loss:.4f} "
        f"accuracy={avg_metrics['accuracy']:.4f} "
        f"precision={avg_metrics['precision']:.4f} "
        f"recall={avg_metrics['recall']:.4f} "
        f"f1={avg_metrics['f1']:.4f}"
    )
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
