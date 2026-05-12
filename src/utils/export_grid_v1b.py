#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn.utils import fuse_conv_bn_eval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.vballnet_grid_v1b import DSConvBlock, VballNetGridV1b


def fuse_model(module: torch.nn.Module) -> int:
    fused_pairs = 0
    for _, child in module.named_children():
        if isinstance(child, torch.nn.Sequential):
            for index in range(len(child) - 1):
                if isinstance(child[index], torch.nn.Conv2d) and isinstance(child[index + 1], torch.nn.BatchNorm2d):
                    child[index] = fuse_conv_bn_eval(child[index], child[index + 1])
                    child[index + 1] = torch.nn.Identity()
                    fused_pairs += 1
        elif isinstance(child, DSConvBlock):
            if isinstance(child.dsconv.pointwise, torch.nn.Conv2d) and isinstance(child.bn, torch.nn.BatchNorm2d):
                child.dsconv.pointwise = fuse_conv_bn_eval(child.dsconv.pointwise, child.bn)
                child.bn = torch.nn.Identity()
                fused_pairs += 1
        fused_pairs += fuse_model(child)
    return fused_pairs


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)


def export_onnx(model: torch.nn.Module, onnx_path: Path, in_dim: int, height: int, width: int) -> None:
    dummy_input = torch.randn(1, in_dim, height, width, device="cpu")
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        opset_version=17,
        input_names=["clip"],
        output_names=["grid"],
        dynamic_axes={"clip": {0: "B"}, "grid": {0: "B"}},
        export_params=True,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,
    )


def export_openvino(onnx_path: Path, openvino_dir: Path) -> Path:
    import openvino as ov

    openvino_dir.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(str(onnx_path))
    xml_path = openvino_dir.with_suffix(".xml")
    ov.save_model(model, str(xml_path))
    return xml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VballNetGridV1b checkpoint to ONNX and OpenVINO")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to .pth checkpoint")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models"),
        help="Directory for exported ONNX files",
    )
    parser.add_argument(
        "--openvino_dir",
        type=Path,
        default=Path("models") / "openvino",
        help="Directory for OpenVINO IR files",
    )
    parser.add_argument("--in_dim", type=int, default=9)
    parser.add_argument("--out_dim", type=int, default=27)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--name", type=str, default=None, help="Override output basename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")

    run_name = args.name or args.model_path.parent.parent.name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.openvino_dir.parent.mkdir(parents=True, exist_ok=True)

    model = VballNetGridV1b(
        input_height=args.height,
        input_width=args.width,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
    )
    load_checkpoint(model, args.model_path)
    model.eval()

    fused_pairs = fuse_model(model)
    onnx_path = args.output_dir / f"{run_name}.onnx"
    export_onnx(model, onnx_path, args.in_dim, args.height, args.width)

    openvino_path = export_openvino(onnx_path, args.openvino_dir / run_name)

    metadata = {
        "checkpoint": str(args.model_path),
        "onnx": str(onnx_path),
        "openvino_xml": str(openvino_path),
        "input_shape": [1, args.in_dim, args.height, args.width],
        "output_shape": [1, args.out_dim, args.height // 16, args.width // 16],
        "fused_conv_bn_pairs": fused_pairs,
    }
    metadata_path = args.output_dir / f"{run_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"ONNX saved: {onnx_path}")
    print(f"OpenVINO saved: {openvino_path}")
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
