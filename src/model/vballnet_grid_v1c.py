import argparse
import os

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VballNetGridV1c(nn.Module):
    """Grid model tuned for grayscale frame stacks with depthwise separable blocks."""

    def __init__(self, input_height=432, input_width=768, in_dim=9, out_dim=27):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.stem = DepthwiseSeparableConv(in_dim, 32)
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(32, 48),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv(48, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv(64, 96),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv(96, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv(128, 512),
            DepthwiseSeparableConv(512, 512),
        )
        self.head = nn.Sequential(
            DepthwiseSeparableConv(512, 512),
            nn.Conv2d(512, out_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        return self.head(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VballNetGridV1c ONNX Exporter")
    parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("--export_onnx", action="store_true", help="Export as ONNX model")
    parser.add_argument("--in_dim", type=int, default=9)
    parser.add_argument("--out_dim", type=int, default=27)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    args = parser.parse_args()

    model = VballNetGridV1c(
        input_height=args.height,
        input_width=args.width,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VballNetGridV1c initialized with {total_params:,} parameters")

    test_input = torch.randn(2, args.in_dim, args.height, args.width)
    test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")

    device = torch.device("cpu")
    model = model.to(device)

    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.model_path}")

    if args.export_onnx:
        model.eval()
        dummy_input = torch.randn(1, args.in_dim, args.height, args.width, device=device)
        onnx_filename = os.path.splitext(args.model_path)[0] + ".onnx" if args.model_path else "vball_net_grid_v1c.onnx"
        torch.onnx.export(
            model,
            (dummy_input,),
            onnx_filename,
            opset_version=17,
            input_names=["clip"],
            output_names=["grid"],
            dynamic_axes={"clip": {0: "B"}, "grid": {0: "B"}},
            export_params=True,
            do_constant_folding=True,
            verbose=False,
            dynamo=False,
        )
        print(f"ONNX model saved: {onnx_filename}")
