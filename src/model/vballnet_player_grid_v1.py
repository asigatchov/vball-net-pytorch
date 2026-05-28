import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import fuse_conv_bn_eval


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dsconv = DepthwiseSeparableConv(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dsconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VballNetPlayerGridV1(nn.Module):
    """Anchor-free player detector for 9-frame grayscale stacks."""

    def __init__(self, input_height=432, input_width=768, in_dim=9, out_dim=5):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.stem = nn.Sequential(
            DSConvBlock(in_dim, 32),
            DSConvBlock(32, 48),
        )

        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DSConvBlock(48, 64),
            DSConvBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DSConvBlock(64, 96),
            DSConvBlock(96, 96),
        )
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DSConvBlock(96, 160),
            DSConvBlock(160, 160),
        )
        self.stage5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DSConvBlock(160, 256),
            DSConvBlock(256, 256),
        )

        self.lateral_s8 = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.lateral_s16 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            DSConvBlock(256, 192),
            DSConvBlock(192, 192),
        )
        self.head = nn.Sequential(
            DSConvBlock(192, 192),
            nn.Conv2d(192, out_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat_s8 = self.stage4(x)
        feat_s16 = self.stage5(feat_s8)

        proj_s8 = self.lateral_s8(feat_s8)
        proj_s16 = self.lateral_s16(feat_s16)
        proj_s16 = F.interpolate(proj_s16, size=proj_s8.shape[-2:], mode="nearest")

        # Keep the prediction grid at stride 8 for dense player localization.
        fused = torch.cat([proj_s8, proj_s16], dim=1)
        fused = self.fusion(fused)
        return self.head(fused)


def fuse_model(module):
    fused_pairs = 0
    for _, child in module.named_children():
        if isinstance(child, nn.Sequential):
            for index in range(len(child) - 1):
                if isinstance(child[index], nn.Conv2d) and isinstance(child[index + 1], nn.BatchNorm2d):
                    child[index] = fuse_conv_bn_eval(child[index], child[index + 1])
                    child[index + 1] = nn.Identity()
                    fused_pairs += 1
        elif isinstance(child, DSConvBlock):
            if isinstance(child.dsconv.pointwise, nn.Conv2d) and isinstance(child.bn, nn.BatchNorm2d):
                child.dsconv.pointwise = fuse_conv_bn_eval(child.dsconv.pointwise, child.bn)
                child.bn = nn.Identity()
                fused_pairs += 1
        fused_pairs += fuse_model(child)
    return fused_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VballNetPlayerGridV1 ONNX Exporter")
    parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("--export_onnx", action="store_true", help="Export as ONNX model")
    parser.add_argument("--in_dim", type=int, default=9)
    parser.add_argument("--out_dim", type=int, default=5)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    args = parser.parse_args()

    model = VballNetPlayerGridV1(
        input_height=args.height,
        input_width=args.width,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VballNetPlayerGridV1 initialized with {total_params:,} parameters")

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
        fused_pairs = fuse_model(model)
        print(f"Fused Conv+BN pairs before ONNX export: {fused_pairs}")
        dummy_input = torch.randn(1, args.in_dim, args.height, args.width, device=device)
        onnx_filename = os.path.splitext(args.model_path)[0] + ".onnx" if args.model_path else "vball_net_player_grid_v1.onnx"
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
