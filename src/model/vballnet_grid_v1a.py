import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class VballNetGridV1a(nn.Module):
    """PyTorch port of GridTrackNet."""

    def __init__(self, input_height=432, input_width=768, in_dim=15, out_dim=15):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.features = nn.Sequential(
            ConvBlock(in_dim, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )
        self.head = nn.Conv2d(512, out_dim, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return self.activation(x)


if __name__ == "__main__":
    import argparse
    import os

    height, width, in_dim, out_dim = 432, 768, 15, 15
    model = VballNetGridV1a(
        input_height=height,
        input_width=width,
        in_dim=in_dim,
        out_dim=out_dim,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VballNetGridV1a initialized with {total_params:,} parameters")

    test_input = torch.randn(2, in_dim, height, width)
    test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print("VballNetGridV1a ready for training")

    parser = argparse.ArgumentParser(description="VballNetGridV1a ONNX Exporter")
    parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("--export_onnx", action="store_true", help="Export as ONNX model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = model.to(device)

    if args.model_path:
        print(f"Loading model from checkpoint: {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
        except Exception as error:
            print(f"Failed to load model: {error}")
            raise

    if args.export_onnx:
        try:
            model.eval()
            dummy_input = torch.randn(1, in_dim, height, width, device=device)
            if args.model_path:
                onnx_filename = os.path.splitext(args.model_path)[0] + ".onnx"
            else:
                onnx_filename = "vball_net_grid_v1a.onnx"

            torch.onnx.export(
                model,
                (dummy_input,),
                onnx_filename,
                opset_version=13,
                input_names=["clip"],
                output_names=["grid"],
                dynamic_axes={
                    "clip": {0: "B"},
                    "grid": {0: "B"},
                },
                export_params=True,
                do_constant_folding=True,
                verbose=False,
                dynamo=False,
            )
            print(f"ONNX model saved: {onnx_filename}")
        except Exception as error:
            print("ONNX export failed:", error)
            import traceback
            traceback.print_exc()
