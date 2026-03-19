import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution with configurable kernel_size."""
    def __init__(self, in_channels, out_channels, name, kernel_size=3, dropout_p=0.0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Single2DConv(nn.Module):
    """Wrapper around DepthwiseSeparableConv."""
    def __init__(self, in_channels, out_channels, name, kernel_size=3, dropout_p=0.0):
        super(Single2DConv, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, name, kernel_size, dropout_p)
        
    def forward(self, x):
        return self.conv(x)

class VballNetFastV2(nn.Module):
    """
    Scalable PyTorch version of VballNet with a GRU to model trajectories across 9 grayscale frames.

    Args:
        input_height (int): Input image height.
        input_width (int): Input image width.
        in_dim (int): Number of input channels (default 9 for grayscale frames).
        out_dim (int): Number of output heatmaps (default 9).
        channels (list): Channel list for the encoder (decoder mirrors it).
        bottleneck_channels (int): Number of channels in the bottleneck.
        bottleneck_kernel_size (int): Kernel size in the bottleneck.
        gru_hidden_size (int): Size of the GRU hidden state.
        dropout_p (float): Dropout probability.

    Returns:
        Model: PyTorch VballNetFastV2 model with GRU.
    """
    def __init__(self, height, width, in_dim=15, out_dim=15, 
                 channels=[8, 16, 32], bottleneck_channels=128, 
                 bottleneck_kernel_size=3, gru_hidden_size=128, dropout_p=0.2):
        super(VballNetFastV2, self).__init__()
        
        self.input_height = height
        self.input_width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channels = channels
        self.bottleneck_channels = bottleneck_channels
        self.gru_hidden_size = gru_hidden_size
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        current_channels = in_dim
        for i, out_channels in enumerate(channels):
            self.down_blocks.append(
                Single2DConv(current_channels, out_channels, f'down_block_{i+1}', kernel_size=3, dropout_p=0.0)
            )
            current_channels = out_channels
        
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(len(channels))
        ])
        
        # Bottleneck
        self.bottleneck = Single2DConv(
            channels[-1], bottleneck_channels, 'bottleneck', 
            kernel_size=bottleneck_kernel_size, dropout_p=dropout_p
        )
        
        # GRU for temporal dependencies
        self.gru = nn.GRU(
            input_size=bottleneck_channels,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True
        )
        # Transition layer after GRU to restore channels
        self.gru_transition = nn.Conv2d(
            gru_hidden_size, bottleneck_channels, kernel_size=1, bias=False
        )
        self.gru_bn = nn.BatchNorm2d(bottleneck_channels)
        self.gru_relu = nn.ReLU(inplace=True)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        decoder_channels = channels[::-1]  # [32, 16, 8]
        current_channels = bottleneck_channels
        
        for i, out_channels in enumerate(decoder_channels):
            in_channels = current_channels + channels[-(i+1)]
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
            self.up_blocks.append(
                Single2DConv(in_channels, out_channels, f'up_block_{i+1}', kernel_size=3, dropout_p=dropout_p)
            )
            current_channels = out_channels
        
        # Final predictor: input channels = channels[0]
        self.predictor = nn.Conv2d(
            channels[0], out_dim, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if x.size(2) != self.input_height or x.size(3) != self.input_width:
            x = F.interpolate(x, size=(self.input_height, self.input_width), 
                            mode='bilinear', align_corners=False)
        
        # Encoder
        skip_connections = []
        for down_block, pool in zip(self.down_blocks, self.pools):
            x = down_block(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)  # (N, bottleneck_channels, H/16, W/16)
        
        # Prepare for GRU: reshape into a sequence
        batch_size, channels, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, H/16, W/16, bottleneck_channels)
        x = x.view(batch_size, h * w, channels)  # (N, seq_len=H/16*W/16, bottleneck_channels)
        
        # GRU
        x, _ = self.gru(x)  # (N, seq_len, gru_hidden_size)
        
        # Restore spatial format
        x = x.view(batch_size, h, w, self.gru_hidden_size)  # (N, H/16, W/16, gru_hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, gru_hidden_size, H/16, W/16)
        x = self.gru_transition(x)  # (N, bottleneck_channels, H/16, W/16)
        x = self.gru_bn(x)
        x = self.gru_relu(x)
        
        # Decoder
        for upsample, up_block, skip in zip(
            self.upsamples, self.up_blocks, skip_connections[::-1]
        ):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = up_block(x)
        
        # Final predictor
        x = self.predictor(x)
        x = self.sigmoid(x)
        
        return x

    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Model initialization and testing
    height, width, in_dim, out_dim = 288, 512, 15, 15
    model = VballNetFastV2(height, width, in_dim, out_dim)
    total_params = model.get_num_parameters()
    print(f"VballNetFastV2 initialized with {total_params:,} parameters")

    # Forward pass test
    test_input = torch.randn(2, in_dim, height, width)
    test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print("✓ VballNetFastV2 ready for training!")

    # Add ONNX export functionality
    import argparse
    import os

    parser = argparse.ArgumentParser(description='VballNetFastV2 ONNX Exporter')
    parser.add_argument('--model_path', type=str, help='Path to the trained model checkpoint')
    parser.add_argument('--export_onnx', action='store_true', help='Export as ONNX model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # For testing purposes, use CPU
    device = 'cpu'

    # If model path is provided, load the trained model
    if args.model_path:
        print(f"Loading model from checkpoint: {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location=device)

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            exit(1)

    if args.export_onnx:
        try:
            model.eval()
            dummy_input = torch.randn(1, 15, 288, 512, device=device)

            if args.model_path:
                # Save ONNX next to the model file, replacing extension with .onnx
                onnx_filename = os.path.splitext(args.model_path)[0] + ".onnx"
            else:
                onnx_filename = "vball_net_fast_v2_trained.onnx"

            torch.onnx.export(
                model,
                (dummy_input,),
                onnx_filename,
                opset_version=13,
                input_names=["clip"],
                output_names=["heatmaps"],
                dynamic_axes={
                    "clip": {0: "B"},
                    "heatmaps": {0: "B"}
                },
                export_params=True,
                do_constant_folding=True,
                verbose=False,
                dynamo=False
            )
            print(f"ONNX model saved: {onnx_filename}")
        except Exception as e:
            print("ONNX export failed:", e)
            import traceback
            traceback.print_exc()
