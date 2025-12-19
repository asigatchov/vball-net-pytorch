import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic blocks
# ============================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class Single2DConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.conv = DepthwiseSeparableConv(
            in_channels, out_channels, dropout_p
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# Motion-aware gate (suppresses hands, boosts ball)
# ============================================================

class CompactMotionGate(nn.Module):
    """
    Усиливает компактное движение (мяч),
    подавляет протяжённое движение (руки).
    """
    def __init__(self, num_frames):
        super().__init__()
        self.temporal_conv = nn.Conv2d(
            num_frames, num_frames,
            kernel_size=1,
            groups=num_frames,
            bias=False
        )
        self.avg_pool = nn.AvgPool2d(
            kernel_size=7, stride=1, padding=3
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, T, H, W]
        diff = torch.abs(x[:, 1:] - x[:, :-1])
        diff = F.pad(diff, (0, 0, 0, 0, 1, 0))  # align T

        # local contrast (kills large smooth motion)
        local = diff - self.avg_pool(diff)
        local = torch.clamp(local, min=0.0)

        gate = self.sigmoid(self.temporal_conv(local))
        return x * (1.0 + gate)


# ============================================================
# Main model
# ============================================================

class VballNetFastV1(nn.Module):
    """
    VballNetFastV1: Fast lightweight model for volleyball tracking.
    Compatible with VballNetV1a interface.
    Supports Grayscale (N input frames, N output heatmaps) and RGB (N×3 input channels, N output heatmaps) modes.
    """
    def __init__(self, height=288, width=512, in_dim=15, out_dim=15):
        super().__init__()
        
        # Determine mode based on input/output dimensions (same logic as VballNetV1a)
        mode = "grayscale" if in_dim == out_dim else "rgb"
        num_frames = in_dim if mode == "grayscale" else in_dim // 3
        
        self.height = height
        self.width = width
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mode = mode
        self.num_frames = num_frames
        
        # Define model parameters
        channels = (8, 16, 32)
        dropout_p = 0.2
        bottleneck_channels = 64

        # motion-aware gate
        self.motion_gate = CompactMotionGate(num_frames)

        # encoder
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = num_frames
        for ch in channels:
            self.down_blocks.append(
                Single2DConv(in_ch, ch, dropout_p=0.0)
            )
            self.pools.append(nn.MaxPool2d(2))
            in_ch = ch

        # bottleneck
        self.bottleneck = Single2DConv(
            channels[-1],
            bottleneck_channels,
            dropout_p=dropout_p
        )

        # decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        in_ch = bottleneck_channels
        for ch in reversed(channels):
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
            self.up_blocks.append(
                Single2DConv(in_ch + ch, ch, dropout_p=dropout_p)
            )
            in_ch = ch

        self.predictor = nn.Conv2d(
            channels[0], out_dim, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()
        
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, imgs_input):
        # Handle input preprocessing to match VballNetV1a interface
        # For RGB mode, convert to grayscale similar to VballNetV1a
        if self.mode == "rgb":
            # Reshape input: [B, C, H, W] -> [B, T, 3, H, W]
            channels_per_frame = 3
            motion_input = imgs_input.view(imgs_input.shape[0], self.num_frames, channels_per_frame, imgs_input.shape[2], imgs_input.shape[3])
            
            # Convert RGB to grayscale using the same weights as VballNetV1a
            gray_scale = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=imgs_input.device)
            x = torch.einsum("btcwh,c->btwh", motion_input, gray_scale)
        else:
            # For grayscale, input is already in the correct format
            x = imgs_input
        
        # resize safety
        if x.shape[-2:] != (self.height, self.width):
            x = F.interpolate(
                x, (self.height, self.width),
                mode="bilinear", align_corners=False
            )

        # motion-aware weighting
        x = self.motion_gate(x)

        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, block, skip in zip(
            self.upsamples, self.up_blocks, reversed(skips)
        ):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        x = self.sigmoid(self.predictor(x))
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Losses (training only)
# ============================================================

def temporal_smoothness_loss(hm):
    return ((hm[:, 1:] - hm[:, :-1]) ** 2).mean()


def compactness_loss(hm, tau=0.3):
    mask = (hm > tau).float()
    return mask.mean()


def trajectory_smoothness_loss(hm):
    B, T, H, W = hm.shape

    y = torch.linspace(0, 1, H, device=hm.device)
    x = torch.linspace(0, 1, W, device=hm.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    cx = (hm * xx).sum(dim=(2, 3)) / (hm.sum(dim=(2, 3)) + 1e-6)
    cy = (hm * yy).sum(dim=(2, 3)) / (hm.sum(dim=(2, 3)) + 1e-6)

    vel = torch.sqrt(
        (cx[:, 1:] - cx[:, :-1]) ** 2 +
        (cy[:, 1:] - cy[:, :-1]) ** 2
    )
    return vel.std()


# ============================================================
# Training loss wrapper
# ============================================================

def vball_loss(pred, target,
               w_temp=0.05,
               w_compact=0.01,
               w_traj=0.02):
    loss_hm = F.mse_loss(pred, target)
    loss = loss_hm
    loss += w_temp * temporal_smoothness_loss(pred)
    loss += w_compact * compactness_loss(pred)
    loss += w_traj * trajectory_smoothness_loss(pred)
    return loss


# ============================================================
# Test / export
# ============================================================

if __name__ == "__main__":
    # Test with grayscale mode (default)
    model = VballNetFastV1()
    print("Parameters:", model.num_parameters())

    x = torch.randn(1, 15, 288, 512)
    y = model(x)
    print("Grayscale input shape:", x.shape)
    print("Output shape:", y.shape)
    
    # Test with RGB mode
    model_rgb = VballNetFastV1(in_dim=45, out_dim=15)  # 15 frames * 3 channels = 45 input channels
    x_rgb = torch.randn(1, 45, 288, 512)
    y_rgb = model_rgb(x_rgb)
    print("RGB input shape:", x_rgb.shape)
    print("RGB output shape:", y_rgb.shape)

    # ONNX export
    model.eval()
    torch.onnx.export(
        model,
        x,
        "vballnet_fast_v1_clean.onnx",
        opset_version=18,
        input_names=["clip"],
        output_names=["heatmaps"],
        dynamic_axes={"clip": {0: "B"}, "heatmaps": {0: "B"}},
        do_constant_folding=True,
        dynamo=False
    )
    print("ONNX export done.")
