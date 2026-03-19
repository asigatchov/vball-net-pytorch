# vball_net_v3_fixed.py
# Full rewrite - cleaned up and corrected version
# - temporal block: preserves temporal resolution and aggregates correctly
# - fusion: attention projection (T -> out_dim) via 1x1 conv
# - smoothness loss: computed inside MotionPromptLayer and exposed via get_extra_losses()

import torch
import torch.nn as nn
import torch.nn.functional as F


def power_normalization(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Nonlinearity adapted for PyTorch.

    x: (...)
    a, b: scalar parameters
    """
    abs_x = torch.abs(x)
    denom = 0.45 * torch.abs(torch.tanh(a)) + 1e-1
    scale = 5.0 / denom
    return 1.0 / (1.0 + torch.exp(-scale * (abs_x - 0.8 * torch.tanh(b))))


class MotionPromptLayer(nn.Module):
    """Motion prompt with trainable a/b and built-in smoothness loss.

    Input: x (B, T, H, W) - grayscale clip.
    Returns: attention (B, T, H, W).
    Smoothness loss is available through get_loss().
    """

    def __init__(self, num_frames: int, penalty_weight: float = 1e-4):
        super().__init__()
        self.num_frames = num_frames
        self.penalty_weight = penalty_weight
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
        # Store the latest smoothness loss value (tensor or None)
        self._smoothness_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x: (B, T, H, W)
        # Shift/scale as in the original
        x = x * 0.225 + 0.45

        T = self.num_frames
        diffs = []
        # Handle edge cases safely
        for t in range(T):
            if T == 1:
                diff = x[:, 0] * 0.0
            elif t == 0:
                diff = x[:, 1] - x[:, 0]
            elif t == T - 1:
                diff = x[:, -1] - x[:, -2]
            else:
                diff = (x[:, t + 1] - x[:, t - 1]) / 2.0
            diffs.append(power_normalization(diff, self.a, self.b))

        attention = torch.stack(diffs, dim=1)  # (B, T, H, W)

        # Smoothness penalty between neighboring temporal maps
        if self.training and self.penalty_weight > 0:
            smoothness = torch.mean((attention[:, 1:] - attention[:, :-1]) ** 2)
            # Keep the tensor in the training graph (gradients flow through a/b)
            self._smoothness_loss = self.penalty_weight * smoothness
        else:
            self._smoothness_loss = None

        return attention

    def get_loss(self):
        return self._smoothness_loss


class FusionLayerTypeA(nn.Module):
    """Project attention (T -> out_dim) via 1x1 conv and fuse with features.

    Input:
      - features: (B, out_dim, H, W)
      - attention: (B, T, H, W)
    """

    def __init__(self, num_frames: int, out_dim: int):
        super().__init__()
        self.num_frames = num_frames
        self.out_dim = out_dim
        # Attention projection: Conv2d(T -> out_dim, kernel=1)
        self.att_proj = nn.Conv2d(num_frames, out_dim, kernel_size=1, bias=True)
        # Initialization
        nn.init.kaiming_normal_(self.att_proj.weight, mode='fan_out', nonlinearity='relu')
        if self.att_proj.bias is not None:
            nn.init.constant_(self.att_proj.bias, 0.0)

    def forward(self, features: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        # features: (B, out_dim, H, W)
        # attention: (B, T, H, W)
        # Project attention
        att_proj = self.att_proj(attention)  # (B, out_dim, H, W)
        weights = torch.sigmoid(att_proj)
        return features * weights


class VballNetV3c(nn.Module):
    """Flexible VBallNet v3 - clean and compatible version.

    in_dim - number of input frames; out_dim - number of output heatmaps.
    """

    def __init__(self, height: int = 288, width: int = 512, in_dim: int = 9, out_dim: int = 9):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.height = height
        self.width = width

        # --- Temporal Conv3D ---
        # out_channels = 10 -> center(1) + temporal(10) = 11 channels, as before
        self.temporal_conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=10,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 1, 1),  # preserve temporal resolution
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm3d(10)

        # Encoder (separable convs)
        def sep_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.enc1 = nn.Sequential(
            sep_conv(11, 32),
            sep_conv(32, 32),
        )
        self.enc2 = sep_conv(32, 64)
        self.enc3 = sep_conv(64, 128)

        # Decoder
        self.dec1 = sep_conv(128 + 64, 64)
        self.dec2 = sep_conv(64 + 32, 32)

        self.final_conv = nn.Conv2d(32, out_dim, kernel_size=1)

        # Motion guidance
        self.motion_prompt = MotionPromptLayer(num_frames=in_dim, penalty_weight=1e-4)
        self.fusion = FusionLayerTypeA(num_frames=in_dim, out_dim=out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim, H, W)"""
        B, C, H, W = x.shape
        assert C == self.in_dim, f"Expected {self.in_dim} channels, got {C}"

        # Convert to (B, T, H, W)
        x = x.view(B, self.in_dim, H, W)
        T = self.in_dim

        # Detach input for motion prompt so gradients do not flow through pixels
        orig_clip = x.detach()

        # === Temporal 3D extractor ===
        x_3d = x.unsqueeze(1)  # (B, 1, T, H, W)
        temporal = self.temporal_conv3d(x_3d)  # (B, 10, T, H, W)
        temporal = self.temporal_bn(temporal)
        temporal = F.relu(temporal, inplace=True)

        # Temporal aggregation replaced with a simple mean over the time axis
        temporal = torch.mean(temporal, dim=2, keepdim=True)  # (B, 10, 1, H, W)
        temporal = temporal.squeeze(2)  # (B, 10, H, W)

        # === Center frame ===
        center_idx = T // 2
        center = x[:, center_idx:center_idx + 1, :, :]  # (B,1,H,W)

        # === concat: center(1) + temporal(10) -> 11 ===
        enc_input = torch.cat([center, temporal], dim=1)  # (B,11,H,W)

        # === Encoder ===
        x_enc = self.enc1(enc_input)
        skip1 = x_enc
        x_enc = F.max_pool2d(x_enc, 2, ceil_mode=True)

        x_enc = self.enc2(x_enc)
        skip2 = x_enc
        x_enc = F.max_pool2d(x_enc, 2, ceil_mode=True)

        x_enc = self.enc3(x_enc)

        # === Decoder ===
        x_dec = F.interpolate(x_enc, scale_factor=2, mode='nearest')
        x_dec = torch.cat([x_dec, skip2], dim=1)
        x_dec = self.dec1(x_dec)

        x_dec = F.interpolate(x_dec, scale_factor=2, mode='nearest')
        x_dec = torch.cat([x_dec, skip1], dim=1)
        x_dec = self.dec2(x_dec)

        x_out = self.final_conv(x_dec)  # (B, out_dim, H, W)

        # === Motion Guidance & Fusion ===
        attention = self.motion_prompt(orig_clip)  # (B, T, H, W)
        out = self.fusion(x_out, attention)  # (B, out_dim, H, W)

        out = torch.sigmoid(out)

        return out

    def get_extra_losses(self):
        losses = {}
        motion_loss = self.motion_prompt.get_loss()
        if motion_loss is not None:
            losses['motion_smooth'] = motion_loss
        return losses


# ====================== Test / usage example ======================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VballNetV3c ONNX Exporter')
    parser.add_argument('--model_path', type=str, help='Path to the trained model checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = 'cpu'
    model = VballNetV3c(height=288, width=512, in_dim=9, out_dim=9).to(device)
    
    # If model path is provided, load the trained model
    if args.model_path:
        print(f"Loading model from checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Handle different checkpoint formats
        # if 'state_dict' in checkpoint:
        #     model.load_state_dict(checkpoint['state_dict'])
        # elif 'model_state_dict' in checkpoint:
        #     model.load_state_dict(checkpoint['model_state_dict'])
        # else:
        #     model.load_state_dict(checkpoint)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        #model.load_state_dict(checkpoint)
        
        print("Model loaded successfully!")

    model.train()

    x = torch.randn(2, 9, 288, 512, device=device)
    y = model(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    print("Extra losses (train):", model.get_extra_losses())

    # Inference/eval check
    model.eval()
    with torch.no_grad():
        y = model(x)
    print("Eval OK. Output range:", float(y.min()), float(y.max()))

    # Export to ONNX (optional)
    try:
        if args.model_path:
            # Save ONNX next to the model, replacing the extension with .onnx
            import os
            onnx_filename = os.path.splitext(args.model_path)[0] + ".onnx"
        else:
            onnx_filename = "vball_net_v3c_random.onnx"
        torch.onnx.export(
            model,
            (x,),
            onnx_filename,
            opset_version=17,
            input_names=["clip"],
            output_names=["heatmaps"],
            dynamic_axes={"clip": {0: "B"}, "heatmaps": {0: "B"}},
        )
        print(f"ONNX saved: {onnx_filename}")
    except Exception as e:
        print("ONNX export failed:", e)
