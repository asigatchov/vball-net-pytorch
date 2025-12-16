import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility functions
def rearrange_tensor(input_tensor, order):
    """Rearranges the dimensions of a tensor according to the specified order."""
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all(dim in order for dim in "BCHWT"), "Order must contain all of BCHWT"
    perm = [order.index(dim) for dim in "BTCHW"]
    return input_tensor.permute(*perm)

def power_normalization(input, a, b) -> torch.Tensor:
    """Power normalization function for attention map generation."""
    return 1 / (1 + torch.exp(-(5 / (0.45 * torch.abs(torch.tanh(a)) + 1e-1)) * (torch.abs(input) - 0.6 * torch.tanh(b))))

# MotionPrompt class
class MotionPrompt(nn.Module):
    """A module for generating attention and direction maps from video sequences."""
    def __init__(self, num_frames, mode="grayscale", penalty_weight=0.0):
        super().__init__()
        self.num_frames = num_frames
        self.mode = mode.lower()
        assert self.mode in ["rgb", "grayscale"], "Mode must be 'rgb' or 'grayscale'"
        self.input_permutation = "BTCHW"
        self.input_color_order = "RGB" if self.mode == "rgb" else None
        self.color_map = {"R": 0, "G": 1, "B": 2}
        self.gray_scale = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.lambda1 = penalty_weight
        self.output_channels = 2  # Two channels: positive and negative

    def forward(self, video_seq):
        loss = torch.tensor(0.0, device=video_seq.device)
        video_seq = rearrange_tensor(video_seq, self.input_permutation)

        if self.mode == "rgb":
            idx_list = [self.color_map[idx] for idx in self.input_color_order]
            weights = self.gray_scale[idx_list].to(video_seq.device)
            grayscale_video_seq = torch.einsum("btcwh,c->btwh", video_seq, weights)
        else:
            grayscale_video_seq = video_seq[:, :, 0, :, :]  # Single channel per frame

        # Compute median across all frames
        median_frame = torch.median(grayscale_video_seq, dim=1, keepdim=True).values

        # Store positive and negative maps
        pos_maps = []
        neg_maps = []

        for t in range(self.num_frames):
            if t == 0:
                # Forward difference for first frame
                diff = grayscale_video_seq[:, t + 1] - grayscale_video_seq[:, t]
            elif t == self.num_frames - 1:
                # Backward difference for last frame
                diff = grayscale_video_seq[:, t] - grayscale_video_seq[:, t - 1]
            else:
                # Central difference for intermediate frames
                diff = (grayscale_video_seq[:, t + 1] - grayscale_video_seq[:, t - 1]) * 0.5

            # Split into positive and negative components
            pos = F.relu(diff)  # Positive changes
            neg = F.relu(-diff)  # Negative changes
            pos_norm = power_normalization(pos, self.a, self.b)
            neg_norm = power_normalization(neg, self.a, self.b)
            pos_maps.append(pos_norm)
            neg_maps.append(neg_norm)

        # Stack into attention and direction maps
        residual_maps = torch.stack(pos_maps, dim=1)  # Shape: (B, T, H, W)
        #residual_maps = torch.stack([pos_norm + neg_norm], dim=1)  # общая активность
        direction_maps = torch.stack([torch.stack(pos_maps, dim=1), torch.stack(neg_maps, dim=1)], dim=2)  # Shape: (B, T, 2, H, W)

        if self.training:
            B, T, _, H, W = direction_maps.shape
            temp_diff = direction_maps[:, 1:] - direction_maps[:, :-1]
            temporal_loss = torch.sum(temp_diff ** 2) / (H * W * (T - 1) * B * 2)
            loss = self.lambda1 * temporal_loss

        return residual_maps, direction_maps, loss

# FusionLayerTypeA class
class FusionLayerTypeA(nn.Module):
    """A module that incorporates motion and direction using attention maps."""
    def __init__(self, num_frames, out_dim):
        super().__init__()
        self.num_frames = num_frames
        self.out_dim = out_dim
        self.pos_weight = nn.Parameter(torch.tensor(0.5))  # Weight for positive channel
        self.neg_weight = nn.Parameter(torch.tensor(0.5))  # Weight for negative channel

    def forward(self, feature_map, residual_maps, direction_maps=None):
        outputs = []
        for t in range(min(self.num_frames, self.out_dim)):
            # Use residual_maps for attention
            att_map = residual_maps[:, t, :, :]  # Shape: (batch, height, width)
            out = feature_map[:, t, :, :] * att_map

            # Incorporate direction if provided
            if direction_maps is not None:
                pos_map = direction_maps[:, t, 0, :, :]  # Positive channel
                neg_map = direction_maps[:, t, 1, :, :]  # Negative channel
                dir_map = self.pos_weight * pos_map + self.neg_weight * neg_map
                dir_map = torch.sigmoid(dir_map)  # Normalize direction map
                out = out * (1 + dir_map)  # Amplify features based on direction

            outputs.append(out)

        return torch.stack(outputs, dim=1)

# VballNetV3 Model
class VballNetV3(nn.Module):
    """VballNetV3: Motion-enhanced U-Net for volleyball tracking with direction."""
    def __init__(self, height=288, width=512, in_dim=9, out_dim=9, fusion_layer_type="TypeA"):
        super().__init__()
        assert fusion_layer_type == "TypeA", "Fusion layer must be 'TypeA'"
        mode = "grayscale" if in_dim == out_dim else "rgb"
        num_frames = in_dim if mode == "grayscale" else in_dim // 3

        # Fusion layer
        self.fusion_layer = FusionLayerTypeA(num_frames=num_frames, out_dim=out_dim)
        # Motion prompt
        self.motion_prompt = MotionPrompt(num_frames=num_frames, mode=mode)
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.enc1_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        # Output layer
        self.out_conv = nn.Conv2d(32, out_dim, kernel_size=1, padding=0)

    def forward(self, imgs_input):
        # Reshape input for motion prompt
        channels_per_frame = imgs_input.shape[1] // self.motion_prompt.num_frames
        motion_input = imgs_input.view(imgs_input.shape[0], self.motion_prompt.num_frames, 
                                     channels_per_frame, imgs_input.shape[2], imgs_input.shape[3])

        # Motion prompt with direction
        residual_maps, direction_maps, _ = self.motion_prompt(motion_input)

        # Encoder
        x1 = self.enc1(imgs_input)
        x1_1 = self.enc1_1(x1)
        x = self.pool1(x1_1)
        x2 = self.enc2(x)
        x = self.pool2(x2)
        x = self.enc3(x)

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x)
        x = self.up2(x)
        x = torch.cat([x, x1_1], dim=1)
        x = self.dec2(x)

        # Output
        x = self.out_conv(x)
        x = self.fusion_layer(x, residual_maps, direction_maps)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    # Model initialization and testing
    height, width, in_dim, out_dim = 288, 512, 9, 9
    model = VballNetV3(height, width, in_dim, out_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VballNetV3 initialized with {total_params:,} parameters")

    # Forward pass test
    test_input = torch.randn(2, in_dim, height, width)
    test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print("✓ VballNetV3 ready for training with direction awareness!")
