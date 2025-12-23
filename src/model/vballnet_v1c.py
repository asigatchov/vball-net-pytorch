import torch
import torch.nn as nn
import torch.nn.functional as F


# Utility functions
def rearrange_tensor(input_tensor, order):
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all(dim in order for dim in "BCHWT"), "Order must contain all of BCHWT"
    perm = [order.index(dim) for dim in "BTCHW"]
    return input_tensor.permute(*perm)


def power_normalization(input, scale, threshold):
    """
    Улучшенная power_normalization.
    Преобразует разность кадров в карту внимания через сигмоиду.
    """
    return torch.sigmoid(scale * (torch.abs(input) - threshold))


class MotionPrompt(nn.Module):
    """
    Улучшенный MotionPrompt с поддержкой stateful ONNX.
    Теперь принимает h0 и возвращает hn.
    """

    def __init__(
        self, num_frames, penalty_weight=0.0, gru_hidden_size=128
    ):
        super().__init__()
        self.num_frames = num_frames
        self.lambda1 = penalty_weight
        self.hidden_size = gru_hidden_size

        # --- Улучшенная нормализация ---
        self.scale = nn.Parameter(torch.tensor(5.0))  # крутизна
        self.threshold = nn.Parameter(torch.tensor(0.6))  # порог

        # --- GRU для временной динамики ---
        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # --- Сжатие пространства ---
        self.pool = nn.MaxPool2d(kernel_size=16, stride=16)  # 288x512 -> 18x32
        self.pooled_height, self.pooled_width = 288 // 16, 512 // 16  # 18, 32
        self.feature_dim = self.pooled_height * self.pooled_width  # 576

        # --- Линейные слои ---
        self.linear_reduce = nn.Linear(self.feature_dim, gru_hidden_size)
        self.linear_expand = nn.Linear(gru_hidden_size, self.feature_dim)

        # --- Восстановление разрешения ---
        self.upsample = nn.Upsample(
            size=(288, 512), mode="bilinear", align_corners=False
        )

    def forward(self, video_seq, h0=None):
        """
        Args:
            video_seq: Tensor of shape (B, T, H, W) - grayscale video sequence
            h0: initial hidden state for GRU, shape (1, B, hidden_size). Optional.

        Returns:
            attention_map: (B, T, H, W)
            loss: scalar (0 if not training)
            hn: final hidden state, shape (1, B, hidden_size) — for ONNX stateful export
        """
        device = video_seq.device
        loss = torch.tensor(0.0, device=device)

        # --- Нормализация ---
        norm_seq = video_seq # * 0.225 + 0.45

        B, T, H, W = norm_seq.shape

        # --- Центральные разности ---
        frame_diffs = []
        for t in range(T):
            if t == 0:
                diff = norm_seq[:, 1] - norm_seq[:, 0]
            elif t == T - 1:
                diff = norm_seq[:, -1] - norm_seq[:, -2]
            else:
                diff = (norm_seq[:, t + 1] - norm_seq[:, t - 1]) / 2
            frame_diffs.append(diff)
        frame_diffs = torch.stack(frame_diffs, dim=1)  # (B, T, H, W)

        # --- Подготовка к GRU ---
        x = frame_diffs.reshape(B * T, 1, H, W)  # (B*T, 1, 288, 512)
        x = self.pool(x)  # (B*T, 1, 18, 32)
        x = x.reshape(B * T, -1)  # (B*T, 576)
        x = self.linear_reduce(x)  # (B*T, hidden_size)
        x = x.reshape(B, T, -1)  # (B, T, hidden_size)

        # --- GRU с h0 ---
        # For ONNX export compatibility, we need to handle the case where h0 might be None
        if h0 is None:
            gru_out, hn = self.gru(x)
        else:
            gru_out, hn = self.gru(x, h0)  # hn: (1, B, hidden_size)

        # --- Восстановление пространственной структуры ---
        x = gru_out.reshape(B * T, -1)  # (B*T, hidden_size)
        x = self.linear_expand(x)  # (B*T, 576)
        x = x.reshape(
            B * T, 1, self.pooled_height, self.pooled_width
        )  # (B*T, 1, 18, 32)
        x = self.upsample(x)  # (B*T, 1, 288, 512)
        x = x.reshape(B, T, H, W)  # (B, T, 288, 512)

        # --- Улучшенная power_normalization ---
        attention_map = torch.sigmoid(self.scale * (torch.abs(x) - self.threshold))

        # --- Temporal loss ---
        if self.training:
            norm_attention = attention_map.unsqueeze(2)
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = temp_diff.pow(2).sum() / (H * W * (T - 1) * B)
            loss = self.lambda1 * temporal_loss

        return attention_map, loss, hn

    def reset_hidden_state(self):
        """Состояние управляется извне (через h0), этот метод можно не использовать."""
        pass


class FusionLayerTypeA(nn.Module):
    def __init__(self, num_frames, out_dim):
        super().__init__()
        self.num_frames = num_frames
        self.out_dim = out_dim

    def forward(self, feature_map, attention_map):
        outputs = []
        for t in range(min(self.num_frames, self.out_dim)):
            outputs.append(feature_map[:, t, :, :] * attention_map[:, t, :, :])
        return torch.stack(outputs, dim=1)


class VballNetV1c(nn.Module):
    """
    Stateful-совместимая версия VballNetV1 для ONNX.
    Теперь принимает h0 и возвращает hn.
    """

    def __init__(
        self, height=288, width=512, in_dim=15, out_dim=15, fusion_layer_type="TypeA"
    ):
        super().__init__()
        assert fusion_layer_type == "TypeA", "Fusion layer must be 'TypeA'"
        num_frames = in_dim  # For grayscale, num_frames equals in_dim

        self.fusion_layer = FusionLayerTypeA(num_frames=num_frames, out_dim=out_dim)
        self.motion_prompt = MotionPrompt(num_frames=num_frames)
        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        self.enc1_1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        # --- Decoder ---
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )

        # --- Output ---
        self.out_conv = nn.Conv2d(32, out_dim, kernel_size=1, padding=0)

    def forward(self, imgs_input, h0=None):
        """
        Args:
            imgs_input: (B, C, H, W) - grayscale frames
            h0: initial hidden state for GRU, shape (1, B, 256)

        Returns:
            output: (B, out_dim, H, W)
            hn: final hidden state (1, B, 256) — for next step
        """
        B, C, H, W = imgs_input.shape
        # For grayscale, each channel represents a separate frame
        # Reshape to (B, T, H, W) for MotionPrompt
        motion_input = imgs_input.view(B, C, H, W)  # Already in correct shape

        # --- Motion Prompt с h0 ---
        residual_maps, _, hn = self.motion_prompt(motion_input, h0)  # получаем hn

        # --- Encoder ---
        x1 = self.enc1(imgs_input)
        x1_1 = self.enc1_1(x1)
        x = self.pool1(x1_1)
        x2 = self.enc2(x)
        x = self.pool2(x2)
        x = self.enc3(x)

        # --- Decoder ---
        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x)
        x = self.up2(x)
        x = torch.cat([x, x1_1], dim=1)
        x = self.dec2(x)

        # --- Output ---
        x = self.out_conv(x)
        x = self.fusion_layer(x, residual_maps)
        x = torch.sigmoid(x)

        return x, hn


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='VballNetV1c ONNX Exporter')
    parser.add_argument('--model_path', type=str, help='Path to the trained model checkpoint')
    parser.add_argument('--stateful', action='store_true', help='Export as stateful ONNX model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # For testing purposes, use CPU
    device = 'cpu'

    # Initialize model with 15 input frames and 15 output heatmaps
    model = VballNetV1c(height=288, width=512, in_dim=15, out_dim=15).to(device)

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

    # Export to ONNX
    try:
        model.eval()
        dummy_input = torch.randn(1, 15, 288, 512, device=device)

        if args.stateful:
            # Stateful export with h0 input and hn output
            dummy_h0 = torch.randn(1, 1, 128, device=device)  # (1, B, hidden_size)

            if args.model_path:
                # Save ONNX next to the model file, replacing extension with .onnx
                onnx_filename = os.path.splitext(args.model_path)[0] + "_stateful.onnx"
            else:
                onnx_filename = "vball_net_v1c_stateful.onnx"

            torch.onnx.export(
                model,
                (dummy_input, dummy_h0),
                onnx_filename,
                opset_version=13,
                input_names=["clip", "h0"],
                output_names=["heatmaps", "hn"],
                dynamic_axes={
                    "clip": {0: "B"},
                    "h0": {1: "B"},
                    "heatmaps": {0: "B"},
                    "hn": {1: "B"}
                },
                export_params=True,
                do_constant_folding=True,
                verbose=False,
            )
            print(f"Stateful ONNX model saved: {onnx_filename}")
        else:
            # Standard export
            if args.model_path:
                # Save ONNX next to the model file, replacing extension with .onnx
                onnx_filename = os.path.splitext(args.model_path)[0] + ".onnx"
            else:
                onnx_filename = "vball_net_v1c_trained.onnx"

            torch.onnx.export(
                model,
                (dummy_input,),
                onnx_filename,
                opset_version=13,
                input_names=["clip"],
                output_names=["heatmaps"],
                dynamic_axes={"clip": {0: "B"}, "heatmaps": {0: "B"}},
                # Add export_kwargs for better ONNX compatibility
                export_params=True,
                do_constant_folding=True,
                # verbose=False,
                # Use traditional export instead of dynamo
                dynamo=False,
            )
            print(f"ONNX model saved: {onnx_filename}")
    except Exception as e:
        print("ONNX export failed:", e)
        import traceback
        traceback.print_exc()
