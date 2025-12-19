import torch
import torch.nn as nn

class MotionPromptLayer(nn.Module):
    """Обобщённый слой для генерации attention map на основе разностей кадров."""
    def __init__(self, penalty_weight=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.lambda1 = penalty_weight

    def power_normalization(self, x):
        tanh_a = torch.tanh(self.a)
        tanh_b = torch.tanh(self.b)
        scale = 5 / (0.45 * torch.abs(tanh_a) + 1e-1)
        threshold = 0.6 * tanh_b
        return 1 / (1 + torch.exp(-scale * (torch.abs(x) - threshold)))

    def forward(self, video_seq):
        # video_seq: B, T, 1, H, W
        grayscale_seq = video_seq.squeeze(2)  # B, T, H, W
        frame_diff = grayscale_seq[:, 1:] - grayscale_seq[:, :-1]  # B, T-1, H, W
        attention_map = self.power_normalization(frame_diff)  # B, T-1, H, W

        loss = 0.0
        if self.training and self.lambda1 > 0 and frame_diff.shape[1] > 1:
            temp_diff = attention_map[:, 1:] - attention_map[:, :-1]
            temporal_loss = torch.mean(temp_diff ** 2)
            loss = self.lambda1 * temporal_loss

        return attention_map, loss


class FusionLayerTypeA(nn.Module):
    """Fusion TypeA: первый кадр без усиления, остальные усиливаются соответствующей attention."""
    def forward(self, feature, attention_map):
        output = feature.clone()
        output[:, 1:] *= attention_map
        return output


class FusionLayerTypeB(nn.Module):
    """Обобщённый TypeB: симметричное усиление с усреднением соседних attention."""
    def forward(self, feature, attention_map):
        B, N, H, W = feature.shape
        if N <= 1:
            return feature
        output = feature.clone()
        output[:, 0] *= attention_map[:, 0]
        if N > 2:
            avg_attention = (attention_map[:, :-1] + attention_map[:, 1:]) / 2.0
            output[:, 1:N-1] *= avg_attention
        if N > 1:
            output[:, N-1] *= attention_map[:, -1]
        return output


class VballNetV1b(nn.Module):
    def __init__(self, height=288, width=512, in_dim=15, out_dim=15,
                 fusion_type='TypeA', penalty_weight=0.0):
        super().__init__()
        if in_dim != out_dim:
            raise ValueError("in_dim должен равняться out_dim")
        self.T = in_dim

        self.motion_prompt = MotionPromptLayer(penalty_weight=penalty_weight)

        if fusion_type == 'TypeA':
            self.fusion = FusionLayerTypeA()
        elif fusion_type == 'TypeB':
            self.fusion = FusionLayerTypeB()
        else:
            raise ValueError("fusion_type должен быть 'TypeA' или 'TypeB'")

        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)  # Added missing pooling layer
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)  # Added missing pooling layer
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)  # Added missing pooling layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )

        # Decoder blocks
        self.dec1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Conv2d(64, out_dim, 1)

    def forward(self, x):
        # x: [B, 15, H, W]
        B, C, H, W = x.shape

        # Motion prompt
        motion_input = x.unsqueeze(2)  # [B, 15, 1, H, W]
        attention_map, _ = self.motion_prompt(motion_input)  # [B, 14, H, W]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bott = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
        d = self.dec1(torch.cat([self.up(bott), e3], dim=1))
        d = self.dec2(torch.cat([self.up(d), e2], dim=1))
        d = self.dec3(torch.cat([self.up(d), e1], dim=1))

        base = self.final_conv(d)  # [B, 15, H, W]

        # Fusion + sigmoid
        out = torch.sigmoid(self.fusion(base, attention_map))

        return out


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='VballNetV1b ONNX Exporter')
    parser.add_argument('--model_path', type=str, help='Path to the trained model checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For testing purposes, use CPU
    device = "cpu"

    model = VballNetV1b(in_dim=15, out_dim=15, fusion_type="TypeA").to(device)

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

    # Проверка компиляции и прохождения тензора
    dummy_input = torch.randn(2, 15, 288, 512).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 15, 288, 512)
    print("Модель успешно скомпилирована и прошла проверку формы выхода!")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего обучаемых параметров: {params:,}")

    # Export to ONNX if model_path is provided
    if args.model_path:
        try:
            model.eval()
            dummy_input = torch.randn(1, 15, 288, 512, device=device)
            onnx_filename = os.path.splitext(args.model_path)[0] + ".onnx"
            torch.onnx.export(
                model,
                (dummy_input,),
                onnx_filename,
                opset_version=18,
                input_names=["clip"],
                output_names=["heatmaps"],
                dynamic_axes={"clip": {0: "B"}, "heatmaps": {0: "B"}},
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
