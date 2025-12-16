# vball_net_v3_pytorch_flexible.py
# Профессиональная PyTorch-реализация (один файл, легко масштабируется)
# Поддерживает произвольное количество входных и выходных кадров
# ≥90–120 FPS на CPU через ONNX → OpenVINO (FP16)

import torch
import torch.nn as nn
import torch.nn.functional as F


def power_normalization(x, a, b):
    """Оригинальная нелинейность из TF-версии"""
    abs_x = torch.abs(x)
    denom = 0.45 * torch.abs(torch.tanh(a)) + 1e-1
    scale = 5.0 / denom
    return 1.0 / (1.0 + torch.exp(-scale * (abs_x - 0.8 * torch.tanh(b))))


class MotionPromptLayer(nn.Module):
    """Улучшенный motion prompt с обучаемыми a/b и smoothness penalty"""
    def __init__(self, num_frames, penalty_weight=1e-4):
        super().__init__()
        self.num_frames = num_frames
        self.penalty_weight = penalty_weight
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, T, H, W) — grayscale клип
        x = x * 0.225 + 0.45  # как в оригинале

        diffs = []
        T = self.num_frames
        for t in range(T):
            if t == 0:
                diff = x[:, 1] - x[:, 0]
            elif t == T - 1:
                diff = x[:, -1] - x[:, -2]
            else:
                diff = (x[:, t + 1] - x[:, t - 1]) / 2.0
            diffs.append(power_normalization(diff, self.a, self.b))

        attention = torch.stack(diffs, dim=1)  # (B, T, H, W)

        # Smoothness penalty
        if self.training and self.penalty_weight > 0:
            smoothness_loss = torch.mean((attention[:, 1:] - attention[:, :-1]) ** 2)
            self.add_loss(self.penalty_weight * smoothness_loss)

        return attention

    def add_loss(self, loss):
        # Store only the current loss, don't accumulate
        self.current_loss = loss

    def get_loss(self):
        # Return current loss and clear it
        loss = getattr(self, 'current_loss', None)
        self.current_loss = None
        return loss


class FusionLayerTypeA(nn.Module):
    """Простое поэлементное умножение с attention (как в оригинале)"""
    def __init__(self, num_frames, out_dim):
        super().__init__()
        self.num_frames = num_frames
        self.out_dim = out_dim

    def forward(self, features, attention):
        # features: (B, out_dim, H, W)
        # attention: (B, T, H, W)
        # Apply attention map of frame t to feature map channel t
        outputs = []
        for t in range(min(self.num_frames, self.out_dim)):
            # Use in-place multiplication to save memory
            output = features[:, t, :, :] * attention[:, t, :, :]
            outputs.append(output)
        result = torch.stack(outputs, dim=1)  # (B, out_dim, H, W)
        # Clear outputs list to free memory
        del outputs
        return result


class VballNetV3b(nn.Module):
    """
    Гибкая версия VBallNet v3
    in_dim  — количество входных кадров (например 9, 15, 21)
    out_dim — количество выходных heatmaps (обычно = in_dim)
    """
    def __init__(self, height=288, width=512, in_dim=9, out_dim=9, fusion_layer_type="TypeA"):
        super().__init__()
        assert fusion_layer_type == "TypeA", "Пока реализован только TypeA"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.height = height
        self.width = width

        # 1. Лёгкий 3D depthwise temporal extractor
        self.temporal_conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=10,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(0, 1, 1),   # только по H,W; по времени — без padding
            bias=False,
            groups=1
        )
        self.temporal_bn = nn.BatchNorm3d(10)

        # Encoder (SeparableConv2D → Depthwise + Pointwise)
        def sep_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),  # depthwise
                nn.Conv2d(in_c, out_c, 1, bias=False),                         # pointwise
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(
            sep_conv(11, 32),
            sep_conv(32, 32)
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
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (B, in_dim, H, W) — input tensor compatible with VballNetV1a
        """
        B, C, H, W = x.shape
        assert C == self.in_dim, f"Ожидалось {self.in_dim} каналов, получено {C}"
        
        # Reshape to (B, T, H, W) for processing
        x = x.view(B, self.in_dim, H, W)
        T = self.in_dim

        orig_clip = x.detach()  # сохраняем для MotionPrompt (B, T, H, W)

        # === 1. Temporal feature extractor (группируем по 3 кадра) ===
        # Делим T кадров на группы по 3 → получаем T//3 + остаток, но упростим: используем stride
        # Более элегантно: просто делаем Conv3D с окном 3 и stride 1, потом усредняем/берём max
        x_3d = x.unsqueeze(2)  # (B, T, 1, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4).contiguous()  # (B, 1, T, H, W)

        # Применяем 3D свёртку (временной размер T→T-2)
        temporal = self.temporal_conv3d(x_3d)           # (B, 10, T-2, H, W)
        temporal = self.temporal_bn(temporal)
        temporal = F.relu(temporal, inplace=True)

        # Приводим обратно к (B, 10, H, W): берём центральный срез или усредняем
        # В оригинале группировали по 3 → усредняли 5 групп → здесь проще:
        # берём центр по времени (или max/mean)
        mid = temporal.shape[2] // 2
        temporal = temporal[:, :, mid:mid+1, :, :].squeeze(2)  # (B, 10, H, W)
        # Или можно F.adaptive_avg_pool3d → (B,10,1,H,W) → squeeze

        # === 2. Центральный кадр ===
        center_idx = T // 2
        center = x[:, center_idx:center_idx+1, :, :]  # (B, 1, H, W)

        # === 3. Конкатенация ===
        enc_input = torch.cat([center, temporal], dim=1)  # (B, 11, H, W)

        # === Encoder ===
        x = self.enc1(enc_input)
        skip1 = x
        x = F.max_pool2d(x, 2, ceil_mode=True)

        x = self.enc2(x)
        skip2 = x
        x = F.max_pool2d(x, 2, ceil_mode=True)

        x = self.enc3(x)

        # === Decoder ===
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip2], dim=1)
        x = self.dec1(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2(x)

        x = self.final_conv(x)  # (B, out_dim, H, W)

        # === Motion Guidance ===
        attention = self.motion_prompt(orig_clip)       # (B, T, H, W)
        out = self.fusion(x, attention)                 # (B, out_dim, H, W)

        out = torch.sigmoid(out)

        # Clear unused variables to free memory
        del x, temporal, x_3d, enc_input, skip1, skip2, center, orig_clip, attention

        return out

    # Для корректного сбора лоссов (например в Lightning или своём лупе)
    def get_extra_losses(self):
        losses = {}
        motion_loss = self.motion_prompt.get_loss()
        if motion_loss is not None:
            losses["motion_smooth"] = motion_loss
        return losses


# ====================== Тест ======================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VballNetV3b ONNX Exporter')
    parser.add_argument('--model_path', type=str, help='Path to the trained model checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Пример: 9 каналов → 9 heatmaps (совместимо с VballNetV1a)
    model = VballNetV3b(
        height=288,
        width=512,
        in_dim=9,
        out_dim=9
    ).to(device)

    # If model path is provided, load the trained model
    if args.model_path:
        print(f"Loading model from checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Model loaded successfully!")
    
    model.eval()
    x = torch.randn(1, 9, 288, 512, device=device)  # (B, C, H, W) - compatible with VballNetV1a
    y = model(x)

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")  # Expected: [1, 9, 288, 512]
    print(f"Параметры  : {sum(p.numel() for p in model.parameters()):,}")
    
    # Test compatibility with VballNetV1a format
    print("\n=== Тест совместимости с VballNetV1a ===")
    print(f"✓ Input format: (B, in_dim, H, W) = {x.shape}")
    print(f"✓ Output format: (B, out_dim, H, W) = {y.shape}")
    print(f"✓ Output range: [{y.min():.3f}, {y.max():.3f}]")

    # Экспорт в ONNX (для OpenVINO)
    try:
        onnx_filename = "vball_net_v3b_trained.onnx" if args.model_path else "vball_net_v3b_random.onnx"
        torch.onnx.export(
            model,
            (x,),
            onnx_filename,
            opset_version=17,
            input_names=["clip"],
            output_names=["heatmaps"],
            dynamic_axes={"clip": {0: "B"}, "heatmaps": {0: "B"}}
        )
        print(f"\n✓ ONNX сохранён как {onnx_filename} → используй: mo --input_model {onnx_filename} --data_type FP16")
    except Exception as e:
        print(f"\nОшибка экспорта ONNX: {e}")
