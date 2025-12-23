import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os


# --- Утилиты ---
def rearrange_tensor(input_tensor, order="BTCHW"):
    """
    Перестановка размерностей тензора: из любого порядка B,C,H,W,T → BTCHW.
    """
    order = order.upper()
    perm = [order.index(dim) for dim in "BTCHW"]
    return input_tensor.permute(*perm)


def power_normalization(input_tensor, a, b):
    """
    Power normalization для motion attention (как в оригинальной TF-версии).
    """
    return 1 / (1 + torch.exp(-(5 / (0.45 * torch.abs(torch.tanh(a)) + 1e-1)) *
                              (torch.abs(input_tensor) - 0.6 * torch.tanh(b))))


# --- Depthwise Separable Convolution Block ---
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise (groups=in_channels) + Pointwise (1x1) свёртка.
    Замена обычного Conv2d(3x3) для значительного снижения параметров.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, stride=stride, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# --- Spatial Attention Module (как в TF-версии) ---
class SpatialAttention(nn.Module):
    """
    CBAM-like spatial attention:
    avg_pool + max_pool → concat → 7x7 conv → sigmoid
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        concat = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        attention = self.conv(concat)                   # (B, 1, H, W)
        return x * self.sigmoid(attention)


# --- Motion Prompt Module ---
class MotionPrompt(nn.Module):
    def __init__(self, num_frames, mode="grayscale", penalty_weight=0.0):
        super().__init__()
        self.num_frames = num_frames
        self.mode = mode.lower()
        assert self.mode in ["rgb", "grayscale"]
        self.gray_scale = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.lambda1 = penalty_weight

    def forward(self, video_seq):
        loss = torch.tensor(0.0, device=video_seq.device)
        # video_seq is already in BTCHW format: (B, T, H, W)
        norm_seq = video_seq * 0.225 + 0.45

        if self.mode == "rgb":
            # For RGB mode, we assume input is (B, T*3, H, W) and reshape to (B, T, 3, H, W)
            B, TC, H, W = norm_seq.shape
            T = self.num_frames
            norm_seq = norm_seq.view(B, T, 3, H, W)
            weights = self.gray_scale.to(norm_seq.device)
            grayscale_seq = torch.einsum("btcwh,c->btwh", norm_seq, weights)
        else:
            # For grayscale mode, input is (B, T, H, W)
            grayscale_seq = norm_seq  # already in correct format (B, T, H, W)

        attention_maps = []
        for t in range(self.num_frames):
            if t == 0:
                diff = grayscale_seq[:, t + 1] - grayscale_seq[:, t]
            elif t == self.num_frames - 1:
                diff = grayscale_seq[:, t] - grayscale_seq[:, t - 1]
            else:
                diff = (grayscale_seq[:, t + 1] - grayscale_seq[:, t - 1]) / 2.0
            attention_maps.append(power_normalization(diff, self.a, self.b))

        attention_map = torch.stack(attention_maps, dim=1)  # (B, T, H, W)

        if self.training and self.lambda1 > 0:
            norm_att = attention_map.unsqueeze(2)  # для temporal loss
            temp_diff = norm_att[:, 1:] - norm_att[:, :-1]
            B, T, _, H, W = grayscale_seq.shape
            temporal_loss = torch.sum(temp_diff ** 2) / (H * W * (T - 1) * B)
            loss = self.lambda1 * temporal_loss

        return attention_map, loss


# --- Fusion Layer Type A (по-кадровое умножение) ---
class FusionLayerTypeA(nn.Module):
    def __init__(self, num_frames, out_dim):
        super().__init__()
        self.num_frames = num_frames
        self.out_dim = out_dim

    def forward(self, feature_map, attention_map):
        # feature_map: (B, out_dim, H, W), attention_map: (B, T, H, W)
        # Берём attention для каждого кадра
        return feature_map * attention_map[:, :self.out_dim]


# --- Улучшенная VballNetV1a ---
class VballNetV2(nn.Module):
    def __init__(self, height=288, width=512, in_dim=15, out_dim=15):
        super().__init__()
        self.height = height
        self.width = width
        mode = "grayscale" if in_dim == out_dim else "rgb"
        num_frames = in_dim if mode == "grayscale" else in_dim // 3

        self.motion_prompt = MotionPrompt(num_frames=num_frames, mode=mode)
        self.fusion_layer = FusionLayerTypeA(num_frames=num_frames, out_dim=out_dim)

        # Encoder с DepthwiseSeparableConv
        self.enc1 = DepthwiseSeparableConv(in_dim, 32)
        self.enc1_1 = DepthwiseSeparableConv(32, 32)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = DepthwiseSeparableConv(32, 64)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc3 = DepthwiseSeparableConv(64, 128)

        # Spatial Attention в bottleneck (как в TF-версии)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DepthwiseSeparableConv(128 + 64, 64)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DepthwiseSeparableConv(64 + 32, 32)

        self.out_conv = nn.Conv2d(32, out_dim, kernel_size=1)

    def forward(self, x):
        B, TC, H, W = x.shape  # TC = total channels (T for grayscale, T*3 for RGB)
        assert H == self.height and W == self.width, f"Expected ({self.height}, {self.width}), got ({H}, {W})"
        
        # Determine actual number of frames based on mode
        if self.motion_prompt.mode == "grayscale":
            T = TC  # For grayscale, T is the same as TC
        else:  # RGB mode
            T = TC // 3  # For RGB, TC = T * 3, so actual frames = TC / 3
        
        assert T == self.motion_prompt.num_frames

        # Motion attention
        motion_maps, _ = self.motion_prompt(x)  # (B, T, H, W)

        # Encoder
        x1 = self.enc1(x)
        x1_skip = self.enc1_1(x1)       # skip connection
        x = self.pool1(x1_skip)

        x2 = self.enc2(x)               # skip connection
        x = self.pool2(x2)

        x = self.enc3(x)
        x = self.spatial_attention(x)   # ← добавлено!

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, x1_skip], dim=1)
        x = self.dec2(x)

        x = self.out_conv(x)
        x = self.fusion_layer(x, motion_maps)
        x = torch.sigmoid(x)

        return x


# --- Main блок ---
if __name__ == "__main__":
    height, width = 288, 512
    in_dim, out_dim = 15, 15  # дефолт: 15 grayscale кадров → 15 heatmaps

    model = VballNetV2(height=height, width=width, in_dim=in_dim, out_dim=out_dim)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Модель VballNetV2 инициализирована")
    print(f"Параметры: {total_params:,} (ожидается ~120–150k благодаря DepthwiseSeparable)")

    # Тестовый прогон
    device = torch.device("cpu")  # для стабильности экспорта
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, in_dim, height, width, device=device)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape:  {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Аргументы для экспорта
    parser = argparse.ArgumentParser(description='VballNetV2 → ONNX экспорт')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Путь к обученному чекпоинту (.pth). Если указан — загрузить и экспортировать в ONNX')
    args = parser.parse_args()

    if args.model_path:
        print(f"Загрузка весов из: {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Веса успешно загружены")
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            exit(1)

        onnx_path = os.path.splitext(args.model_path)[0] + ".onnx"
        print(f"Экспорт в ONNX: {onnx_path}")

        try:
            torch.onnx.export(
                model,
                (dummy_input,),
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["clip"],
                output_names=["heatmaps"],
                dynamic_axes={"clip": {0: "B"}, "heatmaps": {0: "B"}},
                verbose=False
            )
            print("ONNX модель успешно сохранена!")
        except Exception as e:
            print(f"Ошибка экспорта ONNX: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Запуск без --model_path → только тест инициализации. Для экспорта укажите путь к чекпоинту.")