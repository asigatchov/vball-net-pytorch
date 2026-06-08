import torch
import torch.nn as nn
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


class VballNetGridV1c(nn.Module):
    """Grid model for 9-frame grayscale stacks using depthwise separable blocks."""
    def __init__(self, input_height=432, input_width=768, in_dim=9, out_dim=27):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.features = nn.Sequential(
            DSConvBlock(in_dim, 64),
            DSConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            DSConvBlock(64, 128),
            DSConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            DSConvBlock(128, 256),
            DSConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            DSConvBlock(256, 512),
            DSConvBlock(512, 512),
            DSConvBlock(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            DSConvBlock(512, 512),
            DSConvBlock(512, 512),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(512, out_dim, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


def fuse_model(module):
    """Улучшенная функция фьюжинга Conv + BN"""
    fused_pairs = 0
    
    for name, child in module.named_children():
        if isinstance(child, DSConvBlock):
            if isinstance(child.dsconv.pointwise, nn.Conv2d) and isinstance(child.bn, nn.BatchNorm2d):
                child.dsconv.pointwise = fuse_conv_bn_eval(child.dsconv.pointwise, child.bn)
                child.bn = nn.Identity()
                fused_pairs += 1
                
        elif isinstance(child, nn.Sequential):
            i = 0
            while i < len(child) - 1:
                if isinstance(child[i], nn.Conv2d) and isinstance(child[i + 1], nn.BatchNorm2d):
                    child[i] = fuse_conv_bn_eval(child[i], child[i + 1])
                    child[i + 1] = nn.Identity()
                    fused_pairs += 1
                    i += 2
                else:
                    i += 1
                    
        fused_pairs += fuse_model(child)
    
    return fused_pairs
