"""
UNet variants for image restoration.

Implementations:
- UNet-Lite: 180K params (simple encoder-decoder with skip connections)
- UNet-Standard: 650K params (moderate encoder-decoder)
- UNet-Dense: 1.2M params (with dense/residual shortcuts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock(nn.Module):
    """Convolution block: Conv -> GroupNorm -> GELU"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.norm = nn.GroupNorm(num_groups=max(1, out_ch // 8), num_channels=out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DoubleConvBlock(nn.Module):
    """Two consecutive convolution blocks with residual connection"""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch

        self.conv1 = ConvBlock(in_ch, mid_ch)
        self.conv2 = ConvBlock(mid_ch, out_ch)

        # Skip connection if channels match
        self.skip = in_ch == out_ch

    def forward(self, x):
        residual = x if self.skip else None
        x = self.conv1(x)
        x = self.conv2(x)
        if residual is not None:
            x = x + residual
        return x


class DownsampleBlock(nn.Module):
    """Downsampling via max pooling + double conv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpsampleBlock(nn.Module):
    """Upsampling via bilinear interpolation + convolution"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class UNetLite(nn.Module):
    """
    UNet-Lite: 180K parameters

    Lightweight UNet with:
    - 2 encoder levels
    - 1 bottleneck
    - 2 decoder levels
    - Skip connections
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
    ):
        super().__init__()

        self.stem = ConvBlock(in_channels, base_channels)

        # Encoder
        self.enc1 = DoubleConvBlock(base_channels, base_channels)
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)

        self.enc2 = DoubleConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = DoubleConvBlock(base_channels * 4, base_channels * 4)

        # Decoder
        self.up1 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.dec1 = DoubleConvBlock(base_channels * 4, base_channels * 2)

        self.up2 = UpsampleBlock(base_channels * 2, base_channels)
        self.dec2 = DoubleConvBlock(base_channels * 2, base_channels)

        # Output
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Encoder with skip connection storage
        skip1 = self.enc1(x)
        x = self.down1(skip1)

        skip2 = self.enc2(x)
        x = self.down2(skip2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2(x)

        # Output
        x = self.head(x)
        x = self.out_act(x)

        return x


class UNetStandard(nn.Module):
    """
    UNet-Standard: 650K parameters

    Moderate UNet with:
    - 3 encoder levels
    - 1 bottleneck
    - 3 decoder levels
    - Skip connections
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 48,
    ):
        super().__init__()

        self.stem = DoubleConvBlock(in_channels, base_channels)

        # Encoder
        self.enc1 = DoubleConvBlock(base_channels, base_channels)
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)

        self.enc2 = DoubleConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)

        self.enc3 = DoubleConvBlock(base_channels * 4, base_channels * 4)
        self.down3 = DownsampleBlock(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = DoubleConvBlock(base_channels * 8, base_channels * 8)

        # Decoder
        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4)
        self.dec1 = DoubleConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.dec2 = DoubleConvBlock(base_channels * 4, base_channels * 2)

        self.up3 = UpsampleBlock(base_channels * 2, base_channels)
        self.dec3 = DoubleConvBlock(base_channels * 2, base_channels)

        # Output
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Encoder
        skip1 = self.enc1(x)
        x = self.down1(skip1)

        skip2 = self.enc2(x)
        x = self.down2(skip2)

        skip3 = self.enc3(x)
        x = self.down3(skip3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec3(x)

        # Output
        x = self.head(x)
        x = self.out_act(x)

        return x


class DenseBlock(nn.Module):
    """Dense block: multiple parallel paths with concatenation"""

    def __init__(self, in_ch: int, out_ch: int, num_paths: int = 3):
        super().__init__()
        self.paths = nn.ModuleList([ConvBlock(in_ch, out_ch // num_paths) for _ in range(num_paths)])

    def forward(self, x):
        outputs = [path(x) for path in self.paths]
        return torch.cat(outputs, dim=1)


class UNetDense(nn.Module):
    """
    UNet-Dense: 1.2M parameters

    Dense UNet with:
    - 3 encoder levels with dense shortcuts
    - 1 bottleneck
    - 3 decoder levels
    - DenseNet-inspired dense connections
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        self.stem = DoubleConvBlock(in_channels, base_channels)

        # Encoder with dense connections
        self.enc1 = nn.Sequential(
            DoubleConvBlock(base_channels, base_channels),
            DenseBlock(base_channels, base_channels, num_paths=2),
        )
        self.down1 = DownsampleBlock(base_channels * 2, base_channels * 2)

        self.enc2 = nn.Sequential(
            DoubleConvBlock(base_channels * 2, base_channels * 2),
            DenseBlock(base_channels * 2, base_channels * 2, num_paths=2),
        )
        self.down2 = DownsampleBlock(base_channels * 3, base_channels * 4)

        self.enc3 = nn.Sequential(
            DoubleConvBlock(base_channels * 4, base_channels * 4),
            DenseBlock(base_channels * 4, base_channels * 4, num_paths=2),
        )
        self.down3 = DownsampleBlock(base_channels * 6, base_channels * 8)

        # Bottleneck
        self.bottleneck = DoubleConvBlock(base_channels * 8, base_channels * 8)

        # Decoder
        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4)
        self.dec1 = DoubleConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.dec2 = DoubleConvBlock(base_channels * 4, base_channels * 2)

        self.up3 = UpsampleBlock(base_channels * 2, base_channels)
        self.dec3 = DoubleConvBlock(base_channels * 2, base_channels)

        # Output
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Encoder
        skip1 = self.enc1(x)
        x = self.down1(skip1)

        skip2 = self.enc2(x)
        x = self.down2(skip2)

        skip3 = self.enc3(x)
        x = self.down3(skip3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec3(x)

        # Output
        x = self.head(x)
        x = self.out_act(x)

        return x


if __name__ == "__main__":
    # Test UNet variants
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 3, 256, 256).to(device)

    for model_class, name in [(UNetLite, "UNet-Lite"), (UNetStandard, "UNet-Standard"), (UNetDense, "UNet-Dense")]:
        model = model_class().to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        y = model(x)
        print(f"{name}: {params / 1e3:.1f}K params, output shape: {y.shape}")
