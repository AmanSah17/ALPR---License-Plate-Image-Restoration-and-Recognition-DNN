"""
ResNet variants for image restoration.

Implementations:
- RestorationResNet-Small: 250K params (lightweight residual network)
- RestorationResNet-Medium: 750K params (moderate residual network with grouped convolutions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ResidualBlock(nn.Module):
    """Residual block with grouped convolutions and GroupNorm"""

    def __init__(self, channels: int, groups: int = 1, expansion: float = 1.0):
        super().__init__()

        hidden_dim = int(channels * expansion)

        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, groups=groups, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=max(1, hidden_dim // 8), num_channels=hidden_dim)
        self.act = nn.GELU()

        self.conv2 = nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1, groups=groups, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=max(1, channels // 8), num_channels=channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        return x


class BottleneckBlock(nn.Module):
    """Bottleneck block with grouped convolutions"""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1, stride: int = 1):
        super().__init__()

        expansion = 4
        hidden_dim = max(1, out_channels // expansion)

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=max(1, hidden_dim // 8), num_channels=hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=max(1, hidden_dim // 8), num_channels=hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels)

        self.act = nn.GELU()

        # Shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)

        out = out + identity
        out = self.act(out)
        return out


class DownsampleBlock(nn.Module):
    """Downsampling block (stride 2)"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class UpsampleBlock(nn.Module):
    """Upsampling block (scale factor 2)"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=max(1, out_channels // 8), num_channels=out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class RestorationResNetSmall(nn.Module):
    """
    Lightweight ResNet for restoration: 250K parameters

    Architecture:
    - Stem: Conv -> GroupNorm -> GELU
    - 3 residual blocks at base resolution
    - 2 residual blocks at 2x downsampled
    - 2 residual blocks upsampled back
    - Skip connections
    - Global residual learning
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        num_blocks: int = 3,
        groups: int = 1,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, base_channels // 8), num_channels=base_channels),
            nn.GELU(),
        )

        # Encoder
        self.enc_blocks = nn.ModuleList(
            [ResidualBlock(base_channels, groups=groups) for _ in range(num_blocks)]
        )
        self.down = DownsampleBlock(base_channels, base_channels * 2)

        # Bottleneck
        self.bottleneck = nn.ModuleList(
            [ResidualBlock(base_channels * 2, groups=groups) for _ in range(num_blocks // 2)]
        )

        # Decoder
        self.up = UpsampleBlock(base_channels * 2, base_channels)
        self.dec_blocks = nn.ModuleList(
            [ResidualBlock(base_channels, groups=groups) for _ in range(num_blocks)]
        )

        # Output
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Store input for residual learning
        x_input = x

        # Stem
        x = self.stem(x)

        # Encoder
        skip = x
        for block in self.enc_blocks:
            x = block(x)
        x = self.down(x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x)

        # Decoder
        x = self.up(x)
        x = x + skip  # Skip connection
        for block in self.dec_blocks:
            x = block(x)

        # Output with global residual
        x = self.head(x)
        x = torch.clamp(x + x_input, 0, 1)  # Global residual + clamp to [0,1]

        return x


class RestorationResNetMedium(nn.Module):
    """
    Moderate ResNet for restoration: 750K parameters

    Architecture:
    - Stem: Conv -> GroupNorm -> GELU
    - Bottleneck blocks with grouped convolutions
    - Multi-scale encoder-decoder
    - Skip connections at each level
    - Global residual learning
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 48,
        num_levels: int = 3,
        groups: int = 2,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, base_channels // 8), num_channels=base_channels),
            nn.GELU(),
        )

        # Multi-scale encoder
        self.encoder_levels = nn.ModuleList()
        self.downsample_levels = nn.ModuleList()

        in_ch = base_channels
        for level in range(num_levels):
            out_ch = base_channels * (2 ** level)
            self.encoder_levels.append(
                nn.Sequential(
                    *[BottleneckBlock(in_ch if i == 0 else out_ch, out_ch, groups=groups) for i in range(2)]
                )
            )
            self.downsample_levels.append(DownsampleBlock(out_ch, out_ch * 2))
            in_ch = out_ch * 2

        # Bottleneck
        bottleneck_ch = base_channels * (2 ** num_levels)
        self.bottleneck = nn.Sequential(
            *[BottleneckBlock(bottleneck_ch, bottleneck_ch, groups=groups) for _ in range(2)]
        )

        # Multi-scale decoder
        self.decoder_levels = nn.ModuleList()
        self.upsample_levels = nn.ModuleList()

        for level in range(num_levels - 1, -1, -1):
            out_ch = base_channels * (2 ** level)
            in_ch = base_channels * (2 ** (level + 1))

            self.upsample_levels.append(UpsampleBlock(in_ch, out_ch))
            self.decoder_levels.append(
                nn.Sequential(
                    *[BottleneckBlock(out_ch * 2, out_ch, groups=groups) for _ in range(2)]
                )
            )

        # Output
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_input = x

        # Stem
        x = self.stem(x)

        # Encoder with skip connections
        skip_connections = []
        for level_idx, (enc_block, down_block) in enumerate(
            zip(self.encoder_levels, self.downsample_levels)
        ):
            x = enc_block(x)
            skip_connections.append(x)
            x = down_block(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for level_idx, (up_block, dec_block) in enumerate(
            zip(self.upsample_levels, self.decoder_levels)
        ):
            x = up_block(x)
            skip = skip_connections[-(level_idx + 1)]
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        # Output with global residual
        x = self.head(x)
        x = torch.clamp(x + x_input, 0, 1)

        return x


if __name__ == "__main__":
    # Test ResNet variants
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 3, 256, 256).to(device)

    for model_class, name in [
        (RestorationResNetSmall, "ResNet-Small"),
        (RestorationResNetMedium, "ResNet-Medium"),
    ]:
        model = model_class().to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        y = model(x)
        print(f"{name}: {params / 1e3:.1f}K params, output shape: {y.shape}, values in [0,1]: {y.min():.3f}-{y.max():.3f}")
