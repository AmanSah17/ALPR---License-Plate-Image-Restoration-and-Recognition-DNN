"""
Spatiotemporal Hybrid Attention model for multi-frame sequence restoration.

This model utilizes an Early Fusion strategy. By stacking T frames along the
channel dimension, the first convolution explicitly learns the temporal
correlations and optical shifts between neighboring pixels across frames.
The deep SwinIR and Spatial/Channel Attention blocks then operate on this
fused spatiotemporal feature representation, producing a single highly-detailed
restored center frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Be modular: reuse the AttentionBlock from the 2D Hybrid Attention model
from .hybrid_attention import AttentionBlock


class SpatiotemporalHybridRestoration(nn.Module):
    """
    Spatiotemporal Hybrid Attention Restoration Model.

    Architecture:
    - Early Fusion Stem: Conv layers reducing (T * 3) channels to base_channels
    - Encoder: Attention-gated blocks with downsampling
    - Bottleneck: Multi-level attention
    - Decoder: Attention-gated blocks with upsampling
    - Head: Output convolution producing 3 channels (the restored center frame)
    - Global residual learning: Output is added to the center frame of the input sequence.
    """

    def __init__(
        self,
        in_channels: int = 15,  # Default for sequence_length=5 (5 * 3 channels)
        out_channels: int = 3,
        base_channels: int = 56,
        num_blocks: int = 4,
        window_size: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Stem (Early Temporal Fusion)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, base_channels // 8), num_channels=base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, base_channels // 8), num_channels=base_channels),
            nn.GELU(),
        )

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.encoder_blocks.append(
                AttentionBlock(base_channels, window_size, num_heads)
            )

        # Downsampling
        self.down = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, base_channels * 2 // 8), num_channels=base_channels * 2),
            nn.GELU(),
        )

        # Bottleneck (multi-level attention)
        self.bottleneck = nn.ModuleList(
            [
                AttentionBlock(base_channels * 2, window_size, num_heads)
                for _ in range(num_blocks // 2)
            ]
        )

        # Upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=max(1, base_channels // 8), num_channels=base_channels),
            nn.GELU(),
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.decoder_blocks.append(
                AttentionBlock(base_channels, window_size, num_heads)
            )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=max(1, base_channels // 8), num_channels=base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x is (B, 15, H, W) for sequence_length=5
        # The center frame is located at channels [6, 7, 8] if length is 5.
        
        sequence_length = x.shape[1] // 3
        center_idx = sequence_length // 2
        
        # Extract the original center frame for global residual connection
        center_frame = x[:, center_idx * 3 : (center_idx + 1) * 3, :, :]

        # Stem handles the spatiotemporal early fusion
        feats = self.stem(x)

        # Encoder
        skip = feats
        for block in self.encoder_blocks:
            feats = block(feats)
        feats = self.down(feats)

        # Bottleneck
        for block in self.bottleneck:
            feats = block(feats)

        # Decoder
        feats = self.up(feats)
        
        # Match odd/even spatial sizes after down/up (variable RLPR HxW)
        if feats.shape[-2:] != skip.shape[-2:]:
            feats = F.interpolate(feats, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            
        feats = feats + skip  # Skip connection

        for block in self.decoder_blocks:
            feats = block(feats)

        # Head with global residual using the center frame
        out = self.head(feats)
        out = torch.clamp(out + center_frame, 0, 1)

        return out


if __name__ == "__main__":
    # Test spatiotemporal hybrid attention model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # B=1, T=5 -> 15 channels, H=52, W=232
    x = torch.randn(1, 15, 52, 232).to(device)

    model = SpatiotemporalHybridRestoration().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"SpatiotemporalHybridRestoration: {params / 1e3:.1f}K params")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    assert y.shape == (1, 3, 52, 232), "Output shape mismatch"
    assert y.min() >= 0 and y.max() <= 1, "Output values out of range [0,1]"
    print("✓ Model test passed")
