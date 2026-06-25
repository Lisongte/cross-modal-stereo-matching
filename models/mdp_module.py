"""
MDP (Modality-specific Depth Probability) Module.
Paper: Section III-C, Eq. 5.

Predicts per-pixel depth as a Gaussian distribution P(d) = N(μ, σ²).
Used for:
  - MDP-VIS: depth probability → Degradation Masking (remove unreliable matches)
  - MDP-THR: depth probability + last-layer features → concatenate with cost volume
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_extractor import FeatureExtractor


class MDPModule(nn.Module):
    """
    Modality-specific Depth Probability module.

    Architecture:
      Encoder: PSMNet-style FeatureExtractor (output 32ch @ H/4)
      Decoder: Conv2d + Upsample ×2 → 32ch @ H
      Heads:
        - mu_head:    depth mean (m)
        - sigma_head: depth variance σ² (Softplus, always positive)

    Input:  Image tensor (B, C, H, W)  C=3 for RGB, C=1 for NIR
    Output: dict with mu, sigma2, feat_last
    """

    def __init__(self, in_channels=3, feature_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.feature_channels = feature_channels

        # Encoder: PSMNet backbone (output 32ch @ H/4)
        self.encoder = FeatureExtractor(in_channels=in_channels)

        # Decoder: 32ch @ H/4 → 32ch @ H  (4x upscale in 2 steps)
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(feature_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        # Depth mean head: output in meters
        self.mu_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Softplus(),  # depth is positive
        )

        # Depth variance head: output σ² (always positive via Softplus)
        self.sigma_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Softplus(),  # σ² > 0
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input image (B, C, H, W)

        Returns:
            dict:
                'mu':        Depth mean map (B, 1, H, W) in meters
                'sigma2':    Depth variance map (B, 1, H, W) always positive
                'feat_last': Decoder features (B, 32, H, W) for cost volume concatenation
        """
        feat = self.encoder(x)                    # (B, 32, H/4, W/4)
        dec1 = self.decoder_block1(feat)          # (B, 64, H/2, W/2)
        dec2 = self.decoder_block2(dec1)          # (B, 32, H, W)
        mu = self.mu_head(dec2)                   # (B, 1, H, W)
        sigma2 = self.sigma_head(dec2)            # (B, 1, H, W)
        return {
            'mu': mu,
            'sigma2': sigma2,
            'feat_last': dec2,                    # (B, 32, H, W)
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B, C, H, W = 2, 3, 384, 640
    rgb = torch.randn(B, C, H, W).to(device)
    nir = torch.randn(B, 1, H, W).to(device)

    for name, in_ch in [("RGB MDP", 3), ("NIR MDP", 1)]:
        model = MDPModule(in_channels=in_ch).to(device)
        inp = rgb if in_ch == 3 else nir
        with torch.no_grad():
            out = model(inp)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{name} (params={params:,}):")
        print(f"  Input:      {inp.shape}")
        print(f"  mu:         {out['mu'].shape}  range=[{out['mu'].min().item():.2f}, {out['mu'].max().item():.2f}]")
        print(f"  sigma2:     {out['sigma2'].shape}  range=[{out['sigma2'].min().item():.4f}, {out['sigma2'].max().item():.4f}]")
        print(f"  feat_last:  {out['feat_last'].shape}")

    print("\nAll tests passed!")