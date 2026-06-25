"""
Simple RGB + NIR Depth Prediction Model.
Fuses features from both modalities, decodes to dense depth.
No stereo matching, no cost volume.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_extractor import FeatureExtractor


class SimpleDecoderBlock(nn.Module):
    """Conv -> BN -> ReLU x2"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SimpleDepthModel(nn.Module):
    """
    RGB + NIR -> Depth map.
    Input:  rgb (B,3,H,W), nir (B,1,H,W)
    Output: {'depth': (B,1,H,W), 'variance': (B,1,H,W)}
    """
    def __init__(self, feature_channels=32):
        super().__init__()
        self.encoder_rgb = FeatureExtractor(in_channels=3)
        self.encoder_nir = FeatureExtractor(in_channels=1)
        self.decoder_0 = SimpleDecoderBlock(64, 64, 64)
        self.decoder_1 = SimpleDecoderBlock(128, 64, 32)
        self.decoder_2 = SimpleDecoderBlock(36, 32, 16)
        self.depth_head = nn.Sequential(
            nn.Conv2d(20, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Softplus())
        self.var_head = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 1, 3, padding=1), nn.Softplus())

    def forward(self, rgb, nir, **kwargs):
        fr_start = self.encoder_rgb.conv_start(rgb)
        fr_feat  = self.encoder_rgb(rgb)
        fn_start = self.encoder_nir.conv_start(nir)
        fn_feat  = self.encoder_nir(nir)
        f_fused = torch.cat([fr_feat, fn_feat], dim=1)
        d0 = self.decoder_0(f_fused)
        d1 = F.interpolate(d0, scale_factor=2, mode='bilinear', align_corners=False)
        skip_half = torch.cat([fr_start, fn_start], dim=1)
        d1 = torch.cat([d1, skip_half], dim=1)
        d1 = self.decoder_1(d1)
        d2 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        raw = torch.cat([rgb, nir], dim=1)
        d2 = torch.cat([d2, raw], dim=1)
        d2 = self.decoder_2(d2)
        depth = self.depth_head(torch.cat([d2, raw], dim=1))
        variance = self.var_head(depth)
        return {'depth': depth, 'variance': variance}
