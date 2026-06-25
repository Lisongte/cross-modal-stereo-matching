import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """Conv3d + BatchNorm3d + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CostVolumeRegularization(nn.Module):
    """
    3D Conv Regularization - stacked hourglass.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        c = 32  # base channels

        # === Hourglass 1 ===
        # Encoder: (1, D, H, W) -> (32, D/2, H/2, W/2) -> (64, D/4, H/4, W/4) -> (128, D/8, H/8, W/8)
        self.conv1a = Conv3DBlock(in_channels, c, stride=1)
        self.conv1b = Conv3DBlock(c, c, stride=2)
        self.conv1c = Conv3DBlock(c, c*2, stride=2)
        self.conv1d = Conv3DBlock(c*2, c*4, stride=2)
        self.bottleneck = nn.Sequential(
            Conv3DBlock(c*4, c*4),
            Conv3DBlock(c*4, c*4),
        )

        # Decoder - use dynamic skip convs
        # We'll use _make_skip to handle all channel reductions

        # Output classifier 1
        self.out1 = nn.Conv3d(16, 1, kernel_size=3, padding=1, bias=False)

        # === Hourglass 2 (shallower) ===
        self.conv2a = Conv3DBlock(in_channels+1, c, stride=2)
        self.conv2b = Conv3DBlock(c, c*2, stride=2)
        self.bottleneck2 = Conv3DBlock(c*2, c*2)

        # Output classifier 2
        self.out2 = nn.Conv3d(8, 1, kernel_size=3, padding=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _skip(self, x, in_ch, out_ch, name):
        if not hasattr(self, name):
            conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ).to(x.device)
            setattr(self, name, conv)
        return getattr(self, name)(x)

    def forward(self, cv):
        """
        cv: (B, 1, D, H, W)
        returns: (refined, prob, aux1, aux2)
        """
        B, _, D, H, W = cv.shape
        dev = cv.device

        # === HG1 ===
        e1 = self.conv1a(cv)        # 32, D, H, W
        e2 = self.conv1b(e1)        # 32, D/2, H/2, W/2
        e3 = self.conv1c(e2)        # 64, D/4, H/4, W/4
        e4 = self.conv1d(e3)        # 128, D/8, H/8, W/8
        b = self.bottleneck(e4)     # 128, D/8, H/8, W/8

        # Decode with upsample + skip
        d4 = F.interpolate(b, size=e4.shape[2:], mode='trilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)     # 256
        d4 = self._skip(d4, 256, 128, 's4') # 128

        d3 = F.interpolate(d4, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)     # 128+64=192
        d3 = self._skip(d3, 192, 64, 's3')  # 64

        d2 = F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)     # 64+32=96
        d2 = self._skip(d2, 96, 32, 's2')   # 32

        d1 = F.interpolate(d2, size=(D, H, W), mode='trilinear', align_corners=False)
        d1 = torch.cat([d1, cv], dim=1)     # 32+1=33
        d1 = self._skip(d1, 33, 16, 's1')   # 16

        refined1 = self.out1(d1)            # 1, D, H, W

        # === HG2 ===
        hg2_in = torch.cat([refined1, cv], dim=1)  # 2, D, H, W

        f1 = self.conv2a(hg2_in)  # 32, D/2, H/2, W/2
        f2 = self.conv2b(f1)      # 64, D/4, H/4, W/4
        fb = self.bottleneck2(f2) # 64, D/4, H/4, W/4

        g2 = F.interpolate(fb, size=f2.shape[2:], mode='trilinear', align_corners=False)
        g2 = torch.cat([g2, f2], dim=1)     # 128
        g2 = self._skip(g2, 128, 32, 't2')  # 32

        g1 = F.interpolate(g2, size=f1.shape[2:], mode='trilinear', align_corners=False)
        g1 = torch.cat([g1, f1], dim=1)     # 32+32=64
        g1 = self._skip(g1, 64, 16, 't1')   # 16

        g0 = F.interpolate(g1, size=(D, H, W), mode='trilinear', align_corners=False)
        g0 = torch.cat([g0, hg2_in], dim=1) # 16+2=18
        g0 = self._skip(g0, 18, 8, 't0')    # 8

        refined2 = self.out2(g0)            # 1, D, H, W

        refined = refined1 + refined2
        prob = F.softmax(refined.squeeze(1), dim=1)

        return refined, prob, refined1, refined2


class CostVolumeRegularizationLight(nn.Module):
    """Lightweight version for 4GB GPUs."""
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        self.enc1 = Conv3DBlock(in_channels, base_channels, stride=2)
        self.enc2 = Conv3DBlock(base_channels, base_channels*2, stride=2)
        self.bottleneck = Conv3DBlock(base_channels*2, base_channels*2)
        self.dec2 = nn.ConvTranspose3d(base_channels*2, base_channels*2, 3, 2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(base_channels*2)
        self.dec2_conv = Conv3DBlock(base_channels*4, base_channels)
        self.dec1 = nn.ConvTranspose3d(base_channels, base_channels, 3, 2, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.dec1_conv = Conv3DBlock(base_channels + in_channels, base_channels//2)
        self.out = nn.Conv3d(base_channels//2, 1, 3, 1, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cv):
        B, _, D, H, W = cv.shape
        e1 = self.enc1(cv)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = F.relu(self.bn2(self.dec2(b)))
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_conv(d2)
        d1 = F.relu(self.bn1(self.dec1(d2)))
        d1 = F.interpolate(d1, size=(D, H, W), mode='trilinear', align_corners=False)
        d1 = torch.cat([d1, cv], dim=1)
        d1 = self.dec1_conv(d1)
        refined = self.out(d1)
        prob = F.softmax(refined.squeeze(1), dim=1)
        return refined, prob


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    cv = torch.randn(2, 1, 48, 48, 80).to(device)

    for name, Model in [("Full", CostVolumeRegularization), ("Light", CostVolumeRegularizationLight)]:
        print(f"\n=== {name} ===")
        m = Model(in_channels=1).to(device)
        try:
            with torch.no_grad():
                if name == "Full":
                    ref, prob, a1, a2 = m(cv)
                    print(f"  refined: {ref.shape}, prob: {prob.shape}, aux: {a1.shape}, {a2.shape}")
                else:
                    ref, prob = m(cv)
                    print(f"  refined: {ref.shape}, prob: {prob.shape}")
            print(f"  params: {sum(p.numel() for p in m.parameters()):,}")
        except Exception as e:
            import traceback; traceback.print_exc()

    print("\nDone!")