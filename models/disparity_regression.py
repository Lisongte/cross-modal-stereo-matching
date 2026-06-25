import torch
import torch.nn as nn
import torch.nn.functional as F


def disparity_regression(prob_volume, disparity_values):
    """
    Differentiable disparity regression (soft argmin).
    D(u,v) = SUM_k d_k * P(d_k | u,v)
    """
    if disparity_values.dim() == 1:
        disparity_values = disparity_values.view(1, -1, 1, 1)
    return (prob_volume * disparity_values).sum(dim=1, keepdim=True)


def depth_regression(prob_volume, depth_values):
    return disparity_regression(prob_volume, depth_values)


class DepthRegressionModule(nn.Module):
    """
    Depth/Disparity Regression with learnable upsampling.
    Section III-D: Soft argmin + learnable upsampling + variance
    """
    def __init__(self, max_disparity=192, num_candidates=48, fine_channels=3):
        super().__init__()
        self.max_disparity = max_disparity
        self.num_candidates = num_candidates
        self.register_buffer('disparity_values', None)
        self.register_buffer('depth_values', None)
        self.upsample = LearnableUpsample(in_channels=1, guidance_channels=fine_channels)
        self.var_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Softplus(),
        )

    def set_disparity_values(self, max_disp=None, num_candidates=None):
        md = max_disp or self.max_disparity
        nc = num_candidates or self.num_candidates
        self.register_buffer('disparity_values', torch.linspace(0, md, nc))

    def set_depth_values(self, depth_min, depth_max, num_candidates=None):
        nc = num_candidates or self.num_candidates
        inv = torch.linspace(1.0/depth_max, 1.0/depth_min, nc)
        self.register_buffer('depth_values', 1.0 / inv)

    def forward(self, prob_volume, guidance_image=None, return_uncertainty=False):
        B, D, Hl, Wl = prob_volume.shape
        if self.depth_values is not None:
            vals = self.depth_values
        elif self.disparity_values is not None:
            vals = self.disparity_values
        else:
            vals = torch.linspace(0, self.max_disparity, D, device=prob_volume.device)
        dlr = depth_regression(prob_volume, vals)
        dm = self.upsample(dlr, guidance_image) if guidance_image is not None else F.interpolate(dlr, scale_factor=4, mode='bilinear', align_corners=True)
        if return_uncertainty:
            return dm, dlr, self.var_net(dm)
        return dm, dlr


class LearnableUpsample(nn.Module):
    """Guidance-based learnable upsampling (MaGNet [31] style)."""
    def __init__(self, in_channels=1, guidance_channels=3, feat_channels=32):
        super().__init__()
        self.guidance_net = nn.Sequential(
            nn.Conv2d(guidance_channels, feat_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_channels), nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_channels), nn.ReLU(inplace=True),
        )
        self.refine_net = nn.Sequential(
            nn.Conv2d(in_channels + feat_channels, feat_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_channels), nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feat_channels), nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, in_channels, 3, 1, 1),
            nn.Softplus(),
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    def forward(self, depth_lowres, guidance):
        gf = self.guidance_net(guidance)
        dup = F.interpolate(depth_lowres, size=guidance.shape[2:], mode='bilinear', align_corners=True)
        return dup + self.refine_net(torch.cat([dup, gf], dim=1))


class DepthModule(nn.Module):
    """
    Full Depth Module (Section III-D).
    Regularization → Softmax → Regression → Upsampling → Variance
    """
    def __init__(self, max_disparity=192, num_candidates=48, use_lightweight=False):
        super().__init__()
        from models.regularization import CostVolumeRegularization, CostVolumeRegularizationLight
        self.regularization = CostVolumeRegularizationLight(in_channels=1) if use_lightweight else CostVolumeRegularization(in_channels=1)
        self.regression = DepthRegressionModule(max_disparity=max_disparity, num_candidates=num_candidates)
    def set_disparity_values(self, md, nc): self.regression.set_disparity_values(md, nc)
    def set_depth_values(self, dmi, dma, nc): self.regression.set_depth_values(dmi, dma, nc)
    def forward(self, cost_volume, guidance_image=None):
        rcv, pv, a1, a2 = self.regularization(cost_volume)
        dm, dlr, var = self.regression(pv, guidance_image=guidance_image, return_uncertainty=True)
        return {'depth': dm, 'depth_lowres': dlr, 'variance': var, 'prob_volume': pv, 'aux1': a1, 'aux2': a2}


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    B, Dl, Hl, Wl = 2, 48, 48, 80
    H, W = Hl * 4, Wl * 4

    print("\n1. DepthRegressionModule")
    prob = torch.rand(B, Dl, Hl, Wl).to(dev); prob /= prob.sum(1, keepdim=True)
    gui = torch.rand(B, 3, H, W).to(dev)
    reg = DepthRegressionModule().to(dev); reg.set_disparity_values(192, 48)
    dm, dlr, var = reg(prob, gui, True)
    print(f"  depth: {dm.shape}, low-res: {dlr.shape}, var: {var.shape}, params: {sum(p.numel() for p in reg.parameters()):,}")

    print("\n2. DepthModule")
    cv = torch.randn(B, 1, Dl, Hl, Wl).to(dev)
    mod = DepthModule().to(dev); mod.set_disparity_values(192, 48)
    with torch.no_grad():
        o = mod(cv, gui)
    for k, v in o.items(): print(f"  {k}: {v.shape}")
    print(f"  params: {sum(p.numel() for p in mod.parameters()):,}")

    print("\n3. Soft argmin verification")
    p = torch.zeros(1, 5, 1, 1).to(dev); p[:, 2] = 1
    r = disparity_regression(p, torch.tensor([0.,1.,2.,3.,4.]).to(dev))
    assert abs(r[0,0,0,0].item() - 2.) < 1e-5; print("  One-hot @ 2: OK")

    pu = torch.ones(1, 5, 1, 1).to(dev) / 5
    ru = disparity_regression(pu, torch.tensor([0.,1.,2.,3.,4.]).to(dev))
    assert abs(ru[0,0,0,0].item() - 2.) < 1e-5; print("  Uniform: OK")

    print("\nAll tests passed!")
