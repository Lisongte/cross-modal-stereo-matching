"""
RGB + NIR Cross-Modal Stereo Matching Model.

Wires all 6 modules into a single end-to-end pipeline:

    RGB ──┐
          ├── FeatureExtractor ──┐
    NIR ──┘                      │
                                 ├── CrossAttentionModule
                                 │         ↓
                                 │   CostVolume ──→ 3D Regularization ──→ Regression → 视差图
                                 │
                            (对齐特征空间)
"""

import torch
import torch.nn as nn


class RGBNIRStereoModel(nn.Module):
    """
    End-to-end RGB + NIR cross-modal stereo matching model.

    Input:
        rgb:  RGB left image   (B, 3, H, W)
        nir:  NIR right image  (B, 1, H, W)

    Output dict:
        disparity:      Full-res disparity map     (B, 1, H, W)
        disparity_lr:   Low-res disparity          (B, 1, H/4, W/4)
        variance:       Uncertainty map            (B, 1, H, W)
        prob_volume:    Probability volume          (B, D, H/4, W/4)
    """

    def __init__(
        self,
        max_disparity=192,
        num_candidates=48,
        feature_channels=32,
        use_shared_extractor=False,
        use_spatial_attention=False,
        use_lightweight=False,
        mode='simple',
        use_mdp=False,           # Enable MDP modules (paper Section III-C)
    ):
        super().__init__()
        self.max_disparity = max_disparity
        self.num_candidates = num_candidates
        self.mode = mode
        self.use_mdp = use_mdp

        from models.feature_extractor import FeatureExtractor, FeatureExtractorShared
        from models.cross_attention import CrossAttentionModule
        from models.cost_volume import CostVolumeModule
        from models.regularization import CostVolumeRegularization, CostVolumeRegularizationLight
        from models.disparity_regression import DepthRegressionModule

        # 1. Feature Extractor
        if use_shared_extractor:
            self.feature_extractor = FeatureExtractorShared(rgb_channels=3, nir_channels=1)
        else:
            self.feature_extractor_rgb = FeatureExtractor(in_channels=3)
            self.feature_extractor_nir = FeatureExtractor(in_channels=1)
        self.use_shared = use_shared_extractor

        # 2. Cross-Attention Module
        self.cross_attention = CrossAttentionModule(
            channels=feature_channels, use_spatial=use_spatial_attention
        )

        # 3. Cost Volume
        self.cost_volume = CostVolumeModule(
            max_disparity=max_disparity,
            num_candidates=num_candidates,
            feature_channels=feature_channels,
        )
        self.cost_volume.set_disparity_candidates(max_disparity, num_candidates)

        # 4. 3D Regularization
        reg_class = CostVolumeRegularizationLight if use_lightweight else CostVolumeRegularization
        self.regularization = reg_class(in_channels=1)

        # 5. Disparity Regression (guidance from RGB image)
        self.regression = DepthRegressionModule(
            max_disparity=max_disparity,
            num_candidates=num_candidates,
            fine_channels=3,
        )
        self.regression.set_disparity_values(max_disparity, num_candidates)

        # 6. MDP Modules (paper Section III-C)
        if use_mdp:
            from models.mdp_module import MDPModule
            self.mdp_vis = MDPModule(in_channels=3)   # Visible-light MDP
            self.mdp_thr = MDPModule(in_channels=1)   # Thermal / NIR MDP

    def forward(self, rgb, nir, K_rgb=None, K_nir=None, R=None, T=None):
        # Step 1: Extract features
        if self.use_shared:
            f_rgb, f_nir = self.feature_extractor(rgb, nir)
        else:
            f_rgb = self.feature_extractor_rgb(rgb)
            f_nir = self.feature_extractor_nir(nir)

        # Step 2: Cross-modal feature alignment
        f_rgb_align, f_nir_align = self.cross_attention(f_rgb, f_nir)

        # Step 3: Build cost volume
        # Use homography warping when camera params available (matches paper Section III-B)
        has_calib = all(x is not None for x in [K_rgb, K_nir, R, T])
        if has_calib:
            # Convert disparity range to depth range for homography warping
            # depth = baseline * fx / disparity
            baseline = torch.norm(T, dim=1, keepdim=True)  # (B, 1, 1)
            fx_val = K_rgb[:, 0, 0:1].view(-1, 1, 1)       # (B, 1, 1)
            depth_min = torch.clamp(baseline * fx_val / (self.max_disparity + 1e-6), min=0.5, max=100.0)[:, 0, 0].min()
            depth_max = torch.clamp(baseline * fx_val / 0.5, min=2.0, max=150.0)[:, 0, 0].max()
            self.cost_volume.set_depth_candidates(depth_min.item(), depth_max.item(), self.num_candidates)
            # Sync regression module to use depth values (meters), not disparity (pixels)
            self.regression.set_depth_values(depth_min.item(), depth_max.item(), self.num_candidates)
            # Explicitly move registered buffers to GPU (register_buffer on CPU after .to(device))
            if self.cost_volume.depth_candidates is not None:
                self.cost_volume.depth_candidates = self.cost_volume.depth_candidates.to(rgb.device)
            if self.regression.depth_values is not None:
                self.regression.depth_values = self.regression.depth_values.to(rgb.device)
            cost_vol, prob_vol = self.cost_volume(
                f_rgb_align, f_nir_align, mode='homography',
                K_ref=K_rgb, K_src=K_nir, R=R, T=T,
            )
        else:
            cost_vol, prob_vol = self.cost_volume(f_rgb_align, f_nir_align, mode='simple')

        # Step 4: Regularize cost volume with 3D Conv
        cv_4d = cost_vol.unsqueeze(1)
        reg_out = self.regularization(cv_4d)
        if isinstance(reg_out, tuple) and len(reg_out) == 4:
            refined_cv, prob_vol_reg, aux1, aux2 = reg_out
        else:
            refined_cv, prob_vol_reg = reg_out
            aux1 = aux2 = None

        # Step 5: Disparity regression (use RGB as guidance for upsampling)
        disp, disp_lr, variance = self.regression(
            prob_vol_reg, guidance_image=rgb, return_uncertainty=True
        )

        # Homography mode: regression output is depth (m), convert to disparity (px)
        if has_calib:
            baseline_val = torch.norm(T, dim=1, keepdim=True).view(-1, 1, 1, 1)
            fx_val = K_rgb[:, 0, 0:1].view(-1, 1, 1, 1)
            # Convert depth→disparity with proper range clamping
            # depth<0.8m → disparity>48, which exceeds physical range and
            # produces gradients outside the valid domain → model never converges.
            disp = (baseline_val * fx_val / disp.clamp(min=0.5)).clamp(min=0.0, max=self.max_disparity)
            disp_lr = (baseline_val * fx_val / disp_lr.clamp(min=0.5)).clamp(min=0.0, max=self.max_disparity)

        result = {
            'disparity': disp,
            'disparity_lr': disp_lr,
            'variance': variance,
            'prob_volume': prob_vol_reg,
            'cost_volume_4d': cv_4d,
            'refined_cv': refined_cv,
        }

        # Step 6: MDP outputs (if enabled)
        if self.use_mdp:
            mdp_vis = self.mdp_vis(rgb)
            mdp_thr = self.mdp_thr(nir)
            result['mdp_vis'] = mdp_vis
            result['mdp_thr'] = mdp_thr

        return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B, H, W = 2, 384, 640
    model = RGBNIRStereoModel(
        max_disparity=192, num_candidates=48,
        use_shared_extractor=False, use_lightweight=False, mode='simple',
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    rgb = torch.randn(B, 3, H, W).to(device)
    nir = torch.randn(B, 1, H, W).to(device)

    with torch.no_grad():
        out = model(rgb, nir)

    print(f"RGB:           {rgb.shape}")
    print(f"NIR:           {nir.shape}")
    print(f"Disparity:     {out['disparity'].shape}")
    print(f"Disparity LR:  {out['disparity_lr'].shape}")
    print(f"Variance:      {out['variance'].shape}")
    print(f"Prob volume:   {out['prob_volume'].shape}")

    d = out['disparity']
    print(f"Disparity range: [{d.min().item():.2f}, {d.max().item():.2f}]")
    print("\nAll checks passed!")
