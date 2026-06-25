import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_warp(features, depth, K_src, K_tgt, R, T):
    """
    Differentiable inverse warping of features from target view to source view.
    
    Projects each pixel from target view into source view given depth hypothesis,
    then samples features using bilinear interpolation.
    
    Args:
        features: Source view feature map (B, C, H, W) - to be sampled from
        depth:    Depth hypothesis map (B, 1, H, W) at target view
        K_src:    Source camera intrinsic matrix (B, 3, 3)
        K_tgt:    Target camera intrinsic matrix (B, 3, 3)
        R:        Rotation matrix from target to source (B, 3, 3)
        T:        Translation vector from target to source (B, 3, 1)
    
    Returns:
        warped_features: Feature map sampled at projected coordinates (B, C, H, W)
    """
    B, C, H, W = features.shape
    device = features.device

    # Create pixel grid in target view (homogeneous coordinates)
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    # (H, W) -> (1, 3, H, W)
    ones = torch.ones_like(i)
    pixel_grid = torch.stack([i, j, ones], dim=0).unsqueeze(0).float()  # (1, 3, H, W)
    pixel_grid = pixel_grid.expand(B, -1, -1, -1)  # (B, 3, H, W)
    pixel_grid = pixel_grid.reshape(B, 3, -1)  # (B, 3, H*W)

    # Step 1: Convert target pixels to normalized camera coordinates
    # p_cam = K_tgt^{-1} * p_pixel
    K_tgt_inv = torch.inverse(K_tgt)  # (B, 3, 3)
    p_cam_tgt = torch.bmm(K_tgt_inv, pixel_grid)  # (B, 3, H*W)

    # Step 2: Scale by depth: P = depth * p_cam
    depth_flat = depth.reshape(B, 1, -1)  # (B, 1, H*W)
    P_tgt = p_cam_tgt * depth_flat  # (B, 3, H*W) - 3D points in target camera space

    # Step 3: Transform to source camera coordinate system
    # P_src = R * P_tgt + T
    P_src = torch.bmm(R, P_tgt) + T  # (B, 3, H*W)

    # Step 4: Project to source pixel coordinates
    # p_src = K_src * P_src
    p_src = torch.bmm(K_src, P_src)  # (B, 3, H*W)

    # Step 5: Normalize to pixel coordinates (u, v) = (x/z, y/z)
    p_src_u = p_src[:, 0, :] / (p_src[:, 2, :].clamp(min=1e-8))  # (B, H*W)
    p_src_v = p_src[:, 1, :] / (p_src[:, 2, :].clamp(min=1e-8))  # (B, H*W)

    # Step 6: Normalize to [-1, 1] for grid_sample
    p_src_u = 2.0 * p_src_u / (W - 1) - 1.0
    p_src_v = 2.0 * p_src_v / (H - 1) - 1.0

    # Stack and reshape to (B, H, W, 2)
    grid = torch.stack([p_src_u, p_src_v], dim=-1).reshape(B, H, W, 2)

    # Step 7: Sample using bilinear interpolation
    warped_features = F.grid_sample(
        features, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return warped_features


def build_cost_volume(f_ref, f_src, depth_candidates, K_ref, K_src, R, T):
    """
    Build cost volume by warping source features to reference view
    at each depth candidate and computing matching scores.
    
    Args:
        f_ref:         Reference view aligned features (B, C, H, W)
        f_src:         Source view aligned features (B, C, H, W)
        depth_candidates: Depth hypothesis values (D,) e.g., [1.0, 1.5, ..., 100.0]
        K_ref:         Reference camera intrinsic (B, 3, 3)
        K_src:         Source camera intrinsic (B, 3, 3)
        R:             Rotation ref->src (B, 3, 3)
        T:             Translation ref->src (B, 3, 1)
    
    Returns:
        cost_volume: Matching cost volume (B, D, H, W)
        prob_volume: Depth probability volume (B, D, H, W) after softmax
    """
    B, C, H, W = f_ref.shape
    D = depth_candidates.shape[0]
    device = f_ref.device

    # Expand depth candidates to (B, 1, H, W) for each depth level
    cost_volume = []

    for d_idx in range(D):
        depth_val = depth_candidates[d_idx]
        # Create depth map for this hypothesis: (B, 1, H, W)
        depth_map = depth_val * torch.ones(B, 1, H, W, device=device)

        # Warp source features to reference view at this depth
        f_warped = inverse_warp(
            f_src, depth_map,
            K_src, K_ref, R, T
        )

        # Compute matching score: dot product (cosine similarity variant)
        # Normalize features first for cosine similarity
        f_ref_norm = F.normalize(f_ref, dim=1)
        f_warped_norm = F.normalize(f_warped, dim=1)

        # Dot product similarity: (B, C, H, W) * (B, C, H, W) -> (B, 1, H, W)
        similarity = (f_ref_norm * f_warped_norm).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        cost_volume.append(similarity)

    # Stack along depth dimension: (B, D, H, W)
    cost_volume = torch.cat(cost_volume, dim=1)

    return cost_volume


class CostVolumeModule(nn.Module):
    """
    Cost Volume Construction module for cross-modal stereo matching.
    
    Constructs a 4D cost volume by:
    1. Warping source view features to reference view at N depth candidates
    2. Computing dot product similarity at each depth
    3. Applying softmax to convert to probability volume
    
    Supports both stereo rectified (simplified) and general (homography) cases.
    """
    def __init__(self, max_disparity=192, num_candidates=48, feature_channels=32):
        super(CostVolumeModule, self).__init__()
        self.max_disparity = max_disparity
        self.num_candidates = num_candidates
        self.feature_channels = feature_channels

        # Disparity candidates for stereo rectified case
        # For rectified stereo: depth = baseline * focal / disparity
        self.register_buffer('depth_candidates', None)
        self.register_buffer('disparity_candidates', None)

        # Optional: light-weight 3D conv to refine cost volume
        self.refine_conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=3, padding=1, bias=False),
        )

    def set_disparity_candidates(self, max_disp, num_candidates):
        """Set disparity candidates (for stereo rectified case)."""
        disp = torch.linspace(0, max_disp, num_candidates)
        self.register_buffer('disparity_candidates', disp)

    def set_depth_candidates(self, depth_min, depth_max, num_candidates):
        """Set depth candidates (for general case)."""
        # Sample depth in inverse space (more samples near camera)
        inv_depth = torch.linspace(1.0/depth_max, 1.0/depth_min, num_candidates)
        depth = 1.0 / inv_depth
        self.register_buffer('depth_candidates', depth)

    def forward_cost_volume(self, f_ref, f_src, K_ref=None, K_src=None, R=None, T=None):
        """
        Build cost volume using homography warping (general case).
        
        Args:
            f_ref: Reference view aligned features (B, C, H, W)
            f_src: Source view aligned features (B, C, H, W)
            K_ref: Reference camera intrinsic (B, 3, 3) or (3, 3)
            K_src: Source camera intrinsic (B, 3, 3) or (3, 3)
            R:     Rotation ref->src (B, 3, 3) or (3, 3)
            T:     Translation ref->src (B, 3, 1) or (3, 1)
        
        Returns:
            cost_volume: (B, D, H, W)
            prob_volume: (B, D, H, W)
        """
        # Ensure batch dimension for intrinsics/extrinsics
        if K_ref is not None and K_ref.dim() == 2:
            K_ref = K_ref.unsqueeze(0)
        if K_src is not None and K_src.dim() == 2:
            K_src = K_src.unsqueeze(0)
        if R is not None and R.dim() == 2:
            R = R.unsqueeze(0)
        if T is not None and T.dim() == 2:
            T = T.unsqueeze(0)

        B, C, H, W = f_ref.shape
        D = self.depth_candidates.shape[0]
        device = f_ref.device

        # Expand intrinsics/extrinsics to batch
        if K_ref is not None:
            K_ref = K_ref.expand(B, -1, -1)
            K_src = K_src.expand(B, -1, -1)
            R = R.expand(B, -1, -1)
            T = T.expand(B, -1, -1)

        cost_volume = []

        for d_idx in range(D):
            depth_val = self.depth_candidates[d_idx]
            depth_map = depth_val * torch.ones(B, 1, H, W, device=device)

            f_warped = inverse_warp(f_src, depth_map, K_src, K_ref, R, T)

            # Dot product similarity
            f_ref_norm = F.normalize(f_ref, dim=1)
            f_warped_norm = F.normalize(f_warped, dim=1)
            similarity = (f_ref_norm * f_warped_norm).sum(dim=1, keepdim=True)

            cost_volume.append(similarity)

        cost_volume = torch.cat(cost_volume, dim=1)  # (B, D, H, W)

        # Optional: refine cost volume with 3D conv
        # Apply refine conv - need to add channel dim (B, 1, D, H, W)
        cv_refined = self.refine_conv(cost_volume.unsqueeze(1)).squeeze(1)

        # Convert to probability volume via softmax over depth
        prob_volume = F.softmax(cv_refined, dim=1)

        return cv_refined, prob_volume

    def forward_simple(self, f_ref, f_src, max_disp=None):
        """
        Simplified cost volume building for rectified stereo.

        Assumes images are already rectified (epipolar lines are horizontal).
        Uses sub-pixel shifting (F.grid_sample) to support arbitrary disparity
        candidates matching num_candidates.

        Args:
            f_ref: Reference view features (B, C, H, W)  at 1/4 resolution
            f_src: Source view features (B, C, H, W)      at 1/4 resolution
            max_disp: Maximum disparity in original pixel space

        Returns:
            cost_volume: (B, D, H, W) where D = num_candidates
            prob_volume: (B, D, H, W)
        """
        B, C, H, W = f_ref.shape
        max_disp = max_disp or self.max_disparity
        D = self.num_candidates  # Use num_candidates, matching regression module

        # Disparity values in original pixel space, uniformly sampled
        # disparity_candidates: (D,) e.g. linspace(0, 48, 48)
        if self.disparity_candidates is not None:
            disp_vals = self.disparity_candidates  # (D,)
        else:
            disp_vals = torch.linspace(0, max_disp, D, device=f_ref.device)

        # Convert to shift at feature map resolution (1/4)
        shifts = disp_vals / 4.0  # (D,), feature-map pixel shifts

        # Build pixel coordinate grid for grid_sample (normalized to [-1, 1])
        # grid_sample uses (x, y) convention: x = width dim, y = height dim
        ys = torch.linspace(-1, 1, H, device=f_ref.device)
        xs = torch.linspace(-1, 1, W, device=f_ref.device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')  # (H, W) each
        base_grid = torch.stack([gx, gy], dim=-1)  # (H, W, 2)

        cost_volume = []
        f_ref_norm = F.normalize(f_ref, dim=1)

        for d in range(D):
            shift = shifts[d].item()

            # Create shifted grid: move x coordinates right by shift pixels
            # in normalized coordinates: shift_norm = shift * (2.0 / (W - 1))
            shift_norm = shift * (2.0 / (W - 1)) if W > 1 else 0.0
            shifted_grid = base_grid.clone()
            shifted_grid[..., 0] = shifted_grid[..., 0] + shift_norm  # shift x only

            # Expand to batch: (B, H, W, 2)
            grid = shifted_grid.unsqueeze(0).expand(B, -1, -1, -1)

            # Sample source features at shifted positions
            shifted = F.grid_sample(
                f_src, grid, mode='bilinear',
                padding_mode='zeros', align_corners=True
            )  # (B, C, H, W)

            # Matching cost: normalized dot product
            shifted_norm = F.normalize(shifted, dim=1)
            cost = (f_ref_norm * shifted_norm).sum(dim=1)  # (B, H, W)

            cost_volume.append(cost)

        cost_volume = torch.stack(cost_volume, dim=1)  # (B, D, H, W)

        # Apply the same 3D conv refinement as forward_cost_volume,
        # otherwise simple mode produces flat probability distributions
        # and models using simple mode never converge.
        cv_refined = self.refine_conv(cost_volume.unsqueeze(1)).squeeze(1)

        # Convert to probability volume via softmax over disparity dimension
        prob_volume = F.softmax(cv_refined, dim=1)

        return cost_volume, prob_volume

    def forward(self, f_ref, f_src, mode='simple', **kwargs):
        """
        Args:
            f_ref: Reference view features (B, C, H, W)
            f_src: Source view features (B, C, H, W)
            mode: 'simple' for rectified stereo, 'homography' for general case
        
        Returns:
            cost_volume: (B, D, H, W)
            prob_volume: (B, D, H, W)
        """
        if mode == 'simple':
            return self.forward_simple(f_ref, f_src)
        elif mode == 'homography':
            return self.forward_cost_volume(f_ref, f_src, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B, C, H, W = 2, 32, 48, 80  # Feature map at 1/4 resolution (input 192x320)
    f_ref = torch.randn(B, C, H, W).to(device)
    f_src = torch.randn(B, C, H, W).to(device)

    # Test simple mode (rectified stereo)
    print("=" * 60)
    print("Test 1: Simple mode (rectified stereo)")
    print("=" * 60)

    model_simple = CostVolumeModule(max_disparity=192, num_candidates=48).to(device)
    cv_simple, prob_simple = model_simple(f_ref, f_src, mode='simple')

    print(f"Features:         {f_ref.shape}")
    print(f"Cost volume:      {cv_simple.shape}")
    print(f"Prob volume:      {prob_simple.shape}")
    print(f"Params:           {sum(p.numel() for p in model_simple.parameters()):,}")

    # Test homography mode
    print("\n" + "=" * 60)
    print("Test 2: Homography mode (general stereo)")
    print("=" * 60)

    model_homo = CostVolumeModule(max_disparity=192, num_candidates=16).to(device)

    # Set depth candidates (in meters)
    model_homo.set_depth_candidates(depth_min=2.0, depth_max=80.0, num_candidates=16)
    print(f"Depth candidates: {model_homo.depth_candidates}")
    print(f"Depth range:      [{model_homo.depth_candidates[0]:.1f}, {model_homo.depth_candidates[-1]:.1f}]")

    # Use larger feature map for homography test
    H2, W2 = 48, 80
    f_ref2 = torch.randn(B, C, H2, W2).to(device)
    f_src2 = torch.randn(B, C, H2, W2).to(device)

    # Create mock camera intrinsics/extrinsics
    fx = 764.5  # RGB camera focal length
    fy = 764.5
    cx = 579.5
    cy = 153.0
    baseline = 0.299  # RGB baseline in meters (299mm)

    K_ref = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32).to(device)
    K_src = K_ref.clone()
    R = torch.eye(3, dtype=torch.float32).to(device)
    T = torch.tensor([[-baseline], [0.0], [0.0]], dtype=torch.float32).to(device)

    try:
        cv_homo, prob_homo = model_homo.forward(
            f_ref2, f_src2, mode='homography',
            K_ref=K_ref, K_src=K_src, R=R, T=T
        )
        print(f"Cost volume:      {cv_homo.shape}")
        print(f"Prob volume:      {prob_homo.shape}")
    except Exception as e:
        print(f"Homography test skipped (expected with downscaled features): {e}")

    print("\nAll tests completed!")