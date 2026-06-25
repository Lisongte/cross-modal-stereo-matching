"""
Loss Functions for Multi-Spectral Stereo Depth Estimation.

Implements three loss types from the paper (Section III-E):

1. L^MS_1 (Equation 8): L1 loss for CFM Module training
   L^MS_1 = Σ_u Σ_v | Σ_k d_k · p(d) - d^gt_uv |

2. L_NLL (Equation 9): Negative Log-Likelihood for MDP Module training
   L_NLL = Σ_u Σ_v [(d^gt - μ)² / (2σ²) + 1/2 · log(σ²)]

3. L^MS_NLL: Same NLL loss used for Depth Module training (frozen CFM + MDP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# L1 Loss for CFM Module (Equation 8)
# ============================================================================

def l1_depth_loss(depth_pred, depth_gt, mask=None, valid_threshold=0.0):
    """
    L1 loss between predicted depth and ground truth.
    
    Args:
        depth_pred: Predicted depth map (B, 1, H, W) or (B, H, W)
        depth_gt:   Ground truth depth map (B, 1, H, W) or (B, H, W)
        mask:       Optional valid mask (B, 1, H, W) or (B, H, W). If None, uses depth_gt > threshold
        valid_threshold: Minimum depth value to consider valid
    
    Returns:
        loss: Scalar L1 loss
    """
    if depth_pred.dim() == 4 and depth_pred.size(1) == 1:
        depth_pred = depth_pred.squeeze(1)
    if depth_gt.dim() == 4 and depth_gt.size(1) == 1:
        depth_gt = depth_gt.squeeze(1)
    # CRITICAL: mask must be squeezed to match depth shape, otherwise
    # (B,H,W) * (B,1,H,W) triggers cross-batch broadcasting → (B,B,H,W),
    # doubling the loss and mixing gradients across batch samples.
    if mask is not None and mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    
    if mask is None:
        mask = (depth_gt > valid_threshold).float()
    
    diff = torch.abs(depth_pred - depth_gt) * mask
    num_valid = mask.sum()
    
    if num_valid > 0:
        return diff.sum() / num_valid
    # Return a loss connected to the graph even when all masked out,
    # so backward() doesn't fail with "does not require grad".
    return (diff.sum() * 0.0)  # preserves grad_fn


def l1_disparity_loss(disp_pred, disp_gt, mask=None, valid_threshold=0.0):
    """
    L1 loss between predicted disparity and ground truth.
    Same as l1_depth_loss but for disparity values.
    """
    return l1_depth_loss(disp_pred, disp_gt, mask, valid_threshold)


class L1DepthLoss(nn.Module):
    """
    L1 loss for depth/disparity (Equation 8 from paper).
    
    Used in Stage 1: CFM Module training.
    L^MS_1 = Σ_u Σ_v | Σ_k d_k · p(d) - d^gt_uv |
    
    The expected depth is computed from the cost volume via soft argmin,
    then compared against ground truth.
    """
    def __init__(self, valid_threshold=1e-3):
        super().__init__()
        self.valid_threshold = valid_threshold
    
    def forward(self, depth_pred, depth_gt, mask=None):
        return l1_depth_loss(depth_pred, depth_gt, mask, self.valid_threshold)


# ============================================================================
# NLL Loss for MDP Module (Equation 9)
# ============================================================================

def negative_log_likelihood_loss(depth_gt, mean, variance, mask=None, valid_threshold=0.0):
    """
    Negative Log-Likelihood (NLL) loss (Equation 9).
    
    L_NLL = Σ_u Σ_u [(d^gt - μ)² / (2σ²) + 1/2 · log(σ²)]
    
    Models depth uncertainty as a Gaussian distribution:
        P(d) = 1/√(2πσ²) · exp(-(d - μ)² / (2σ²))
    
    Args:
        depth_gt:    Ground truth depth (B, 1, H, W) or (B, H, W)
        mean:        Predicted mean μ (B, 1, H, W) or (B, H, W)
        variance:    Predicted variance σ² (B, 1, H, W) or (B, H, W)
                     Must be positive (enforced by Softplus in regression)
        mask:        Optional valid mask
        valid_threshold: Minimum depth to consider valid
    
    Returns:
        loss: Scalar NLL loss
    """
    # Squeeze channel dim if needed
    if depth_gt.dim() == 4 and depth_gt.size(1) == 1:
        depth_gt = depth_gt.squeeze(1)
    if mean.dim() == 4 and mean.size(1) == 1:
        mean = mean.squeeze(1)
    if variance.dim() == 4 and variance.size(1) == 1:
        variance = variance.squeeze(1)
    
    if mask is None:
        mask = (depth_gt > valid_threshold).float()
    
    # Clamp variance to avoid numerical instability
    variance = variance.clamp(min=1e-6)
    
    # NLL loss components
    # Term 1: (d_gt - μ)² / (2σ²)  — penalizes inaccurate mean
    # Term 2: 1/2 log(σ²)           — penalizes high uncertainty
    nll = ((depth_gt - mean) ** 2) / (2 * variance) + 0.5 * torch.log(variance)
    nll = nll * mask
    
    num_valid = mask.sum()
    if num_valid > 0:
        return nll.sum() / num_valid
    return torch.tensor(0.0, device=depth_gt.device)


class NLLLoss(nn.Module):
    """
    Negative Log-Likelihood loss (Equation 9).
    
    Used in Stage 2: MDP Module training.
    Two separate losses: L^VIS_NLL for visible image, L^THR_NLL for thermal image.
    Also used in Stage 3: Depth Module training as L^MS_NLL.
    
    The log-likelihood of the ground truth depth under the predicted
    Gaussian distribution is maximized (= NLL minimized).
    """
    def __init__(self, valid_threshold=1e-3):
        super().__init__()
        self.valid_threshold = valid_threshold
    
    def forward(self, depth_gt, mean, variance, mask=None):
        return negative_log_likelihood_loss(
            depth_gt, mean, variance, mask, self.valid_threshold
        )


# ============================================================================
# Combined Loss for Full Pipeline
# ============================================================================

class MultiSpectrumDepthLoss(nn.Module):
    """
    Combined loss for multi-spectral stereo depth estimation.
    
    Supports all three training stages:
    
    Stage 1 (CFM Module): L_total = L^MS_1
    Stage 2 (MDP Module): L_total = L^VIS_NLL + L^THR_NLL
    Stage 3 (Depth Module): L_total = L^MS_NLL
    
    The losses can also be combined for joint training.
    """
    def __init__(self, valid_threshold=1e-3, lambda_l1=1.0, lambda_nll=1.0):
        super().__init__()
        self.valid_threshold = valid_threshold
        self.lambda_l1 = lambda_l1
        self.lambda_nll = lambda_nll
        self.l1_loss = L1DepthLoss(valid_threshold)
        self.nll_loss = NLLLoss(valid_threshold)
    
    def forward_stage1(self, depth_pred, depth_gt, mask=None):
        """
        Stage 1 loss: L^MS_1 for CFM Module.
        
        Args:
            depth_pred: Expected depth from cost volume (B, 1, H, W)
            depth_gt:   Ground truth depth (B, 1, H, W)
        
        Returns:
            loss: Scalar L1 loss
        """
        return self.l1_loss(depth_pred, depth_gt, mask)
    
    def forward_stage2(self, depth_gt, mean_vis, var_vis, mean_thr, var_thr, mask=None):
        """
        Stage 2 loss: L^VIS_NLL + L^THR_NLL for MDP Module.
        
        Args:
            depth_gt:  Ground truth depth (B, 1, H, W)
            mean_vis:  Visible image predicted mean μ_vis (B, 1, H, W)
            var_vis:   Visible image predicted variance σ²_vis (B, 1, H, W)
            mean_thr:  Thermal image predicted mean μ_thr (B, 1, H, W)
            var_thr:   Thermal image predicted variance σ²_thr (B, 1, H, W)
        
        Returns:
            loss_vis:  NLL loss for visible modality
            loss_thr:  NLL loss for thermal modality
            loss:      Combined loss
        """
        loss_vis = self.nll_loss(depth_gt, mean_vis, var_vis, mask)
        loss_thr = self.nll_loss(depth_gt, mean_thr, var_thr, mask)
        return loss_vis, loss_thr, loss_vis + loss_thr
    
    def forward_stage3(self, depth_gt, mean_ms, var_ms, mask=None):
        """
        Stage 3 loss: L^MS_NLL for Depth Module.
        
        Args:
            depth_gt:  Ground truth depth (B, 1, H, W)
            mean_ms:   Multi-spectral predicted mean μ_ms (B, 1, H, W)
            var_ms:    Multi-spectral predicted variance σ²_ms (B, 1, H, W)
        
        Returns:
            loss: NLL loss for final depth module
        """
        return self.nll_loss(depth_gt, mean_ms, var_ms, mask)


# ============================================================================
# Additional Utility Losses
# ============================================================================

def silog_loss(depth_pred, depth_gt, mask=None, valid_threshold=1e-3, alpha=0.15):
    """
    Scale-Invariant Logarithmic Loss (SI-Log).
    
    Commonly used in depth estimation (DORN, BTS, AdaBins).
    
    L_silog = α · sqrt(1/N Σ g² - 1/N² (Σ g)²)
    where g = log(d_pred) - log(d_gt)
    
    Args:
        depth_pred: Predicted depth (B, 1, H, W) or (B, H, W)
        depth_gt:   Ground truth depth (B, 1, H, W) or (B, H, W)
        mask:       Valid mask (optional)
        alpha:      Scaling factor (default: 0.15 from BTS paper)
    
    Returns:
        loss: SI-Log loss scalar
    """
    if depth_pred.dim() == 4:
        depth_pred = depth_pred.squeeze(1)
    if depth_gt.dim() == 4:
        depth_gt = depth_gt.squeeze(1)
    # Fix cross-batch broadcasting: mask must match squeezed shape
    if mask is not None and mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    
    if mask is None:
        mask = (depth_gt > valid_threshold).float()
    
    # Clamp depths to avoid log(0)
    depth_pred = depth_pred.clamp(min=1e-6)
    depth_gt = depth_gt.clamp(min=1e-6)
    
    # Logarithmic difference
    g = torch.log(depth_pred) - torch.log(depth_gt)
    g = g * mask
    
    N = mask.sum()
    if N > 0:
        # Variance: E[g²] - E[g]²
        loss = torch.sqrt((g ** 2).sum() / N - (g.sum() / N) ** 2 + 1e-6) * alpha
        return loss
    return torch.tensor(0.0, device=depth_pred.device)


def disparity_metrics(disp_pred, disp_gt, mask=None, valid_threshold=0.0):
    """
    Compute stereo disparity evaluation metrics.

    Metrics (KITTI-style, evaluated on disparity domain):
    - EPE:     End-Point Error = mean(|d_pred - d_gt|), in pixels
    - D1-all:  Bad pixel rate: pixels where |error| > max(3, 0.05 * d_gt)

    Args:
        disp_pred:  Predicted disparity (B, 1, H, W) or (B, H, W)
        disp_gt:    Ground truth disparity (B, 1, H, W) or (B, H, W)
        mask:       Valid mask
        valid_threshold: Minimum disparity to consider valid

    Returns:
        dict: {'epe': float, 'd1_all': float}
    """
    if disp_pred.dim() == 4 and disp_pred.size(1) == 1:
        disp_pred = disp_pred.squeeze(1)
    if disp_gt.dim() == 4 and disp_gt.size(1) == 1:
        disp_gt = disp_gt.squeeze(1)
    # Fix cross-batch broadcasting: mask must match squeezed shape
    if mask is not None and mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)

    if mask is None:
        mask = (disp_gt > valid_threshold).float()

    # EPE: mean absolute error in pixels
    abs_error = torch.abs(disp_pred - disp_gt) * mask
    n_valid = mask.sum()
    epe = (abs_error.sum() / n_valid).item() if n_valid > 0 else 0.0

    # D1-all: bad pixel rate
    # |error| > max(3px, 5% of GT)
    threshold = torch.maximum(
        torch.tensor(3.0, device=disp_gt.device),
        0.05 * disp_gt
    )
    bad = ((abs_error > threshold) * mask).sum()
    d1_all = (bad / n_valid).item() if n_valid > 0 else 0.0

    return {'epe': epe, 'd1_all': d1_all}


class SILogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss wrapper."""
    def __init__(self, alpha=0.15, valid_threshold=1e-3):
        super().__init__()
        self.alpha = alpha
        self.valid_threshold = valid_threshold
    
    def forward(self, depth_pred, depth_gt, mask=None):
        return silog_loss(depth_pred, depth_gt, mask, self.valid_threshold, self.alpha)


def depth_metrics(depth_pred, depth_gt, mask=None, valid_threshold=1e-3):
    """
    Compute standard depth estimation metrics.
    
    Metrics from paper (Table I):
    - Abs Rel:    |d_pred - d_gt| / d_gt
    - Sq Rel:     (d_pred - d_gt)² / d_gt
    - RMSE:       sqrt(mean((d_pred - d_gt)²))
    - RMSE log:   sqrt(mean((log d_pred - log d_gt)²))
    - δ < 1.25:   max(d_pred/d_gt, d_gt/d_pred) < 1.25
    - δ < 1.25²:  max(d_pred/d_gt, d_gt/d_pred) < 1.25²
    - δ < 1.25³:  max(d_pred/d_gt, d_gt/d_pred) < 1.25³
    
    Args:
        depth_pred: Predicted depth (B, 1, H, W) or (B, H, W)
        depth_gt:   Ground truth depth (B, 1, H, W) or (B, H, W)
        mask:       Valid mask
        valid_threshold: Minimum depth considered valid
    
    Returns:
        dict: Dictionary of metric name -> scalar value
    """
    if depth_pred.dim() == 4 and depth_pred.size(1) == 1:
        depth_pred = depth_pred.squeeze(1)
    if depth_gt.dim() == 4 and depth_gt.size(1) == 1:
        depth_gt = depth_gt.squeeze(1)
    # Fix cross-batch broadcasting: mask must match squeezed shape,
    # otherwise (B,H,W) * (B,1,H,W) → (B,B,H,W), inflating metrics 4-16×.
    if mask is not None and mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    
    if mask is None:
        mask = (depth_gt > valid_threshold).float()
    
    depth_pred = depth_pred.clamp(min=1e-6)
    depth_gt = depth_gt.clamp(min=1e-6)
    
    diff = depth_pred - depth_gt
    rel_diff = diff / depth_gt
    
    # Abs Rel
    abs_rel = (torch.abs(rel_diff) * mask).sum() / mask.sum()
    
    # Sq Rel
    sq_rel = ((rel_diff ** 2) * mask).sum() / mask.sum()
    
    # RMSE
    rmse = torch.sqrt(((diff ** 2) * mask).sum() / mask.sum())
    
    # RMSE log
    log_diff = torch.log(depth_pred) - torch.log(depth_gt)
    rmse_log = torch.sqrt(((log_diff ** 2) * mask).sum() / mask.sum())
    
    # Threshold accuracy δ
    max_ratio = torch.max(depth_pred / depth_gt, depth_gt / depth_pred)
    delta1 = ((max_ratio < 1.25).float() * mask).sum() / mask.sum()
    delta2 = ((max_ratio < 1.25 ** 2).float() * mask).sum() / mask.sum()
    delta3 = ((max_ratio < 1.25 ** 3).float() * mask).sum() / mask.sum()
    
    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'delta1': delta1.item(),
        'delta2': delta2.item(),
        'delta3': delta3.item(),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 60)
    
    B, H, W = 2, 48, 80
    
    # Generate synthetic GT depth
    depth_gt = torch.rand(B, H, W).to(device) * 50 + 1  # 1-51 meters
    
    # ============================================
    print("Test 1: L1 Loss (Equation 8)")
    print("-" * 40)
    depth_pred = depth_gt + torch.randn(B, H, W).to(device) * 2 + 0.5
    l1 = L1DepthLoss()
    loss = l1(depth_pred, depth_gt)
    print(f"  L1 loss: {loss.item():.4f}")
    
    # With mask
    mask = (depth_gt > 10).float()
    loss_masked = l1(depth_pred, depth_gt, mask)
    print(f"  L1 loss (masked): {loss_masked.item():.4f}")
    
    # ============================================
    print("\nTest 2: NLL Loss (Equation 9)")
    print("-" * 40)
    mean = depth_gt + torch.randn(B, H, W).to(device) * 2
    variance = torch.rand(B, H, W).to(device) * 5 + 0.1  # positive
    nll = NLLLoss()
    loss_nll = nll(depth_gt, mean, variance)
    print(f"  NLL loss: {loss_nll.item():.4f}")
    
    # Perfect prediction should give lower NLL
    loss_perfect = nll(depth_gt, depth_gt, torch.ones_like(variance) * 0.1)
    print(f"  NLL loss (perfect): {loss_perfect.item():.4f}  (should be lower)")
    assert loss_perfect < loss_nll, "Perfect prediction should have lower NLL"
    
    # ============================================
    print("\nTest 3: MultiSpectrumDepthLoss (3 stages)")
    print("-" * 40)
    
    ms_loss = MultiSpectrumDepthLoss()
    
    # Stage 1
    loss_s1 = ms_loss.forward_stage1(depth_pred, depth_gt)
    print(f"  Stage 1 (L^MS_1): {loss_s1.item():.4f}")
    
    # Stage 2
    mean_vis = depth_gt + torch.randn(B, H, W).to(device) * 2
    var_vis = torch.rand(B, H, W).to(device) * 3 + 0.1
    mean_thr = depth_gt + torch.randn(B, H, W).to(device) * 3
    var_thr = torch.rand(B, H, W).to(device) * 4 + 0.1
    l_vis, l_thr, l_s2 = ms_loss.forward_stage2(depth_gt, mean_vis, var_vis, mean_thr, var_thr)
    print(f"  Stage 2: L^VIS_NLL={l_vis.item():.4f}, L^THR_NLL={l_thr.item():.4f}, total={l_s2.item():.4f}")
    
    # Stage 3
    mean_ms = depth_gt + torch.randn(B, H, W).to(device) * 1.5
    var_ms = torch.rand(B, H, W).to(device) * 2 + 0.1
    l_s3 = ms_loss.forward_stage3(depth_gt, mean_ms, var_ms)
    print(f"  Stage 3 (L^MS_NLL): {l_s3.item():.4f}")
    
    # ============================================
    print("\nTest 4: Depth Metrics (Table I)")
    print("-" * 40)
    depth_pred_2 = depth_gt + torch.randn(B, H, W).to(device) * 3
    metrics = depth_metrics(depth_pred_2, depth_gt)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    
    # Perfect prediction -> delta1 = 1.0, abs_rel = 0.0
    metrics_perfect = depth_metrics(depth_gt, depth_gt)
    print(f"\n  Perfect prediction:")
    print(f"  abs_rel:     {metrics_perfect['abs_rel']:.4f} (expected 0.0)")
    print(f"  delta1:      {metrics_perfect['delta1']:.4f} (expected 1.0)")
    assert abs(metrics_perfect['abs_rel']) < 1e-5
    assert abs(metrics_perfect['delta1'] - 1.0) < 1e-5
    
    # ============================================
    print("\nTest 5: SILog Loss")
    print("-" * 40)
    silog = SILogLoss(alpha=0.15)
    loss_silog = silog(depth_pred, depth_gt)
    print(f"  SI-Log loss: {loss_silog.item():.4f}")
    
    # ============================================
    print("\nTest 6: CFM Loss (L^MS_1 via expected depth from cost volume)")
    print("-" * 40)
    from models.disparity_regression import disparity_regression
    
    D = 48
    B2, H2_low, W2_low = 2, 48, 80
    
    # Simulate probability volume (after softmax)
    prob_vol = torch.rand(B2, D, H2_low, W2_low).to(device)
    prob_vol = prob_vol / prob_vol.sum(dim=1, keepdim=True)
    
    # Disparity candidates
    disp_vals = torch.linspace(0, 96, D).to(device)
    
    # Expected disparity via soft argmin
    expected_disp = disparity_regression(prob_vol, disp_vals)
    print(f"  Expected disparity: {expected_disp.shape}")
    print(f"  Range: [{expected_disp.min().item():.2f}, {expected_disp.max().item():.2f}]")
    
    # L1 loss between expected disparity and GT
    gt_disp = torch.rand(B2, 1, H2_low, W2_low).to(device) * 96
    cfm_loss = l1_disparity_loss(expected_disp, gt_disp)
    print(f"  CFM L^MS_1 loss: {cfm_loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
