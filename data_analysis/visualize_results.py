"""
Visualization: RGB + NIR Stereo Matching Model Performance.
Generates comprehensive comparison figures, including regression analysis,
classification metrics, and EDA analysis.

Usage:
    python data_analysis/visualize_results.py
"""

import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps, font_manager
# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rgb_nir_model import RGBNIRStereoModel
from models.losses import l1_disparity_loss, depth_metrics

# Import our new data analysis modules
from data_analysis.regression_analysis import run_regression_analysis
from data_analysis.classification_metrics import run_classification_analysis
from data_analysis.dataset_eda import run_dataset_eda
from data_analysis.data_storage import ExperimentDataStore


def generate_all_visualizations(model, rgb, nir, gt_disp, mask, objects, save_dir='visualizations'):
    """
    Generate all visualizations: model performance, pipeline architecture,
    regression analysis, classification metrics, and EDA.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    print("Model inference...")
    with torch.no_grad():
        out = model(rgb.to(device), nir.to(device))

    pred_disp = out['disparity'][0, 0].cpu().numpy()
    variance = out['variance'][0, 0].cpu().numpy()
    gt_np = gt_disp[0, 0].cpu().numpy()
    mask_np = mask[0, 0].cpu().numpy() > 0.5
    max_disp = model.max_disparity

    # ========== 1. Metrics ==========
    metrics = depth_metrics(out['disparity'], gt_disp.to(device), mask.to(device))
    loss_val = l1_disparity_loss(out['disparity'], gt_disp.to(device), mask.to(device))

    print(f"\n{'='*55}")
    print(f"  Evaluation Metrics (valid pixels only)")
    print(f"{'='*55}")
    print(f"  L1 Loss:           {loss_val.item():.4f}")
    print(f"  Abs Rel:           {metrics['abs_rel']:.4f}")
    print(f"  Sq Rel:            {metrics['sq_rel']:.4f}")
    print(f"  RMSE:              {metrics['rmse']:.2f}")
    print(f"  RMSE log:          {metrics['rmse_log']:.4f}")
    print(f"  delta < 1.25:      {metrics['delta1']:.4f}")
    print(f"  delta < 1.25^2:    {metrics['delta2']:.4f}")
    print(f"  delta < 1.25^3:    {metrics['delta3']:.4f}")
    print(f"{'='*55}")

    # ========== 2. Model Performance Visualization ==========
    print("\nGenerating model performance visualization...")
    fig = plt.figure(figsize=(18, 14))

    # Row 1: Inputs
    plt.subplot(3, 4, 1)
    rgb_img = rgb[0].permute(1, 2, 0).cpu().numpy()
    plt.imshow(rgb_img)
    plt.title("RGB Left View (Input)", fontsize=12)
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(nir[0, 0].cpu().numpy(), cmap='gray')
    plt.title("NIR Right View (Input)", fontsize=12)
    plt.axis('off')

    # Row 2: Disparity comparison
    ax = plt.subplot(3, 4, 3)
    vmax = max_disp / 2
    im = ax.imshow(gt_np, cmap='jet', vmin=0, vmax=vmax)
    plt.title("GT Disparity (LiDAR)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.axis('off')

    ax = plt.subplot(3, 4, 4)
    im = ax.imshow(pred_disp, cmap='jet', vmin=0, vmax=vmax)
    plt.title("Predicted Disparity", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.axis('off')

    # Row 3: Error analysis
    ax = plt.subplot(3, 4, 5)
    error = np.abs(pred_disp - gt_np)
    error_vis = np.where(mask_np, error, np.nan)
    cmap_err = colormaps['hot'].copy()
    cmap_err.set_bad('white', alpha=0)
    im = ax.imshow(error_vis, cmap=cmap_err, vmin=0, vmax=10)
    plt.title(f"Abs Error (valid pixels)\nMean={error[mask_np].mean():.2f}", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.axis('off')

    ax = plt.subplot(3, 4, 6)
    im = ax.imshow(variance, cmap='hot')
    plt.title("Uncertainty (Variance)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.axis('off')

    # Row 4: Sparse supervision
    ax = plt.subplot(3, 4, 7)
    gt_sparse = np.where(mask_np, gt_np, 0)
    im = ax.imshow(gt_sparse, cmap='jet', vmin=0, vmax=vmax)
    plt.title(f"Sparse GT Disparity\n({mask_np.sum()/mask_np.size*100:.1f}% valid)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.axis('off')

    ax = plt.subplot(3, 4, 8)
    pred_sparse = np.where(mask_np, pred_disp, 0)
    im = ax.imshow(pred_sparse, cmap='jet', vmin=0, vmax=vmax)
    plt.title("Sparse Predicted\n(at GT locations)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.axis('off')

    # Row 5: Histogram
    plt.subplot(3, 4, (9, 10))
    valid_gt = gt_np[mask_np]
    valid_pred = pred_disp[mask_np]
    bins = np.linspace(0, max_disp, 50)
    plt.hist(valid_gt, bins=bins, alpha=0.6, label='GT', color='tab:blue', density=True)
    plt.hist(valid_pred, bins=bins, alpha=0.6, label='Predicted', color='tab:orange', density=True)
    plt.xlabel('Disparity (pixels)')
    plt.ylabel('Density')
    plt.title('Disparity Distribution (Valid Pixels)')
    plt.legend()
    plt.grid(alpha=0.3)

    # Row 6: Scatter plot
    plt.subplot(3, 4, (11, 12))
    sample_idx = np.random.choice(valid_gt.shape[0], min(5000, valid_gt.shape[0]), replace=False)
    plt.scatter(valid_gt[sample_idx], valid_pred[sample_idx], alpha=0.3, s=2, c='tab:blue')
    plt.plot([0, max_disp], [0, max_disp], 'r--', alpha=0.7, label='Perfect')
    plt.xlabel('GT Disparity')
    plt.ylabel('Predicted Disparity')
    plt.title(f'GT vs Predicted (AbsRel={metrics["abs_rel"]:.4f}, RMSE={metrics["rmse"]:.2f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.xlim(0, max_disp)
    plt.ylim(0, max_disp)

    plt.tight_layout()
    path_perf = f'{save_dir}/model_performance.png'
    plt.savefig(path_perf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {path_perf}")

    # ========== 3. Pipeline Architecture Diagram ==========
    print("\nGenerating pipeline diagram...")
    fig3, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    import matplotlib.patches as mpatches

    blocks = [
        (0.5, 7.0, 3.0, 1.5, '#E8F5E9', 'RGB Image\n(B, 3, H, W)', 'Input'),
        (0.5, 4.0, 3.0, 1.5, '#FFF3E0', 'NIR Image\n(B, 1, H, W)', 'Input'),
        (4.5, 6.5, 2.5, 1.2, '#E3F2FD', 'Feature\nExtractor\n(PSMNet)', 'Stage 1'),
        (4.5, 4.0, 2.5, 1.2, '#E3F2FD', 'Feature\nExtractor\n(PSMNet)', 'Stage 1'),
        (8.0, 5.0, 2.5, 1.2, '#F3E5F5', 'Cross-Attention\nModule', 'Stage 2'),
        (11.5, 5.0, 2.5, 1.2, '#FCE4EC', 'Cost Volume\n(D x H/4 x W/4)', 'Stage 3'),
        (11.5, 2.0, 2.5, 1.2, '#E0F7FA', '3D Regularization\n(Hourglass)', 'Stage 4'),
        (11.5, -1.0, 2.5, 1.2, '#FFF9C4', 'Disparity\nRegression', 'Stage 5'),
        (15.0, -1.0, 1.5, 1.2, '#C8E6C9', 'Output\n(B,1,H,W)', 'Output'),
    ]

    for x, y, w, h, color, label, stage in blocks:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='gray', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=8, linespacing=1.3)

    arrows = [
        (3.5, 7.5, 4.5, 7.1),
        (3.5, 4.5, 4.5, 4.6),
        (7.0, 7.1, 8.0, 5.6),
        (7.0, 4.6, 8.0, 5.6),
        (10.5, 5.6, 11.5, 5.6),
        (12.75, 4.4, 12.75, 3.2),
        (12.75, 2.0, 12.75, -0.4),
        (14.0, -0.4, 15.0, -0.4),
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    legend_text = (
        "Architecture Overview\n"
        "----------------------\n"
        "Input: RGB (3ch) + NIR (1ch)\n"
        "1. Feature Extraction (PSMNet + SPP)\n"
        "2. Cross-Attention Feature Alignment\n"
        "3. Cost Volume Construction\n"
        "4. 3D Hourglass Regularization\n"
        "5. Disparity Regression + Upsampling\n"
        "Output: Full-res disparity + variance"
    )
    ax.text(0.5, 2.0, legend_text, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    pipeline_path = f'{save_dir}/pipeline_architecture.png'
    plt.savefig(pipeline_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {pipeline_path}")

    # ========== 4. Data Analysis Modules ==========
    print("\n=== Running Data Analysis Modules ===")

    # 4a. Regression analysis (对标 Lab0525)
    try:
        print("\n[1/4] Regression Analysis...")
        run_regression_analysis(pred_disp, gt_np, mask_np, save_dir=save_dir)
    except Exception as e:
        print(f"[WARN] Regression analysis failed: {e}")

    # 4b. Classification metrics (对标 Lab0608)
    try:
        print("\n[2/4] Classification Metrics...")
        run_classification_analysis(pred_disp, gt_np, mask_np,
                                     max_disparity=max_disp, save_dir=save_dir)
    except Exception as e:
        print(f"[WARN] Classification metrics failed: {e}")

    # 4c. EDA analysis (对标 Lab0518)
    try:
        print("\n[3/4] Dataset EDA...")
        run_dataset_eda(rgb, nir, pred_disp, gt_np, mask_np, save_dir=save_dir)
    except Exception as e:
        print(f"[WARN] Dataset EDA failed: {e}")

    # 4d. Save data to SQLite + CSV (对标 Lab0511)
    try:
        print("\n[4/4] Data Storage...")
        store = ExperimentDataStore(db_path=f'{save_dir}/experiment_data.db',
                                     csv_dir=save_dir)
        store.save_model_summary('cross_modal_stereo',
                                  model_params=sum(p.numel() for p in model.parameters()),
                                  best_l1=loss_val.item(),
                                  best_epoch=0,
                                  r2_score=None,
                                  mse=metrics['rmse'])
        store.export_all_to_csv()
        store.close()
    except Exception as e:
        print(f"[WARN] Data storage failed: {e}")

    # ========== 5. Save combined results ==========
    np.savez(f'{save_dir}/results.npz',
             rgb=rgb.cpu().numpy(), nir=nir.cpu().numpy(),
             gt_disp=gt_np, pred_disp=pred_disp,
             variance=variance, mask=mask_np,
             abs_rel=metrics['abs_rel'], rmse=metrics['rmse'],
             delta1=metrics['delta1'])

    print(f"\n{'='*55}")
    print(f"  Visualization Complete!")
    print(f"{'='*55}")
    print(f"\nGenerated files in {save_dir}/:")
    print(f"  model_performance.png      - Model inference comparison")
    print(f"  pipeline_architecture.png   - Pipeline architecture diagram")
    print(f"  regression_scatter.png      - Regression analysis (Lab0525)")
    print(f"  residual_analysis.png       - Residual analysis (Lab0525)")
    print(f"  confusion_matrix.png        - Confusion matrix (Lab0608)")
    print(f"  precision_recall.png        - Precision/Recall (Lab0608)")
    print(f"  class_distribution.png      - Class distribution (Lab0608)")
    print(f"  channel_distribution.png    - Channel pixel distribution (Lab0518)")
    print(f"  disparity_analysis.png      - Disparity analysis (Lab0518)")
    print(f"  experiment_data.db          - SQLite database (Lab0511)")
    print(f"  *.csv                       - Exported CSV tables (Lab0511)")
    print(f"  results.npz                 - All data for further analysis")
    print(f"\nTotal model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return metrics


def make_scene(H=192, W=320, max_disp=96, device='cpu', sparsity=0.05):
    """Generate synthetic structured scene for testing."""
    u = torch.linspace(0, 1, W).view(1, -1).expand(H, -1)
    v = torch.linspace(0, 1, H).view(-1, 1).expand(-1, W)

    rgb = torch.zeros(1, 3, H, W)
    sky = v < 0.35
    rgb[0, 0][sky] = 0.4 + 0.2 * u[sky]
    rgb[0, 1][sky] = 0.5 + 0.2 * v[sky]
    rgb[0, 2][sky] = 0.8

    road = v > 0.65
    road_pattern = 0.5 + 0.1 * torch.sin(5 * u[road]) + 0.05 * torch.cos(20 * v[road])
    for c in range(3):
        rgb[0, c][road] = road_pattern

    mid = ~sky & ~road
    for c in range(3):
        rgb[0, c][mid] = 0.5 + 0.1 * torch.sin(10 * u[mid] + 5 * v[mid])

    objects = [
        (0.3, 0.5, 0.12, 0.08, [0.7, 0.2, 0.2], True),
        (0.7, 0.55, 0.10, 0.07, [0.2, 0.5, 0.8], True),
        (0.5, 0.3, 0.15, 0.25, [0.6, 0.6, 0.6], False),
        (0.85, 0.5, 0.08, 0.20, [0.3, 0.7, 0.3], False),
        (0.15, 0.7, 0.06, 0.06, [0.9, 0.9, 0.1], True),
    ]

    gt_disp = torch.zeros(1, 1, H, W)
    bg_disp = max_disp * (0.15 + 0.7 * (1 - v) + 0.15 * u)
    gt_disp[0, 0] = bg_disp

    for cu, cv, cw, ch, color, is_car in objects:
        mask_u = (u - cu).abs() < cw
        mask_v = (v - cv).abs() < ch
        obj_mask = mask_u & mask_v
        for c in range(3):
            rgb[0, c][obj_mask] = color[c]
        if is_car:
            gt_disp[0, 0][obj_mask] = bg_disp[obj_mask] * 1.4 + 10
        else:
            gt_disp[0, 0][obj_mask] = bg_disp[obj_mask] * 0.7

    detail = 0.03 * torch.randn(1, 3, H, W)
    for c in range(3):
        rgb[0, c] = (rgb[0, c] + detail[0, c]).clamp(0, 1)

    nir = torch.zeros(1, 1, H, W)
    nir[0, 0] = 0.4 + 0.2 * (1 - v)
    for cu, cv, cw, ch, color, is_car in objects:
        mask_u = (u - cu).abs() < cw
        mask_v = (v - cv).abs() < ch
        obj_mask = mask_u & mask_v
        if is_car:
            nir[0, 0][obj_mask] = 0.8 + 0.15 * torch.randn(obj_mask.sum())
        else:
            nir[0, 0][obj_mask] = 0.5
    nir[0, 0] = nir[0, 0].clamp(0, 1)

    gt_disp = gt_disp.clamp(0, max_disp)

    mask = torch.zeros(1, 1, H, W)
    n_valid = int(sparsity * H * W)
    mask[0, 0, torch.randint(0, H, (n_valid,)), torch.randint(0, W, (n_valid,))] = 1.0

    return rgb.to(device), nir.to(device), gt_disp.to(device), mask.to(device), objects


def main():
    """Main entry point for standalone visualization."""
    device = torch.device('cpu')
    B, H, W = 1, 192, 320
    max_disp = 96

    print("Creating model...")
    model = RGBNIRStereoModel(
        max_disparity=max_disp,
        num_candidates=24,
        use_shared_extractor=False,
        use_lightweight=True,
        mode='simple',
    ).to(device)
    model.eval()

    print("Generating synthetic scene...")
    rgb, nir, gt_disp, mask, objects = make_scene(H, W, max_disp, device)

    generate_all_visualizations(model, rgb, nir, gt_disp, mask, objects,
                                 save_dir='visualizations')


if __name__ == "__main__":
    main()