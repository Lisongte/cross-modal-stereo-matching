"""
Inference script for RGB + NIR cross-modal stereo disparity estimation.

Uses RGBNIRStereoModel (cost volume + disparity regression).

Usage:
    python infer.py --checkpoint checkpoints/best.pth --rgb_dir ./images
    python infer.py --checkpoint checkpoints/best.pth --rgb rgb.png --nir nir.png
    python infer.py --checkpoint checkpoints/best.pth --rgb_dir ./images --gt_dir ./depth_filtered
    python infer.py --checkpoint checkpoints/best.pth --data_root D:/MS2_Dataset/train_data
"""

import os, sys, glob, argparse, time, yaml, torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_image(path, target_h, target_w):
    img = Image.open(path).convert('RGB')
    img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  # (H,W,3)


def load_nir(path, target_h, target_w):
    img = Image.open(path).convert('L')
    img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  # (H,W)


def load_gt_depth(path, target_h, target_w):
    """Load uint16 depth GT, convert to meters (/200), return (H,W) float32."""
    img = Image.open(path)
    img = img.resize((target_w, target_h), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 200.0
    arr = np.clip(arr, 0, 80.0)
    return arr


def load_calibration(data_root, fallback_fx=None, fallback_baseline=None):
    """
    Load MS2 calibration from data_root/../calib.npy.
    Returns (K_rgb, K_nir, R, T, fx, baseline_m) tensors,
    or None if calibration is unavailable (falls back to simple mode).
    """
    calib_candidates = [
        os.path.join(os.path.dirname(data_root), 'calib.npy'),
        os.path.join(data_root, 'calib.npy'),
        os.path.join(data_root, '..', 'calib.npy'),
    ]
    calib_path = None
    for c in calib_candidates:
        if os.path.exists(c):
            calib_path = c
            break
    if calib_path is None:
        print(f"[infer] No calib.npy found near {data_root}, using simple mode (disparity space)")
        return None

    calib = np.load(calib_path, allow_pickle=True).item()
    K_rgb = torch.from_numpy(calib['K_rgbL'].astype(np.float32))  # (3,3)
    K_nir = torch.from_numpy(calib.get('K_nirR', calib['K_rgbL']).astype(np.float32))
    fx = float(K_rgb[0, 0])

    T_rgbL = calib['T_rgbL'].astype(np.float32)  # (3,)
    T_nirR = calib['T_nirR'].astype(np.float32)  # (3,)
    baseline_mm = float(np.linalg.norm(T_nirR - T_rgbL))
    if baseline_mm < 1.0:
        baseline_mm = 299.0
    baseline_m = baseline_mm / 1000.0

    R = torch.eye(3, dtype=torch.float32)
    T_vec = (T_nirR - T_rgbL).astype(np.float32) / 1000.0  # NIR - RGB, meters
    T = torch.from_numpy(T_vec.reshape(3, 1))

    print(f"[infer] Loaded calib: {calib_path}")
    print(f"        fx={fx:.2f}, baseline={baseline_m:.5f}m → homography cost volume (depth-space)")

    return K_rgb, K_nir, R, T, fx, baseline_m


def create_model(checkpoint_path, device, cfg=None):
    """Create and load RGBNIRStereoModel. Returns (model, camera_config)."""
    from models.rgb_nir_model import RGBNIRStereoModel

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    camera_cfg = ckpt.get('camera_config', {}) if isinstance(ckpt, dict) else {}
    # Also check model_config key (used by save_model.py format)
    model_cfg_ckpt = ckpt.get('model_config', {}) if isinstance(ckpt, dict) else {}

    if cfg is not None:
        camera_cfg = cfg.get('camera', camera_cfg)

    max_disp = int(camera_cfg.get('max_disparity', 48))
    num_cand = int(camera_cfg.get('num_candidates', 48))
    # feature_channels may be in camera_config (train.py) or model_config (save_model.py)
    feat_ch = int(camera_cfg.get('feature_channels') or model_cfg_ckpt.get('feature_channels', 32))

    # Read use_lightweight / use_mdp from checkpoint first, then config.yaml, then default
    use_lightweight_ckpt = camera_cfg.get('use_lightweight') if 'use_lightweight' in camera_cfg else None
    use_mdp_ckpt = camera_cfg.get('use_mdp') if 'use_mdp' in camera_cfg else None
    if use_lightweight_ckpt is None and cfg is not None:
        use_lightweight_ckpt = cfg.get('model', {}).get('use_lightweight', None)
    if use_lightweight_ckpt is None:
        use_lightweight_ckpt = True  # default
    if use_mdp_ckpt is None and cfg is not None:
        use_mdp_ckpt = cfg.get('model', {}).get('use_mdp', None)
    if use_mdp_ckpt is None:
        use_mdp_ckpt = False  # default

    model = RGBNIRStereoModel(
        max_disparity=max_disp,
        num_candidates=num_cand,
        feature_channels=feat_ch,
        use_shared_extractor=False,
        use_lightweight=use_lightweight_ckpt,
        use_mdp=use_mdp_ckpt,
        mode='simple',
    ).to(device)

    sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    return model, camera_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    parser.add_argument('--config', type=str, default=None,
                        help='config.yaml to override camera params')
    parser.add_argument('--rgb_dir', type=str, default=None)
    parser.add_argument('--rgb', type=str, default=None)
    parser.add_argument('--nir', type=str, default=None)
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Path to depth_filtered/ directory (uint16 PNG) for EPE/D1-all')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root dir with img_left/img_right/depth_filtered/. Auto-detects GT.')
    parser.add_argument('--output_dir', type=str, default='inference_results')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--var_threshold', type=float, default=None,
                        help='Uncertainty threshold for confidence-filtered depth (e.g. 1.0)')
    parser.add_argument('--degradation_k', type=float, default=None,
                        help='Paper Degradation Masking: GT filtered by '
                             '|GT - MDP-VIS mu| < k * sigma. Paper uses k=1.')
    parser.add_argument('--samples', type=str, default=None,
                        help='Comma-separated sample stems to process (e.g. 000163,000439)')
    parser.add_argument('--export_depth', action='store_true',
                        help='Export depth/disparity maps as .npy files')
    args = parser.parse_args()

    # Parse sample filter
    sample_filter = None
    if args.samples:
        sample_filter = set(s.strip() for s in args.samples.split(',') if s.strip())
        print(f"[infer] Filtering to {len(sample_filter)} samples: {sorted(sample_filter)}")

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[infer] Device: {device}")

    # Load config if provided (or auto-detected for MDP features)
    cfg = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    elif args.degradation_k is not None and os.path.exists('config.yaml'):
        # Degradation Masking requires MDP (use_mdp=True from config.yaml)
        with open('config.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        print("[infer] Auto-loaded config.yaml (needed for Degradation Masking / MDP)")

    # Create model (stereo only)
    model, camera_cfg = create_model(args.checkpoint, device, cfg)
    fx = float(camera_cfg.get('fx', 764.5))
    baseline_m = float(camera_cfg.get('baseline_m', 0.0503))
    max_disp = int(camera_cfg.get('max_disparity', 48))
    print(f"[infer] Camera: fx={fx:.2f}, baseline={baseline_m:.5f}m, max_disp={max_disp}")
    print(f"[infer] Model: RGBNIRStereoModel, params: {sum(p.numel() for p in model.parameters()):,}")

    os.makedirs(args.output_dir, exist_ok=True)

    H, W = 384, 640

    # Resolve GT directory
    gt_dir = args.gt_dir
    if gt_dir is None and args.data_root is not None:
        candidate = os.path.join(args.data_root, 'depth_filtered')
        if os.path.isdir(candidate):
            gt_dir = candidate
            print(f"[infer] Auto-detected GT from data_root: {gt_dir}")

    gt_available = gt_dir is not None and os.path.isdir(gt_dir)
    if gt_available:
        from models.losses import disparity_metrics
        print(f"[infer] GT directory: {gt_dir}  → EPE / D1-all will be computed")
    else:
        print(f"[infer] No GT directory provided. EPE / D1-all will be skipped.")
        print(f"         Use --gt_dir depth_filtered/ or --data_root D:/MS2_Dataset/train_data")

    # Load calibration if using data_root (training used homography cost volume!)
    calib = None
    if args.data_root:
        calib = load_calibration(args.data_root, fallback_fx=fx, fallback_baseline=baseline_m)
        if calib is not None:
            # Override fx/baseline with calibrated values
            K_rgb_cal, K_nir_cal, R_cal, T_cal, fx, baseline_m = calib
            max_disp = int(camera_cfg.get('max_disparity', 48))  # keep max_disp from ckpt

    # Collect image pairs
    pairs = []
    if args.data_root:
        # MS2 dataset layout: data_root/img_left/ (RGB) + data_root/img_right/ (NIR)
        left_dir = os.path.join(args.data_root, 'img_left')
        right_dir = os.path.join(args.data_root, 'img_right')
        if not os.path.isdir(left_dir) or not os.path.isdir(right_dir):
            print(f"ERROR: data_root missing img_left/ or img_right/: {args.data_root}")
            sys.exit(1)
        rgb_files = sorted(glob.glob(os.path.join(left_dir, '*.png')))
        for f in rgb_files:
            stem = os.path.splitext(os.path.basename(f))[0]
            nir_path = os.path.join(right_dir, os.path.basename(f))
            if os.path.exists(nir_path):
                pairs.append((stem, f, nir_path))
        print(f"[infer] Found {len(pairs)} pairs in {left_dir} / {right_dir}")
    elif args.rgb_dir:
        rgb_files = sorted(glob.glob(os.path.join(args.rgb_dir, '*rgb*')))
        if not rgb_files:
            rgb_files = sorted(glob.glob(os.path.join(args.rgb_dir, '*.png')))
        for f in rgb_files:
            stem, ext = os.path.splitext(os.path.basename(f))
            nir_candidates = [
                os.path.join(args.rgb_dir, stem.replace('rgb', 'nir').replace('left', 'right') + ext),
                os.path.join(args.rgb_dir, stem + '_nir' + ext),
                os.path.join(args.rgb_dir, stem.replace('_rgb', '_nir') + ext),
                # Fallback: try without extension (backward compat)
                os.path.join(args.rgb_dir, stem.replace('rgb', 'nir').replace('left', 'right')),
                os.path.join(args.rgb_dir, stem.replace('_rgb', '_nir')),
            ]
            nir_path = None
            for nc in nir_candidates:
                if os.path.exists(nc):
                    nir_path = nc
                    break
            if nir_path:
                pairs.append((stem, f, nir_path))
        print(f"[infer] Found {len(pairs)} pairs in {args.rgb_dir}")
    elif args.rgb and args.nir:
        pairs.append(("result", args.rgb, args.nir))
    else:
        print("ERROR: provide --data_root, --rgb_dir, or --rgb+--nir")
        sys.exit(1)

    for base, rgb_p, nir_p in pairs:
        if sample_filter is not None and base not in sample_filter:
            continue
        print(f"\n[infer] Processing: {base}")

        # Load images
        rgb = load_image(rgb_p, H, W)                     # (H,W,3)
        nir = load_nir(nir_p, H, W)                       # (H,W)

        # Convert to tensors
        rgb_t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().to(device)
        nir_t = torch.from_numpy(nir).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)

        t0 = time.time()
        with torch.no_grad():
            # Pass camera params for homography cost volume (matches training)
            if calib is not None:
                K_rgb_b = K_rgb_cal.unsqueeze(0).to(device)
                K_nir_b = K_nir_cal.unsqueeze(0).to(device)
                R_b = R_cal.unsqueeze(0).to(device)
                T_b = T_cal.unsqueeze(0).to(device)
                out = model(rgb_t, nir_t, K_rgb=K_rgb_b, K_nir=K_nir_b, R=R_b, T=T_b)
            else:
                out = model(rgb_t, nir_t)
        elapsed = time.time() - t0

        # Disparity -> depth conversion
        disp_pred = out['disparity'][0,0].cpu().numpy()
        disp_tensor = out['disparity'].cpu()  # (1,1,H,W) for metrics
        depth = baseline_m * fx / np.maximum(disp_pred, 0.5)  # min disp=0.5 → max depth=76.8m
        print(f"  inference: {elapsed:.3f}s")
        print(f"  disparity range: [{disp_pred.min():.2f}, {disp_pred.max():.2f}]")
        print(f"  depth range:     [{depth.min():.2f}, {depth.max():.2f}], mean={depth.mean():.2f}")

        # Compute EPE / D1-all if GT available
        epe = None
        d1_all = None
        if gt_available:
            gt_path = os.path.join(gt_dir, f"{base}.png")
            if os.path.exists(gt_path):
                gt_depth_m = load_gt_depth(gt_path, H, W)  # meters
                valid_mask = (gt_depth_m > 0).astype(np.float32)
                depth_safe = np.maximum(gt_depth_m, 0.01)
                gt_disp = baseline_m * fx / depth_safe
                gt_disp = np.clip(gt_disp, 0, max_disp)

                gt_disp_t = torch.from_numpy(gt_disp).unsqueeze(0).float()
                mask_t = torch.from_numpy(valid_mask).unsqueeze(0).float()
                dm = disparity_metrics(disp_tensor, gt_disp_t, mask_t)
                epe = dm['epe']
                d1_all = dm['d1_all']

        # Print metrics line
        if epe is not None:
            print(f"  EPE: {epe:.4f} px  |  D1-all: {d1_all:.4f}  |  (GT OK)")
        else:
            print(f"  EPE: N/A        |  D1-all: N/A         |  (no GT)")

        var = out['variance'][0,0].cpu().numpy() if 'variance' in out else None

        # MDP outputs: modality-specific depth predictions + uncertainty
        mdp_vis_depth = None
        mdp_vis_sigma = None
        mdp_thr_depth = None
        mdp_thr_sigma = None
        if 'mdp_vis' in out:
            mdp_vis_depth = out['mdp_vis']['mu'][0,0].cpu().numpy()
            mdp_vis_sigma = out['mdp_vis']['sigma2'][0,0].cpu().numpy()
        if 'mdp_thr' in out:
            mdp_thr_depth = out['mdp_thr']['mu'][0,0].cpu().numpy()
            mdp_thr_sigma = out['mdp_thr']['sigma2'][0,0].cpu().numpy()

        # Confidence-filtered depth (if variance threshold specified)
        depth_filtered = None
        if args.var_threshold is not None and var is not None:
            confident_mask = var < args.var_threshold
            depth_filtered = np.where(confident_mask, depth, np.nan)
            n_confident = confident_mask.sum()
            n_total = confident_mask.size
            print(f"  variance filter: {n_confident:,}/{n_total:,} ({n_confident/n_total*100:.1f}%) "
                  f"pixels below σ²<{args.var_threshold}")

        # Export depth/disparity maps
        if args.export_depth:
            np.save(os.path.join(args.output_dir, f'{base}_depth.npy'), depth)
            np.save(os.path.join(args.output_dir, f'{base}_disparity.npy'), disp_pred)
            if var is not None:
                np.save(os.path.join(args.output_dir, f'{base}_variance.npy'), var)
            if depth_filtered is not None:
                np.save(os.path.join(args.output_dir, f'{base}_depth_filtered.npy'), depth_filtered)
            print(f"  exported: .npy maps → {args.output_dir}")

        # Save visualization
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 2×2 layout: RGB Left | NIR Right | Filtered GT | Full GT
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0,0].imshow(rgb); axes[0,0].set_title('RGB Left'); axes[0,0].axis('off')
        axes[0,1].imshow(nir, cmap='gray'); axes[0,1].set_title('NIR Right'); axes[0,1].axis('off')

        gt_loaded = None
        if gt_available:
            gt_path = os.path.join(gt_dir, f"{base}.png")
            if os.path.exists(gt_path):
                gt_loaded = load_gt_depth(gt_path, H, W)

        # Bottom-left: Filtered GT
        if gt_loaded is not None and args.degradation_k is not None and mdp_vis_depth is not None:
            mdp_sigma = np.sqrt(np.maximum(mdp_vis_sigma, 1e-6))
            # Guard against uncalibrated tiny sigma: floor at 0.5m
            mdp_sigma = np.maximum(mdp_sigma, 0.5)
            reliable = np.abs(gt_loaded - mdp_vis_depth) < (args.degradation_k * mdp_sigma)
            gt_masked = np.where(reliable & (gt_loaded > 0), gt_loaded, np.nan)
            n_reliable = (reliable & (gt_loaded > 0)).sum()
            n_valid = (gt_loaded > 0).sum()
            im3 = axes[1,0].imshow(gt_masked, cmap='jet', vmin=0, vmax=80)
            axes[1,0].set_title(f'Filtered GT (k={args.degradation_k})\n'
                                f'{n_reliable}/{n_valid}={n_reliable/n_valid*100:.1f}%')
            print(f"  Degradation Mask (k={args.degradation_k}): "
                  f"{n_reliable}/{n_valid} ({n_reliable/n_valid*100:.1f}%) reliable GT pixels")
        elif gt_loaded is not None:
            im3 = axes[1,0].imshow(gt_loaded, cmap='jet', vmin=0, vmax=80)
            axes[1,0].set_title('GT Depth (full)')
        else:
            im3 = axes[1,0].imshow(np.zeros((H, W)), cmap='jet')
            axes[1,0].set_title('GT not available')
        axes[1,0].axis('off')
        plt.colorbar(im3, ax=axes[1,0], fraction=0.046)

        # Bottom-right: Full GT
        if gt_loaded is not None:
            im4 = axes[1,1].imshow(gt_loaded, cmap='jet', vmin=0, vmax=80)
            axes[1,1].set_title('GT Depth (full)')
        else:
            im4 = axes[1,1].imshow(np.zeros((H, W)), cmap='jet')
            axes[1,1].set_title('GT not available')
        axes[1,1].axis('off')
        plt.colorbar(im4, ax=axes[1,1], fraction=0.046)

        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f'{base}_depth.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  saved: {save_path}")

    print(f"\n[infer] Done! Results in {args.output_dir}")


if __name__ == "__main__":
    main()