"""
Training script for RGB + NIR Cross-Modal Stereo Matching → Disparity Estimation.

Uses RGBNIRStereoModel with cost volume + 3D regularization.

Usage:
    python train.py
    python train.py --config config.yaml
    python train.py --resume checkpoints/best.pth
    python train.py --debug
"""

import os, sys, time, argparse, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None


def compute_loss(model_out, batch, cfg, device):
    """
    Compute training loss for disparity prediction.

    Loss combination:
      L_total = w_l1 * L1(pred, gt) + w_nll * NLL(gt, pred, variance) + w_edge * silog(pred, gt)
    """
    from models.losses import l1_disparity_loss, negative_log_likelihood_loss, silog_loss
    import torch.nn.functional as F

    max_disp = cfg['camera'].get('max_disparity', 48)
    pred_raw = model_out['disparity']          # (B,1,H,W) raw output
    gt       = batch['disparity'].to(device)   # (B,1,H,W) disparity in pixels

    variance = model_out.get('variance', None)       # (B,1,H,W) or None
    mask     = batch['valid_mask'].to(device)         # (B,1,H,W)

    # Range penalty (soft): guide model to stay within valid range
    # Without gradient cutoff of hard clamp
    loss_range = (F.relu(pred_raw - max_disp) ** 2 + F.relu(-pred_raw + 0.5) ** 2)[mask > 0].mean() * 0.5

    # L1 loss (on raw output, no gradient cutoff)
    loss_l1 = l1_disparity_loss(pred_raw, gt, mask, valid_threshold=0.0)

    # NLL loss (uncertainty-aware)
    loss_nll = torch.tensor(0.0, device=device)
    w_nll = cfg['loss'].get('nll_weight', 0.0)
    if w_nll > 0 and variance is not None:
        loss_nll = negative_log_likelihood_loss(gt, pred_raw, variance, mask)

    # Edge-aware smoothness loss
    loss_edge = torch.tensor(0.0, device=device)
    if cfg['loss'].get('edge_smooth_weight', 0) > 0:
        loss_edge = silog_loss(pred_raw, gt, mask) * cfg['loss']['edge_smooth_weight']

    # MDP NLL loss (Eq.9): supervised on depth GT in meters
    loss_mdp = torch.tensor(0.0, device=device)
    w_mdp = cfg['loss'].get('nll_mdp_weight', 0.0)
    if w_mdp > 0 and 'mdp_vis' in model_out and 'mdp_thr' in model_out:
        depth_gt = batch['depth'].to(device)  # (B,1,H,W)
        for mdp_out in [model_out['mdp_vis'], model_out['mdp_thr']]:
            loss_mdp = loss_mdp + negative_log_likelihood_loss(
                depth_gt, mdp_out['mu'], mdp_out['sigma2'], mask
            )
        loss_mdp = loss_mdp * 0.5  # average over two modalities

    total_loss = (cfg['loss']['smoothl1_weight'] * loss_l1
                  + w_nll * loss_nll
                  + loss_edge
                  + loss_range
                  + w_mdp * loss_mdp)

    return total_loss, {
        'loss_l1':   loss_l1.item(),
        'loss_nll':  loss_nll.item(),
        'loss_edge': loss_edge.item(),
        'loss_range': loss_range.item(),
        'loss_mdp':  loss_mdp.item(),
    }


def validate(model, val_loader, cfg, device):
    """Run validation and return depth + disparity metrics."""
    model.eval()
    from models.losses import depth_metrics, disparity_metrics, l1_disparity_loss

    total_l1 = 0.0
    all_depth_metrics = []
    all_disp_metrics = []
    num_batches = 0

    baseline_m = cfg['camera']['baseline_m']
    fx = cfg['camera']['fx']

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch['rgb'].to(device)
            nir = batch['nir'].to(device)

            calib_args = {}
            if 'K_rgb' in batch:
                calib_args = {
                    'K_rgb': batch['K_rgb'].to(device),
                    'K_nir': batch['K_nir'].to(device),
                    'R': batch['R'].to(device),
                    'T': batch['T'].to(device),
                }
            out = model(rgb, nir, **calib_args)

            pred = out['disparity'].clamp(min=0.5, max=48.0)  # clamp to prevent depth explosion
            gt = batch['disparity'].to(device)
            mask = batch['valid_mask'].to(device)

            total_l1 += l1_disparity_loss(pred, gt, mask).item()

            # Disparity metrics: EPE, D1-all
            disp_m = disparity_metrics(pred, gt, mask)
            all_disp_metrics.append(disp_m)

            # Convert disparity to depth for standard metrics
            pred_depth = baseline_m * fx / (pred.clamp(min=0.5))  # 0.5→max depth 76.8m
            gt_depth = batch['depth'].to(device)
            depth_m = depth_metrics(pred_depth, gt_depth, mask)
            all_depth_metrics.append(depth_m)

            num_batches += 1

    avg = {}
    for k in all_depth_metrics[0].keys():
        avg[k] = np.mean([m[k] for m in all_depth_metrics])
    for k in all_disp_metrics[0].keys():
        avg[k] = np.mean([m[k] for m in all_disp_metrics])
    avg['l1'] = total_l1 / num_batches
    return avg


def create_model(cfg, device):
    """Create RGBNIRStereoModel."""
    from models.rgb_nir_model import RGBNIRStereoModel
    camera_cfg = cfg.get('camera', {})
    model = RGBNIRStereoModel(
        max_disparity=camera_cfg.get('max_disparity', 48),
        num_candidates=camera_cfg.get('num_candidates', 48),
        feature_channels=cfg['model'].get('feature_channels', 32),
        use_shared_extractor=False,
        use_lightweight=cfg['model'].get('use_lightweight', True),
        use_mdp=cfg['model'].get('use_mdp', False),
        mode='simple',
    ).to(device)
    print(f"  Model: RGBNIRStereoModel (max_disp={camera_cfg.get('max_disparity', 48)})")
    return model


def train():
    # Linux multiprocessing fix: must use 'spawn' to avoid CUDA fork error
    # 'fork' is default on Linux and breaks CUDA in DataLoader workers
    import platform
    if platform.system() != 'Windows':
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # already set

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print("Model: RGBNIRStereoModel (stereo disparity estimation)")

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Device: {device}")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}, {gpu_mem:.1f} GB")

    # ============================================
    # Dataset
    # ============================================
    from dataset.train_data_dataset import TrainDataDataset

    print("Loading dataset...")
    camera_cfg = cfg.get('camera', {})

    # Enable GPU-side resize to reduce CPU bottleneck
    gpu_resize = cfg['data'].get('gpu_resize', True) and use_cuda
    if gpu_resize:
        print("  GPU resize: ON (resize operations executed on GPU)")
    full_dataset = TrainDataDataset(
        root=cfg['data']['root'],
        input_height=cfg['data']['input_height'],
        input_width=cfg['data']['input_width'],
        normalize_depth=True,
        output_mode='disparity',
        fx=camera_cfg.get('fx', 764.5),
        baseline_m=camera_cfg.get('baseline_m', 0.0503),
        max_disparity=camera_cfg.get('max_disparity', 48),
        resize_on_gpu=gpu_resize,
    )

    # Train/val split
    total_n = len(full_dataset)
    val_root = cfg['data'].get('val_root', None)
    if val_root and val_root != cfg['data']['root']:
        train_dataset = full_dataset
        val_dataset = TrainDataDataset(
            root=val_root,
            input_height=cfg['data']['input_height'],
            input_width=cfg['data']['input_width'],
            normalize_depth=True,
            output_mode='disparity',
            fx=camera_cfg.get('fx', 764.5),
            baseline_m=camera_cfg.get('baseline_m', 0.0503),
            max_disparity=camera_cfg.get('max_disparity', 48),
            resize_on_gpu=gpu_resize,
        )
    else:
        train_ratio = cfg['data'].get('train_ratio', 0.8)
        val_size = max(1, int(total_n * (1 - train_ratio)))
        train_size = total_n - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    # Apply max_samples limits
    max_train = cfg['data'].get('max_train_samples', None)
    max_val   = cfg['data'].get('max_val_samples', None)
    if max_train is not None and len(train_dataset) > max_train:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(max_train)))
    if max_val is not None and len(val_dataset) > max_val:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(max_val)))

    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(min(8, len(train_dataset)))))
        val_dataset   = torch.utils.data.Subset(val_dataset,   list(range(min(4, len(val_dataset)))))
        cfg['output']['print_freq'] = 1  # debug mode: log every batch
        cfg['output']['vis_freq'] = max(1, cfg['output'].get('vis_freq', 100) // 10)
        print("  Debug mode: print_freq=1")

    # Collate function: use GPU resize collate when enabled
    collate_fn = None
    if gpu_resize:
        from dataset.train_data_dataset import gpu_resize_collate_fn
        collate_fn = gpu_resize_collate_fn
        print("  Using GPU-resize collate function")

    nw = cfg['data']['num_workers'] if use_cuda else 0
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['train']['batch_size'],
        shuffle=True, num_workers=nw, pin_memory=use_cuda,
        drop_last=True, persistent_workers=(nw > 0),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['train']['batch_size'],
        shuffle=False, num_workers=nw, pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        collate_fn=collate_fn,
    ) if len(val_dataset) > 0 else None

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples")

    # ============================================
    # Model
    # ============================================
    model = create_model(cfg, device)

    # DataParallel disabled: conflicts with dynamic depth_candidates buffer
    # registered inside RGBNIRStereoModel.forward() via set_depth_candidates()
    use_dp = cfg['train'].get('use_data_parallel', False)
    if use_dp and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs, using GPU 0 only "
              f"(DataParallel disabled for homography+MDP compatibility)")

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ============================================
    # Optimizer & Scheduler
    # ============================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['train']['learning_rate'],
        weight_decay=cfg['train']['weight_decay'],
    )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        start_epoch = ckpt.get('epoch', 0)
        optimizer.load_state_dict(ckpt.get('optimizer', optimizer.state_dict()))
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    if cfg['train']['lr_scheduler'] == 'cycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg['train']['cycle_max_lr'],
            total_steps=cfg['train']['epochs'] * len(train_loader),
            pct_start=cfg['train']['cycle_pct_start'],
        )
    elif cfg['train']['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg['train']['step_size'],
            gamma=cfg['train']['step_gamma'],
        )
    else:
        scheduler = None

    # ============================================
    # Logger
    # ============================================
    writer = SummaryWriter(cfg['output']['log_dir']) if HAS_TENSORBOARD else None
    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['output']['vis_dir'], exist_ok=True)

    # ============================================
    # Training Loop
    # ============================================
    print(f"\n{'='*60}")
    print(f"Training for {cfg['train']['epochs']} epochs")
    print(f"{'='*60}")

    global_step = 0
    best_l1 = float('inf')

    train_history = {'step': [], 'epoch': [], 'loss': [], 'l1': [], 'lr': []}
    val_history   = {'epoch': [], 'abs_rel': [], 'sq_rel': [], 'rmse': [],
                     'rmse_log': [], 'delta1': [], 'delta2': [], 'delta3': [], 'l1': [],
                     'epe': [], 'd1_all': []}

    # Gradient accumulation
    grad_accum_steps = cfg['train'].get('grad_accumulation_steps', 1)
    if grad_accum_steps > 1:
        print(f"  Gradient accumulation: {grad_accum_steps} steps "
              f"(effective batch_size={cfg['train']['batch_size'] * grad_accum_steps})")

    for epoch in range(start_epoch, cfg['train']['epochs']):
        model.train()
        epoch_start = time.time()
        optimizer.zero_grad()

        gpu_data_time = 0.0
        gpu_compute_time = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # --- GPU timing: data transfer ---
            if use_cuda:
                torch.cuda.synchronize()
                t0 = time.time()
            rgb = batch['rgb'].to(device, non_blocking=use_cuda)
            nir = batch['nir'].to(device, non_blocking=use_cuda)
            if use_cuda:
                torch.cuda.synchronize()
                gpu_data_time += time.time() - t0

            # --- GPU timing: forward + backward ---
            if use_cuda:
                torch.cuda.synchronize()
                t0 = time.time()
            # Pass camera params for homography cost volume
            calib_kwargs = {}
            if 'K_rgb' in batch:
                calib_kwargs = {
                    'K_rgb': batch['K_rgb'].to(device),
                    'K_nir': batch['K_nir'].to(device),
                    'R': batch['R'].to(device),
                    'T': batch['T'].to(device),
                }
            out = model(rgb, nir, **calib_kwargs)
            total_loss, loss_dict = compute_loss(out, batch, cfg, device)
            total_loss = total_loss / grad_accum_steps
            total_loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None and cfg['train']['lr_scheduler'] == 'cycle':
                    scheduler.step()

            if use_cuda:
                torch.cuda.synchronize()
                gpu_compute_time += time.time() - t0

            # Logging
            if global_step % cfg['output']['print_freq'] == 0:
                lr = optimizer.param_groups[0]['lr']
                loss_unscaled = total_loss.item() * grad_accum_steps
                gpu_info = ""
                if use_cuda:
                    alloc = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    if global_step > 0:
                        total_steps = global_step + 1
                        avg_data = gpu_data_time / total_steps * 1000
                        avg_comp = gpu_compute_time / total_steps * 1000
                        avg_total = avg_data + avg_comp
                        gpu_pct = avg_comp / avg_total * 100 if avg_total > 0 else 0
                        gpu_info = (f"| GPU mem: {alloc:.2f}/{reserved:.2f}GB "
                                    f"| Data {avg_data:.1f}ms | Compute {avg_comp:.1f}ms "
                                    f"| GPU%={gpu_pct:.0f}%")
                    else:
                        gpu_info = (f"| GPU mem: {alloc:.2f}/{reserved:.2f}GB "
                                    f"| (timing starts next step)")
                print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss {loss_unscaled:.4f} | "
                      f"L1 {loss_dict['loss_l1']:.4f} | "
                      f"LR {lr:.2e} "
                      f"{gpu_info}")
                if use_cuda:
                    print(f"  disp range: "
                          f"[{out['disparity'].min().item():.2f}, {out['disparity'].max().item():.2f}]")
                if writer is not None:
                    writer.add_scalar('train/loss', loss_unscaled, global_step)
                    writer.add_scalar('train/l1', loss_dict['loss_l1'], global_step)
                    writer.add_scalar('train/lr', lr, global_step)
                    if use_cuda:
                        writer.add_scalar('gpu/memory_allocated_gb',
                                          torch.cuda.memory_allocated() / 1e9, global_step)
                train_history['step'].append(global_step)
                train_history['epoch'].append(epoch + 1)
                train_history['loss'].append(loss_unscaled)
                train_history['l1'].append(loss_dict['loss_l1'])
                train_history['lr'].append(lr)

            # Visualization
            if global_step % cfg['output']['vis_freq'] == 0:
                _save_visualization(rgb, nir, out, batch, global_step, cfg)

            global_step += 1

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} finished in {epoch_time:.1f}s")

        if scheduler is not None and cfg['train']['lr_scheduler'] == 'step':
            scheduler.step()

        # Validation
        if val_loader is not None and (epoch + 1) % cfg['train']['val_interval'] == 0:
            metrics = validate(model, val_loader, cfg, device)
            print(f"\nValidation @ epoch {epoch+1}:")
            for k, v in metrics.items():
                print(f"  {k:12s}: {v:.4f}")
                if writer is not None:
                    writer.add_scalar(f'val/{k}', v, epoch)

            val_history['epoch'].append(epoch + 1)
            for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'delta1', 'delta2', 'delta3', 'l1', 'epe', 'd1_all']:
                if k in metrics:
                    val_history[k].append(metrics[k])

            if metrics['l1'] < best_l1:
                best_l1 = metrics['l1']
                best_path = os.path.join(cfg['output']['checkpoint_dir'], 'best.pth')
                # Persist architecture flags into camera_config so infer.py can reconstruct the model
                save_camera_cfg = dict(camera_cfg)
                save_camera_cfg['use_lightweight'] = cfg['model'].get('use_lightweight', True)
                save_camera_cfg['use_mdp'] = cfg['model'].get('use_mdp', False)
                save_camera_cfg['feature_channels'] = cfg['model'].get('feature_channels', 32)
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'metrics': metrics,
                    'model_type': 'stereo',
                    'camera_config': save_camera_cfg,
                }, best_path)
                print(f"  -> New best model saved to {best_path}")

        # Save checkpoint
        if (epoch + 1) % cfg['train']['save_interval'] == 0:
            ckpt_path = os.path.join(cfg['output']['checkpoint_dir'], f'epoch_{epoch+1:03d}.pth')
            # Persist architecture flags into camera_config so infer.py can reconstruct the model
            save_camera_cfg = dict(camera_cfg)
            save_camera_cfg['use_lightweight'] = cfg['model'].get('use_lightweight', True)
            save_camera_cfg['use_mdp'] = cfg['model'].get('use_mdp', False)
            save_camera_cfg['feature_channels'] = cfg['model'].get('feature_channels', 32)
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_type': 'stereo',
                'camera_config': save_camera_cfg,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # ============================================
    # Save history & generate plots
    # ============================================
    if train_history['step']:
        np.savez(os.path.join(cfg['output']['vis_dir'], 'training_history.npz'), **train_history)
    if val_history['epoch']:
        np.savez(os.path.join(cfg['output']['vis_dir'], 'validation_history.npz'), **val_history)
    _plot_training_curves(train_history, val_history, cfg)

    # ============================================
    # Inference Demo
    # ============================================
    infer_num = cfg['output'].get('infer_num_samples', 0)
    if infer_num > 0:
        print(f"\n{'='*60}")
        print(f"  Running Post-Training Inference Demo ({infer_num} samples)...")
        print(f"{'='*60}")
        if val_loader is not None and len(val_dataset) > 0:
            _run_inference_demo(model, val_dataset, cfg, device, infer_num)
        else:
            print("  [WARN] No validation data, skipping inference demo.")
    else:
        print(f"  [INFO] Inference demo skipped (infer_num_samples={infer_num})")

    # ============================================
    # Data Analysis
    # ============================================
    print(f"\n{'='*60}")
    print(f"  Running Data Analysis Modules...")
    print(f"{'='*60}")
    try:
        from data_analysis.data_storage import ExperimentDataStore
        store = ExperimentDataStore(
            db_path=os.path.join(cfg['output']['vis_dir'], 'experiment_data.db'),
            csv_dir=cfg['output']['vis_dir'],
        )
        store.save_config(cfg['output']['exp_name'], cfg)
        for i in range(len(train_history['step'])):
            store.save_training_record(
                cfg['output']['exp_name'],
                train_history['epoch'][i], 0,
                train_history['loss'][i],
                train_history['l1'][i],
                train_history['lr'][i],
            )
        for i in range(len(val_history['epoch'])):
            vm = {}
            for k in ['abs_rel','sq_rel','rmse','rmse_log','delta1','delta2','delta3','l1','epe','d1_all']:
                if k in val_history and len(val_history[k]) > i:
                    vm[k] = val_history[k][i]
            store.save_validation_record(cfg['output']['exp_name'], val_history['epoch'][i], vm)
        store.save_model_summary(
            cfg['output']['exp_name'],
            model_params=sum(p.numel() for p in model.parameters()),
            best_l1=best_l1 if best_l1 != float('inf') else 0,
            best_epoch=val_history['epoch'][-1] if val_history['epoch'] else 0,
        )
        store.export_all_to_csv()
        store.close()
        print(f"  [data_storage] Export complete")
    except Exception as e:
        print(f"  [WARN] Data storage module failed: {e}")
        import traceback; traceback.print_exc()

    if writer is not None:
        writer.close()
    print(f"\nTraining complete! Best L1: {best_l1:.4f}")


# ============================================================================
# Visualization helpers
# ============================================================================

def _get_disp_depth(out, batch, cfg):
    """Extract display values: convert disparity -> depth in meters for visualization."""
    camera_cfg = cfg.get('camera', {})
    fx = camera_cfg.get('fx', 764.5)
    baseline_m = camera_cfg.get('baseline_m', 0.0503)
    pred_disp = out['disparity']
    pred_d = baseline_m * fx / pred_disp.clamp(min=1e-6)
    gt_d = batch['depth'].to(pred_d.device)
    return pred_d, gt_d


def _save_visualization(rgb, nir, out, batch, step, cfg):
    """Save RGB, NIR, GT depth, predicted depth side-by-side."""
    import matplotlib; matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    import matplotlib.pyplot as plt

    pred_d, gt_d = _get_disp_depth(out, batch, cfg)

    bs = min(rgb.shape[0], 2)
    for b in range(bs):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Row 1
        rgb_img = rgb[b].cpu().permute(1,2,0).numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
        axes[0,0].imshow(rgb_img); axes[0,0].set_title('RGB Left')

        nir_img = nir[b,0].cpu().numpy()
        axes[0,1].imshow(nir_img, cmap='gray'); axes[0,1].set_title('NIR Right')

        gt_d_np = gt_d[b,0].cpu().numpy()
        mask = batch['valid_mask'][b,0].cpu().numpy() > 0
        gt_v = np.where(mask, gt_d_np, 0)
        im = axes[0,2].imshow(gt_v, cmap='jet', vmin=0, vmax=80)
        axes[0,2].set_title(f'GT Depth ({mask.sum()/mask.size*100:.1f}% valid)')
        plt.colorbar(im, ax=axes[0,2])

        # Row 2
        pred_d_np = pred_d[b,0].cpu().detach().numpy()
        im = axes[1,0].imshow(pred_d_np, cmap='jet', vmin=0, vmax=80)
        axes[1,0].set_title('Predicted Depth (from disp)')
        plt.colorbar(im, ax=axes[1,0])

        var = out.get('variance', None)
        if var is not None:
            v = var[b,0].cpu().detach().numpy()
            im = axes[1,1].imshow(v, cmap='hot')
            axes[1,1].set_title('Uncertainty (variance)')
            plt.colorbar(im, ax=axes[1,1])
        else:
            axes[1,1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Uncertainty (variance)')

        if mask.any():
            error = np.abs(pred_d_np - gt_d_np) * mask
            im = axes[1,2].imshow(error, cmap='hot', vmin=0, vmax=10)
            axes[1,2].set_title(f'Abs Error (mean: {error[mask].mean():.2f})')
            plt.colorbar(im, ax=axes[1,2])

        for ax in axes.flat: ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg['output']['vis_dir'], f'step_{step:06d}_batch{b}.png'),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)


def _plot_training_curves(train_history, val_history, cfg):
    """Generate training/validation plots."""
    import matplotlib; matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    import matplotlib.pyplot as plt

    vis_dir = cfg['output']['vis_dir']

    # All plots go into a single dashboard folder
    dashboard_dir = os.path.join(vis_dir, 'training_dashboard')
    os.makedirs(dashboard_dir, exist_ok=True)

    # --- Training metrics ---
    if train_history['step']:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_history['step'], train_history['loss'], 'b-', lw=1.5)
        ax.set_title('Training Loss'); ax.set_xlabel('Step'); ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(dashboard_dir, 'training_loss.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_history['step'], train_history['l1'], 'g-', lw=1.5)
        ax.set_title('L1 Loss (Disparity)'); ax.set_xlabel('Step'); ax.set_ylabel('L1')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(dashboard_dir, 'l1_loss.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_history['step'], train_history['lr'], 'r-', lw=1.5)
        ax.set_title('Learning Rate'); ax.set_xlabel('Step'); ax.set_ylabel('LR')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(dashboard_dir, 'learning_rate.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    # --- Validation metrics ---
    if val_history['epoch']:
        ep = val_history['epoch']

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ep, val_history['l1'], 'o-', color='royalblue', lw=2, ms=8)
        ax.set_title('Validation L1 (Disparity)'); ax.set_xlabel('Epoch'); ax.set_ylabel('L1')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(dashboard_dir, 'val_l1.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

        if 'abs_rel' in val_history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ep, val_history['abs_rel'], 'o-', color='darkorange', lw=2, ms=8)
            ax.set_title('Abs Rel (Depth)'); ax.set_xlabel('Epoch'); ax.set_ylabel('Abs Rel')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(dashboard_dir, 'abs_rel.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

        if 'sq_rel' in val_history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ep, val_history['sq_rel'], 'o-', color='purple', lw=2, ms=8)
            ax.set_title('Sq Rel (Depth)'); ax.set_xlabel('Epoch'); ax.set_ylabel('Sq Rel')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(dashboard_dir, 'sq_rel.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

        if 'rmse' in val_history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ep, val_history['rmse'], 'o-', color='crimson', lw=2, ms=8)
            ax.set_title('RMSE (Depth)'); ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(dashboard_dir, 'rmse.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

        if 'delta1' in val_history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ep, val_history['delta1'], 'o-', color='seagreen', lw=2, ms=8)
            ax.set_title('delta < 1.25 (Depth)'); ax.set_xlabel('Epoch'); ax.set_ylabel('delta1')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(dashboard_dir, 'delta1.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

        if 'epe' in val_history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ep, val_history['epe'], 'o-', color='teal', lw=2, ms=8)
            ax.set_title('EPE (Disparity, px)'); ax.set_xlabel('Epoch'); ax.set_ylabel('EPE')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(dashboard_dir, 'epe.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

        if 'd1_all' in val_history:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ep, val_history['d1_all'], 'o-', color='brown', lw=2, ms=8)
            ax.set_title('D1-all (Bad Pixel Rate)'); ax.set_xlabel('Epoch'); ax.set_ylabel('D1-all')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(dashboard_dir, 'd1_all.png'), dpi=120, bbox_inches='tight')
            plt.close(fig)

    print(f"[vis] Individual dashboard plots saved to {dashboard_dir}/")


def _run_inference_demo(model, val_dataset, cfg, device, num_samples=4):
    """
    Post-training inference: one image per validation sample.
    Each image: 2x2 panel: RGB / NIR / GT Depth / Predicted Depth.
    Saved to visualizations/inference_dashboard/.
    """
    import matplotlib; matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    import matplotlib.pyplot as plt

    model.eval()
    vis_dir = cfg['output']['vis_dir']
    camera_cfg = cfg.get('camera', {})
    fx = camera_cfg.get('fx', 764.5)
    baseline_m = camera_cfg.get('baseline_m', 0.0503)
    th = cfg['data']['input_height']
    tw = cfg['data']['input_width']

    total_n = len(val_dataset)
    if total_n == 0:
        print("  [WARN] No validation samples available for inference demo.")
        return
    ns = min(num_samples, total_n)
    if ns >= total_n:
        indices = list(range(total_n))
    else:
        indices = sorted(random.sample(range(total_n), ns))

    infer_dir = os.path.join(vis_dir, 'inference_dashboard')
    os.makedirs(infer_dir, exist_ok=True)

    from models.losses import disparity_metrics

    print(f"\n  {'='*50}")
    print(f"  Per-sample metrics (depth L1 in meters, EPE and D1-all in pixels):")
    print(f"  {'='*50}")

    # Check if samples need GPU resize (gpu_resize_collate mode)
    sample0 = val_dataset[0]
    need_resize = 'orig_size' in sample0 and sample0['orig_size'] != (th, tw)

    with torch.no_grad():
        for i in range(ns):
            batch = val_dataset[indices[i]]
            stem = batch.get('stem', f'sample_{i+1}')

            # Handle GPU-resize mode: resize individual sample to target size
            if 'orig_size' in batch and batch['orig_size'] != (th, tw):
                from torchvision.transforms import functional as TF
                rgb_t = TF.resize(batch['rgb'].unsqueeze(0), (th, tw), antialias=True).to(device)
                nir_t = TF.resize(batch['nir'].unsqueeze(0), (th, tw), antialias=True).to(device)
                gt_t  = TF.resize(batch['depth'].unsqueeze(0), (th, tw),
                                  interpolation=TF.InterpolationMode.NEAREST).to(device)
                msk_t = TF.resize(batch['valid_mask'].unsqueeze(0), (th, tw),
                                  interpolation=TF.InterpolationMode.NEAREST).to(device)
                disp_t = TF.resize(batch['disparity'].unsqueeze(0), (th, tw),
                                   interpolation=TF.InterpolationMode.NEAREST).to(device)
                # Display images: use resized versions
                rgb_disp = TF.resize(batch['rgb'].unsqueeze(0), (th, tw), antialias=True)
                nir_disp = TF.resize(batch['nir'].unsqueeze(0), (th, tw), antialias=True)
                gt_d_np = TF.resize(batch['depth'].unsqueeze(0), (th, tw),
                                    interpolation=TF.InterpolationMode.NEAREST)[0, 0].numpy()
                gt_msk_np = TF.resize(batch['valid_mask'].unsqueeze(0), (th, tw),
                                      interpolation=TF.InterpolationMode.NEAREST)[0, 0].numpy() > 0
                rgb_img = rgb_disp[0].permute(1,2,0).cpu().numpy()
                nir_img = nir_disp[0, 0].cpu().numpy()
            else:
                rgb_t = batch['rgb'].unsqueeze(0).to(device)
                nir_t = batch['nir'].unsqueeze(0).to(device)
                gt_t  = batch['depth'].unsqueeze(0).to(device)
                msk_t = batch['valid_mask'].unsqueeze(0).to(device)
                disp_t = batch['disparity'].unsqueeze(0).to(device)
                gt_d_np = batch['depth'][0].numpy()
                gt_msk_np = batch['valid_mask'][0].numpy() > 0
                rgb_img = batch['rgb'].permute(1,2,0).numpy()
                nir_img = batch['nir'][0].numpy()

            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

            out = model(rgb_t, nir_t)

            pred_disp = out['disparity'].clamp(min=0.5)[0, 0].cpu().numpy()
            pred_d = baseline_m * fx / np.maximum(pred_disp, 0.5)

            gt_v   = np.where(gt_msk_np, gt_d_np, np.nan)
            pred_v = np.where(gt_msk_np, pred_d, np.nan)

            # Create per-sample 2x2 figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            axes[0,0].imshow(rgb_img)
            axes[0,0].set_title(f'{stem}: RGB Left'); axes[0,0].axis('off')

            axes[0,1].imshow(nir_img, cmap='gray')
            axes[0,1].set_title(f'{stem}: NIR Right'); axes[0,1].axis('off')

            im1 = axes[1,0].imshow(gt_v, cmap='jet', vmin=0, vmax=80)
            axes[1,0].set_title(f'{stem}: GT Depth'); axes[1,0].axis('off')
            plt.colorbar(im1, ax=axes[1,0], fraction=0.046)

            im2 = axes[1,1].imshow(pred_v, cmap='jet', vmin=0, vmax=80)
            axes[1,1].set_title(f'{stem}: Predicted Depth'); axes[1,1].axis('off')
            plt.colorbar(im2, ax=axes[1,1], fraction=0.046)

            plt.tight_layout()
            save_path = os.path.join(infer_dir, f'{stem}_inference.png')
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close(fig)

            # Metrics: depth L1, EPE, D1-all
            pred_d_t = baseline_m * fx / out['disparity'].clamp(min=0.5)
            loss = nn.L1Loss(reduction='mean')
            valid = msk_t > 0
            l1v = loss(pred_d_t[valid], gt_t[valid]).item() if valid.sum() > 0 else float('nan')

            dm = disparity_metrics(out['disparity'].cpu(), disp_t.cpu(), msk_t.cpu())
            epe = dm['epe']
            d1 = dm['d1_all']
            nv = int(valid.sum().item())
            print(f"    {stem}: L1={l1v:.4f}  EPE={epe:.4f}  D1-all={d1:.4f}  Valid={nv}  saved to {save_path}")

    print(f"  {'='*50}")
    print(f"  [infer] {ns} inference images saved to {infer_dir}/")


if __name__ == "__main__":
    train()