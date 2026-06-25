"""
Model Save & Load Utility for RGBNIRStereoModel.

Usage:
    # Train 50 steps and save
    python save_model.py --mode train_and_save --save_path checkpoints/my_model.pth

    # Just save a fresh model (random weights)
    python save_model.py --mode save --save_path checkpoints/fresh_model.pth

    # Load and test
    python save_model.py --mode load_test --load_path checkpoints/my_model.pth

    # Train 100 steps, save, then load and visualize
    python save_model.py --mode full_demo
"""

import os, sys, argparse, time
import torch
import numpy as np

from models.rgb_nir_model import RGBNIRStereoModel
from models.losses import l1_disparity_loss, depth_metrics


def create_model(max_disparity=96, num_candidates=24, use_lightweight=True, device='cpu'):
    """Create a fresh model instance."""
    model = RGBNIRStereoModel(
        max_disparity=max_disparity,
        num_candidates=num_candidates,
        use_shared_extractor=False,
        use_lightweight=use_lightweight,
        mode='simple',
    ).to(device)
    return model


def save_checkpoint(model, optimizer=None, epoch=0, loss=None, metrics=None, save_path='checkpoints/model.pth'):
    """Save model checkpoint with full training state."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'stereo',
        'model_config': {
            'max_disparity': model.max_disparity,
            'num_candidates': model.num_candidates,
            'use_shared': model.use_shared,
            'mode': model.mode,
        },
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    print(f"  [SAVE] Model saved to: {save_path}")
    print(f"         Epoch: {epoch}, Loss: {loss:.4f}" if loss is not None else "")
    return save_path


def load_checkpoint(load_path, device='cpu'):
    """Load model checkpoint and return model + metadata."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    checkpoint = torch.load(load_path, map_location=device, weights_only=True)

    # Create model with saved config
    config = checkpoint.get('model_config', {})
    model = create_model(
        max_disparity=config.get('max_disparity', 96),
        num_candidates=config.get('num_candidates', 24),
        use_lightweight=config.get('use_lightweight', True),
        device=device,
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  [LOAD] Model loaded from: {load_path}")
    print(f"         Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"         Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"         Loss:  {checkpoint.get('loss', 'N/A')}")
    print(f"         Time:  {checkpoint.get('timestamp', 'N/A')}")

    return model, checkpoint


def train_steps(model, num_steps=50, B=2, H=192, W=320, max_disp=96, lr=1e-3, device='cpu'):
    """Train model for N steps on synthetic data and return metrics history."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=num_steps, pct_start=0.3)

    history = {'loss': [], 'abs_rel': [], 'rmse': [], 'delta1': []}

    print(f"\n  Training for {num_steps} steps (B={B}, {H}x{W})...")
    t0 = time.time()

    for step in range(num_steps):
        rgb = torch.randn(B, 3, H, W).to(device)
        nir = torch.randn(B, 1, H, W).to(device)

        # GT: tilted plane
        u = torch.linspace(0, 1, W).view(1, -1).expand(H, -1)
        v = torch.linspace(0, 1, H).view(-1, 1).expand(-1, W)
        gt_disp = (30 + 40 * u + 10 * v).unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1).to(device)

        # Sparse mask (5%)
        mask = torch.zeros(B, 1, H, W).to(device)
        for b in range(B):
            n = int(0.05 * H * W)
            mask[b, 0, torch.randint(0, H, (n,)), torch.randint(0, W, (n,))] = 1.0

        opt.zero_grad()
        out = model(rgb, nir)
        loss = l1_disparity_loss(out['disparity'], gt_disp, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        scheduler.step()

        if (step + 1) % 10 == 0 or step == num_steps - 1:
            with torch.no_grad():
                m = depth_metrics(out['disparity'], gt_disp, mask)
            history['loss'].append(loss.item())
            history['abs_rel'].append(m['abs_rel'])
            history['rmse'].append(m['rmse'])
            history['delta1'].append(m['delta1'])
            print(f"    Step {step+1:3d}/{num_steps} | Loss: {loss.item():.4f} | "
                  f"AbsRel: {m['abs_rel']:.4f} | RMSE: {m['rmse']:.2f}")

    elapsed = time.time() - t0
    print(f"  Training complete! Time: {elapsed:.1f}s, Final Loss: {history['loss'][-1]:.4f}")
    return history


def load_and_visualize(model, save_dir='visualizations'):
    """Run inference and print metrics."""
    from visualize_results import make_scene

    B, H, W = 1, 192, 320
    max_disp = model.max_disparity
    device = next(model.parameters()).device

    # Generate scene
    print("\n  Generating scene for inference...")
    # We'll simple recreate the scene inline
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
    gt_disp = torch.zeros(1, 1, H, W)
    bg_disp = max_disp * (0.15 + 0.7 * (1 - v) + 0.15 * u)
    gt_disp[0, 0] = bg_disp
    objects = [
        (0.3, 0.5, 0.12, 0.08, [0.7, 0.2, 0.2], True),
        (0.7, 0.55, 0.10, 0.07, [0.2, 0.5, 0.8], True),
        (0.5, 0.3, 0.15, 0.25, [0.6, 0.6, 0.6], False),
    ]
    for cu, cv, cw, ch, color, is_car in objects:
        mask_u = (u - cu).abs() < cw
        mask_v = (v - cv).abs() < ch
        mask = mask_u & mask_v
        for c in range(3):
            rgb[0, c][mask] = color[c]
        if is_car:
            gt_disp[0, 0][mask] = bg_disp[mask] * 1.4 + 10
        else:
            gt_disp[0, 0][mask] = bg_disp[mask] * 0.7
    gt_disp = gt_disp.clamp(0, max_disp)
    nir = torch.zeros(1, 1, H, W)
    nir[0, 0] = 0.4 + 0.2 * (1 - v)
    nir[0, 0] = nir[0, 0].clamp(0, 1)
    mask = torch.zeros(1, 1, H, W)
    n_valid = int(0.05 * H * W)
    mask[0, 0, torch.randint(0, H, (n_valid,)), torch.randint(0, W, (n_valid,))] = 1.0

    rgb, nir, gt_disp, mask = rgb.to(device), nir.to(device), gt_disp.to(device), mask.to(device)

    with torch.no_grad():
        out = model(rgb, nir)

    metrics = depth_metrics(out['disparity'], gt_disp, mask)
    loss = l1_disparity_loss(out['disparity'], gt_disp, mask)

    print(f"\n  {'='*45}")
    print(f"    Inference Metrics")
    print(f"  {'='*45}")
    print(f"    L1 Loss:    {loss.item():.4f}")
    print(f"    Abs Rel:    {metrics['abs_rel']:.4f}")
    print(f"    RMSE:       {metrics['rmse']:.2f}")
    print(f"    δ<1.25:     {metrics['delta1']:.4f}")
    print(f"    δ<1.25^2:   {metrics['delta2']:.4f}")
    print(f"    δ<1.25^3:   {metrics['delta3']:.4f}")
    print(f"  {'='*45}")

    return metrics, loss.item()


# =====================================================================
# Main modes
# =====================================================================

def mode_save(device, save_path):
    """Save a model with random weights."""
    print(f"\n{'='*55}")
    print("  Mode: SAVE - Create and save a fresh model")
    print(f"{'='*55}")
    model = create_model(device=device)
    save_checkpoint(model, save_path=save_path)
    print(f"\n  Done! File size: {os.path.getsize(save_path) / 1024:.1f} KB")
    return model


def mode_train_and_save(device, save_path, num_steps=50):
    """Train for N steps and save."""
    print(f"\n{'='*55}")
    print(f"  Mode: TRAIN_AND_SAVE - {num_steps} steps -> {save_path}")
    print(f"{'='*55}")

    model = create_model(device=device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    history = train_steps(model, num_steps=num_steps, device=device)

    # Save with metrics
    save_checkpoint(
        model, epoch=num_steps, loss=history['loss'][-1],
        metrics={'abs_rel': history['abs_rel'][-1], 'rmse': history['rmse'][-1]},
        save_path=save_path,
    )
    print(f"  File size: {os.path.getsize(save_path) / 1024:.1f} KB")

    # Quick inference test
    load_and_visualize(model)

    return model


def mode_load_test(device, load_path):
    """Load a saved checkpoint and test."""
    print(f"\n{'='*55}")
    print(f"  Mode: LOAD_TEST - Loading from {load_path}")
    print(f"{'='*55}")

    model, checkpoint = load_checkpoint(load_path, device=device)
    load_and_visualize(model)
    return model


def mode_full_demo(device):
    """Full demo: train 100 steps, save multiple checkpoints, load best."""
    print(f"\n{'='*55}")
    print("  Mode: FULL_DEMO - Train, save at milestones, load best")
    print(f"{'='*55}")

    os.makedirs('checkpoints', exist_ok=True)
    model = create_model(device=device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=100, pct_start=0.3)

    best_loss = float('inf')
    best_metrics = None

    print("\n  Training 100 steps, saving at milestones...")
    for step in range(100):
        rgb = torch.randn(2, 3, 192, 320).to(device)
        nir = torch.randn(2, 1, 192, 320).to(device)
        u = torch.linspace(0, 1, 320).view(1, -1).expand(192, -1)
        v = torch.linspace(0, 1, 192).view(-1, 1).expand(-1, 320)
        gt_disp = (30 + 40 * u + 10 * v).unsqueeze(0).unsqueeze(0).expand(2, -1, -1, -1).to(device)
        mask = torch.zeros(2, 1, 192, 320).to(device)
        for b in range(2):
            n = int(0.05 * 192 * 320)
            mask[b, 0, torch.randint(0, 192, (n,)), torch.randint(0, 320, (n,))] = 1.0

        opt.zero_grad()
        out = model(rgb, nir)
        loss = l1_disparity_loss(out['disparity'], gt_disp, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        scheduler.step()

        # Save at milestones
        if (step + 1) in [10, 25, 50, 75, 100]:
            with torch.no_grad():
                m = depth_metrics(out['disparity'], gt_disp, mask)
            ckpt_path = f'checkpoints/demo_step{step+1}.pth'
            save_checkpoint(model, opt, epoch=step+1, loss=loss.item(), metrics=m, save_path=ckpt_path)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_metrics = m
                # Also save as best
                torch.save(model.state_dict(), 'checkpoints/demo_best_weights.pth')
                print(f"    -> New best! Saving weights to checkpoints/demo_best_weights.pth")

            print(f"    Step {step+1:3d} | Loss: {loss.item():.4f}")

    print(f"\n  {'='*45}")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Checkpoints saved:")
    print(f"    checkpoints/demo_step10.pth, demo_step25.pth, demo_step50.pth,")
    print(f"    demo_step75.pth, demo_step100.pth, demo_best_weights.pth")
    print(f"  {'='*45}")

    # Load the best and test
    print("\n  Loading best checkpoint...")
    model2, _ = load_checkpoint('checkpoints/demo_step100.pth', device=device)
    load_and_visualize(model2)

    return model2


# =====================================================================
# CLI
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGB-NIR Model Save/Load Utility')
    parser.add_argument('--mode', type=str, default='full_demo',
                        choices=['save', 'train_and_save', 'load_test', 'full_demo'],
                        help='Operation mode')
    parser.add_argument('--save_path', type=str, default='checkpoints/model.pth',
                        help='Path to save checkpoint')
    parser.add_argument('--load_path', type=str, default='checkpoints/model.pth',
                        help='Path to load checkpoint from')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of training steps')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.mode == 'save':
        mode_save(device, args.save_path)
    elif args.mode == 'train_and_save':
        mode_train_and_save(device, args.save_path, args.steps)
    elif args.mode == 'load_test':
        mode_load_test(device, args.load_path)
    elif args.mode == 'full_demo':
        mode_full_demo(device)

    print("\nDone!")
