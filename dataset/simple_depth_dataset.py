"""
Simple RGB + NIR Depth Dataset.
Loads from a flat directory structure:
  root/
    rgb_left/   -> 000101.png, 000102.png, ...
    nir_right/  -> 000101.png, 000102.png, ...
    gt_depth/   -> 000101.png, 000102.png, ... (uint16 PNG)
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SimpleDepthDataset(Dataset):
    """
    Dataset for dataset_testing:
      root/rgb_left/xxx.png
      root/nir_right/xxx.png
      root/gt_depth/xxx.png   (uint16 dense depth)

    Supports two output modes:
      - 'depth' mode: returns depth in meters (for SimpleDepthModel)
      - 'disparity' mode: returns disparity in pixels (for RGBNIRStereoModel)
    """

    def __init__(self, root, input_height=384, input_width=640,
                 max_samples=None, normalize_depth=True,
                 output_mode='disparity', fx=764.5, baseline_m=0.0503,
                 max_disparity=48):
        super().__init__()
        self.root = root
        self.input_height = input_height
        self.input_width = input_width
        self.normalize_depth = normalize_depth
        self.output_mode = output_mode
        self.fx = fx
        self.baseline_m = baseline_m
        self.max_disparity = max_disparity

        rgb_dir = os.path.join(root, 'rgb_left')
        nir_dir = os.path.join(root, 'nir_right')
        gt_dir = os.path.join(root, 'gt_depth')

        if not all(os.path.isdir(d) for d in [rgb_dir, nir_dir, gt_dir]):
            raise FileNotFoundError(
                f"Missing directories in {root}. Expected: rgb_left/, nir_right/, gt_depth/"
            )

        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        self.samples = []
        for rgb_path in rgb_files:
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            nir_path = os.path.join(nir_dir, f"{stem}.png")
            gt_path = os.path.join(gt_dir, f"{stem}.png")
            if os.path.exists(nir_path) and os.path.exists(gt_path):
                self.samples.append({
                    'rgb': rgb_path,
                    'nir': nir_path,
                    'gt': gt_path,
                    'stem': stem,
                })

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"  SimpleDepthDataset: {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load RGB (H, W, 3) uint8
        rgb = np.array(Image.open(sample['rgb'])).astype(np.float32)

        # Load NIR (H, W) or (H, W, 1) uint8
        nir_img = Image.open(sample['nir'])
        nir = np.array(nir_img).astype(np.float32)
        if nir.ndim == 3:
            nir = nir[:, :, 0]

        # Load GT depth (H, W) uint16
        gt = np.array(Image.open(sample['gt'])).astype(np.float32)

        # Resize if needed
        orig_h, orig_w = rgb.shape[:2]
        target_h, target_w = self.input_height, self.input_width

        if (orig_h, orig_w) != (target_h, target_w):
            from torchvision.transforms import functional as TF
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
            rgb_t = TF.resize(rgb_t, (target_h, target_w), antialias=True)
            rgb = rgb_t.permute(1, 2, 0).numpy()

            nir_t = torch.from_numpy(nir).unsqueeze(0)
            nir_t = TF.resize(nir_t, (target_h, target_w), antialias=True)
            nir = nir_t[0].numpy()

            gt_t = torch.from_numpy(gt).unsqueeze(0)
            gt_t = TF.resize(gt_t, (target_h, target_w), antialias=False,
                             interpolation=TF.InterpolationMode.NEAREST)
            gt = gt_t[0].numpy()

        # Normalize
        rgb = rgb / 255.0
        nir = nir / 255.0

        if self.normalize_depth:
            # uint16 depth values map to 0~80m range (divide by 200)
            gt = gt / 200.0
            gt = np.clip(gt, 0, 80.0)

        valid_mask = (gt > 0).astype(np.float32)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        nir_t = torch.from_numpy(nir).unsqueeze(0).float()
        mask_t = torch.from_numpy(valid_mask).unsqueeze(0).float()

        result = {
            'rgb': rgb_t,
            'nir': nir_t,
            'valid_mask': mask_t,
            'stem': sample['stem'],
        }

        if self.output_mode == 'depth':
            # Depth in meters (for SimpleDepthModel)
            result['depth'] = torch.from_numpy(gt).unsqueeze(0).float()
        elif self.output_mode == 'disparity':
            # Disparity in pixels (for RGBNIRStereoModel)
            # disparity = baseline_m * fx / depth_m
            depth_safe = np.maximum(gt, 0.01)
            disp = self.baseline_m * self.fx / depth_safe
            disp = np.clip(disp, 0, self.max_disparity)
            result['disparity'] = torch.from_numpy(disp).unsqueeze(0).float()
            # Also keep depth for visualization
            result['depth'] = torch.from_numpy(gt).unsqueeze(0).float()
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

        return result
