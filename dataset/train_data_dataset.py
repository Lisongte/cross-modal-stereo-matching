"""
TrainDataDataset for preprocessed MS2 data (flat directory structure).

Actual data layout:
    root/
        img_left/         ← RGB left  (reference view)
        img_right/        ← NIR right (stereo pair)
        depth_filtered/   ← dense depth GT (uint16 PNG, meters * 200)

Calibration is loaded from <root>/../calib.npy (MS2 session calib format).

Outputs disparity for RGBNIRStereoModel:
    - 'rgb':          (3, H, W) float32, range [0,1]
    - 'nir':          (1, H, W) float32, range [0,1]
    - 'disparity':    (1, H, W) float32, computed from depth GT
    - 'valid_mask':   (1, H, W) float32, 1.0 where depth > 0
    - 'depth':        (1, H, W) float32, depth in meters (for metrics)
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def gpu_resize_collate_fn(samples):
    """
    Collate function that batches samples and performs resize on GPU.
    Uses torch.cuda if available, otherwise falls back to CPU resize.

    For homogeneous batch sizes (same orig_size), this batches the resize
    operations via torchvision.transforms.functional.resize on GPU tensors.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Collect all individual tensors
    rgb_list = []
    nir_list = []
    disp_list = []
    depth_list = []
    mask_list = []
    stems = []
    K_rgb_list = []
    K_nir_list = []
    R_list = []
    T_list = []
    has_calib = False

    for sample in samples:
        # Check if resize is needed (orig != target)
        if 'orig_size' in sample and 'target_size' in sample:
            orig_h, orig_w = sample['orig_size']
            target_h, target_w = sample['target_size']

            # Group by original size for batched resize
            # For simplicity, we process each sample individually but on GPU
            if (orig_h, orig_w) != (target_h, target_w):
                from torchvision.transforms import functional as TF
                rgb = sample['rgb'].to(device)
                nir = sample['nir'].to(device)
                disp = sample['disparity'].to(device)
                depth = sample['depth'].to(device)
                mask = sample['valid_mask'].to(device)

                rgb = TF.resize(rgb, (target_h, target_w), antialias=True)
                nir = TF.resize(nir, (target_h, target_w), antialias=True)
                disp = TF.resize(disp, (target_h, target_w), antialias=False,
                                 interpolation=TF.InterpolationMode.NEAREST)
                depth = TF.resize(depth, (target_h, target_w), antialias=False,
                                  interpolation=TF.InterpolationMode.NEAREST)
                mask = TF.resize(mask, (target_h, target_w), antialias=False,
                                 interpolation=TF.InterpolationMode.NEAREST)

                rgb_list.append(rgb.cpu())
                nir_list.append(nir.cpu())
                disp_list.append(disp.cpu())
                depth_list.append(depth.cpu())
                mask_list.append(mask.cpu())
            else:
                rgb_list.append(sample['rgb'])
                nir_list.append(sample['nir'])
                disp_list.append(sample['disparity'])
                depth_list.append(sample['depth'])
                mask_list.append(sample['valid_mask'])
        else:
            rgb_list.append(sample['rgb'])
            nir_list.append(sample['nir'])
            disp_list.append(sample['disparity'])
            depth_list.append(sample['depth'])
            mask_list.append(sample['valid_mask'])
        stems.append(sample['stem'])
        if 'K_rgb' in sample:
            K_rgb_list.append(sample['K_rgb'])
            K_nir_list.append(sample['K_nir'])
            R_list.append(sample['R'])
            T_list.append(sample['T'])
            has_calib = True

    batch = {
        'rgb': torch.stack(rgb_list),
        'nir': torch.stack(nir_list),
        'disparity': torch.stack(disp_list),
        'depth': torch.stack(depth_list),
        'valid_mask': torch.stack(mask_list),
        'stem': stems,
    }
    if has_calib:
        batch['K_rgb'] = torch.stack(K_rgb_list)
        batch['K_nir'] = torch.stack(K_nir_list)
        batch['R'] = torch.stack(R_list)
        batch['T'] = torch.stack(T_list)
    return batch


def _load_ms2_calib(calib_path):
    """
    Load MS2 calibration from calib.npy.
    Returns (fx, baseline_m, K_rgbL, K_nirR, R, T).
    """
    import numpy as np
    calib = np.load(calib_path, allow_pickle=True).item()
    K_rgbL = calib['K_rgbL'].astype(np.float32)     # (3,3)
    K_nirR = calib.get('K_nirR', calib['K_rgbL']).astype(np.float32)  # fallback to rgb K
    fx = float(K_rgbL[0, 0])
    T_rgbL_mm = calib['T_rgbL'].astype(np.float32)  # (3,)
    T_nirR_mm = calib['T_nirR'].astype(np.float32)  # (3,)
    baseline_mm = float(np.linalg.norm(T_nirR_mm - T_rgbL_mm))
    if baseline_mm < 1.0:
        baseline_mm = 299.0
    baseline_m = baseline_mm / 1000.0
    # Rotation: identity (rectified stereo)
    # T = source_position - reference_position = NIR - RGB (in meters)
    # inverse_warp does: P_src = R * P_ref + T
    R = np.eye(3, dtype=np.float32)
    T = (T_nirR_mm - T_rgbL_mm).astype(np.float32) / 1000.0  # NIR - RGB
    T = T.reshape(3, 1)
    return fx, baseline_m, K_rgbL, K_nirR, R, T


class TrainDataDataset(Dataset):
    """
    Dataset for preprocessed MS2 training data (flat structure):
        root/img_left/*.png       ← RGB left (reference)
        root/img_right/*.png      ← NIR right (stereo pair)
        root/depth_filtered/*.png ← uint16 dense depth GT

    If calib_path is provided (or auto-detected), loads camera parameters
    to convert depth → disparity for stereo model training.

    Args:
        root:            Path to data directory containing img_left/, img_right/, depth_filtered/
        input_height:    Target image height
        input_width:     Target image width
        normalize_depth: If True, divide uint16 depth by 200 → meters
        output_mode:     'disparity' (default, for stereo model)
        fx:              Focal length in pixels (overridden by calib if available)
        baseline_m:      Baseline in meters (overridden by calib if available)
        max_disparity:   Maximum disparity in pixels
        max_samples:     Limit number of samples (None = all)
    """

    def __init__(self, root, input_height=384, input_width=640,
                 normalize_depth=True, output_mode='disparity',
                 fx=764.5, baseline_m=0.0503, max_disparity=48,
                 max_samples=None, is_train=True, resize_on_gpu=False):
        super().__init__()
        self.root = root
        self.input_height = input_height
        self.input_width = input_width
        self.normalize_depth = normalize_depth
        self.output_mode = output_mode
        self.max_disparity = max_disparity
        self.is_train = is_train
        self.resize_on_gpu = resize_on_gpu
        self.target_size_set = (input_height, input_width) != (0, 0)

        # Directories
        img_left_dir = os.path.join(root, 'img_left')
        img_right_dir = os.path.join(root, 'img_right')
        depth_dir = os.path.join(root, 'depth_filtered')

        if not all(os.path.isdir(d) for d in [img_left_dir, img_right_dir, depth_dir]):
            raise FileNotFoundError(
                f"Missing directories in {root}. "
                f"Expected: img_left/, img_right/, depth_filtered/"
            )

        # Auto-detect calibration from parent directory
        self.fx = fx
        self.baseline_m = baseline_m
        calib_candidates = [
            os.path.join(os.path.dirname(root), 'calib.npy'),
            os.path.join(root, 'calib.npy'),
            os.path.join(root, '..', 'calib.npy'),
        ]
        calib_path = None
        for c in calib_candidates:
            if os.path.exists(c):
                calib_path = c
                break
        self.has_calib = False
        if calib_path is not None:
            try:
                fx_calib, bl_calib, K_rgb, K_nir, R, T = _load_ms2_calib(calib_path)
                self.fx = fx_calib
                self.baseline_m = bl_calib
                self.K_rgb = torch.from_numpy(K_rgb)
                self.K_nir = torch.from_numpy(K_nir)
                self.R = torch.from_numpy(R)
                self.T_vec = torch.from_numpy(T)
                self.has_calib = True
                print(f"  Loaded calibration from {calib_path}: fx={self.fx:.2f}, baseline={self.baseline_m:.5f}m")
            except Exception as e:
                print(f"  [WARN] Failed to load calibration from {calib_path}: {e}")
                print(f"  Using defaults: fx={self.fx:.2f}, baseline={self.baseline_m:.5f}m")
        else:
            print(f"  No calib.npy found, using defaults: fx={self.fx:.2f}, baseline={self.baseline_m:.5f}m")

        # Build sample list
        left_files = sorted(glob.glob(os.path.join(img_left_dir, '*.png')))
        self.samples = []
        for left_path in left_files:
            stem = os.path.splitext(os.path.basename(left_path))[0]
            right_path = os.path.join(img_right_dir, f"{stem}.png")
            depth_path = os.path.join(depth_dir, f"{stem}.png")
            if os.path.exists(right_path) and os.path.exists(depth_path):
                self.samples.append({
                    'rgb': left_path,
                    'nir': right_path,
                    'depth': depth_path,
                    'stem': stem,
                })

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        print(f"  TrainDataDataset: {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load RGB (H, W, 3) uint8 → float32 [0,255]
        rgb = np.array(Image.open(sample['rgb'])).astype(np.float32)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)

        # Load NIR right (H, W) or (H, W, 1) → float32 [0,255]
        nir_img = Image.open(sample['nir'])
        nir = np.array(nir_img).astype(np.float32)
        if nir.ndim == 3:
            nir = nir[:, :, 0]

        # Load depth (H, W) uint16 → float32
        gt = np.array(Image.open(sample['depth'])).astype(np.float32)

        # Resize (skip if using GPU collate)
        orig_h, orig_w = rgb.shape[:2]
        target_h, target_w = self.input_height, self.input_width
        need_resize = (orig_h, orig_w) != (target_h, target_w) and self.target_size_set

        if need_resize and not self.resize_on_gpu:
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

        # Normalize RGB and NIR to [0, 1]
        rgb = rgb / 255.0
        nir = nir / 255.0

        # Depth: uint16, 0~80m range (divide by 200)
        if self.normalize_depth:
            gt = gt / 200.0
            gt = np.clip(gt, 0, 80.0)

        valid_mask = (gt > 0).astype(np.float32)

        # Convert to disparity for stereo model
        depth_safe = np.maximum(gt, 0.01)
        disp = self.baseline_m * self.fx / depth_safe
        disp = np.clip(disp, 0, self.max_disparity)

        # To tensors
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        nir_t = torch.from_numpy(nir).unsqueeze(0).float()
        disp_t = torch.from_numpy(disp).unsqueeze(0).float()
        depth_t = torch.from_numpy(gt).unsqueeze(0).float()
        mask_t = torch.from_numpy(valid_mask).unsqueeze(0).float()

        result = {
            'rgb': rgb_t,
            'nir': nir_t,
            'disparity': disp_t,
            'depth': depth_t,
            'valid_mask': mask_t,
            'stem': sample['stem'],
        }
        if self.has_calib:
            result['K_rgb'] = self.K_rgb.clone()
            result['K_nir'] = self.K_nir.clone()
            result['R'] = self.R.clone()
            result['T'] = self.T_vec.clone()
        if self.resize_on_gpu:
            result['orig_size'] = (orig_h, orig_w)
            result['target_size'] = (target_h, target_w)
        return result
