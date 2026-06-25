"""
MS2 Dataset Loader for RGB + NIR Cross-Modal Stereo Matching.

Real MS2 directory structure:
    session/
        calib.npy                     ← camera calibration dict (MM units)
        rgb/img_left/   *.png         ← RGB left  (reference)
        rgb/img_right/  *.png
        nir/img_left/   *.png
        nir/img_right/  *.png         ← NIR right (stereo pair)
        lidar/left/     *.mat         ← LiDAR point clouds (METERS)
        lidar/right/    *.mat
        thr/img_left/   *.png
        thr/img_right/  *.png

calib.npy keys:
    K_rgbL:     (3,3) RGB left intrinsics (pixels, full-res)
    K_nirR:     (3,3) NIR right intrinsics (pixels, full-res)
    R_nir2lidarL: (3,3) Rotation: NIR left → LiDAR
    T_nir2lidarL: (3,1) Translation: NIR left → LiDAR (MILLIMETERS)
    R_nir2rgb:    (3,3) Rotation: NIR left → RGB left
    T_nir2rgb:    (3,1) Translation: NIR left → RGB left (MILLIMETERS)
    T_rgbL, T_nirR: camera positions in NIR left frame (mm)

LiDAR .mat: key 'data', shape (N, 4) = (x, y, z, intensity)

LiDAR coordinate system: x=forward, y=left, z=up (meters)
Camera coordinate system: x=right, y=down, z=forward

Projection chain:
    LiDAR (meters)
    → Convert to mm
    → [R_nir2lidarL^T @ P_mm - ...] to NIR left frame (mm)
    → [R_nir2rgb @ P_nir + T_nir2rgb] to RGB left frame (mm)
    → Divide by 1000 → camera frame (meters)
    → K @ [x/z, y/z, 1] → image pixels (full-res)
    → Scale to input_height/input_width
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import glob
import scipy.io as sio


def load_calibration(calib_path):
    """
    Load MS2 calibration dict from calib.npy.

    MS2 stores calibration matrices in MILLIMETERS, while LiDAR points
    are in METERS. We convert T to meters where needed.

    Returns:
        K_rgbL:         (3,3) RGB left intrinsics (full-res pixels)
        K_nirR:         (3,3) NIR right intrinsics (full-res pixels)
        R_lidar2rgb:    (3,3) rotation: LiDAR → RGB left
        T_lidar2rgb:    (3,1) translation: LiDAR → RGB left (METERS)
        baseline_mm:    baseline between RGB left and NIR right (mm)
                        disparity = baseline_mm * fx_pixels / Z_mm
                        Since we compute in meters: disparity = baseline_m * fx / Z_m
        img_size:       (W, H) original image size
    """
    calib = np.load(calib_path, allow_pickle=True).item()

    K_rgbL = calib['K_rgbL'].astype(np.float32)      # (3, 3) in pixels

    # R_nir2lidarL: NIR → LiDAR
    # P_lidar_mm = R_nir2lidarL @ P_nir_mm + T_nir2lidarL_mm
    # → P_nir_mm = R_nir2lidarL^T @ (P_lidar_mm - T_nir2lidarL_mm)
    R_nir2lidar = calib['R_nir2lidarL'].astype(np.float32)
    T_nir2lidar_mm = calib['T_nir2lidarL'].astype(np.float32)   # mm

    # R_nir2rgb, T_nir2rgb: NIR left → RGB left
    R_nir2rgb = calib['R_nir2rgb'].astype(np.float32)
    T_nir2rgb_mm = calib['T_nir2rgb'].astype(np.float32)        # mm

    # Combined: LiDAR → RGB left (all in mm first, then T converted to m)
    # P_rgb_mm = R_nir2rgb @ [R_nir2lidar^T @ (P_lidar_mm - T_nir2lidar_mm)] + T_nir2rgb_mm
    #         = (R_nir2rgb @ R_nir2lidar^T) @ P_lidar_mm + (-R_nir2rgb @ R_nir2lidar^T @ T_nir2lidar_mm + T_nir2rgb_mm)
    R_nir2lidar_T = R_nir2lidar.T  # LiDAR → NIR rotation
    R_lidar2rgb = R_nir2rgb @ R_nir2lidar_T                    # (3, 3)
    T_combined_mm = -R_lidar2rgb @ T_nir2lidar_mm + T_nir2rgb_mm  # (3, 1)

    # Convert T to meters (points are in meters)
    T_combined_m = T_combined_mm / 1000.0

    # Original image size from principal point
    W_orig = int(round(K_rgbL[0, 2] * 2))
    H_orig = int(round(K_rgbL[1, 2] * 2))

    # Baseline in mm (for disparity computation)
    T_rgbL_mm = calib['T_rgbL'].astype(np.float32)
    T_nirR_mm = calib['T_nirR'].astype(np.float32)
    baseline_mm = float(np.linalg.norm(T_nirR_mm - T_rgbL_mm))
    if baseline_mm < 1.0:
        baseline_mm = 299.0  # fallback ~30cm

    return {
        'K_rgbL': K_rgbL,
        'K_nirR': calib['K_nirR'].astype(np.float32),
        'R_lidar2rgb': R_lidar2rgb,
        'T_lidar2rgb_m': T_combined_m,   # METERS
        'baseline_mm': baseline_mm,
        'img_size': (W_orig, H_orig),
    }


class MS2Dataset(Dataset):
    """
    MS2 dataset for RGB + NIR stereo matching.
    LiDAR ground truth projected onto RGB left image plane → sparse disparity.
    """

    def __init__(self, root, sessions, input_height=384, input_width=640,
                 max_disparity=192, max_train_samples=None, max_val_samples=None, is_train=True):

        super().__init__()
        self.root = root
        self.input_height = input_height
        self.input_width = input_width
        self.max_disparity = max_disparity
        self.is_train = is_train

        self.samples = []
        for session in sessions:
            session_dir = os.path.join(root, session)

            calib_path = os.path.join(session_dir, 'calib.npy')
            if not os.path.exists(calib_path):
                alt = glob.glob(os.path.join(session_dir, '*calib*.npy'))
                calib_path = alt[0] if alt else None
            if calib_path is None or not os.path.exists(calib_path):
                print(f"[!] No calib.npy found for {session}, skipping")
                continue

            calib = load_calibration(calib_path)

            rgb_dir = os.path.join(session_dir, 'rgb', 'img_left')
            nir_dir = os.path.join(session_dir, 'nir', 'img_right')
            lidar_dir = os.path.join(session_dir, 'lidar', 'left')

            if not all(os.path.isdir(d) for d in [rgb_dir, nir_dir, lidar_dir]):
                print(f"[!] Missing directories in {session}, skipping")
                continue

            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            matched = 0
            for rgb_path in rgb_files:
                stem = os.path.splitext(os.path.basename(rgb_path))[0]
                nir_path = os.path.join(nir_dir, f"{stem}.png")
                lidar_path = os.path.join(lidar_dir, f"{stem}.mat")
                if os.path.exists(nir_path) and os.path.exists(lidar_path):
                    self.samples.append({
                        'rgb': rgb_path, 'nir': nir_path, 'lidar': lidar_path,
                        'calib': calib,
                    })
                    matched += 1

            print(f"  {session}: {len(rgb_files)} rgb, matched {matched}")

        if max_train_samples is not None and len(self.samples) > max_train_samples:
            self.samples = self.samples[:max_train_samples]

        if max_val_samples is not None and len(self.samples) > max_val_samples:
            self.samples = self.samples[:max_val_samples]

        print(f"\nTotal samples: {len(self.samples)}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path, is_nir=False):
        """Load image, resize, convert to tensor."""
        img = Image.open(path)
        img = img.convert('L' if is_nir else 'RGB')
        img = img.resize((self.input_width, self.input_height), Image.BILINEAR)
        return self.to_tensor(img)

    def _project_lidar_to_disparity(self, points, calib):
        """
        Project LiDAR points (N, 3) to RGB left image → sparse disparity.

        Steps:
          1. LiDAR (meters, x=forward/y=left/z=up) → RGB camera frame (meters, x=right/y=down/z=forward)
             using calibration matrices (with mm→m conversion)
          2. Project to original image plane using full-resolution K
          3. Scale coordinates to input_height/input_width

        disparity = baseline_mm * fx / Z_mm  (both baseline and Z in same unit = mm)
                  = baseline_mm * fx / (Z_m * 1000)
                  = baseline_mm * fx / Z_m / 1000

        So if baseline_mm=299, fx=764, Z_m=30m:
          disparity = 299 * 764 / 30000 / 1000 ... hmm that's tiny.

        Actually for disparity: d = baseline * fx / Z (all in same unit)
        If baseline=0.3m, fx=764, Z=30m:
          d = 0.3 * 764 / 30 ≈ 7.64 pixels → reasonable

        So we need baseline in meters: baseline_m = baseline_mm / 1000
        """
        H, W = self.input_height, self.input_width
        K_full = calib['K_rgbL']          # full-res intrinsics
        R = calib['R_lidar2rgb']
        T_m = calib['T_lidar2rgb_m']      # meters
        baseline_mm = calib['baseline_mm']

        if points.shape[0] == 0:
            return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

        W_orig, H_orig = calib['img_size']

        # Points in meters → apply R and T (both in meter space)
        xyz = points[:, :3].T  # (3, N)
        cam_pts = R @ xyz + T_m  # (3, N), meters
        X, Y, Z_m = cam_pts[0], cam_pts[1], cam_pts[2]

        # Filter front-facing
        valid = Z_m > 0.1
        X, Y, Z_m = X[valid], Y[valid], Z_m[valid]
        if len(Z_m) == 0:
            return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

        # Project to full-resolution image plane
        fx, fy = K_full[0, 0], K_full[1, 1]
        cx, cy = K_full[0, 2], K_full[1, 2]
        u_full = (X * fx / Z_m + cx).astype(np.int32)
        v_full = (Y * fy / Z_m + cy).astype(np.int32)

        # Scale to target resolution
        scale_x = W / W_orig
        scale_y = H / H_orig
        u = (u_full * scale_x).astype(np.int32)
        v = (v_full * scale_y).astype(np.int32)

        # Bound check
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, Z_m = u[in_bounds], v[in_bounds], Z_m[in_bounds]
        if len(Z_m) == 0:
            return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

        # Depth (meters) → disparity (pixels)
        # d = baseline_mm * fx / Z_mm  where Z_mm = Z_m * 1000
        # d = baseline_mm * fx / (Z_m * 1000)
        baseline_m = baseline_mm / 1000.0
        disparity = baseline_m * fx / Z_m
        disparity = np.clip(disparity, 0, self.max_disparity)

        # Nearest-point filtering
        order = np.argsort(Z_m)
        u, v, disparity = u[order], v[order], disparity[order]

        disp_map = np.zeros((H, W), dtype=np.float32)
        mask = np.zeros((H, W), dtype=bool)
        disp_map[v, u] = disparity
        mask[v, u] = True

        return disp_map, mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        rgb = self._load_image(sample['rgb'], is_nir=False)      # (3, H, W)
        nir = self._load_image(sample['nir'], is_nir=True)       # (1, H, W)

        # LiDAR .mat → points (N, 4) → project
        lidar_data = sio.loadmat(sample['lidar'])
        points = lidar_data['data']
        disp_map, mask = self._project_lidar_to_disparity(points, sample['calib'])

        disparity = torch.from_numpy(disp_map).unsqueeze(0).float()     # (1, H, W)
        valid_mask = torch.from_numpy(mask).unsqueeze(0).float()        # (1, H, W)

        calib = sample['calib']
        K_rgbL = torch.from_numpy(calib['K_rgbL']).float()
        K_nirR = torch.from_numpy(calib['K_nirR']).float()

        return {
            'rgb': rgb,
            'nir': nir,
            'disparity': disparity,
            'valid_mask': valid_mask,
            'K_rgbL': K_rgbL,
            'K_nirR': K_nirR,
            'path': sample['rgb'],
        }


if __name__ == "__main__":
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    dataset = MS2Dataset(
        root=cfg['data']['root'],
        sessions=cfg['data']['train_sessions'],
        input_height=cfg['data']['input_height'],
        input_width=cfg['data']['input_width'],
        max_disparity=cfg['data']['max_disparity'],
        max_train_samples=5,
    )

    if len(dataset) > 0:
        s = dataset[0]
        for k, v in s.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
            elif isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {v}")
        d, m = s['disparity'], s['valid_mask']
        print(f"  Valid pixels: {int(m.sum().item())}/{m.numel()} ({100*m.sum()/m.numel():.1f}%)")
        if m.sum() > 0:
            print(f"  Disp range: [{d[m>0].min().item():.2f}, {d[m>0].max().item():.2f}]")
    else:
        print("Dataset is empty!")
