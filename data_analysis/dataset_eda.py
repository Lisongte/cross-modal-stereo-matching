"""
数据集探索性分析（EDA）模块（对标 Lab0518）。
对视差估计数据集进行统计分析，包括视差分布、稀疏度分析等。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import pandas as pd


def run_dataset_eda(rgb, nir, pred_disp=None, gt_disp=None, mask=None,
                    dataset_name="MS2", save_dir='visualizations'):
    """
    对视差估计数据集进行探索性分析。

    Args:
        rgb:        RGB图像 (B, 3, H, W) 或 (3, H, W) 或 (H, W, 3)
        nir:        NIR图像 (B, 1, H, W) 或 (1, H, W) 或 (H, W)
        pred_disp:  预测视差 (可选)
        gt_disp:    真实视差 (可选)
        mask:       有效像素掩码 (可选)
        dataset_name: 数据集名称
        save_dir:      保存目录
    """
    # 统一转为 numpy
    tensors = [rgb, nir]
    names = ['rgb', 'nir']
    for i, t in enumerate(tensors):
        if hasattr(t, 'cpu'):
            tensors[i] = t.cpu().numpy()
        while tensors[i].ndim > 3:
            tensors[i] = tensors[i].squeeze(0)
    rgb, nir = tensors

    # RGB: (C, H, W) -> (H, W, C)
    if rgb.ndim == 3 and rgb.shape[0] in [1, 3]:
        rgb = rgb.transpose(1, 2, 0)
    nir = nir.squeeze()

    H, W = rgb.shape[:2]

    # 如果有GT和mask，做分析
    if gt_disp is not None:
        if hasattr(gt_disp, 'cpu'):
            gt_disp = gt_disp.cpu().numpy()
        while gt_disp.ndim > 2:
            gt_disp = gt_disp.squeeze(0)

    if mask is not None:
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        while mask.ndim > 2:
            mask = mask.squeeze(0)

    # ========== 1. 通道像素值分布 ==========
    fig1, axes = plt.subplots(2, 2, figsize=(12, 8))

    channels = [
        ('R 通道', rgb[:, :, 0].flatten(), 'red'),
        ('G 通道', rgb[:, :, 1].flatten(), 'green'),
        ('B 通道', rgb[:, :, 2].flatten(), 'blue'),
        ('NIR 通道', nir.flatten(), 'gray'),
    ]

    for idx, (name, data, color) in enumerate(channels):
        ax = axes[idx // 2, idx % 2]
        ax.hist(data, bins=100, color=color, alpha=0.7, density=True)
        ax.set_xlabel('像素值')
        ax.set_ylabel('概率密度')
        ax.set_title(f'{name} 像素值分布\n均值={data.mean():.3f}, 标准差={data.std():.3f}')
        ax.grid(alpha=0.3)

    plt.suptitle(f'{dataset_name} 数据集 - 各通道像素值分布', fontsize=13)
    plt.tight_layout()
    path1 = f'{save_dir}/channel_distribution.png'
    fig1.savefig(path1, dpi=120, bbox_inches='tight')
    plt.close(fig1)
    print(f"[dataset_eda] -> {path1}")

    # ========== 2. 如果提供了GT视差和mask ==========
    if gt_disp is not None and mask is not None:
        mask_bool = mask > 0.5
        valid_disp = gt_disp[mask_bool]

        if len(valid_disp) > 0:
            # 视差分布直方图
            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

            # 2a. 视差分布
            axes2[0, 0].hist(valid_disp, bins=80, color='steelblue', alpha=0.8, density=True)
            axes2[0, 0].set_xlabel('视差值 (pixels)')
            axes2[0, 0].set_ylabel('概率密度')
            axes2[0, 0].set_title(f'有效像素视差分布\n有效像素数={len(valid_disp)}, '
                                   f'均值={valid_disp.mean():.1f}, 标准差={valid_disp.std():.1f}')
            axes2[0, 0].grid(alpha=0.3)

            # 2b. 稀疏度 (有效像素比例)
            valid_ratio = mask_bool.sum() / mask_bool.size
            axes2[0, 1].text(0.5, 0.5,
                             f'稀疏度 = 有效像素 / 总像素\n'
                             f'= {mask_bool.sum()} / {mask_bool.size}\n'
                             f'= {valid_ratio*100:.2f}%',
                             ha='center', va='center', fontsize=14,
                             transform=axes2[0, 1].transAxes)
            axes2[0, 1].set_title('LiDAR 标注稀疏度')
            axes2[0, 1].axis('off')

            # 2c. 视差空间分布 (二维直方图)
            y_valid, x_valid = np.where(mask_bool)
            if len(x_valid) > 0:
                h = axes2[1, 0].hist2d(x_valid, y_valid, bins=(50, 50),
                                        cmap='hot', cmin=1)
                axes2[1, 0].set_xlabel('x 坐标 (pixels)')
                axes2[1, 0].set_ylabel('y 坐标 (pixels)')
                axes2[1, 0].set_title('有效像素空间分布')
                plt.colorbar(h[3], ax=axes2[1, 0])

            # 2d. 如果有预测值，显示误差 vs 视差关系
            if pred_disp is not None:
                if hasattr(pred_disp, 'cpu'):
                    pred_disp_local = pred_disp.cpu().numpy()
                else:
                    pred_disp_local = pred_disp
                while pred_disp_local.ndim > 2:
                    pred_disp_local = pred_disp_local.squeeze(0)

                pred_valid = pred_disp_local[mask_bool]
                error = np.abs(pred_valid - valid_disp)

                # 误差 vs 真实视差 散点图
                axes2[1, 1].scatter(valid_disp, error, alpha=0.2, s=1, c='coral')
                axes2[1, 1].set_xlabel('真实视差 (pixels)')
                axes2[1, 1].set_ylabel('绝对误差 (pixels)')
                axes2[1, 1].set_title(f'误差 vs 真实视差\n平均误差={error.mean():.2f}')
                axes2[1, 1].grid(alpha=0.3)

                # 按视差分箱统计平均误差
                bin_edges = np.linspace(0, valid_disp.max(), 20)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                mean_error_by_disp = []
                for i in range(len(bin_edges) - 1):
                    in_bin = (valid_disp >= bin_edges[i]) & (valid_disp < bin_edges[i + 1])
                    if in_bin.sum() > 0:
                        mean_error_by_disp.append(error[in_bin].mean())
                    else:
                        mean_error_by_disp.append(0)
                axes2[1, 1].plot(bin_centers, mean_error_by_disp, 'r-', linewidth=2, label='平均误差')
                axes2[1, 1].legend()
            else:
                axes2[1, 1].text(0.5, 0.5, '无预测数据',
                                 ha='center', va='center', fontsize=14,
                                 transform=axes2[1, 1].transAxes)
                axes2[1, 1].axis('off')

            plt.suptitle(f'{dataset_name} 数据集 - 视差分析', fontsize=13)
            plt.tight_layout()
            path2 = f'{save_dir}/disparity_analysis.png'
            fig2.savefig(path2, dpi=120, bbox_inches='tight')
            plt.close(fig2)
            print(f"[dataset_eda] -> {path2}")

    # ========== 3. 图像统计特征 ==========
    # 计算一些图像统计量
    stats = {
        'Brightness': rgb.mean(axis=2).mean() if rgb.ndim == 3 else rgb.mean(),
        'Contrast': rgb.std(),
        'Valid_Ratio': mask_bool.sum() / mask_bool.size if (mask is not None) else 0,
        'Disp_Mean': valid_disp.mean() if (gt_disp is not None and len(valid_disp) > 0) else 0,
        'Disp_Std': valid_disp.std() if (gt_disp is not None and len(valid_disp) > 0) else 0,
    }

    print(f"\n[dataset_eda] 图像统计信息:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # 保存统计信息到 CSV
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(f'{save_dir}/image_statistics.csv', index=False)
    print(f"[dataset_eda] -> {save_dir}/image_statistics.csv")

    return stats


if __name__ == "__main__":
    print("Testing dataset_eda...")
    H, W = 100, 160
    rgb = np.random.rand(H, W, 3).astype(np.float32)
    nir = np.random.rand(H, W).astype(np.float32)
    gt = np.random.rand(H, W) * 96
    pred = gt + np.random.randn(H, W) * 5
    mask = np.random.rand(H, W) > 0.9
    run_dataset_eda(rgb, nir, pred, gt, mask)
    print("Done!")