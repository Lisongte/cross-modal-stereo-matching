"""
回归分析与残差分析模块（对标 Lab0525）。
分析模型预测视差与真实视差之间的线性关系和残差分布。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE, r2_score


def run_regression_analysis(pred_disp, gt_disp, mask, save_dir='visualizations'):
    """
    对模型预测结果执行回归分析，生成散点图、拟合线和残差分析图。

    Args:
        pred_disp: 预测视差图 (B, 1, H, W) 或 (H, W) numpy数组
        gt_disp:   真实视差图 (B, 1, H, W) 或 (H, W) numpy数组
        mask:      有效像素掩码 (B, 1, H, W) 或 (H, W) numpy数组 (bool/float)
        save_dir:  图像保存目录
    """
    # 统一转为 numpy 并降维
    if hasattr(pred_disp, 'cpu'):
        pred_disp = pred_disp.cpu().numpy()
    if hasattr(gt_disp, 'cpu'):
        gt_disp = gt_disp.cpu().numpy()
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()

    # 降维到 (H, W)
    while pred_disp.ndim > 2:
        pred_disp = pred_disp.squeeze(0)
    while gt_disp.ndim > 2:
        gt_disp = gt_disp.squeeze(0)
    while mask.ndim > 2:
        mask = mask.squeeze(0)

    mask_bool = mask > 0.5

    # 提取有效像素
    pred_valid = pred_disp[mask_bool].flatten()
    gt_valid = gt_disp[mask_bool].flatten()

    if len(pred_valid) < 10:
        print("[regression_analysis] 有效像素不足，跳过分析")
        return

    # 限制采样数量防止绘图卡死
    max_samples = 10000
    if len(pred_valid) > max_samples:
        idx = np.random.choice(len(pred_valid), max_samples, replace=False)
        pred_valid = pred_valid[idx]
        gt_valid = gt_valid[idx]

    X = pred_valid.reshape(-1, 1)
    Y = gt_valid.reshape(-1, 1)

    # ========== 1. 线性回归 ==========
    model = LinearRegression()
    model.fit(X, Y)
    k = model.coef_[0, 0]
    b = model.intercept_[0]
    r2 = r2_score(Y, model.predict(X))
    mse_val = MSE(Y, model.predict(X))

    print(f"[regression_analysis] Linear regression: y = {k:.4f}x + {b:.4f}")
    print(f"[regression_analysis] R2 = {r2:.4f}, MSE = {mse_val:.4f}")

    # 绘制散点图 + 拟合线
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    ax1.scatter(X, Y, alpha=0.3, s=2, c='steelblue', label='有效像素点')
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    ax1.plot(x_line, model.predict(x_line), 'r-', linewidth=2,
             label=f'线性拟合: y={k:.3f}x+{b:.3f}')
    ax1.plot(x_line, x_line, 'k--', alpha=0.5, label='y=x（完美预测）')
    ax1.set_xlabel('预测视差 (pixels)')
    ax1.set_ylabel('真实视差 (pixels)')
    ax1.set_title(f'预测视差 vs 真实视差\nR²={r2:.4f}, MSE={mse_val:.2f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    path1 = f'{save_dir}/regression_scatter.png'
    fig1.savefig(path1, dpi=120, bbox_inches='tight')
    plt.close(fig1)
    print(f"[regression_analysis] -> {path1}")

    # ========== 2. 残差分析 ==========
    residuals = Y - model.predict(X)

    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 残差直方图
    axes[0].hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('残差 (真实值 - 预测值)')
    axes[0].set_ylabel('频数')
    axes[0].set_title(f'残差分布直方图\n均值={residuals.mean():.3f}, 标准差={residuals.std():.3f}')
    axes[0].grid(alpha=0.3)

    # 残差 vs 拟合值
    axes[1].scatter(model.predict(X), residuals, alpha=0.3, s=2, c='coral')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('拟合值 (预测视差)')
    axes[1].set_ylabel('残差')
    axes[1].set_title('残差 vs 拟合值')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path2 = f'{save_dir}/residual_analysis.png'
    fig2.savefig(path2, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f"[regression_analysis] -> {path2}")

    return {'k': k, 'b': b, 'r2': r2, 'mse': mse_val}


if __name__ == "__main__":
    # 快速测试
    print("Testing regression_analysis...")
    H, W = 100, 160
    gt = np.random.rand(H, W) * 96
    pred = gt + np.random.randn(H, W) * 5 + 2
    mask = np.random.rand(H, W) > 0.9
    run_regression_analysis(pred, gt, mask)
    print("Done!")