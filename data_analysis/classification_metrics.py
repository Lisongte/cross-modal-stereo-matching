"""
分类指标与混淆矩阵分析模块（对标 Lab0608）。
将连续视差值离散化为距离区间，构建混淆矩阵并计算分类指标。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd


def run_classification_analysis(pred_disp, gt_disp, mask, max_disparity=192,
                                 num_classes=4, save_dir='visualizations'):
    """
    将视差离散化为距离类别，生成混淆矩阵和分类报告。

    Args:
        pred_disp:     预测视差图 (B, 1, H, W) 或 (H, W)
        gt_disp:       真实视差图 (B, 1, H, W) 或 (H, W)
        mask:          有效像素掩码 (B, 1, H, W) 或 (H, W)
        max_disparity: 最大视差
        num_classes:   距离类别数
        save_dir:      保存目录
    """
    # 统一转为 numpy
    if hasattr(pred_disp, 'cpu'):
        pred_disp = pred_disp.cpu().numpy()
    if hasattr(gt_disp, 'cpu'):
        gt_disp = gt_disp.cpu().numpy()
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()

    while pred_disp.ndim > 2:
        pred_disp = pred_disp.squeeze(0)
    while gt_disp.ndim > 2:
        gt_disp = gt_disp.squeeze(0)
    while mask.ndim > 2:
        mask = mask.squeeze(0)

    mask_bool = mask > 0.5
    pred_valid = pred_disp[mask_bool]
    gt_valid = gt_disp[mask_bool]

    if len(pred_valid) < 10:
        print("[classification_metrics] 有效像素不足，跳过分析")
        return

    # 定义距离区间的边界
    bin_edges = np.linspace(0, max_disparity, num_classes + 1)
    # 让最后一个区间包含 max_disparity
    bin_edges[-1] = max_disparity + 1e-6
    labels = []
    for i in range(num_classes):
        start = int(bin_edges[i])
        end = int(bin_edges[i + 1])
        if i == num_classes - 1:
            labels.append(f'极远({start}-{int(max_disparity)})')
        elif i == 0:
            labels.append(f'近距离({start}-{end})')
        elif i == 1:
            labels.append(f'中距离({start}-{end})')
        else:
            labels.append(f'远距离({start}-{end})')

    # 离散化
    gt_class = np.digitize(gt_valid, bins=bin_edges) - 1
    pred_class = np.digitize(pred_valid, bins=bin_edges) - 1

    # 越界修正
    gt_class = np.clip(gt_class, 0, num_classes - 1)
    pred_class = np.clip(pred_class, 0, num_classes - 1)

    # ========== 1. 混淆矩阵 ==========
    cm = confusion_matrix(gt_class, pred_class)

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlBu_r',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('预测距离区间')
    plt.ylabel('真实距离区间')
    plt.title('视差估计混淆矩阵')
    plt.tight_layout()
    path1 = f'{save_dir}/confusion_matrix.png'
    plt.savefig(path1, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[classification_metrics] -> {path1}")

    # ========== 2. 分类报告 ==========
    report_dict = classification_report(
        gt_class, pred_class, target_names=labels, digits=4, output_dict=True
    )
    report_text = classification_report(
        gt_class, pred_class, target_names=labels, digits=4
    )
    print("[classification_metrics] 分类报告:")
    print(report_text)

    # 保存报告为 CSV
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f'{save_dir}/classification_report.csv')
    print(f"[classification_metrics] -> {save_dir}/classification_report.csv")

    # ========== 3. 各类别精确率和召回率柱状图 ==========
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    x = np.arange(num_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision_list, width, label='精确率 (Precision)', color='steelblue')
    bars2 = ax.bar(x, recall_list, width, label='召回率 (Recall)', color='coral')
    bars3 = ax.bar(x + width, f1_list, width, label='F1-score', color='seagreen')

    ax.set_xlabel('距离区间')
    ax.set_ylabel('分数')
    ax.set_title('各类别精确率 / 召回率 / F1-score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path2 = f'{save_dir}/precision_recall.png'
    fig.savefig(path2, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[classification_metrics] -> {path2}")

    # ========== 4. 各类别样本数量分布 ==========
    unique_gt, counts_gt = np.unique(gt_class, return_counts=True)
    unique_pred, counts_pred = np.unique(pred_class, return_counts=True)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x2 = np.arange(num_classes)
    gt_counts_full = np.zeros(num_classes)
    pred_counts_full = np.zeros(num_classes)
    for u, c in zip(unique_gt, counts_gt):
        gt_counts_full[u] = c
    for u, c in zip(unique_pred, counts_pred):
        pred_counts_full[u] = c

    ax2.bar(x2 - 0.2, gt_counts_full, 0.4, label='真实分布', color='steelblue', alpha=0.8)
    ax2.bar(x2 + 0.2, pred_counts_full, 0.4, label='预测分布', color='coral', alpha=0.8)
    ax2.set_xlabel('距离区间')
    ax2.set_ylabel('像素数量')
    ax2.set_title('各类别样本数量分布')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, rotation=30, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path3 = f'{save_dir}/class_distribution.png'
    fig2.savefig(path3, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[classification_metrics] -> {path3}")

    return {
        'confusion_matrix': cm,
        'precision': precision_list,
        'recall': recall_list,
        'f1': f1_list,
    }


if __name__ == "__main__":
    print("Testing classification_metrics...")
    H, W = 100, 160
    gt = np.random.rand(H, W) * 96
    pred = gt + np.random.randn(H, W) * 10
    mask = np.random.rand(H, W) > 0.9
    run_classification_analysis(pred, gt, mask, max_disparity=96)
    print("Done!")