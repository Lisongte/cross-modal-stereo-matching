# RGB-NIR Cross-Modal Stereo Depth Estimation

基于论文 **"Adaptive Stereo Depth Estimation with Multi-Spectral Images Across All Lighting Conditions"** (Qin et al., arXiv 2411.03638) 的 PyTorch 实现。

利用可见光(RGB)和近红外(NIR)图像作为立体对，通过 Cross-modal Feature Matching (CFM)、Degradation Masking 和 Depth Module 实现多光谱立体深度估计。

---

## 环境要求

- Python 3.8+
- PyTorch 2.0+ (CUDA 推荐)
- 依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

## 数据集结构

数据集采用 MS2 格式，目录布局如下：

```
D:/MS2_Dataset/
├── calib.npy                    # 相机标定参数
└── train_data/
    ├── img_left/                # RGB 左图 (参考视角)
    ├── img_right/               # NIR 右图 (立体对)
    └── depth_filtered/          # 稠密深度 GT (uint16 PNG, meters × 200)
```

## 训练

```bash
python train.py --config config.yaml
```

### config.yaml 关键配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data.root` | 训练数据根目录 | — |
| `data.input_height/width` | 输入分辨率 | 384×640 |
| `model.use_lightweight` | 轻量级正则化 (True=CostVolumeRegularizationLight) | true |
| `model.use_mdp` | 启用 MDP 模块 (Degradation Masking) | true |
| `train.batch_size` | 批大小 | 4 |
| `train.epochs` | 训练轮数 | 50 |
| `loss.nll_weight` | NLL 损失权重 | 0.05 |
| `loss.nll_mdp_weight` | MDP NLL 损失权重 | 0.05 |

### 三阶段训练

| 阶段 | 模块 | 损失 | 论文 Eq. |
|------|------|------|----------|
| 1 | CFM | L1 (期望深度 vs GT) | Eq.(8) |
| 2 | MDP-VIS + MDP-THR | NLL (μ, σ² vs GT) | Eq.(9) |
| 3 | Depth Module | NLL | Eq.(9) |

## 推理

### 单个图像对

```bash
python infer.py --checkpoint checkpoints/best.pth --rgb left.png --nir right.png
```

### 批量推理（MS2 数据集）

```bash
python infer.py --checkpoint checkpoints/best.pth --data_root D:/MS2_Dataset/train_data
```

### 指定样本推理

```bash
python infer.py --checkpoint checkpoints/best.pth \
    --data_root D:/MS2_Dataset/train_data \
    --samples 000163,000439,001762
```

### Degradation Masking（论文 Eq.5-6）

通过 MDP-VIS 模块的深度不确定度筛选可靠像素：

```bash
python infer.py --checkpoint checkpoints/best.pth \
    --data_root D:/MS2_Dataset/train_data \
    --degradation_k 1.0 \
    --samples 000163
```

筛选条件：`|GT - μ_vis| < k · σ_vis`（论文使用 k=1）

### 其他推理选项

```bash
# 导出 .npy 原始数据
python infer.py --checkpoint checkpoints/best.pth --data_root D:/MS2_Dataset/train_data --export_depth

# CPU 推理
python infer.py --checkpoint checkpoints/best.pth --data_root D:/MS2_Dataset/train_data --device cpu

# 方差阈值过滤（视差域）
python infer.py --checkpoint checkpoints/best.pth --data_root D:/MS2_Dataset/train_data --var_threshold 1.0

# 指定输出目录
python infer.py --checkpoint checkpoints/best.pth --data_root D:/MS2_Dataset/train_data --output_dir results
```

### 推理输出

每张图像生成一张 2×2 可视化 PNG：
- **左上**: RGB 左图
- **右上**: NIR 右图
- **左下**: 经过 Degradation Mask 筛选的 GT 深度图（或完整 GT）
- **右下**: 完整 GT 深度图

控制台同时输出 EPE、D1-all 和可靠像素占比。

## 文件结构

```
.
├── config.yaml                  # 训练配置
├── train.py                     # 训练脚本
├── infer.py                     # 推理脚本
├── save_model.py                # 模型保存/加载工具
├── clean.py                     # 数据清洗工具
├── requirements.txt             # 依赖
├── models/
│   ├── rgb_nir_model.py         # 主模型 (RGBNIRStereoModel)
│   ├── feature_extractor.py     # PSMNet 特征提取器
│   ├── cross_attention.py       # 跨模态特征对齐
│   ├── cost_volume.py           # Cost Volume (支持 homography)
│   ├── regularization.py        # 3D 卷积正则化
│   ├── disparity_regression.py  # 视差回归 + 上采样
│   ├── mdp_module.py            # MDP 模块 (单目深度+不确定度)
│   └── losses.py                # L1 / NLL / SILog 损失 & 评估指标
├── dataset/
│   ├── train_data_dataset.py    # MS2 训练数据集加载器
│   └── ms2_dataset.py           # MS2 原始数据集
├── data_analysis/               # 数据分析和可视化
├── checkpoints/                 # 训练检查点
├── inference_results/           # 推理输出
└── visualizations/              # 训练过程可视化
```

## 模型架构

```
RGB ──┐
      ├── FeatureExtractor ── Cross-Attention ──┐
NIR ──┘                                        │
                                                ├── Cost Volume ── 3DConv ── Regression ── 视差/深度
MDP-VIS (RGB) ── Degradation Mask ─────────────┘
MDP-THR (NIR) ── Feature Concat ────────────────┘
```

## 评估结果 (MS2 训练集, Epoch 50)

| 指标 | 值 |
|------|-----|
| EPE (视差) | 0.30 px |
| D1-all | 0.35% |
| AbsRel | 0.151 |
| RMSE | 7.02 m |
| δ < 1.25 | 76.7% |

## 引用

```bibtex
@article{qin2024adaptive,
  title={Adaptive Stereo Depth Estimation with Multi-Spectral Images Across All Lighting Conditions},
  author={Qin, Zihan and Xu, Jialei and Zhao, Wenbo and Jiang, Junjun and Liu, Xianming},
  journal={arXiv preprint arXiv:2411.03638},
  year={2024}
}