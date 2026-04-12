# starVLA for Motion Generation

## 简介

基于 starVLA 框架的人体动作生成项目，使用 Diffusion Transformer (DiT) 在潜在空间中进行动作建模。

## 主要特性

- **64 维动作潜在空间表示** - 高效的动作编码与解码
- **DiT-B 架构** - 1024 hidden dimension, 16 layers
- **Qwen2.5-VL-3B-Instruct** - 作为视觉编码器，支持多模态理解
- **扩散模型动作预测** - 8 步推理，高效生成

## 训练配置（当前运行）

```yaml
GPU: NVIDIA A100-SXM4-80GB (双卡训练)
Batch Size: 32 (per device)
总训练步数：100,000
学习率：1e-5 (cosine scheduler, min 5e-7)
Warmup: 10% (10,000 步)
数据集：motion_latent (robot_humanml_data_v2)
```

## 训练结果

| 指标 | 初始值 | 最终值 | 下降幅度 |
|------|--------|--------|----------|
| action_dit_loss | 1.99 (step 10) | 0.98 (step 100k) | 50.8% |
| mse_score | 0.00807 (step 5k) | 0.00535 (step 100k) | 33.7% |

**训练时长**：约 10.5 小时  
**最终 checkpoint**: `results/Checkpoints/motion_ar_cuda0_bs32_full_20260411/final_model`

## 目录结构

```
starVLA/
├── starVLA/              # 核心代码
│   ├── training/         # 训练脚本
│   ├── models/           # 模型定义
│   └── config/           # 配置文件
├── results/              # 训练结果（已 gitignore）
├── examples/             # 示例脚本
└── README.md
```

## 使用方法

### 训练

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate starVLA
cd starVLA/training
python train_starvla.py --config_path ../config/your_config.yaml
```

### 推理

```python
from starVLA.models.MotionAR import MotionAR

model = MotionAR.from_pretrained("path/to/checkpoint")
output = model.predict_action(examples, use_ddim=True, num_ddim_steps=20)
```

## 注意事项

- ⚠️ **评估代码需要修复**：添加 `model.eval()` 和 `torch.inference_mode()`
- 📦 **大文件已排除**：checkpoints、wandb 日志等已通过 `.gitignore` 排除

## 作者

Hxxxz0

---

本项目基于 [starVLA](https://github.com/starVLA/starVLA) 框架开发。
