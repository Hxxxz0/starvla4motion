# starVLA for Motion Generation

## 简介

基于 starVLA 框架的人体动作生成项目，使用 Diffusion Transformer (DiT) 在潜在空间中进行动作建模。

## 主要特性

- **64 维动作潜在空间表示** - 高效的动作编码与解码
- **DiT-B 架构** - 1024 hidden dimension, 16 layers
- **Qwen2.5-VL-3B-Instruct** - 作为视觉编码器，支持多模态理解
- **扩散模型动作预测** - 8 步推理，高效生成
- **World Model** - 预测未来摘要 token (C_t: DCT plan tokens, P_t: dynamics tokens)，为 MotionAR 提供结构化条件

## World Model (Phase 1 — standalone)

World Model 从文本和历史序列中预测未来摘要 token，作为 MotionAR 的结构化先验条件：

```
(text_hidden, z_past) → WorldModel → (C_t, P_t)
  z_past: 变长历史 latent [T_past, 64] (T_past ≤ 150，带 padding mask)
  C_t: 4×64 — DCT 投影未来 latent plan tokens
  P_t: 4×38 — 未来 action 动力学统计量
```

**架构**: TextCompressor(8 queries) → LatentProjector(64→512) → CausalTransformerEncoder(6 layers, mixed visibility) → PlanDecoder + DynamicsDecoder
**参数**: 38.5M trainable, d_model=512, H=16 future horizon (> MotionAR chunk 15)
**监督**: 确定性构造目标 — C_t* (DCT-II 投影), P_t* (mean/change/velocity/acceleration)
**训练配置**: 50k steps, batch_size=64, lr=1e-4 (cosine), warmup=5k

```bash
# World Model 训练
accelerate launch -m starVLA.training.train_world_model \
  --config_yaml starVLA/config/training/world_model_train.yaml
```

## World Model Integration (Phase 2)

训练好的 WorldModel 作为冻结条件提供器嵌入 MotionAR：

```
text → Qwen → text_hidden [B, L, 2048]
                      ↓
obs_latent (0~150帧) → WM(text_hidden, z_past=obs_latent) → c_cond [B,4,512] + p_cond [B,4,512]
                                                            ↓
                                                  world_token_proj → [B, 8, 2048]
                                                            ↓
condition = [text_hidden; obs_hidden; world_tokens] → DiT-B → action
```

**关键设计**:
- WorldModel 完全冻结（eval + no_grad），仅 `world_token_proj` (~1M) 可训练
- WM 与 DiT 共用同一段 `obs_latent`（0~150 帧），分布与 Phase 1 训练一致
- DiT 输入不变，仅在 condition 末尾拼接 8 个 world token
- 推理时每个 chunk 重新跑一次 WM 前向（输入 history_latents 随生成增长）

```yaml
# motion_ar_train.yaml 配置
framework:
  use_world_model: true
  world_model_checkpoint: results/Checkpoints/world_model/checkpoints/steps_50000_pytorch_model.pt
trainer:
  freeze_modules: qwen_vl_interface,world_model
```

## 训练配置（当前运行）

```yaml
GPU: NVIDIA A100-SXM4-80GB (单卡)
Batch Size: 64 (per device)
总训练步数：50,000
学习率：1e-4 (cosine scheduler, min 5e-7)
Warmup: 5,000 步
数据集：robot_humanml_data_v2 (train split, eval on test split)
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

- 评估使用独立 test split dataloader，平均 5 个 batch 的 loss
- z_past 带 padding mask，无效帧不参与注意力计算
- 多 GPU 下 Qwen 自动分配到对应 rank 的 GPU
- 📦 **大文件已排除**：checkpoints、wandb 日志等已通过 `.gitignore` 排除

## 作者

Hxxxz0

---

本项目基于 [starVLA](https://github.com/starVLA/starVLA) 框架开发。
