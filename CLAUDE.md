# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

starVLA — a Lego-like Vision-Language-Action framework for robotics and motion generation. This repo uses the **MotionAR** framework to generate human motion from text descriptions using a Diffusion Transformer (DiT) in latent space, with a **World Model** providing structured future summary tokens as conditions.

## Architecture

### MotionAR (Actor)

```
Text prompt → Qwen2.5-VL-3B (frozen) → text_hidden [B, seq, 2048]
                                     ↗
obs_latent [T, 64] → LatentObsEncoder → obs_hidden [T_obs, 2048]
                                     ↘
                              condition → FlowmatchingActionHead (DiT-B)
                                         → predicted motion latent [15, 64]
```

### World Model (Phase 1 — standalone)

```
(text_hidden, z_past) → WorldModel → (C_t, P_t)
  z_past: variable-length [T_past, 64] past motion latents (padded, with attention mask)
  C_t: 4×64 — DCT-projected future latent plan tokens
  P_t: 4×38 — future action dynamics summary tokens
```

**Training phases**:
1. **Phase 1**: Train WorldModel standalone (current)
2. **Phase 2**: Actor uses GT world tokens as condition
3. **Phase 3**: Gradual replacement of GT with predicted world tokens

### Key Components
- **Qwen2.5-VL-3B-Instruct**: Frozen VLM backbone (text encoding), located at `./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct`
- **LatentObsEncoder**: MLP that maps 64-dim motion latents to VLM hidden space (trained)
- **FlowmatchingActionHead**: Flow Matching + DiT-B (16 layers, 1024 dim) for motion prediction (trained)
- **WorldModel**: 38.5M parameters, TextCompressor → LatentProjector → CausalTransformerEncoder → PlanDecoder + DynamicsDecoder (`starVLA/model/world_model/`)

### Data Flow
- **Training**: Text + obs_latent → encode → condition → DiT predicts velocity field → MSE loss
- **Inference**: Autoregressive chunk generation — each chunk is 15 frames, default 20 chunks = 300 frames total
- **Flow Matching**: 4-step Euler integration per chunk, 8x repeated during training for efficiency

## Directory Structure (key files)

```
starVLA/
├── config/
│   ├── training/
│   │   ├── motion_ar_train.yaml        # Main MotionAR training config
│   │   └── world_model_train.yaml      # World Model Phase 1 training config
│   └── deepseeds/                      # DeepSpeed zero2/zero3 configs
├── dataloader/
│   ├── motion_latent_datasets.py       # Robot HumanML latent dataset
│   ├── world_model_datasets.py         # World Model dataset (variable z_past, 38D action)
│   ├── dataloader_manager.py           # Multi-dataset support (vla/vlm co-training)
│   ├── lerobot_datasets.py             # LeRobot format datasets
│   └── vlm_datasets.py                 # VLM-only datasets
├── model/
│   ├── framework/
│   │   ├── base_framework.py           # Base framework class + build_framework() entry point
│   │   ├── share_tools.py              # Config merging utilities
│   │   ├── VLM4A/
│   │   │   └── MotionAR.py             # MotionAR framework (primary)
│   │   └── WM4A/                       # World-Model variants
│   ├── modules/
│   │   ├── action_model/
│   │   │   ├── GR00T_ActionHeader.py       # FlowmatchingActionHead + DiT + ActionEncoder
│   │   │   └── flow_matching_head/         # DiT + ActionEncoder sub-modules
│   │   └── vlm/QWen2_5.py                  # Qwen2.5-VL wrapper
│   ├── world_model/                      # World Model module (Phase 1)
│   │   ├── __init__.py                   # Module exports
│   │   ├── world_model.py                # WorldModel main class
│   │   ├── world_blocks.py               # TextCompressor, LatentProjector, CausalTransformerEncoder, QueryDecoder
│   │   ├── build_targets.py              # Deterministic target construction (DCT, dynamics, aux)
│   │   └── losses.py                     # C_t: SmoothL1+Cosine, P_t: SmoothL1, aux: MSE
│   └── tools.py                    # FRAMEWORK_REGISTRY
└── training/
    ├── train_starvla.py            # Single-framework training (VLA-only)
    ├── train_world_model.py        # World Model standalone training (Phase 1)
    └── train_unified.py            # Unified trainer (VLA+VLM co-training via DataLoaderManager)

playground/Pretrained_models/Qwen2.5-VL-3B-Instruct  # VLM weights (7GB)
robot_humanml_data_v2/                                 # Dataset (symlink)
```

## Development Commands

### Environment Setup

```bash
conda activate starVLA
```

### Code Formatting & Linting

```bash
make autoformat   # Apply black + ruff fixes in place
make check        # Run black --check + ruff check (dry-run)
make clean        # Remove *.pyc and __pycache__
```

### Training

**Single-framework (VLA-only):**
```bash
cd /limx_embap/tos/user/Jensen/project/Starworld
accelerate launch -m starVLA.training.train_starvla \
  --config_yaml starVLA/config/training/motion_ar_train.yaml \
  datasets.vla_data.per_device_batch_size 32
```

**With DeepSpeed:**
```bash
STARVLA_USE_DEEPSPEED=1 accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/motion_ar_train.yaml
```

**Unified trainer (co-training):**
```bash
accelerate launch -m starVLA.training.train_unified \
  --config_yaml <config_with_vla_and_vlm_data>
```

**Override any config value via CLI dotlist:**
```bash
--trainer.max_train_steps 200000 --trainer.base_lr 5e-5
```

**World Model (Phase 1):**
```bash
# Smoke test (single GPU, 50 steps)
CUDA_VISIBLE_DEVICES=0 python -m starVLA.training.train_world_model \
  --config_yaml starVLA/config/training/world_model_train.yaml \
  data.per_device_batch_size 32 \
  trainer.max_train_steps 50

# Full training
accelerate launch -m starVLA.training.train_world_model \
  --config_yaml starVLA/config/training/world_model_train.yaml
```

### Inference

```python
from starVLA.model.framework.base_framework import build_framework
from omegaconf import OmegaConf

config = OmegaConf.load("path/to/config.full.yaml")
model = build_framework(config)
state_dict = torch.load("path/to/checkpoint.pt", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model = model.cuda().eval()

result = model.predict_action(
    examples=[{"lang": "a person walks in a circle", "image": [], "obs_latent": np.zeros((0, 64), dtype=np.float32)}],
    num_chunks=20,
    max_obs_frames=150
)
motion = result["normalized_actions"]  # shape: (1, 300, 64)
```

**Quick test script:**
```bash
python generate_walk_motion.py  # Loads checkpoint and generates "a man walks in a circle"
```

### Training Config Highlights (`motion_ar_train.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `framework.name` | `MotionAR` | Framework to use |
| `latent_dim` | `64` | Motion latent dimension |
| `action_model_type` | `DiT-B` | DiT model size |
| `hidden_size` | `1024` | DiT hidden dimension |
| `num_layers` | `16` | DiT transformer layers |
| `action_horizon` | `15` | Frames per chunk |
| `repeated_diffusion_steps` | `8` | Repeat factor during training |
| `num_inference_timesteps` | `4` | Euler integration steps |
| `freeze_modules` | `qwen_vl_interface` | Frozen modules |
| `max_train_steps` | `100000` | Total training steps |
| `base_lr` | `1e-5` | Learning rate |
| `lr_scheduler_type` | `cosine_with_min_lr` | Scheduler with `min_lr: 5e-7` |

## Dataset

Located at `robot_humanml_data_v2/` (symlink to `/limx_embap/tos/user/Jensen/dataset/robot_humanml_data_v2`):
- `train.txt` / `test.txt` — sample lists
- `latent/{name}.npz` — motion latent arrays, shape [T, 64]
- `texts/{name}.txt` — text annotations (caption#start#end format)
- `npz/{name}.npz` — raw motion data (joint_pos T×29, body_pos_w T×30×3, body_quat_w T×30×4)
- `Mean_38d.npy` / `Std_38d.npy` — 38D action normalization
- `Mean_64d.npy` / `Std_64d.npy` — latent normalization

**MotionAR sample format**:
- `obs_latent`: past motion frames [0~150, 64] (variable length)
- `action`: target next 15 frames [15, 64]
- `lang`: text description string
- `image`: empty list (text-only for MotionAR)

**World Model sample format** (constructed at runtime from npz):
- `z_past`: [T_past, 64] normalized past latents (variable length 1~150, padded with `z_past_lengths`)
- `z_fut`: [H, 64] normalized future latents (H=16)
- `a_fut`: [H, 38] normalized future actions (H=16)
- 38D action: 29 joint_pos + 2 root_vel_xy + 1 root_z + 6 root_rot_6d (for P_t target only)

## Framework Registry

New frameworks are added via `@FRAMEWORK_REGISTRY.register("Name")` decorator on subclasses of `baseframework`. Two families exist:
- **VLM4A/**: Vision-Language-Model for Action (Qwen + action head)
- **WM4A/**: World-Model for Action (Cosmos/Wan2 + action head)

The entry point is `build_framework(cfg)` in `base_framework.py`, which auto-imports all framework modules and instantiates based on `cfg.framework.name`.

## Trainer Architecture

- **`train_starvla.py`**: Single dataloader, single framework. Uses `VLATrainer` class. Supports checkpointing, wandb logging, gradient accumulation, mixed precision.
- **`train_world_model.py`**: World Model standalone training (Phase 1). Uses frozen Qwen for text encoding (device-aware: `cuda:{local_rank}`), builds deterministic targets (DCT + action dynamics) from future windows. Separate eval dataloader from test split. ~4.5 it/s on single A100 with batch_size=64.
- **`train_unified.py`**: Multi-dataloader via `DataLoaderManager`. Uses `UnifiedTrainer` class. Supports VLA-only, VLM-only, or co-training based on which `*_data` sections exist in config.
- Training uses `accelerate` + optional `DeepSpeed` (controlled by `STARVLA_USE_DEEPSPEED` env var).
- Evaluation: periodic forward pass on validation set (up to 5 batches averaged), reporting loss_c, loss_p, loss_aux.

### World Model Design Notes

- **Input**: `text_hidden [B, L, 2048]` + `z_past [B, T_past, 64]` (variable-length, no `a_past`)
- **Coverage window**: H=16 (> MotionAR's 15-frame chunk length)
- **Attention mask**: Explicit `nn.MultiheadAttention` with combined mask — do NOT use `BasicTransformerBlock` (cannot express mixed visibility: text↔text full, motion→text full, motion→motion causal)
- **Padding mask**: `z_past_lengths` tracked per batch; padding positions masked out in attention (diagonal self-attention to avoid softmax NaN)
- **Dataset filtering**: `WorldModelDataset.__init__` filters T < H+1 samples immediately
- **Loss scope**: Only `c_pred`, `p_pred`, `aux_pred` receive gradients; `c_cond`/`p_cond` are adapter outputs reserved for Phase 2 (no loss)
- **38D action**: Constructed from npz at runtime (29 joint_pos + 2 root_vel_xy + 1 root_z + 6 root_rot_6d), used ONLY for P_t supervision, NOT as model input
- **C_t token weights**: [1.0, 0.75, 0.5, 0.25] — assert M_c ≤ 4

## Important Notes

- Qwen-VL weights are at `./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct` — relative path from repo root
- Results and checkpoints are in `.gitignore` — not tracked in git (`*.pt`, `*.pth`, `*.bin`, `results/`, `playground/Pretrained_models`)
- Conda env: `starVLA`
- Training requires `accelerate`, `torch`, `wandb`, `omegaconf`, `transformers`
- For multi-node training, see `run_oxe_train.sh` for NCCL and slurm configuration
