# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

starVLA — a Lego-like Vision-Language-Action framework for robotics and motion generation. This repo uses the **MotionAR** framework to generate human motion from text descriptions using a Diffusion Transformer (DiT) in latent space.

## Architecture

```
Text prompt → Qwen2.5-VL-3B (frozen) → text_hidden [B, seq, 2048]
                                     ↗
obs_latent [T, 64] → LatentObsEncoder → obs_hidden [T_obs, 2048]
                                     ↘
                              condition → FlowmatchingActionHead (DiT-B)
                                         → predicted motion latent [15, 64]
```

### Key Components
- **Qwen2.5-VL-3B-Instruct**: Frozen VLM backbone (text encoding), located at `./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct`
- **LatentObsEncoder**: MLP that maps 64-dim motion latents to VLM hidden space (trained)
- **FlowmatchingActionHead**: Flow Matching + DiT-B (16 layers, 1024 dim) for motion prediction (trained)

### Data Flow
- **Training**: Text + obs_latent → encode → condition → DiT predicts velocity field → MSE loss
- **Inference**: Autoregressive chunk generation — each chunk is 15 frames, default 20 chunks = 300 frames total
- **Flow Matching**: 4-step Euler integration per chunk, 8x repeated during training for efficiency

## Directory Structure (key files)

```
starVLA/
├── config/
│   ├── training/
│   │   └── motion_ar_train.yaml    # Main MotionAR training config
│   └── deepseeds/                  # DeepSpeed zero2/zero3 configs
├── dataloader/
│   ├── motion_latent_datasets.py   # Robot HumanML latent dataset
│   └── dataloader_manager.py       # Multi-dataset support
├── model/
│   ├── framework/
│   │   ├── base_framework.py       # Base framework class
│   │   ├── VLM4A/
│   │   │   └── MotionAR.py         # MotionAR framework (primary)
│   │   └── WM4A/                   # World-Model variants
│   └── modules/
│       ├── action_model/
│       │   ├── GR00T_ActionHeader.py     # FlowmatchingActionHead
│       │   └── flow_matching_head/       # DiT + ActionEncoder
│       └── vlm/QWen2_5.py                # Qwen2.5-VL wrapper
└── training/
    ├── train_starvla.py            # Single-framework training
    └── train_unified.py            # Unified trainer (VLA+VLM co-training)

playground/Pretrained_models/Qwen2.5-VL-3B-Instruct  # VLM weights (7GB)
robot_humanml_data_v2/                                 # Dataset (symlink)
```

## How to Train

```bash
cd /limx_embap/tos/user/Jensen/project/Starworld
accelerate launch -m starVLA.training.train_starvla \
  --config_yaml starVLA/config/training/motion_ar_train.yaml \
  datasets.vla_data.per_device_batch_size 32
```

Config highlights (`motion_ar_train.yaml`):
- `framework.name: MotionAR`
- `latent_dim: 64`, `action_dim: 64`
- `action_model_type: DiT-B`, `num_layers: 16`, `hidden_size: 1024`
- `action_horizon: 15`, `repeated_diffusion_steps: 8`
- `num_inference_timesteps: 4` (Euler steps)
- `freeze_modules: qwen_vl_interface` (Qwen frozen)
- `max_train_steps: 100000`, `base_lr: 1e-5`, `cosine scheduler`

## How to Inference

```python
from starVLA.model.framework.base_framework import build_framework
from omegaconf import OmegaConf

config = OmegaConf.load("path/to/config.full.yaml")
model = build_framework(config)
model.load_state_dict(torch.load("path/to/checkpoint.pt", map_location="cpu"), strict=False)
model = model.cuda().eval()

result = model.predict_action(
    examples=[{"lang": "a person walks in a circle", "image": [], "obs_latent": np.zeros((0, 64), dtype=np.float32)}],
    num_chunks=20,
    max_obs_frames=150
)
motion = result["normalized_actions"]  # shape: (1, 300, 64)
```

## Dataset

Located at `robot_humanml_data_v2/` (symlink to `/limx_embap/tos/user/Jensen/dataset/robot_humanml_data_v2`):
- `train.txt` / `test.txt` — sample lists
- `latent/{name}.npz` — motion latent arrays, shape [T, 64]
- `texts/{name}.txt` — text annotations (caption#start#end format)

Each sample provides:
- `obs_latent`: past motion frames [0~150, 64] (variable length)
- `action`: target next 15 frames [15, 64]
- `lang`: text description string
- `image`: empty list (text-only for MotionAR)

## Framework Registry

New frameworks can be added via `@FRAMEWORK_REGISTRY.register("Name")` decorator. Two families:
- **VLM4A/**: Vision-Language-Model for Action (Qwen + action head)
- **WM4A/**: World-Model for Action (Cosmos/Wan2 + action head)

## Important Notes

- Qwen-VL weights are at `./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct` — relative path from repo root
- Results and checkpoints are in `.gitignore` — not tracked in git
- `.gitignore` intentionally excludes `*.pt`, `*.pth`, `*.bin`, `results/`, `playground/Pretrained_models` to keep repo small
- For large model files, copy manually to `playground/Pretrained_models/`
