# starVLA for Motion Generation

基于 starVLA 框架的动作生成项目。

## 项目简介

本项目基于 [starVLA](https://github.com/starVLA/starVLA) 框架，专注于动作生成（Motion Generation）任务。starVLA 是一个模块化、可扩展的 Vision-Language-Action (VLA) 模型开发平台，支持快速原型设计和独立调试。

## 主要功能

- **模块化设计**：模型、数据、训练器、配置等组件高度解耦，支持即插即用
- **多种 VLA 架构支持**：
  - StarVLA-FAST：自回归离散动作 token
  - StarVLA-OFT：并行连续动作解码
  - StarVLA-PI：基于 Flow-Matching 的扩散动作
  - StarVLA-GR00T：双系统架构（VLM 作为 System 2，Flow-Matching 作为 System 1）
- **灵活的训练方案**：支持 SFT、多目标联合训练、跨躯体联合训练等
- **多基准集成**：支持 LIBERO、SimplerEnv、RoboTwin、VLA-Arena 等基准

## 快速开始

### 环境安装

```bash
# 克隆仓库
git clone git@github.com:Hxxxz0/starvla4motion.git
cd starvla4motion

# 安装依赖
pip install -e .
```

### 训练

```bash
# 使用示例配置进行训练
bash run_oxe_train.sh

# 或使用自定义配置
python starVLA/training/train_starvla.py \
    --config-path starVLA/config \
    --config-name your_config.yaml
```

### 评估

```bash
# 在特定基准上评估
python examples/LIBERO/eval_files/eval_libero.py \
    --model_path /path/to/checkpoint
```

## 目录结构

```
starVLA/
├── starVLA/              # 核心代码
│   ├── model/            # 模型定义
│   ├── dataloader/       # 数据加载器
│   ├── training/         # 训练脚本
│   └── config/           # 配置文件
├── examples/             # 示例和基准
├── results/              # 训练结果（已忽略）
├── wandb/                # W&B 日志（已忽略）
└── README.md             # 本文件
```

## 训练说明

1. **数据准备**：根据目标基准准备数据集
2. **配置修改**：在 `starVLA/config/` 下修改训练配置
3. **启动训练**：使用训练脚本启动，支持单卡/多卡训练
4. **监控进度**：通过 W&B 或 TensorBoard 监控训练进度

## 评估说明

1. **加载 checkpoint**：从 `results/` 目录选择最佳 checkpoint
2. **运行评估**：使用对应的评估脚本
3. **结果分析**：评估结果将保存在 `results/` 目录

## 注意事项

- 大文件（checkpoints、模型文件、数据集等）已通过 `.gitignore` 排除
- 训练结果和日志不会提交到仓库
- 请确保有足够的 GPU 内存进行训练

## 致谢

本项目基于 [starVLA](https://github.com/starVLA/starVLA) 框架开发。

## License

继承原项目 License，详见 [LICENSE](LICENSE) 文件。
