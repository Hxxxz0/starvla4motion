"""
Training script for World Model (Phase 1: standalone training).

Training loop:
    1. text → Qwen (frozen) → text_hidden [B, L, 2048]
    2. (text_hidden, z_past, a_past) → WorldModel → (c_pred, p_pred, aux_pred)
    3. C* = build_c_target(z_fut, Bc), P* = build_p_target(a_fut), aux* = build_aux_target(a_fut)
    4. loss = world_loss(outputs, targets)
    5. backward, step, scheduler

Usage:
    accelerate launch -m starVLA.training.train_world_model \
        --config_yaml starVLA/config/training/world_model_train.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.dataloader.world_model_datasets import build_world_model_dataloader
from starVLA.model.world_model.build_targets import (
    build_aux_target,
    build_c_target,
    build_p_target,
    make_dct_basis,
)
from starVLA.model.world_model.losses import world_loss
from starVLA.model.world_model.world_model import WorldModel, WorldModelConfig

logger = get_logger(__name__)


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_rank0() -> bool:
    return (not _dist_ready()) or dist.get_rank() == 0


class FrozenQwenTextEncoder(torch.nn.Module):
    """Frozen Qwen model that extracts text hidden states [B, L, 2048]."""

    def __init__(self, model_id: str, attn_implementation: str = "sdpa", device: str = "cuda"):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype="auto",
            device_map={"": device},
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of text strings to Qwen's last hidden state.

        Args:
            texts: list of B text strings.

        Returns:
            hidden_states: [B, L, 2048] — last layer hidden states.
        """
        # Build chat messages (text-only, no images)
        messages = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in texts]
        texts_tokenized = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages
        ]
        batch = self.processor(text=texts_tokenized, padding=True, return_tensors="pt")
        batch = batch.to(self.model.device)

        outputs = self.model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
        )
        # Last hidden state
        return outputs.hidden_states[-1]  # [B, L, 2048]


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if _is_rank0():
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def setup_optimizer_and_scheduler(model, cfg) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set up AdamW optimizer and cosine LR scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.base_lr,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.trainer.max_train_steps,
        eta_min=cfg.trainer.scheduler_specific_kwargs.min_lr,
    )

    # Warmup: manually adjust LR for first N steps
    warmup_steps = cfg.trainer.num_warmup_steps

    return optimizer, lr_scheduler, warmup_steps


def save_checkpoint(model, optimizer, lr_scheduler, step, output_dir: Path, cfg):
    """Save model checkpoint."""
    if _is_rank0():
        checkpoint_path = output_dir / "checkpoints" / f"steps_{step}"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state_dict, str(checkpoint_path) + "_pytorch_model.pt")

        # Log to summary
        summary_data = {"steps": step}
        with open(output_dir / "summary.jsonl", "a") as f:
            f.write(json.dumps(summary_data) + "\n")

        logger.info(f"✅ Checkpoint saved at {checkpoint_path}")


def main(cfg) -> None:
    accelerator = Accelerator()
    set_seed(cfg.seed)

    # Setup directories
    output_dir = setup_directories(cfg)

    # Log config
    if _is_rank0():
        logger.info("***** World Model Training *****")
        logger.info(f"  Total optimization steps = {cfg.trainer.max_train_steps}")
        logger.info(f"  Per device batch size = {cfg.data.per_device_batch_size}")
        logger.info(f"  Max past frames = {cfg.data.max_obs_frames}, Future horizon H = {cfg.data.H}")
        logger.info(f"  d_model = {cfg.model.d_model}, layers = {cfg.model.n_layers_enc}")

    # Build WorldModel
    wm_config = WorldModelConfig(
        H=cfg.data.H,
        M_c=cfg.model.M_c,
        M_p=cfg.model.M_p,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers_enc=cfg.model.n_layers_enc,
        n_layers_dec=cfg.model.n_layers_dec,
        d_ffn=cfg.model.d_ffn,
        dropout=cfg.model.dropout,
        D_text=cfg.model.D_text,
        D_action=cfg.model.D_action,
        D_latent=cfg.model.D_latent,
        d_actor=cfg.model.d_actor,
    )
    model = WorldModel(wm_config)

    # Print trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"WorldModel trainable parameters: {n_params:,}")

    # Build dataloader
    dataloader = build_world_model_dataloader(
        data_root_dir=cfg.data.data_root_dir,
        split=cfg.data.train_split,
        batch_size=cfg.data.per_device_batch_size,
        H=cfg.data.H,
        max_obs_frames=cfg.data.max_obs_frames,
        num_workers=cfg.data.num_workers,
    )

    # Build eval dataloader (separate split for real validation)
    eval_dataloader = build_world_model_dataloader(
        data_root_dir=cfg.data.data_root_dir,
        split=cfg.data.eval_split,
        batch_size=cfg.data.per_device_batch_size,
        H=cfg.data.H,
        max_obs_frames=cfg.data.max_obs_frames,
        num_workers=cfg.data.num_workers,
    )

    # Build DCT basis (fixed, on CPU, moved to device during training)
    Bc = make_dct_basis(H=cfg.data.H, M_c=cfg.model.M_c)  # [M_c, H]

    # Build frozen Qwen text encoder (use local rank for multi-GPU)
    if _is_rank0():
        logger.info(f"Loading Qwen from {cfg.qwenvl.base_vlm} (frozen)...")
    qwenvl_model_id = cfg.qwenvl.base_vlm
    attn_impl = cfg.qwenvl.get("attn_implementation", "sdpa")
    local_device = f"cuda:{accelerator.local_process_index}"
    text_encoder = FrozenQwenTextEncoder(qwenvl_model_id, attn_implementation=attn_impl, device=local_device)

    # Set up optimizer and scheduler
    optimizer, lr_scheduler, warmup_steps = setup_optimizer_and_scheduler(model, cfg)

    # Distribute with accelerator
    model, optimizer, dataloader, eval_dataloader = accelerator.prepare(model, optimizer, dataloader, eval_dataloader)

    # Init wandb (graceful fallback if entity/project not configured)
    use_wandb = False
    if accelerator.is_main_process and hasattr(cfg, "wandb_project"):
        try:
            wandb.init(
                name=cfg.run_id,
                dir=os.path.join(cfg.output_dir, "wandb"),
                project=cfg.wandb_project,
                entity=cfg.get("wandb_entity", None),
                group="world-model-train",
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            use_wandb = True
        except Exception as e:
            logger.warning(f"wandb init failed ({e}), continuing without logging")

    # Training loop
    completed_steps = 0
    progress_bar = tqdm(range(cfg.trainer.max_train_steps), disable=not accelerator.is_local_main_process)

    # Loss weights
    lambda_c = cfg.loss.lambda_c
    lambda_p = cfg.loss.lambda_p
    lambda_aux = cfg.loss.lambda_aux
    alpha_cos = cfg.loss.alpha_cos

    # Data iterator
    data_iter = iter(dataloader)
    epoch_count = 0

    while completed_steps < cfg.trainer.max_train_steps:
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            epoch_count += 1
            batch = next(data_iter)

        t_start = time.perf_counter()

        # Encode text with frozen Qwen
        texts = batch["text"]  # list of str
        with torch.no_grad():
            text_hidden = text_encoder(texts)  # [B, L, 2048]

        z_past = batch["z_past"]  # [B, T_past, 64] (padded, variable length)
        z_past_lengths = batch.get("z_past_lengths")  # [B]
        z_fut = batch["z_fut"]  # [B, H, 64]
        a_fut = batch["a_fut"]  # [B, H, 38]

        # World model forward
        # text_hidden is already bfloat16 from Qwen; convert to float32 for WorldModel
        text_hidden = text_hidden.to(torch.float32)
        outputs = model(text_hidden, z_past, z_past_lengths=z_past_lengths)

        # Build deterministic targets
        Bc_device = Bc.to(z_fut.device)
        c_star = build_c_target(z_fut, Bc_device)  # [B, M_c, 64]
        p_star = build_p_target(a_fut)  # [B, M_p, 38]
        aux_star = build_aux_target(a_fut)  # [B, 3]

        # Compute loss
        loss, loss_dict = world_loss(
            outputs,
            {"c_star": c_star, "p_star": p_star, "aux_star": aux_star},
            alpha_cos=alpha_cos,
            lambda_c=lambda_c,
            lambda_p=lambda_p,
            lambda_aux=lambda_aux,
        )

        # Backward
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)
        optimizer.step()

        # LR scheduler step (with warmup)
        if completed_steps < warmup_steps:
            # Linear warmup
            lr_scale = (completed_steps + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.trainer.base_lr * lr_scale
        else:
            lr_scheduler.step()

        optimizer.zero_grad()

        t_end = time.perf_counter()

        # Progress
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        # Logging
        if completed_steps % cfg.trainer.logging_frequency == 0 and accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]["lr"]
            log_dict = {
                "loss": loss.item(),
                "lr": current_lr,
                "epoch": epoch_count,
                "data_time": t_end - t_start,
            }
            log_dict.update(loss_dict)
            if use_wandb:
                wandb.log(log_dict, step=completed_steps)
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}", "time": f"{t_end - t_start:.3f}s"}
            )

        # Evaluation (on validation set — just metrics, no backward)
        if completed_steps % cfg.trainer.eval_interval == 0 and completed_steps > 0:
            model.eval()
            eval_loss_c, eval_loss_p, eval_loss_aux = 0.0, 0.0, 0.0
            n_eval_batches = 0
            with torch.no_grad():
                for eval_batch in eval_dataloader:
                    eval_texts = eval_batch["text"]
                    eval_text_hidden = text_encoder(eval_texts)
                    eval_text_hidden = eval_text_hidden.to(torch.float32)
                    eval_z_past = eval_batch["z_past"]
                    eval_z_past_lengths = eval_batch.get("z_past_lengths")
                    eval_z_fut = eval_batch["z_fut"]
                    eval_a_fut = eval_batch["a_fut"]

                    eval_Bc = Bc.to(eval_z_fut.device)
                    eval_c_star = build_c_target(eval_z_fut, eval_Bc)
                    eval_p_star = build_p_target(eval_a_fut)
                    eval_aux_star = build_aux_target(eval_a_fut)

                    eval_outputs = model(eval_text_hidden, eval_z_past, z_past_lengths=eval_z_past_lengths)
                    _, eval_loss_dict = world_loss(
                        eval_outputs,
                        {"c_star": eval_c_star, "p_star": eval_p_star, "aux_star": eval_aux_star},
                        alpha_cos=alpha_cos,
                        lambda_c=lambda_c,
                        lambda_p=lambda_p,
                        lambda_aux=lambda_aux,
                    )
                    eval_loss_c += eval_loss_dict["loss_c"]
                    eval_loss_p += eval_loss_dict["loss_p"]
                    eval_loss_aux += eval_loss_dict["loss_aux"]
                    n_eval_batches += 1
                    if n_eval_batches >= 5:
                        break  # average over up to 5 batches

                if n_eval_batches == 0:
                    logger.warning("Eval dataloader is empty, skipping evaluation")
                elif accelerator.is_main_process:
                    avg_loss_c = eval_loss_c / n_eval_batches
                    avg_loss_p = eval_loss_p / n_eval_batches
                    avg_loss_aux = eval_loss_aux / n_eval_batches
                    avg_eval_loss = lambda_c * avg_loss_c + lambda_p * avg_loss_p + lambda_aux * avg_loss_aux
                    eval_log = {
                        "eval_loss": avg_eval_loss,
                        "eval_loss_c": avg_loss_c,
                        "eval_loss_p": avg_loss_p,
                        "eval_loss_aux": avg_loss_aux,
                    }
                    if use_wandb:
                        wandb.log(eval_log, step=completed_steps)
                    logger.info(
                        f"Eval at step {completed_steps}: loss={avg_eval_loss:.4f} "
                        f"(c={avg_loss_c:.4f}, p={avg_loss_p:.4f}, aux={avg_loss_aux:.4f})"
                    )
            model.train()

        # Save checkpoint
        if completed_steps % cfg.trainer.save_interval == 0 and completed_steps > 0:
            save_checkpoint(model, optimizer, lr_scheduler, completed_steps, output_dir, cfg)

        if completed_steps >= cfg.trainer.max_train_steps:
            break

    # Final save
    if accelerator.is_main_process:
        save_checkpoint(model, optimizer, lr_scheduler, completed_steps, output_dir, cfg)
        if use_wandb:
            wandb.finish()
        logger.info(f"Training complete. Final model at {output_dir}/checkpoints/steps_{completed_steps}")

    accelerator.wait_for_everyone()

    if _dist_ready():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/world_model_train.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    # Handle both --key value and key value formats
    normalized = normalize_dotlist_args(clipargs)
    # Also handle bare dotlist args (without -- prefix): key1 val1 key2 val2
    i = 0
    while i < len(clipargs):
        arg = clipargs[i]
        if arg.startswith("--"):
            i += 2  # skip --key value (already handled by normalize_dotlist_args)
        elif i + 1 < len(clipargs) and not clipargs[i + 1].startswith("--"):
            # bare key value pair
            normalized.append(f"{arg}={clipargs[i + 1]}")
            i += 2
        else:
            i += 1

    if normalized:
        cli_cfg = OmegaConf.from_dotlist(normalized)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
