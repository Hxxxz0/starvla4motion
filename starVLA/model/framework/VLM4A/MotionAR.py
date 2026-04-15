from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.world_model.world_model import WorldModel, WorldModelConfig
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@dataclass
class MotionARDefaultConfig:
    name: str = "MotionAR"
    latent_dim: int = 64
    max_obs_frames: int = 150
    obs_encoder_hidden_dim: int = 768
    # WorldModel (Phase 2)
    use_world_model: bool = False
    world_model_checkpoint: str = ""
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct",
            "attn_implementation": "sdpa",
            "vl_hidden_dim": 2048,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 64,
            "state_dim": None,
            "future_action_window_size": 14,
            "action_horizon": 15,
            "past_action_window_size": 0,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )
    obs_image_size: Optional[list] = None


class LatentObsEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, mlp_hidden_dim: int, max_obs_frames: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = mlp_hidden_dim
        self.max_obs_frames = max_obs_frames
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.pos_embed = nn.Embedding(max_obs_frames, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, obs_latent: torch.Tensor) -> torch.Tensor:
        if obs_latent.ndim != 2:
            raise ValueError(f"Expected obs_latent to have shape [T, {self.latent_dim}], got {tuple(obs_latent.shape)}")

        if obs_latent.shape[0] > self.max_obs_frames:
            obs_latent = obs_latent[-self.max_obs_frames :]

        pos_ids = torch.arange(obs_latent.shape[0], device=obs_latent.device)
        hidden = self.mlp(obs_latent) + self.pos_embed(pos_ids)
        return self.norm(hidden)


@FRAMEWORK_REGISTRY.register("MotionAR")
class MotionAR(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(MotionARDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # Get VLM hidden size for DiT cross_attention_dim
        qwen_cfg = self.qwen_vl_interface.model.config
        vlm_hidden_size = getattr(qwen_cfg, "hidden_size", None) or getattr(qwen_cfg.text_config, "hidden_size", 2048)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = vlm_hidden_size

        self.latent_dim = getattr(self.config.framework, "latent_dim", self.config.framework.action_model.action_dim)
        self.max_obs_frames = self.config.framework.max_obs_frames
        self.latent_obs_encoder = LatentObsEncoder(
            latent_dim=self.latent_dim,
            hidden_size=vlm_hidden_size,
            mlp_hidden_dim=self.config.framework.obs_encoder_hidden_dim,
            max_obs_frames=self.max_obs_frames,
        )
        self.bos_obs_token = nn.Parameter(torch.zeros(1, vlm_hidden_size))
        nn.init.normal_(self.bos_obs_token, mean=0.0, std=0.02)
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # WorldModel（Phase 2）
        self.use_world_model = getattr(self.config.framework, "use_world_model", False)
        if self.use_world_model:
            wm_cfg = WorldModelConfig()
            self.world_model = WorldModel(config=wm_cfg)
            wm_ckpt = getattr(self.config.framework, "world_model_checkpoint", "")
            if wm_ckpt:
                self._load_world_ckpt(wm_ckpt)
            for p in self.world_model.parameters():
                p.requires_grad = False
            self.world_model.eval()
            self.world_token_proj = nn.Linear(wm_cfg.d_actor, vlm_hidden_size)
        else:
            self.world_model = None
            self.world_token_proj = None

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.action_horizon = self.config.framework.action_model.action_horizon
        self.chunk_len = self.future_action_window_size + 1
        if self.action_horizon != self.chunk_len:
            raise ValueError(
                f"Expected action_horizon == future_action_window_size + 1, got "
                f"{self.action_horizon} vs {self.future_action_window_size + 1}"
            )

    def _normalize_batch_images(self, examples: List[dict]) -> List[list]:
        batch_images = []
        for example in examples:
            images = example.get("image", [])
            if images is None:
                images = []
            if not isinstance(images, list):
                images = [images]
            batch_images.append(images)
        return batch_images

    def _encode_obs_batch(self, obs_latents, device, dtype):
        encoded = []
        lengths = []

        for obs in obs_latents:
            obs_array = np.asarray(obs, dtype=np.float32)
            if obs_array.ndim == 1 and obs_array.size == 0:
                obs_array = obs_array.reshape(0, self.latent_dim)
            elif obs_array.ndim == 1:
                obs_array = obs_array.reshape(1, self.latent_dim)
            obs_tensor = torch.as_tensor(obs_array, device=device, dtype=dtype)
            if obs_tensor.shape[0] == 0:
                obs_hidden = self.bos_obs_token.to(device=device, dtype=dtype)
            else:
                obs_hidden = self.latent_obs_encoder(obs_tensor)
            encoded.append(obs_hidden)
            lengths.append(obs_hidden.shape[0])

        obs_hidden = pad_sequence(encoded, batch_first=True)
        obs_lengths = torch.tensor(lengths, device=device, dtype=torch.long)
        obs_mask = torch.arange(obs_hidden.shape[1], device=device).unsqueeze(0) < obs_lengths.unsqueeze(1)
        return obs_hidden, obs_mask

    def _pad_obs_latents(self, obs_latents, device, dtype):
        """Pad list of obs_latent → [B, T_max, 64] + lengths [B]."""
        tensors = []
        lengths = []
        for obs in obs_latents:
            arr = np.asarray(obs, dtype=np.float32)
            if arr.ndim == 1 and arr.size == 0:
                arr = arr.reshape(0, self.latent_dim)
            elif arr.ndim == 1:
                arr = arr.reshape(1, self.latent_dim)
            # Force float32 to match WM training dtype
            t = torch.as_tensor(arr, device=device, dtype=torch.float32)
            tensors.append(t)
            lengths.append(t.shape[0])
        padded = pad_sequence(tensors, batch_first=True)  # [B, T_max, 64]
        lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
        return padded, lengths_tensor

    def _build_world_condition(self, text_hidden, obs_latents):
        """
        Run WorldModel to produce world condition tokens.

        Args:
            text_hidden: [B, L, 2048] from Qwen.
            obs_latents: list of per-sample obs_latent arrays (each [T_i, 64]).

        Returns:
            world_hidden: [B, 8, 2048] projected world tokens.
            world_mask: [B, 8] bool mask (all True).
        """
        device = text_hidden.device
        dtype = text_hidden.dtype
        padded, lengths = self._pad_obs_latents(obs_latents, device, dtype)

        # Convert text_hidden to float32 to match WM training dtype
        text_hidden_fp32 = text_hidden.to(torch.float32)

        with torch.no_grad():
            wm_out = self.world_model(text_hidden_fp32, padded, z_past_lengths=lengths)

        # c_cond [B, 4, 512] + p_cond [B, 4, 512] → [B, 8, 512]
        world_tokens = torch.cat([wm_out["c_cond"], wm_out["p_cond"]], dim=1)
        # Project to DiT cross_attention_dim
        world_hidden = self.world_token_proj(world_tokens)
        world_mask = torch.ones(text_hidden.shape[0], world_hidden.shape[1], device=device, dtype=torch.bool)
        return world_hidden, world_mask

    def _load_world_ckpt(self, ckpt_path: str):
        """Load WorldModel checkpoint."""
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            ckpt = load_file(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Handle key prefix
        filtered = {}
        for k, v in ckpt.items():
            if k.startswith("world_model."):
                filtered[k[len("world_model.") :]] = v
            else:
                filtered[k] = v
        missing, unexpected = self.world_model.load_state_dict(filtered, strict=False)
        if missing:
            logger.warning(f"WM missing keys: {missing}")
        if unexpected:
            logger.warning(f"WM unexpected keys: {unexpected}")
        logger.info(f"WM checkpoint loaded from: {ckpt_path}")

    def _build_condition(self, examples: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_images = self._normalize_batch_images(examples)
        instructions = [example["lang"] for example in examples]
        obs_latents = [example.get("obs_latent", np.zeros((0, self.latent_dim), dtype=np.float32)) for example in examples]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            text_hidden = qwenvl_outputs.hidden_states[-1]
            text_mask = qwen_inputs["attention_mask"].to(device=text_hidden.device, dtype=torch.bool)
            obs_hidden, obs_mask = self._encode_obs_batch(obs_latents, text_hidden.device, text_hidden.dtype)
            condition = torch.cat([text_hidden, obs_hidden], dim=1)
            encoder_attention_mask = torch.cat([text_mask, obs_mask], dim=1)

            # WorldModel condition
            if self.use_world_model:
                world_hidden, world_mask = self._build_world_condition(text_hidden, obs_latents)
                condition = torch.cat([condition, world_hidden], dim=1)
                encoder_attention_mask = torch.cat([encoder_attention_mask, world_mask], dim=1)

        return condition, encoder_attention_mask

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        condition, encoder_attention_mask = self._build_condition(examples)
        actions = [example["action"] for example in examples]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=condition.device, dtype=torch.float32)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            condition_repeated = condition.repeat(repeated_diffusion_steps, 1, 1)
            encoder_attention_mask_repeated = encoder_attention_mask.repeat(repeated_diffusion_steps, 1)

            action_loss = self.action_model(
                condition_repeated,
                actions_target_repeated,
                None,
                encoder_attention_mask=encoder_attention_mask_repeated,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], num_chunks: int = 20, max_obs_frames: Optional[int] = None, **kwargs):
        if type(examples) is not list:
            examples = [examples]

        max_obs_frames = max_obs_frames or self.max_obs_frames
        batch_images = self._normalize_batch_images(examples)
        instructions = [example["lang"] for example in examples]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            text_hidden = qwenvl_outputs.hidden_states[-1]
            text_mask = qwen_inputs["attention_mask"].to(device=text_hidden.device, dtype=torch.bool)

        history_latents = []
        for example in examples:
            obs_latent = example.get("obs_latent", np.zeros((0, self.latent_dim), dtype=np.float32))
            obs_latent = np.asarray(obs_latent, dtype=np.float32)
            if obs_latent.ndim == 1 and obs_latent.size == 0:
                obs_latent = obs_latent.reshape(0, self.latent_dim)
            elif obs_latent.ndim == 1:
                obs_latent = obs_latent.reshape(1, self.latent_dim)
            history_latents.append(obs_latent)

        generated_chunks = [[] for _ in examples]

        for _ in range(num_chunks):
            obs_windows = []
            for history in history_latents:
                if history.shape[0] > max_obs_frames:
                    obs_windows.append(history[-max_obs_frames:])
                else:
                    obs_windows.append(history)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                obs_hidden, obs_mask = self._encode_obs_batch(obs_windows, text_hidden.device, text_hidden.dtype)
                condition = torch.cat([text_hidden, obs_hidden], dim=1)
                encoder_attention_mask = torch.cat([text_mask, obs_mask], dim=1)

                # WorldModel condition
                if self.use_world_model:
                    world_hidden, world_mask = self._build_world_condition(text_hidden, obs_windows)
                    condition = torch.cat([condition, world_hidden], dim=1)
                    encoder_attention_mask = torch.cat([encoder_attention_mask, world_mask], dim=1)

            with torch.autocast("cuda", dtype=torch.float32):
                next_chunk = self.action_model.predict_action(
                    condition,
                    state=None,
                    encoder_attention_mask=encoder_attention_mask,
                )

            next_chunk_np = next_chunk.detach().cpu().float().numpy()
            for idx, chunk in enumerate(next_chunk_np):
                history_latents[idx] = np.concatenate([history_latents[idx], chunk], axis=0)
                generated_chunks[idx].append(chunk)

        generated = []
        for chunks in generated_chunks:
            if chunks:
                generated.append(np.concatenate(chunks, axis=0).astype(np.float32))
            else:
                generated.append(np.zeros((0, self.latent_dim), dtype=np.float32))

        normalized_actions = np.stack(generated, axis=0)
        return {"normalized_actions": normalized_actions}
