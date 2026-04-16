"""
World Model main class.

Predicts future summary tokens (C_t and P_t) from text + history:
    (C_t, P_t) = WM(text_hidden, z_past)

Architecture:
    TextCompressor  →  LatentProjector  →  CausalTransformerEncoder  →  PlanDecoder / DynamicsDecoder
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .world_blocks import (
    CausalTransformerEncoder,
    LatentProjector,
    QueryDecoder,
    TextCompressor,
)


@dataclass
class WorldModelConfig:
    """Configuration for the World Model."""

    # Context lengths
    H: int = 16  # future horizon (must be > MotionAR's 15-frame chunk)

    # Token counts
    M_c: int = 4  # number of plan tokens
    M_p: int = 4  # number of dynamics tokens
    n_text_queries: int = 8  # number of compressed text tokens

    # Model dimensions
    d_model: int = 512
    n_heads: int = 8
    n_layers_enc: int = 6
    n_layers_dec: int = 2
    d_ffn: int = 2048
    dropout: float = 0.1

    # Input dimensions
    D_text: int = 2048  # Qwen hidden size
    D_action: int = 38  # 38D action for P_t target
    D_latent: int = 64

    # Actor adapter dimensions
    d_actor: int = 512

    # Sequence length limit
    max_seq_len: int = 1024


class WorldModel(nn.Module):
    """
    World Model that predicts future plan (C_t) and dynamics (P_t) tokens.

    Input:  text_hidden [B, L, 2048], z_past [B, T_past, 64] (variable length)
    Output: dict with c_pred, p_pred, aux_pred, c_cond, p_cond
    """

    def __init__(self, config: WorldModelConfig | None = None):
        super().__init__()
        self.config = config or WorldModelConfig()
        cfg = self.config

        # 1. Text compressor: Qwen hidden states → [B, 8, d_model]
        self.text_compressor = TextCompressor(
            D_text=cfg.D_text,
            d_model=cfg.d_model,
            n_queries=cfg.n_text_queries,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )

        # 2. Latent projector: z_past → [B, T_past, d_model]
        self.latent_projector = LatentProjector(
            d_model=cfg.d_model,
            dropout=cfg.dropout,
        )

        # 3. Causal Transformer encoder
        self.encoder = CausalTransformerEncoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers_enc,
            d_ffn=cfg.d_ffn,
            dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len,
        )

        # 4. Plan decoder: 4 queries → cross-attend → [B, 4, d_model]
        self.plan_decoder = QueryDecoder(
            d_model=cfg.d_model,
            n_queries=cfg.M_c,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers_dec,
            d_ffn=cfg.d_ffn,
            dropout=cfg.dropout,
        )

        # 5. Dynamics decoder: 4 queries → cross-attend → [B, 4, d_model]
        self.dynamics_decoder = QueryDecoder(
            d_model=cfg.d_model,
            n_queries=cfg.M_p,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers_dec,
            d_ffn=cfg.d_ffn,
            dropout=cfg.dropout,
        )

        # 6. Prediction heads
        self.head_c = nn.Linear(cfg.d_model, cfg.D_latent)  # 512 → 64
        self.head_p = nn.Linear(cfg.d_model, cfg.D_action)  # 512 → 38
        self.head_aux = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 3),  # 3 scalar aux targets
        )

        # 7. Adapters for actor condition interface (no loss in Phase 1)
        self.adapter_c = nn.Linear(cfg.D_latent, cfg.d_actor)  # 64 → 512
        self.adapter_p = nn.Linear(cfg.D_action, cfg.d_actor)  # 38 → 512

    def forward(
        self,
        text_hidden: torch.Tensor,
        z_past: torch.Tensor,
        z_past_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            text_hidden: [B, L, D_text] — Qwen hidden states from text.
            z_past: [B, T_past, D_latent] — past motion latents (variable length, padded).
            z_past_lengths: [B] — actual lengths for each sequence (0 = all padding → no motion).

        Returns:
            dict with:
                c_pred:     [B, M_c, D_latent]  — predicted plan tokens
                p_pred:     [B, M_p, D_action]  — predicted dynamics tokens
                aux_pred:   [B, 3]              — auxiliary scalar predictions
                c_cond:     [B, M_c, d_actor]   — adapter output (no loss)
                p_cond:     [B, M_p, d_actor]   — adapter output (no loss)
        """
        # Step 1: Compress text
        text_tokens = self.text_compressor(text_hidden)  # [B, n_text_queries, d_model]
        n_text = text_tokens.shape[1]

        # Step 2: Project latent to d_model
        motion_tokens = self.latent_projector(z_past)  # [B, T_past, d_model]

        # Build motion padding mask: [B, T_past], True = padding position
        motion_padding_mask = None
        if z_past_lengths is not None:
            T_past = z_past.shape[1]
            arange = torch.arange(T_past, device=z_past_lengths.device).unsqueeze(0)  # [1, T_past]
            motion_padding_mask = arange >= z_past_lengths.unsqueeze(1)  # [B, T_past]

        # Step 3: Encode with mixed-visibility causal transformer
        encoder_output = self.encoder(text_tokens, motion_tokens, n_text, motion_padding_mask=motion_padding_mask)

        # Step 4: Plan decoding
        plan_hidden = self.plan_decoder(encoder_output)  # [B, M_c, d_model]
        c_pred = self.head_c(plan_hidden)  # [B, M_c, 64]

        # Step 5: Dynamics decoding
        dyn_hidden = self.dynamics_decoder(encoder_output)  # [B, M_p, d_model]
        p_pred = self.head_p(dyn_hidden)  # [B, M_p, 38]

        # Step 6: Auxiliary prediction from dynamics hidden state
        # Use mean over query dimension for scalar prediction
        aux_pred = self.head_aux(dyn_hidden.mean(dim=1))  # [B, 3]

        # Step 7: Adapter outputs for actor condition (no loss in Phase 1)
        c_cond = self.adapter_c(c_pred)  # [B, M_c, d_actor]
        p_cond = self.adapter_p(p_pred)  # [B, M_p, d_actor]

        return {
            "c_pred": c_pred,
            "p_pred": p_pred,
            "aux_pred": aux_pred,
            "c_cond": c_cond,
            "p_cond": p_cond,
        }
