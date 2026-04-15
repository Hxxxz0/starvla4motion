"""
Building blocks for the World Model.

Key design: CausalTransformerEncoder uses explicit nn.MultiheadAttention with a
manually constructed combined attention mask. It does NOT reuse BasicTransformerBlock
because that class cannot express the mixed visibility pattern we need:
  - text token ↔ text token: full bidirectional visibility
  - motion token → text token: motion can see all text
  - motion token ↔ motion token: strictly causal (lower triangular)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Lower triangular causal mask. True = can attend."""
    mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    return mask  # [L, L]


def _make_combined_mask(
    n_text: int, n_motion: int, device: torch.device,
    motion_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Construct the combined attention mask for the CausalTransformerEncoder.

    Sequence layout: [text_0, ..., text_{n_text-1}, motion_0, ..., motion_{n_motion-1}]

    Mask rules (True = can attend):
    - text ↔ text: full (all True)
    - text → motion: text does NOT need to attend to motion (False)
    - motion → text: motion CAN see all text (True)
    - motion → motion: causal/lower triangular (only past motion)
    - padding positions: no one can attend to them, they can't attend to anyone

    nn.MultiheadAttention with attn_mask expects:
      - If bool: True positions are masked OUT (i.e., -inf attention)
      - We want the OPPOSITE: True = can attend, so we invert for nn.MultiheadAttention

    Args:
        motion_padding_mask: bool tensor [B, n_motion], True = padding position.

    Returns:
        attn_mask for nn.MultiheadAttention (bool, True = MASKED OUT)
        Shape: [B, total, total] if padding mask provided, else [total, total].
    """
    total = n_text + n_motion

    # Start with what CAN attend (True = can attend)
    can_attend = torch.zeros(total, total, dtype=torch.bool, device=device)

    # text ↔ text: full visibility
    can_attend[:n_text, :n_text] = True

    # motion → text: motion can see all text
    can_attend[n_text:, :n_text] = True

    # motion → motion: causal
    can_attend[n_text:, n_text:] = _make_causal_mask(n_motion, device)

    # If padding mask provided, expand per-sample and mask out padding positions
    if motion_padding_mask is not None:
        B = motion_padding_mask.shape[0]
        # [B, total, total] — broadcast the [total, total] pattern per sample
        attn_mask = (~can_attend).unsqueeze(0).expand(B, -1, -1).clone()  # [B, total, total]
        # No one can attend TO padding motion positions (column)
        padding_cols = motion_padding_mask  # [B, n_motion]
        attn_mask[:, :, n_text:].masked_fill_(padding_cols.unsqueeze(1), True)
        # Padding positions can't attend to anyone (row)
        padding_rows = motion_padding_mask  # [B, n_motion]
        attn_mask[:, n_text:, :].masked_fill_(padding_rows.unsqueeze(2), True)
        # Let padding positions see themselves on diagonal to avoid softmax NaN
        diag_idx = torch.arange(total, device=device)
        for b in range(B):
            attn_mask[b, diag_idx, diag_idx] = False
        return attn_mask  # [B, total, total]

    # nn.MultiheadAttention: True = masked out (attention = -inf)
    attn_mask = ~can_attend  # invert: True = cannot attend
    return attn_mask  # [total, total]


class TextCompressor(nn.Module):
    """
    Compress variable-length text hidden states into a fixed number of
    learnable query tokens via cross-attention.

    Input:  text_hidden [B, L_text, D_text]
    Output: text_tokens [B, n_queries, d_model]
    """

    def __init__(self, D_text: int, d_model: int, n_queries: int = 8, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.query_proj = nn.Linear(D_text, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_hidden: [B, L_text, D_text] — Qwen hidden states.

        Returns:
            text_tokens: [B, n_queries, d_model]
        """
        B = text_hidden.shape[0]

        # Project text to d_model
        text_proj = self.query_proj(text_hidden)  # [B, L_text, d_model]

        # Learnable queries broadcasted to batch
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, n_queries, d_model]

        # Cross-attention: queries attend to text
        attn_out, _ = self.attn(q, text_proj, text_proj)  # [B, n_queries, d_model]

        # Residual + norm
        out = self.norm(q + self.dropout(attn_out))
        return out


class LatentProjector(nn.Module):
    """
    Project 64D motion latent into d_model space.

    Input:  z [B, T, 64]
    Output: m [B, T, d_model]
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, T, 64]

        Returns:
            m: [B, T, d_model]
        """
        return self.net(z)


class FeedForward(nn.Module):
    """Standard FFN: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalTransformerEncoder(nn.Module):
    """
    Transformer encoder with mixed visibility:
      - text tokens: full bidirectional attention
      - motion tokens: can see all text tokens, but only causal attention among motion tokens

    Implemented explicitly with nn.MultiheadAttention (not BasicTransformerBlock)
    because BasicTransformerBlock cannot express per-region visibility rules.

    Architecture per layer:
        x = LayerNorm(x)
        x = self_attn(x, x, x, attn_mask=combined_mask) + residual
        x = LayerNorm(x)
        x = FFN(x) + residual
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ffn: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(
                            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
                        ),
                        "norm2": nn.LayerNorm(d_model),
                        "ffn": FeedForward(d_model, d_ffn, dropout),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_tokens: torch.Tensor,
        motion_tokens: torch.Tensor,
        n_text: int,
        motion_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            text_tokens: [B, n_text, d_model] — compressed text from TextCompressor.
            motion_tokens: [B, T, d_model] — projected motion from LatentProjector.
            n_text: number of text tokens (must match text_tokens.shape[1]).
            motion_padding_mask: bool tensor [B, T], True = padding position.

        Returns:
            encoder_output: [B, n_text + T, d_model]
        """
        n_motion = motion_tokens.shape[1]
        total = n_text + n_motion

        # Concatenate: [text_tokens; motion_tokens]
        x = torch.cat([text_tokens, motion_tokens], dim=1)  # [B, total, d_model]

        # Add positional embeddings
        pos = self.pos_embed.weight[:total]  # [total, d_model]
        x = x + pos.unsqueeze(0)  # [B, total, d_model]
        x = self.dropout(x)

        # Build combined attention mask (shared across layers — mask pattern is the same)
        # If padding mask provided: [B, total, total], else: [total, total]
        combined_mask = _make_combined_mask(n_text, n_motion, device=x.device, motion_padding_mask=motion_padding_mask)
        has_batched_mask = motion_padding_mask is not None

        # Transformer layers
        for layer in self.layers:
            # Self-attention with pre-LN
            normed = layer["norm1"](x)
            if has_batched_mask:
                # PyTorch MHA with batch_first expects 3D mask of shape [B * n_heads, seq_q, seq_k]
                float_mask = combined_mask.float().masked_fill(combined_mask, float("-inf"))  # [B, total, total]
                float_mask = float_mask.repeat_interleave(self.n_heads, dim=0)  # [B*n_heads, total, total]
                attn_out, _ = layer["attn"](
                    normed, normed, normed, attn_mask=float_mask
                )
            else:
                attn_out, _ = layer["attn"](
                    normed, normed, normed, attn_mask=combined_mask
                )
            x = x + attn_out

            # FFN with pre-LN
            normed = layer["norm2"](x)
            ffn_out = layer["ffn"](normed)
            x = x + ffn_out

        return x  # [B, total, d_model]


class QueryDecoder(nn.Module):
    """
    M learnable queries cross-attend to encoder memory.

    Input:  encoder_memory [B, seq_len, d_model]
    Output: decoded [B, M, d_model]
    """

    def __init__(
        self,
        d_model: int = 512,
        n_queries: int = 4,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ffn: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
                        ),
                        "norm2": nn.LayerNorm(d_model),
                        "self_attn": nn.MultiheadAttention(
                            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
                        ),
                        "norm3": nn.LayerNorm(d_model),
                        "ffn": FeedForward(d_model, d_ffn, dropout),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_memory: [B, seq_len, d_model]

        Returns:
            decoded: [B, n_queries, d_model]
        """
        B = encoder_memory.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, M, d_model]

        for layer in self.layers:
            # Cross-attention: queries attend to encoder memory (pre-LN)
            normed_q = layer["norm1"](q)
            normed_mem = layer["norm1"](encoder_memory)
            cross_out, _ = layer["cross_attn"](normed_q, normed_mem, normed_mem)
            q = q + self.dropout(cross_out)

            # Self-attention among queries
            normed_q = layer["norm2"](q)
            self_out, _ = layer["self_attn"](normed_q, normed_q, normed_q)
            q = q + self.dropout(self_out)

            # FFN
            normed_q = layer["norm3"](q)
            ffn_out = layer["ffn"](normed_q)
            q = q + self.dropout(ffn_out)

        return q  # [B, M, d_model]
