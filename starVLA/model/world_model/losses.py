"""
Loss functions for World Model training.

Loss decomposition:
    L_wm = lambda_c * L_C + lambda_p * L_P + lambda_aux * L_aux

C_t loss: SmoothL1 + Cosine similarity per token, weighted [1.0, 0.75, 0.5, 0.25]
P_t loss: SmoothL1
Aux loss: MSE

IMPORTANT: c_cond and p_cond (adapter outputs) do NOT participate in any loss.
"""

import torch
import torch.nn.functional as F


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity. Returns scalar per sample."""
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    # Cosine similarity along the last dimension
    cos_sim = (pred_norm * target_norm).sum(dim=-1)  # [B, M]
    return (1.0 - cos_sim).mean()  # scalar


def world_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    alpha_cos: float = 0.2,
    lambda_c: float = 1.0,
    lambda_p: float = 0.5,
    lambda_aux: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the world model loss.

    Args:
        outputs: dict from WorldModel.forward() with c_pred, p_pred, aux_pred.
        targets: dict with:
            c_star: [B, M_c, D_latent] — DCT plan targets
            p_star: [B, M_p, D_action] — dynamics targets
            aux_star: [B, 3] — auxiliary scalar targets

    Returns:
        total_loss: scalar
        loss_dict: {"loss_c": ..., "loss_p": ..., "loss_aux": ...}
    """
    c_pred = outputs["c_pred"]  # [B, M_c, D_latent]
    p_pred = outputs["p_pred"]  # [B, M_p, D_action]
    aux_pred = outputs["aux_pred"]  # [B, 3]

    c_star = targets["c_star"]  # [B, M_c, D_latent]
    p_star = targets["p_star"]  # [B, M_p, D_action]
    aux_star = targets["aux_star"]  # [B, 3]

    # Token weights for C_t: lower-frequency tokens are more important
    M_c = c_pred.shape[1]
    assert M_c <= 4, f"Only 4 token weights defined, but M_c={M_c}"
    c_weights = torch.tensor([1.0, 0.75, 0.5, 0.25], device=c_pred.device)

    # C_t loss: SmoothL1 + Cosine per token
    loss_c_smoothl1 = torch.tensor(0.0, device=c_pred.device)
    loss_c_cos = torch.tensor(0.0, device=c_pred.device)

    for m in range(M_c):
        w = c_weights[m] if m < len(c_weights) else 0.25
        loss_c_smoothl1 += w * F.smooth_l1_loss(c_pred[:, m], c_star[:, m])
        loss_c_cos += w * cosine_loss(c_pred[:, m], c_star[:, m])

    loss_c = loss_c_smoothl1 + alpha_cos * loss_c_cos

    # P_t loss: SmoothL1
    loss_p = F.smooth_l1_loss(p_pred, p_star)

    # Aux loss: MSE
    loss_aux = F.mse_loss(aux_pred, aux_star)

    # Total loss
    total_loss = lambda_c * loss_c + lambda_p * loss_p + lambda_aux * loss_aux

    loss_dict = {
        "loss_c": loss_c.item(),
        "loss_p": loss_p.item(),
        "loss_aux": loss_aux.item(),
        "loss_c_smoothl1": loss_c_smoothl1.item(),
        "loss_c_cos": loss_c_cos.item(),
    }

    return total_loss, loss_dict
