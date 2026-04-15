"""
Deterministic target construction for World Model training.

All functions are pure (no nn.Module, no learnable parameters).
Targets are constructed from ground-truth future windows and used as
supervision signals for the World Model predictions.
"""

import torch


def make_dct_basis(H: int = 8, M_c: int = 4) -> torch.Tensor:
    """
    Construct the first M_c coefficients of the DCT-II basis for a window of length H.

    The DCT-II basis matrix B has shape [M_c, H] where:
        B[m, h] = cos(pi * m * (2h + 1) / (2H)) / sqrt(H/2)

    We use orthonormal normalization so that B @ B^T = I.

    Args:
        H: Window length (future horizon).
        M_c: Number of low-frequency basis vectors to extract.

    Returns:
        DCT basis matrix of shape [M_c, H].
    """
    h = torch.arange(H, dtype=torch.float32)  # [H]
    m = torch.arange(M_c, dtype=torch.float32)  # [M_c]

    # DCT-II: cos(pi * m * (2h + 1) / (2H))
    basis = torch.cos(torch.outer(m, 2 * h + 1) * (torch.pi / (2 * H)))  # [M_c, H]

    # Orthonormal normalization
    basis[0] *= 1.0 / torch.sqrt(torch.tensor(H, dtype=torch.float32))
    basis[1:] *= torch.sqrt(torch.tensor(2.0 / H, dtype=torch.float32))

    return basis  # [M_c, H]


def build_c_target(z_fut: torch.Tensor, Bc: torch.Tensor) -> torch.Tensor:
    """
    Construct plan targets by projecting future latent chunk onto fixed DCT basis.

    Args:
        z_fut: Future latent window, shape [B, H, 64].
        Bc: DCT basis matrix, shape [M_c, H] (from make_dct_basis).

    Returns:
        Plan targets C_t* of shape [B, M_c, 64].
    """
    # einsum: Bc[m,h] * z_fut[b,h,d] -> c_star[b,m,d]
    return torch.einsum("mh,bhd->bmd", Bc.to(z_fut.device), z_fut)  # [B, M_c, 64]


def build_p_target(a_fut: torch.Tensor) -> torch.Tensor:
    """
    Construct dynamics targets from future action chunk using fixed statistics.

    Four tokens per action dimension:
        Token 0: mean (future action block average position)
        Token 1: net change (final - initial action)
        Token 2: mean first-order difference (average velocity)
        Token 3: mean second-order difference (average acceleration/curvature)

    Args:
        a_fut: Future action window, shape [B, H, D_action].

    Returns:
        Dynamics targets P_t* of shape [B, 4, D_action].
    """
    B, H, D = a_fut.shape

    # Token 0: mean
    p1 = a_fut.mean(dim=1)  # [B, D]

    # Token 1: net change
    p2 = a_fut[:, -1] - a_fut[:, 0]  # [B, D]

    # Token 2: mean first-order difference
    da = a_fut[:, 1:] - a_fut[:, :-1]  # [B, H-1, D]
    p3 = da.mean(dim=1)  # [B, D]

    # Token 3: mean second-order difference
    if H >= 3:
        d2a = da[:, 1:] - da[:, :-1]  # [B, H-2, D]
        p4 = d2a.mean(dim=1)  # [B, D]
    else:
        p4 = torch.zeros_like(p1)

    return torch.stack([p1, p2, p3, p4], dim=1)  # [B, 4, D]


def build_aux_target(a_fut: torch.Tensor) -> torch.Tensor:
    """
    Construct auxiliary scalar targets for the dynamics branch.

    Three scalars:
        - Mean squared action magnitude (energy)
        - Mean squared first-order difference (velocity energy)
        - Mean squared second-order difference (jerk energy)

    Args:
        a_fut: Future action window, shape [B, H, D_action].

    Returns:
        Auxiliary targets of shape [B, 3].
    """
    # Energy: mean of squared actions
    energy = (a_fut**2).mean(dim=(1, 2))  # [B]

    # Velocity energy: mean of squared first-order differences
    da = a_fut[:, 1:] - a_fut[:, :-1]  # [B, H-1, D]
    vel_energy = (da**2).mean(dim=(1, 2))  # [B]

    # Jerk energy: mean of squared second-order differences
    H = a_fut.shape[1]
    if H >= 3:
        d2a = da[:, 1:] - da[:, :-1]  # [B, H-2, D]
        jerk = (d2a**2).mean(dim=(1, 2))  # [B]
    else:
        jerk = torch.zeros_like(energy)

    return torch.stack([energy, vel_energy, jerk], dim=1)  # [B, 3]
