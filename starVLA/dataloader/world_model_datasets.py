"""
Dataset and dataloader for World Model training.

WorldModelDataset:
    - Filters T < H at __init__ time (constraint: no short samples)
    - z_past: variable-length [0:t], truncated to max_obs_frames (like MotionAR)
    - z_fut: fixed H-frame future latent window
    - a_fut: H-frame future action (38D, constructed from npz for P_t target)
    - Does NOT take a_past as input (MotionAR doesn't have it at inference time)

Qwen text encoding is done in the training loop (not in __getitem__),
since Qwen needs GPU and is frozen.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation as R


def collate_fn(batch: list[dict]) -> dict:
    """Stack tensors; z_past is variable-length so we pad it and track lengths."""
    out = {}
    for key in batch[0]:
        if key == "text":
            out[key] = [item[key] for item in batch]
        elif key == "z_past":
            # Variable-length: pad_sequence expects list of [T_i, 64]
            z_past_list = [item[key] for item in batch]
            lengths = [z.shape[0] for z in z_past_list]
            out[key] = pad_sequence(z_past_list, batch_first=True)
            out["z_past_lengths"] = torch.tensor(lengths, dtype=torch.long)
        else:
            out[key] = torch.stack([item[key] for item in batch])
    return out


def _quat_wxyz_to_6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) to 6D continuous representation."""
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    rotmat = R.from_quat(quat_xyzw).as_matrix()
    return rotmat[..., :2, :].reshape(*quat_wxyz.shape[:-1], 6).astype(np.float32)


def _npz_to_38d(data: dict) -> np.ndarray:
    """Convert npz data to 38D action representation.
    29 joint_pos + 2 root_vel_xy + 1 root_z + 6 root_rot_6d = 38
    """
    joint_pos = data["joint_pos"].astype(np.float32)
    body_pos_w = data["body_pos_w"].astype(np.float32)
    body_quat_w = data["body_quat_w"].astype(np.float32)

    T = joint_pos.shape[0]
    root_pos = body_pos_w[:, 0, :]
    root_quat = body_quat_w[:, 0, :]

    root_pos[:, :2] -= root_pos[0, :2]

    root_vel_xy = np.zeros((T, 2), dtype=np.float32)
    root_vel_xy[1:] = root_pos[1:, :2] - root_pos[:-1, :2]

    root_z = root_pos[:, 2:3]
    root_rot_6d = _quat_wxyz_to_6d(root_quat)

    return np.concatenate([joint_pos, root_vel_xy, root_z, root_rot_6d], axis=1)


class WorldModelDataset(Dataset):
    """
    Dataset for World Model training.

    Each sample provides:
        text: str — text description
        z_past: [T_past, 64] — past latents (variable length, 1~max_obs_frames)
        z_fut: [H, 64] — future latents (normalized)
        a_fut: [H, 38] — future actions (normalized, for P_t target)
    """

    def __init__(
        self,
        data_root_dir: str,
        split: str = "train",
        H: int = 16,
        max_obs_frames: int = 150,
        debug_max_samples: int | None = None,
    ):
        self.data_root = Path(data_root_dir)
        self.H = H
        self.max_obs_frames = max_obs_frames
        self.min_T = H + 1  # need at least 1 past frame + H future

        split_file = self.data_root / f"{split}.txt"
        if not split_file.exists():
            split_file = self.data_root / "test.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            sample_ids = [line.strip() for line in f if line.strip()]

        self.latent_mean = np.load(self.data_root / "Mean_64d.npy").astype(np.float32)
        self.latent_std = np.load(self.data_root / "Std_64d.npy").astype(np.float32)
        self.action_mean = np.load(self.data_root / "Mean_38d.npy").astype(np.float32)
        self.action_std = np.load(self.data_root / "Std_38d.npy").astype(np.float32)

        self.records: list[dict] = []
        self.filtered_short = 0
        self.filtered_missing = 0

        for sid in sample_ids:
            latent_path = self.data_root / "latent" / f"{sid}.npz"
            npz_path = self.data_root / "npz" / f"{sid}.npz"
            text_path = self.data_root / "texts" / f"{sid}.txt"

            if not latent_path.exists() or not npz_path.exists() or not text_path.exists():
                self.filtered_missing += 1
                continue

            try:
                latent_data = np.load(latent_path, allow_pickle=True)
                T = latent_data["latent"].shape[0]
            except Exception:
                self.filtered_missing += 1
                continue

            if T < self.min_T:
                self.filtered_short += 1
                continue

            self.records.append(
                {
                    "sample_id": sid,
                    "latent_path": str(latent_path),
                    "npz_path": str(npz_path),
                    "text_path": str(text_path),
                    "T": T,
                }
            )

        if debug_max_samples:
            self.records = self.records[:debug_max_samples]

        if not self.records:
            raise RuntimeError(
                f"No valid samples after filtering. "
                f"Short (T<{self.min_T}): {self.filtered_short}, "
                f"Missing files: {self.filtered_missing}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]

        latent_data = np.load(record["latent_path"], allow_pickle=True)
        latent = latent_data["latent"].astype(np.float32)

        npz_data = np.load(record["npz_path"], allow_pickle=True)
        action_38d = _npz_to_38d(npz_data)

        latent_norm = (latent - self.latent_mean) / (self.latent_std + 1e-6)
        action_norm = (action_38d - self.action_mean) / (self.action_std + 1e-6)

        T = record["T"]

        # Sample random timestep t: 1 <= t <= T - H
        max_t = T - self.H
        t = random.randint(1, max_t)

        # z_past: all frames before t, truncated to max_obs_frames (like MotionAR)
        z_past_full = torch.from_numpy(latent_norm[:t])  # [t, 64]
        if z_past_full.shape[0] > self.max_obs_frames:
            z_past_full = z_past_full[-self.max_obs_frames:]  # keep most recent

        z_fut = torch.from_numpy(latent_norm[t : t + self.H])  # [H, 64]
        a_fut = torch.from_numpy(action_norm[t : t + self.H])  # [H, 38]

        text = self._load_text(record["text_path"])

        return {
            "text": text,
            "z_past": z_past_full,
            "z_fut": z_fut,
            "a_fut": a_fut,
        }

    def _load_text(self, text_path: str) -> str:
        with open(text_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        captions = [line.split("#")[0] for line in lines if "#" in line]
        if not captions:
            captions = [line for line in lines]
        return random.choice(captions)


def build_world_model_dataloader(
    data_root_dir: str,
    split: str = "train",
    batch_size: int = 64,
    H: int = 16,
    max_obs_frames: int = 150,
    num_workers: int = 4,
    debug_max_samples: int | None = None,
) -> DataLoader:
    """Build a DataLoader for World Model training."""
    dataset = WorldModelDataset(
        data_root_dir=data_root_dir,
        split=split,
        H=H,
        max_obs_frames=max_obs_frames,
        debug_max_samples=debug_max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
