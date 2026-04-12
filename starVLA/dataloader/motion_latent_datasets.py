import codecs as cs
import random
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset


def collate_fn(batch):
    return batch


class MotionLatentDataset(Dataset):
    def __init__(
        self,
        data_root_dir,
        split="train",
        chunk_size=15,
        max_obs_frames=150,
        max_motion_length=600,
        fps=20,
        debug_max_samples=None,
    ):
        self.data_root = Path(data_root_dir)
        self.split = split
        self.chunk_size = chunk_size
        self.max_obs_frames = max_obs_frames
        self.max_motion_length = max_motion_length
        self.fps = fps

        split_file = self.data_root / f"{split}.txt"
        if not split_file.exists() and split == "val":
            split_file = self.data_root / "test.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with cs.open(split_file, "r") as f:
            sample_names = [line.strip() for line in f if line.strip()]

        if debug_max_samples:
            sample_names = sample_names[: debug_max_samples]

        self.records = []
        self.filtered_count = 0
        self.filtered_out_of_bounds_count = 0
        for name in sample_names:
            latent_path = self.data_root / "latent" / f"{name}.npz"
            text_path = self.data_root / "texts" / f"{name}.txt"
            if (not latent_path.exists()) or (not text_path.exists()):
                self.filtered_count += 1
                continue

            try:
                latent_file = np.load(latent_path, allow_pickle=True)
                latent_shape = latent_file["latent"].shape
                steps_shape = latent_file["steps"].shape
            except Exception:
                self.filtered_count += 1
                continue

            if len(latent_shape) != 2 or len(steps_shape) != 1:
                self.filtered_count += 1
                continue
            if latent_shape[0] != steps_shape[0]:
                self.filtered_count += 1
                continue
            if latent_shape[1] != 64:
                self.filtered_count += 1
                continue
            latent_length = latent_shape[0]

            text_data = []
            full_sample_added = False
            with cs.open(text_path, "r") as f:
                for line in f.readlines():
                    line_split = line.strip().split("#")
                    if len(line_split) < 4:
                        continue

                    caption = line_split[0]
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    if f_tag == 0.0 and to_tag == 0.0:
                        text_data.append(caption)
                        full_sample_added = True
                        continue

                    start = int(f_tag * self.fps)
                    end = int(to_tag * self.fps)
                    if start >= end:
                        continue
                    if start >= latent_length or end > latent_length:
                        self.filtered_out_of_bounds_count += 1
                        continue

                    segment_length = end - start
                    if not (self.chunk_size <= segment_length <= self.max_motion_length):
                        continue

                    self.records.append(
                        {
                            "sample_id": f"{name}_{f_tag:.6f}_{to_tag:.6f}",
                            "latent_path": latent_path,
                            "start": start,
                            "end": end,
                            "captions": [caption],
                        }
                    )

            if full_sample_added and self.chunk_size <= latent_shape[0] <= self.max_motion_length:
                self.records.append(
                    {
                        "sample_id": name,
                        "latent_path": latent_path,
                        "start": 0,
                        "end": latent_shape[0],
                        "captions": text_data,
                    }
                )

        if not self.records:
            raise RuntimeError(f"No valid motion latent samples loaded from {self.data_root}")

    def __len__(self):
        return len(self.records)

    def _build_item(self, record):
        latent_file = np.load(record["latent_path"], allow_pickle=True)
        latent = latent_file["latent"].astype(np.float32)
        latent = latent[record["start"] : record["end"]]
        if latent.shape[0] < self.chunk_size:
            return None

        max_split = latent.shape[0] - self.chunk_size
        split_point = random.randint(0, max_split) if max_split > 0 else 0
        obs_start = max(0, split_point - self.max_obs_frames)

        obs_latent = latent[obs_start:split_point]
        target = latent[split_point : split_point + self.chunk_size]
        caption = random.choice(record["captions"])

        return {
            "lang": caption,
            "image": [],
            "obs_latent": obs_latent,
            "action": target,
            "meta": {
                "sample_id": record["sample_id"],
                "split_point": split_point,
                "obs_start": obs_start,
            },
        }

    def __getitem__(self, index):
        num_records = len(self.records)
        for offset in range(num_records):
            record = self.records[(index + offset) % num_records]
            item = self._build_item(record)
            if item is not None:
                return item

        raise RuntimeError(
            f"No valid samples available for chunk_size={self.chunk_size}. "
            f"Filtered records={self.filtered_count}, out_of_bounds={self.filtered_out_of_bounds_count}."
        )


def get_motion_latent_dataset(data_cfg, mode="train"):
    split = data_cfg.get("train_split", "train") if mode == "train" else data_cfg.get("eval_split", "test")
    return MotionLatentDataset(
        data_root_dir=data_cfg.data_root_dir,
        split=split,
        chunk_size=data_cfg.get("chunk_size", 15),
        max_obs_frames=data_cfg.get("max_obs_frames", 150),
        max_motion_length=data_cfg.get("max_motion_length", 600),
        fps=data_cfg.get("fps", 20),
        debug_max_samples=data_cfg.get("debug_max_samples", None),
    )


def build_motion_latent_dataloader(cfg):
    data_cfg = cfg.datasets.vla_data
    dataset = get_motion_latent_dataset(data_cfg=data_cfg, mode="train")
    return DataLoader(
        dataset,
        batch_size=data_cfg.per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=data_cfg.get("num_workers", 4),
        shuffle=True,
    )
