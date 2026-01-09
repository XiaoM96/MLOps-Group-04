from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class EkgDataset(Dataset):
    """Dataset for folder-of-classes containing .npy files.

    Expected layout:
      data_path/
        class_a/*.npy
        class_b/*.npy
        ...
    """

    def __init__(self, data_path: Path, target_hw: Tuple[int, int] = (224, 224)) -> None:
        self.data_path = Path(data_path)
        self.target_hw = target_hw

        # Raise error if no data found
        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} does not exist")

        # Find class subfolders and names
        self.classes = sorted([p.name for p in self.data_path.iterdir() if p.is_dir()])
        if not self.classes:
            raise ValueError(f"No class subfolders found in {self.data_path}")

        # Map class names to integer labels
        class_to_idx = {name: i for i, name in enumerate(self.classes)}

        samples: List[Sample] = []
        for cls in self.classes:
            for f in sorted((self.data_path / cls).glob("*.npy")):
                samples.append(Sample(path=f, label=class_to_idx[cls]))


        if not samples:
            raise ValueError(f"No .npy files found under {self.data_path}")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        item_np_array = np.load(s.path)  # expects numeric array
        item_torch = self._to_chw_float32(item_np_array, self.target_hw)
        label = s.label
        return item_torch, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess all raw .npy -> float32 CHW resized .npy, saved by class.

        Output layout:
          output_folder/
            class_a/<same_filename>.npy
            class_b/<same_filename>.npy
            ...
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        for cls in self.classes:
            (output_folder / cls).mkdir(parents=True, exist_ok=True)

        for s in self.samples:
            cls_name = self.classes[s.label]

            # same filename, but .pt
            out_path = (output_folder / cls_name / s.path.stem).with_suffix(".pt")

            x = np.load(s.path)
            x_t = self._to_chw_float32(x, self.target_hw)  # torch.Tensor (C,H,W), float32

            torch.save(x_t, out_path)

    @staticmethod
    def _to_chw_float32(x: np.ndarray, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Convert numpy array to torch float32 tensor in CHW and resize if needed."""
        
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(x)}")

        # Handle common shapes:
        # - (H, W) -> (1, H, W)
        # - (H, W, C) -> (C, H, W)
        # - (C, H, W) -> keep
        if x.ndim == 2:
            x = x[None, :, :]
        elif x.ndim == 3:
            # Heuristic: if last dim looks like channels (1/3), assume HWC
            if x.shape[-1] in (1, 3):
                x = np.transpose(x, (2, 0, 1))
            # else assume already CHW
        else:
            raise ValueError(f"Unsupported array shape {x.shape}; expected 2D or 3D")

        x = x.astype(np.float32, copy=False)

        # Replace NaN/Inf with finite values
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        t = torch.from_numpy(x)  # (C, H, W)

        _, h, w = t.shape
        th, tw = target_hw
        if (h, w) != (th, tw):
            # interpolate expects NCHW
            t4 = t.unsqueeze(0)  # (1, C, H, W)
            t4 = F.interpolate(t4, size=(th, tw), mode="bilinear", align_corners=False)
            t = t4.squeeze(0)

        return t
    
def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = EkgDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        data_path = Path(sys.argv[1])
        output_folder = Path(sys.argv[2])
    else:
        data_path = Path("./data/raw/time_series")
        output_folder = Path("./data/processed")
    preprocess(data_path, output_folder)
