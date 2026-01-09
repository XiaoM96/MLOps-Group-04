from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import typer
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class EkgDataset(Dataset):
    """Dataset for folder-of-classes containing .npy files.

    Expected layout:
      data_path/
        class_a/*.pt
        class_b/*.pt
        ...
    """

    def __init__(
        self,
        data_path: Path,
        target_hw: Tuple[int, int] = (224, 224),
        num_samples: int | None = None,
        use_min_class_size: bool = False,
        random_seed: int = 42,
        ignore_classes: List[str] | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.target_hw = target_hw
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.ignore_classes = ignore_classes or []

        # Raise error if no data found
        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} does not exist")

        # Find class subfolders and names
        all_classes = sorted([p.name for p in self.data_path.iterdir() if p.is_dir()])
        self.classes = [cls for cls in all_classes if cls not in self.ignore_classes]
        
        if not self.classes:
            raise ValueError(f"No class subfolders found in {self.data_path}")

        # Map class names to integer labels
        class_to_idx = {name: i for i, name in enumerate(self.classes)}

        # Get all files per class
        files_per_class: dict[str, list[Path]] = {}
        for cls in self.classes:
            files = sorted((self.data_path / cls).glob("*.pt"))
            files_per_class[cls] = files

        # Determine target sample count
        if use_min_class_size:
            num_samples = min(len(files) for files in files_per_class.values())
        elif num_samples is None:
            num_samples = max(len(files) for files in files_per_class.values())

        # Build samples with random selection
        samples: List[Sample] = []
        for cls in self.classes:
            files = files_per_class[cls]
            # Randomly select num_samples files from this class
            selected_files = self.rng.choice(
                files, size=min(num_samples, len(files)), replace=False
            )
            for f in selected_files:
                samples.append(Sample(path=Path(f), label=class_to_idx[cls]))

        if not samples:
            raise ValueError(f"No .npy files found under {self.data_path}")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]
        item_torch = torch.load(s.path)  # already a torch tensor (C, H, W, float32)
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
    

class EkgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = './data/processed/',
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        test_split: float = 0.2,
        use_weighted_sampler: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.use_weighted_sampler = use_weighted_sampler
        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None):
        if self.dataset is None:
            self.dataset = EkgDataset(Path(self.data_dir))
            total_size = len(self.dataset)
            train_size = int((1 - self.val_split - self.test_split) * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size

            self.train_ds, self.val_ds, self.test_ds = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        ) 
    
def preprocess(
    data_path: str,
    output_folder: str,
    num_samples: int | None = None,
    use_min_class_size: bool = False,
    random_seed: int = 42,
    ignore_classes: List[str] | None = None,
) -> None:
    """Preprocess raw .npy files to standardized tensors.

    Args:
        data_path: Path to raw data directory with class subfolders.
        output_folder: Path to save processed tensors.
        num_samples: Target number of samples per class. If None, uses max class size.
        use_min_class_size: If True, uses the size of the smallest class.
        random_seed: Seed for reproducible random selection.
        ignore_classes: List of class folder names to exclude from processing.
    """
    print("Preprocessing data...")
    dataset = EkgDataset(
        Path(data_path),
        num_samples=num_samples,
        use_min_class_size=use_min_class_size,
        random_seed=random_seed,
        ignore_classes=ignore_classes,
    )
    dataset.preprocess(Path(output_folder))


if __name__ == "__main__":
    typer.run(preprocess)
