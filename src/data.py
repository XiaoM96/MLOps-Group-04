
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from typing import Optional, List

class CACHETDataset(Dataset):
    """
    Dataset for CACHET-CADB ECG data.
    Assumes data is stored as pre-computed 2D spectrograms (.npy files) in class-specific subdirectories.
    Structure: root_dir/{class_name}/*.npy
    """
    CLASSES = ['AF', 'Noise', 'NSR', 'Other']
    CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir (str): Path to the directory containing class subdirectories.
        """
        self.data_dir = data_dir
        self.file_paths = []
        self.labels = []

        # Glob all files
        for class_name in self.CLASSES:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory {class_dir} not found.")
                continue
            
            # Find all .npy files
            files = glob.glob(os.path.join(class_dir, "*.npy"))
            self.file_paths.extend(files)
            self.labels.extend([self.CLASS_TO_IDX[class_name]] * len(files))

        print(f"Found {len(self.file_paths)} files in {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # Load npy file (H, W) -> (224, 224)
        try:
            # We use mmap_mode='r' for potentially faster access, or just standard load
            spectrogram = np.load(path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy tensor in case of corruption, or raise error. 
            # For now, let's just crash so we know something is wrong.
            raise e

        # Convert to tensor
        # Input shape expected: (224, 224)
        # Output shape needed for ResNet: (3, 224, 224)
        
        tensor = torch.from_numpy(spectrogram)
        
        # Add channel dimension: (1, 224, 224)
        tensor = tensor.unsqueeze(0)
        
        # Replicate to 3 channels: (3, 224, 224)
        tensor = tensor.repeat(3, 1, 1)

        # Basic Normalization (Ad-hoc based on observed range [-5, 0])
        # ResNet expects roughly standard normal inputs. 
        # Mean/Std for ImageNet is roughly specific, but simple min-max or shift might suffice.
        # Let's simple normalize approx to [0, 1] then standardize?
        # Actually, for transfer learning on spectrograms, standardizing instance-wise is often good.
        # Let's try simple instance normalization: (x - mean) / std
        mean = tensor.mean()
        std = tensor.std() 
        if std > 0:
            tensor = (tensor - mean) / std
        
        return tensor, label


class ECGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data/time_series', batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None):
        if self.dataset is None:
            self.dataset = CACHETDataset(self.data_dir)
            total_size = len(self.dataset)
            train_size = int(0.8 * total_size)
            val_size = int(0.1 * total_size)
            test_size = total_size - train_size - val_size
            
            self.train_ds, self.val_ds, self.test_ds = random_split(
                self.dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=True)
