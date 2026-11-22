import os
import numpy as np
import tifffile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class AgriPotentialDataset(Dataset):
    """PyTorch Dataset for 12-channel Sentinel-2 AgriPotential imagery (10 bands + NDVI + NDWI)."""

    def __init__(self, data_root: str, split_file: str, augment: bool = False):
        self.data_root = data_root
        self.augment = augment
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split index file not found: {split_file}")

        self.samples = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                basename, source_split = line.split(",")
                self.samples.append((basename, source_split))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        basename, source_split = self.samples[idx]
        img_path = os.path.join(self.data_root, source_split, "images", f"{basename}.tif")
        mask_path = os.path.join(self.data_root, source_split, "masks", f"{basename}.png")

        img = tifffile.imread(img_path).astype(np.float32)  # (H, W, 10)
        if img.ndim == 2:
            img = img[..., np.newaxis]

        # Derive NDVI and NDWI from Green (idx 1), Red (idx 2), and NIR (idx 6)
        green = img[:, :, 1]
        red = img[:, :, 2]
        nir = img[:, :, 6]
        eps = 1e-8
        ndvi = (nir - red) / (nir + red + eps)
        ndwi = (green - nir) / (green + nir + eps)

        img_12ch = np.concatenate([img, ndvi[..., None], ndwi[..., None]], axis=-1)
        img_12ch = np.nan_to_num(img_12ch, nan=0.0, posinf=0.0, neginf=0.0)

        mask = np.array(Image.open(mask_path), dtype=np.int64)

        if self.augment:
            if np.random.rand() > 0.5:
                img_12ch = np.flip(img_12ch, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                img_12ch = np.flip(img_12ch, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

        image_tensor = torch.from_numpy(img_12ch.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor


def create_dataloaders(data_root: str, split_dir: str, batch_size: int = 32, num_workers: int = 4):
    """Create train and validation DataLoaders from split index files."""
    train_ds = AgriPotentialDataset(data_root, os.path.join(split_dir, "train.txt"), augment=True)
    val_ds = AgriPotentialDataset(data_root, os.path.join(split_dir, "val.txt"), augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
