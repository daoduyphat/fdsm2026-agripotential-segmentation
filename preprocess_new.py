import os
import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob

IMG_SIZE = 128
NUM_CLASSES = 5
INPUT_CHANNELS = 12


class LandCoverDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.images = sorted(glob(os.path.join(root_dir, split, "images", "*.tif")))
        self.masks = sorted(glob(os.path.join(root_dir, split, "masks", "*.tif")))

        if not self.masks:
            self.masks = sorted(glob(os.path.join(root_dir, split, "masks", "*.png")))

        if len(self.images) != len(self.masks):
            print(f"[WARN] Mismatch count in {split}: {len(self.images)} images vs {len(self.masks)} masks")

    def __len__(self):
        return len(self.images)

    @staticmethod
    def add_spectral_indices(img):
        epsilon = 1e-6
        if img.shape[2] >= 4:
            green = img[:, :, 1]
            red = img[:, :, 2]
            nir = img[:, :, 3]

            ndvi = (nir - red) / (nir + red + epsilon)
            ndwi = (green - nir) / (green + nir + epsilon)

            img = np.concatenate((
                img,
                np.expand_dims(ndvi, axis=-1),
                np.expand_dims(ndwi, axis=-1)
            ), axis=-1)
        return img

    @staticmethod
    def robust_normalize(img):
        lower = np.percentile(img, 2, axis=(0, 1), keepdims=True)
        upper = np.percentile(img, 98, axis=(0, 1), keepdims=True)
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower + 1e-6)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        try:
            img = tifffile.imread(img_path).astype(np.float32)

            # Xử lý channel: (C, H, W) -> (H, W, C) nếu cần
            if img.ndim == 3 and img.shape[0] <= 20:
                img = np.transpose(img, (1, 2, 0))

            img = self.robust_normalize(img)

            if img.shape[2] == 10:
                img = self.add_spectral_indices(img)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = tifffile.imread(mask_path)

            mask = np.clip(mask, 0, NUM_CLASSES - 1)

            if self.transform:
                if self.split == "train":
                    # Chỉ lật và xoay, KHÔNG CROP, KHÔNG RESIZE
                    aug = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2()
                    ])
                else:
                    # Val/Test giữ nguyên gốc
                    aug = A.Compose([
                        ToTensorV2()
                    ])

                transformed = aug(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"].long()
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                mask = torch.from_numpy(mask).long()

            return {"image": img, "mask": mask}

        except Exception as e:
            print(f"[ERROR] File: {img_path} | {e}")
            return self.__getitem__((idx + 1) % len(self.images))


def create_landcover_loaders(data_path, batch_size=16, num_workers=2):
    train_ds = LandCoverDataset(data_path, split="train", transform=True)
    val_ds = LandCoverDataset(data_path, split="val", transform=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def check_existing_splits(data_path):
    return os.path.exists(os.path.join(data_path, "train")) and \
        os.path.exists(os.path.join(data_path, "val"))


if __name__ == "__main__":
    ROOT = "dataset_final"
    if check_existing_splits(ROOT):
        tr, val = create_landcover_loaders(ROOT, batch_size=2)
        try:
            batch = next(iter(tr))
            print(f"[SUCCESS] Image shape: {batch['image'].shape}")
            print(f"[SUCCESS] Mask shape:  {batch['mask'].shape}")
        except Exception as e:
            print(f"[ERROR] {e}")
