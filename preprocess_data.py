import os
import cv2
import torch
import numpy as np
import tifffile
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.colors

# =========================
# GLOBAL CONFIG
# =========================
NUM_CLASSES = 5

labels_cmap = matplotlib.colors.ListedColormap([
    "#000000",  # 0 Background
    "#8B0000",  # 1 Very Low
    "#FFA500",  # 2 Low
    "#FFFF00",  # 3 Average
    "#32CD32"  # 4 High
])


# =========================
# 1. DATASET 10-KENH (LOAD TU TXT)
# =========================
class LandCoverDataset(Dataset):
    def __init__(self, data_root, mode="train", augment=True):
        self.data_root = data_root
        self.mode = mode

        # Kiem tra su ton tai cua file split
        txt_path = os.path.join(data_root, f"{mode}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(
                f"Loi: Khong tim thay file split {txt_path}. Hay dam bao ban da co san cac file .txt.")

        with open(txt_path) as f:
            self.names = f.read().splitlines()

        print(f"[INFO] Da load tap {mode.upper()}: {len(self.names)} mau.")

        # Augmentation cho tap train
        if augment and mode == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(0.05, 0.1, 15, p=0.5),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        base = self.names[idx]

        # 1. Doc anh TIF 10 kenh va chuan hoa Min-Max
        img_path = os.path.join(self.data_root, "images", f"{base}.tif")
        img = tifffile.imread(img_path).astype(np.float32)

        min_v, max_v = img.min(), img.max()
        if max_v > min_v:
            img = (img - min_v) / (max_v - min_v)

        # 2. Doc mat na PNG
        mask_path = os.path.join(self.data_root, "masks", f"{base}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Dam bao label nam trong khoang 0-4
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()


# =========================
# 2. KIEM TRA TINH TRANG SPLIT
# =========================
def check_existing_splits(data_root):
    required_files = ["train.txt", "val.txt", "test.txt"]
    all_exist = True
    print(f"\n--- KIEM TRA FILE SPLIT TAI: {data_root} ---")

    for f in required_files:
        path = os.path.join(data_root, f)
        if os.path.exists(path):
            with open(path) as f_in:
                count = len(f_in.read().splitlines())
            print(f"  OK {f}: Ton tai ({count} mau)")
        else:
            print(f"  LOI {f}: THIEU")
            all_exist = False
    return all_exist


# =========================
# MAIN (CHI CHAY TEST)
# =========================
if __name__ == "__main__":
    ROOT = "dataset_final/train"

    if check_existing_splits(ROOT):
        try:
            ds = LandCoverDataset(ROOT, mode="train", augment=False)
            img, mask = ds[0]
            print("\n[CHECK SAMPLE SUCCESS]")
            print(f"  - Image Shape: {img.shape}")
            print(f"  - Mask Shape : {mask.shape}")
            print(f"  - Unique Labels: {torch.unique(mask).tolist()}")
        except Exception as e:
            print(f"\nLoi khi load du lieu: {e}")
    else:
        print("\nVui long bo sung cac file .txt vao thu muc du lieu truoc khi train.")