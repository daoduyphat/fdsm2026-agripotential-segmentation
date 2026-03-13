import os
import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================
# GLOBAL CONFIG
# =========================
NUM_CLASSES = 5
IGNORE_INDEX = 255

class LandCoverDataset(Dataset):
    def __init__(self, data_root, mode="train", augment=True):
        self.data_root = data_root
        self.mode = mode

        # Đọc file list (.txt)
        txt_path = os.path.join(data_root, f"{mode}.txt")
        if not os.path.exists(txt_path):
            txt_path = f"{mode}.txt" # Fallback nếu file nằm ở root

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file {mode}.txt")

        with open(txt_path, "r") as f:
            lines = [line.strip() for line in f.read().splitlines() if line.strip()]

        self.samples = []
        for line in lines:
            if "," in line:
                name, source_folder = line.split(",")
            else:
                name = line
                source_folder = mode
            self.samples.append((name, source_folder))

        # Augmentation
        if augment and mode == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def add_spectral_indices(img):
        """Tính toán thêm NDVI và NDWI để mở rộng lên 12 kênh"""
        epsilon = 1e-6
        # Giả định index kênh: 1:Green, 2:Red, 3:NIR
        green = img[:, :, 1]
        red = img[:, :, 2]
        nir = img[:, :, 3]

        ndvi = (nir - red) / (nir + red + epsilon)
        ndwi = (green - nir) / (green + nir + epsilon)

        # Gộp vào ảnh gốc (10 kênh + 1 NDVI + 1 NDWI = 12 kênh)
        img_new = np.concatenate((
            img,
            np.expand_dims(ndvi, axis=-1),
            np.expand_dims(ndwi, axis=-1)
        ), axis=-1)
        return img_new

    @staticmethod
    def robust_normalize(img):
        """Chuẩn hóa ảnh vệ tinh dùng Percentile để loại bỏ nhiễu sáng"""
        lower = np.percentile(img, 2, axis=(0, 1), keepdims=True)
        upper = np.percentile(img, 98, axis=(0, 1), keepdims=True)
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower + 1e-6)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        name, source_folder = self.samples[idx]

        img_path = os.path.join(self.data_root, source_folder, "images", f"{name}.tif")
        mask_path = os.path.join(self.data_root, source_folder, "masks", f"{name}.png")

        # Đọc ảnh TIF
        try:
            img = tifffile.imread(img_path).astype(np.float32)
            # Chuyển về format (H, W, C) nếu đang là (C, H, W)
            if img.ndim == 3 and img.shape[0] < img.shape[2]:
                img = np.transpose(img, (1, 2, 0))
        except Exception:
            img = np.zeros((128, 128, 10), dtype=np.float32)

        # Xử lý kênh
        img = self.robust_normalize(img)
        img = self.add_spectral_indices(img) # Tăng lên 12 kênh

        # Đọc Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((128, 128), dtype=np.uint8)
        
        # Đảm bảo label nằm trong range (0 -> 4)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        # Augment
        augmented = self.transform(image=img, mask=mask)
        
        # QUAN TRỌNG: Trả về tuple (image, mask) để khớp với train.py
        return augmented["image"], augmented["mask"].long()

def check_existing_splits(data_root):
    """Hàm kiểm tra dữ liệu trước khi train"""
    print(f"\n--- CHECKING DATASET AT: '{data_root}' ---")
    status = True
    for f in ["train.txt", "val.txt"]:
        fpath = os.path.join(data_root, f)
        if os.path.exists(fpath):
            print(f"  {f:<10}: FOUND")
        else:
            print(f"  {f:<10}: NOT FOUND")
            status = False
    return status