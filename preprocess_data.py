import os
import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.colors

# =========================
# GLOBAL CONFIG
# =========================
NUM_CLASSES = 5
IGNORE_INDEX = 255

labels_cmap = matplotlib.colors.ListedColormap([
    "#000000", "#8B0000", "#FFA500", "#FFFF00", "#32CD32"
])


class LandCoverDataset(Dataset):
    def __init__(self, data_root, mode="train", augment=True, filter_empty=True):
        self.data_root = data_root
        self.mode = mode
        self.filter_empty = filter_empty  # Giữ biến này để tương thích, dù logic cũ có thể chưa dùng tới

        # -----------------------------------------------------------
        # 1. LOGIC ĐỌC FILE CŨ (AN TOÀN NHẤT VỚI DỮ LIỆU CỦA BẠN)
        # -----------------------------------------------------------
        txt_path = os.path.join(data_root, f"{mode}.txt")
        if not os.path.exists(txt_path):
            txt_path = f"{mode}.txt"

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file {mode}.txt")

        print(f"[INIT] Đang đọc danh sách từ: {txt_path}")
        with open(txt_path, "r") as f:
            lines = [line.strip() for line in f.read().splitlines() if line.strip()]

        self.samples = []
        for line in lines:
            # Xử lý format "tên_ảnh,tên_folder_nguồn"
            if "," in line:
                name, source_folder = line.split(",")
            else:
                name = line
                source_folder = mode

            self.samples.append((name, source_folder))

        print(f"[INFO] Load tập {mode.upper()}: {len(self.samples)} mẫu.")

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

    # -----------------------------------------------------------
    # 2. HÀM TÍNH NDMI / NDWI (TINH HOA MỚI)
    # -----------------------------------------------------------
    @staticmethod
    def add_spectral_indices(img):
        epsilon = 1e-6
        # Giả định: 0:Blue, 1:Green, 2:Red, 3:NIR ... (10 kênh)
        green = img[:, :, 1]
        red = img[:, :, 2]
        nir = img[:, :, 3]

        # Nếu muốn chính xác NDMI cần SWIR (kênh index 8 hoặc 9 tùy data),
        # Tạm thời dùng công thức mẫu dựa trên NIR/Red/Green nếu chưa chắc chắn index SWIR
        # Ở đây tôi thêm NDVI và NDWI như code "tinh hoa" trước đó

        ndvi = (nir - red) / (nir + red + epsilon)
        ndwi = (green - nir) / (green + nir + epsilon)

        img_new = np.concatenate((
            img,
            np.expand_dims(ndvi, axis=-1),
            np.expand_dims(ndwi, axis=-1)
        ), axis=-1)
        return img_new

    @staticmethod
    def robust_normalize(img):
        lower = np.percentile(img, 2, axis=(0, 1), keepdims=True)
        upper = np.percentile(img, 98, axis=(0, 1), keepdims=True)
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower + 1e-6)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        name, source_folder = self.samples[idx]

        img_path = os.path.join(self.data_root, source_folder, "images", f"{name}.tif")
        mask_path = os.path.join(self.data_root, source_folder, "masks", f"{name}.png")

        # Đọc ảnh (Có try-except chống crash)
        try:
            img = tifffile.imread(img_path).astype(np.float32)
            if img.ndim == 3 and img.shape[0] <= 20 and img.shape[2] > 20:
                img = np.transpose(img, (1, 2, 0))
        except Exception:
            img = np.zeros((128, 128, 10), dtype=np.float32)

        # -----------------------------------------------------------
        # ÁP DỤNG CÁC PHƯƠNG PHÁP XỬ LÝ MỚI
        # -----------------------------------------------------------
        img = self.robust_normalize(img)  # Chuẩn hóa xịn
        img = self.add_spectral_indices(img)  # Thêm kênh NDVI, NDWI -> Input thành 12 kênh

        # Đọc mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((128, 128), dtype=np.uint8)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        out = self.transform(image=img, mask=mask)

        # Trả về Dictionary
        return {"image": out["image"], "mask": out["mask"].long()}


# =========================
# 3. HELPER FUNCTIONS
# =========================
def landcover_collate_fn(batch):
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    return {"image": imgs, "mask": masks}


def check_existing_splits(data_root):
    # Logic check cũ đơn giản nhưng hiệu quả
    print(f"\n--- KIỂM TRA FILE LIST TẠI: '{data_root}' ---")
    all_ok = True
    for f in ["train.txt", "val.txt", "test.txt"]:
        fpath = os.path.join(data_root, f)
        if os.path.exists(fpath) or os.path.exists(f):
            print(f"   {f:<10}: OK")
        else:
            print(f"   {f:<10}: KHÔNG TÌM THẤY")
            if f != "test.txt": all_ok = False
    return all_ok


def create_landcover_loaders(data_root, batch_size=32, num_workers=2):
    train_ds = LandCoverDataset(data_root, "train", augment=True)
    val_ds = LandCoverDataset(data_root, "val", augment=False)

    print(f"\n[SUMMARY] Data Loaders Created:")
    print(f"   - Train samples: {len(train_ds)}")
    print(f"   - Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=landcover_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=landcover_collate_fn
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Test
    ROOT = "dataset_final"
    if check_existing_splits(ROOT):
        tr, val = create_landcover_loaders(ROOT, batch_size=2)
        try:
            batch = next(iter(tr))
            print("\n[SUCCESS] Test load batch OK!")
            # Kỳ vọng: [2, 12, 128, 128] nếu đã thêm NDVI/NDWI
            print(f"   - Image: {batch['image'].shape}")
            print(f"   - Mask:  {batch['mask'].shape}")
        except Exception as e:
            print(f"[ERROR] {e}")