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
        """
        data_root: Duong dan den thu muc chua cac file .txt (vd: 'dataset_final')
        mode: 'train', 'val', hoặc 'test'
        """
        self.data_root = data_root
        self.mode = mode

        # Su dung os.path.join de dam bao tinh tuong thich tren moi OS
        txt_path = os.path.join(data_root, f"{mode}.txt")

        # Duong dan den thu muc con chua anh/mask (vd: dataset_final/train)
        self.specific_dir = os.path.join(data_root, mode)

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Loi: Khong tim thay file chi muc {txt_path}")

        with open(txt_path, "r") as f:
            self.names = f.read().splitlines()

        print(f"[INFO] Da load tap {mode.upper()}: {len(self.names)} mau tu {txt_path}")

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
        return len(self.names)

    def __getitem__(self, idx):
        base = self.names[idx]

        # Duong dan tuong doi den tung file anh va mask
        img_path = os.path.join(self.specific_dir, "images", f"{base}.tif")
        mask_path = os.path.join(self.specific_dir, "masks", f"{base}.png")

        # Doc va chuan hoa anh TIF 10 kenh
        img = tifffile.imread(img_path).astype(np.float32)
        min_v, max_v = img.min(), img.max()
        if max_v > min_v:
            img = (img - min_v) / (max_v - min_v)

        # Doc mask PNG va ep vao khoang label 0-4
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()


# =========================
# 2. KIEM TRA TINH TRANG SPLIT
# =========================
def check_existing_splits(data_root):
    required_files = ["train.txt", "val.txt", "test.txt"]
    print(f"\n--- KIEM TRA CAU TRUC DU LIEU TAI: {data_root} ---")

    all_exist = True
    for f in required_files:
        txt_path = os.path.join(data_root, f)
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f_in:
                count = len(f_in.read().splitlines())
            print(f"  OK {f}: Ton tai ({count} mau)")

            # Kiem tra thu muc du lieu vat ly (train, val, test)
            sub_dir = os.path.join(data_root, f.replace(".txt", ""))
            if not os.path.exists(sub_dir):
                print(f"  LOI: Thieu thu muc du lieu thuc te '{sub_dir}'")
                all_exist = False
        else:
            print(f"  LOI {f}: Khong tim thay file .txt")
            all_exist = False
    return all_exist


# =========================
# MAIN (TEST LOGIC)
# =========================
if __name__ == "__main__":
    # Su dung duong dan tuong doi (gia su ban dang dung o thu muc datdai)
    ROOT = "dataset_final"

    if check_existing_splits(ROOT):
        try:
            ds = LandCoverDataset(ROOT, mode="train", augment=False)
            if len(ds) > 0:
                img, mask = ds[0]
                print("\n[CHECK SAMPLE SUCCESS]")
                print(f"  - Image Shape: {img.shape}")
                print(f"  - Mask Shape : {mask.shape}")
                print(f"  - Unique Labels: {torch.unique(mask).tolist()}")
        except Exception as e:
            print(f"\nLoi trong qua trinh doc mau: {e}")
    else:
        print("\nVui long kiem tra lai vi tri folder 'dataset_final' so voi file script nay.")