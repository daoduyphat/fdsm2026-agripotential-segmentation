import os
import glob
import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

# =========================
# GLOBAL CONFIG
# =========================
TARGET_SIZE = 128
NUM_CLASSES = 5  # 0–4

labels_cmap = matplotlib.colors.ListedColormap([
    "#000000",  # 0 Background
    "#8B0000",  # 1 Very Low
    "#FFA500",  # 2 Low
    "#FFFF00",  # 3 Average
    "#32CD32"   # 4 High
])

# =========================
# 1. SPLIT DATASET (SAFE)
# =========================
def generate_splits(data_root, ratio=(0.8, 0.1, 0.1), seed=42):
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")

    img_names = {Path(p).stem for p in glob.glob(os.path.join(img_dir, "*.tif"))}
    mask_names = {Path(p).stem for p in glob.glob(os.path.join(mask_dir, "*.png"))}

    valid_names = sorted(img_names & mask_names)
    assert len(valid_names) > 0, "❌ Không tìm thấy cặp ảnh–mask hợp lệ"

    np.random.seed(seed)
    np.random.shuffle(valid_names)

    n = len(valid_names)
    t_end = int(ratio[0] * n)
    v_end = int((ratio[0] + ratio[1]) * n)

    splits = {
        "train.txt": valid_names[:t_end],
        "val.txt": valid_names[t_end:v_end],
        "test.txt": valid_names[v_end:]
    }

    print(f"\n[INFO] Chia {n} ảnh:")
    for fname, data in splits.items():
        path = os.path.join(data_root, fname)
        if os.path.exists(path):
            print(f"  ⚠️ {fname} đã tồn tại → bỏ qua")
            continue
        with open(path, "w") as f:
            f.write("\n".join(data))
        print(f"  ✅ Tạo {fname}: {len(data)} mẫu")

# =========================
# 2. DATASET 10-KÊNH
# =========================
class LandCoverDataset(Dataset):
    def __init__(self, data_root, mode="train", augment=True):
        self.data_root = data_root
        self.mode = mode

        txt = os.path.join(data_root, f"{mode}.txt")
        if not os.path.exists(txt):
            raise FileNotFoundError(f"❌ Thiếu file split: {txt}")

        with open(txt) as f:
            self.names = f.read().splitlines()

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

        img = tifffile.imread(
            os.path.join(self.data_root, "images", f"{base}.tif")
        ).astype(np.float32)

        min_v, max_v = img.min(), img.max()
        if max_v > min_v:
            img = (img - min_v) / (max_v - min_v)

        mask = cv2.imread(
            os.path.join(self.data_root, "masks", f"{base}.png"),
            cv2.IMREAD_GRAYSCALE
        )

        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()

# =========================
# 3. VERIFY DATASET
# =========================
def verify_dataset(data_root, mode="train", max_check=20):
    txt = os.path.join(data_root, f"{mode}.txt")
    with open(txt) as f:
        names = f.read().splitlines()

    print(f"[VERIFY] {mode}: {len(names)} samples")

    for name in names[:max_check]:
        assert os.path.exists(os.path.join(data_root, "images", f"{name}.tif"))
        assert os.path.exists(os.path.join(data_root, "masks", f"{name}.png"))

    print(f"[OK] {mode}: dữ liệu hợp lệ")

# =========================
# 4. VISUALIZE
# =========================
def visualize_dataset(data_root, num_samples=8, mode="train"):
    txt = os.path.join(data_root, f"{mode}.txt")
    with open(txt) as f:
        names = f.read().splitlines()

    idxs = np.random.choice(len(names), min(num_samples, len(names)), replace=False)

    sns.set_style("white")
    fig, axes = plt.subplots(len(idxs), 2, figsize=(8, 4 * len(idxs)))

    for i, idx in enumerate(idxs):
        base = names[idx]
        img = tifffile.imread(os.path.join(data_root, "images", f"{base}.tif"))[..., :3]
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        mask = cv2.imread(os.path.join(data_root, "masks", f"{base}.png"), 0)

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("RGB")

        axes[i, 1].imshow(mask, cmap=labels_cmap, vmin=-0.5, vmax=4.5)
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Mask")

    plt.tight_layout()
    plt.savefig("visualize_dataset.png", dpi=300)
    print("[INFO] Đã lưu visualize_dataset.png")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ROOT = "dataset_final/train"

    generate_splits(ROOT)

    verify_dataset(ROOT, "train")
    verify_dataset(ROOT, "val")

    ds = LandCoverDataset(ROOT, "train")
    img, mask = ds[0]

    print("\n[CHECK SAMPLE]")
    print("Image:", img.shape, img.dtype)
    print("Mask:", mask.shape, torch.unique(mask))

    visualize_dataset(ROOT)