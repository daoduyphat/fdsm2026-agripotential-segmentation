import os
import glob
import cv2
import numpy as np
import tifffile  # Thư viện đọc ảnh TIF 10 kênh
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

# --- THƯ VIỆN VẼ BIỂU ĐỒ ---
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

# CONFIG: 5 CLASSES THEO THỰC TẾ DỮ LIỆU
TARGET_SIZE = 128
NUM_CLASSES = 5  # 0: Background, 1: Very Low, 2: Low, 3: Average, 4: High

# Bảng màu Nông nghiệp cho 5 Class (Đã cắt bỏ lớp thứ 6)
labels_cmap = matplotlib.colors.ListedColormap([
    "#000000",  # 0: Background (Đen)
    "#8B0000",  # 1: Very Low (Đỏ sậm)
    "#FFA500",  # 2: Low (Cam)
    "#FFFF00",  # 3: Average (Vàng)
    "#32CD32"   # 4: High / Very High (Xanh lá)
])


# 1. BỘ CHIA DỮ LIỆU
def generate_splits(data_root, ratio=(0.8, 0.1, 0.1), seed=42):
    """
    Quét thư mục, chia ngẫu nhiên thành train.txt, val.txt, test.txt
    """
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    assert len(img_paths) == len(mask_paths), "Lệch số lượng Ảnh TIF và Mask PNG!"

    valid_names = [Path(p).stem for p in img_paths]

    np.random.seed(seed)
    np.random.shuffle(valid_names)

    n = len(valid_names)
    train_end = int(ratio[0] * n)
    val_end = int((ratio[0] + ratio[1]) * n)

    splits = {
        "train.txt": valid_names[:train_end],
        "val.txt": valid_names[train_end:val_end],
        "test.txt": valid_names[val_end:]
    }

    print(f"\n[INFO] Đang chia {n} ảnh...")
    for name, data in splits.items():
        with open(os.path.join(data_root, name), "w") as f:
            f.write("\n".join(data))
        print(f"  -> Đã tạo {name}: {len(data)} mẫu")


 #2. DATASET ĐA PHỔ 10-KÊNH
class LandCoverDataset(Dataset):
    def __init__(self, data_root, mode="train", augment=True):
        self.data_root = data_root
        self.mode = mode
        self.augment = augment

        txt = os.path.join(data_root, f"{mode}.txt")
        if not os.path.exists(txt):
            raise FileNotFoundError(f"Chưa có file chia dữ liệu: {txt}")

        with open(txt) as f:
            self.names = f.read().splitlines()

        # Augmentation hỗ trợ N-Dimensional (10 Kênh)
        if augment and mode == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                ToTensorV2(),  # Chuyển [H, W, 10] thành [10, H, W]
            ])
        else:
            self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        base = self.names[idx]

        # 1. ĐỌC VÀ CHUẨN HÓA ẢNH TIF 10 KÊNH (Float32)
        img_path = os.path.join(self.data_root, "images", f"{base}.tif")
        img = tifffile.imread(img_path).astype(np.float32)

        # Min-Max Scaling [0, 1] cho 10 kênh
        min_val, max_val = img.min(), img.max()
        if max_val - min_val > 1e-5:
            img = (img - min_val) / (max_val - min_val)

        # 2. ĐỌC NHÃN PNG
        mask_path = os.path.join(self.data_root, "masks", f"{base}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 3. AUGMENT VÀ TRẢ VỀ TENSOR
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()



# 3. CÔNG CỤ TRỰC QUAN HÓA
def visualize_dataset(data_root, num_samples=8, mode="train"):
    txt = os.path.join(data_root, f"{mode}.txt")
    with open(txt) as f:
        names = f.read().splitlines()

    indices = np.random.choice(len(names), num_samples, replace=False)

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(12, 12), nrows=num_samples // 2, ncols=4)

    for i, idx in enumerate(indices):
        base = names[idx]
        r, rem = divmod(i, 2)

        # Lấy 3 kênh RGB (0, 1, 2) từ file TIF để hiển thị
        img_10ch = tifffile.imread(os.path.join(data_root, "images", f"{base}.tif"))
        rgb = img_10ch[:, :, 0:3]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        mask = cv2.imread(os.path.join(data_root, "masks", f"{base}.png"), cv2.IMREAD_GRAYSCALE)

        # Vẽ Ảnh RGB
        ax[r, 2 * rem].imshow(rgb)
        ax[r, 2 * rem].set_title(f"RGB: {base[-12:]}")
        ax[r, 2 * rem].axis('off')

        # Vẽ Mask 5 Lớp
        ax[r, 2 * rem + 1].imshow(mask, cmap=labels_cmap, interpolation='nearest', vmin=-0.5, vmax=4.5)
        ax[r, 2 * rem + 1].set_title("Mask (Level 0-4)")
        ax[r, 2 * rem + 1].axis('off')

    plt.suptitle(f"AgriPotential Dataset (5 Classes) - {mode.upper()}", fontsize=20, weight='bold')
    plt.tight_layout(pad=1.0)
    plt.savefig("visualize_dataset.png", dpi=300)
    print("\n[INFO] Đã lưu ảnh báo cáo: visualize_dataset.png")


# MAIN
if __name__ == "__main__":
    ROOT = "dataset_final/train"

    # 1. Tạo file txt chia dataset
    generate_splits(ROOT)

    # 2. Test DataLoader
    dataset = LandCoverDataset(ROOT, mode="train")
    img, mask = dataset[0]
    print(f"\n--- KIỂM TRA ĐẦU VÀO CHO AI ---")
    print(f"Ảnh TIF 10 kênh: Shape {img.shape}, Type: {img.dtype}")
    print(f"Nhãn Mask 1 kênh: Shape {mask.shape}, Type: {mask.dtype}, Các Level có: {torch.unique(mask).tolist()}")

    # 3. Vẽ ảnh báo cáo
    visualize_dataset(ROOT, num_samples=8)