import os
import glob
import cv2
import shutil
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path

# CONFIG
TARGET_SIZE = 128
NUM_CLASSES = 5  # 0–4

# PIXEL-PERFECT PREPROCESS
def pixel_perfect_preprocess(img_dir, mask_dir, output_dir, target_size=TARGET_SIZE):
    """
    Pixel-perfect preprocessing for semantic segmentation.
    - Strict size control
    - Exact class mapping
    - Mask integrity preserved
    - Deterministic train/val/test split
    """

    if os.path.exists(output_dir):
        print(f"[INFO] Recreating directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    img_exts = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    mask_exts = ["*.png", "*.tif", "*.tiff", "*.jpg"]

    img_paths, mask_paths = [], []

    for ext in img_exts:
        img_paths += glob.glob(os.path.join(img_dir, ext))
        img_paths += glob.glob(os.path.join(img_dir, ext.upper()))

    for ext in mask_exts:
        mask_paths += glob.glob(os.path.join(mask_dir, ext))
        mask_paths += glob.glob(os.path.join(mask_dir, ext.upper()))

    img_paths = sorted(set(img_paths))
    mask_paths = sorted(set(mask_paths))

    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    mask_map = {Path(p).stem: p for p in mask_paths}
    valid_names = []
    ignored = 0

    print(f"[INFO] Pixel-perfect validation started: {len(img_paths)} samples")

    for img_path in tqdm(img_paths, desc="Validating"):
        stem = Path(img_path).stem
        if stem not in mask_map:
            ignored += 1
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_map[stem], cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            ignored += 1
            continue

        # Resize strictly (mask always NEAREST)
        if img.shape[:2] != (target_size, target_size):
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != (target_size, target_size):
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        # Class mapping (pixel-safe)
        mask = mask.astype(np.uint8)
        mask[mask == 5] = 4
        mask[mask >= NUM_CLASSES] = 0

        # Remove empty masks
        if np.all(mask == 0):
            ignored += 1
            continue

        base = f"{stem}_valid"

        # Save lossless
        cv2.imwrite(os.path.join(output_dir, f"{base}.png"), img)
        cv2.imwrite(os.path.join(output_dir, f"{base}_m.png"), mask)

        valid_names.append(base)

    print(f"[INFO] Done | Valid: {len(valid_names)} | Ignored: {ignored}")


    # SPLIT (80 / 10 / 10)
    np.random.seed(42)
    np.random.shuffle(valid_names)

    n = len(valid_names)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    splits = {
        "train.txt": valid_names[:train_end],
        "val.txt": valid_names[train_end:val_end],
        "test.txt": valid_names[val_end:]
    }

    for name, data in splits.items():
        with open(os.path.join(output_dir, name), "w") as f:
            f.write("\n".join(data))

    return True

# DATASET
class LandCoverDataset(Dataset):
    def __init__(self, data_root, mode="train", img_size=(128, 128), augment=True):
        self.data_root = data_root
        self.mode = mode
        self.img_size = img_size
        self.augment = augment

        txt = os.path.join(data_root, f"{mode}.txt")
        if not os.path.exists(txt):
            raise FileNotFoundError(f"Missing split file: {txt}")

        with open(txt) as f:
            self.names = f.read().splitlines()

        if augment and mode == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    mask_value=0,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,  # 🔒 CRITICAL
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        base = self.names[idx]

        img = cv2.imread(os.path.join(self.data_root, f"{base}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            os.path.join(self.data_root, f"{base}_m.png"),
            cv2.IMREAD_GRAYSCALE
        )

        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()
