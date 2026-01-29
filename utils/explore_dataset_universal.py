import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import glob
import random
import tifffile
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN MỚI ---
# [SỬA QUAN TRỌNG]: Đã thêm \train vào cuối đường dẫn
DATASET_ROOT = r"E:\datdai\dataset_final\train"

IMG_DIR_NAME = "images"
MASK_DIR_NAME = "masks"

# Bảng màu Nông nghiệp cho 6 Class
import matplotlib.colors

labels_cmap = matplotlib.colors.ListedColormap([
    "#000000",  # 0: Background
    "#8B0000",  # 1: Very Low
    "#FFA500",  # 2: Low
    "#FFFF00",  # 3: Average
    "#32CD32",  # 4: High
    "#006400"  # 5: Very High
])


def get_file_list(folder_path, ext):
    return sorted(glob.glob(os.path.join(folder_path, "**", f"*{ext}"), recursive=True))


def analyze_dataset():
    if not os.path.exists(DATASET_ROOT):
        print(f"[LỖI] Không tìm thấy thư mục: {DATASET_ROOT}")
        return

    print(f"--- ĐANG PHÂN TÍCH DATASET (10-KÊNH TIF) TẠI: {DATASET_ROOT} ---\n")

    # 1. Quét file: Ảnh TIF và Nhãn PNG
    all_images = get_file_list(os.path.join(DATASET_ROOT, IMG_DIR_NAME), ".tif")
    all_masks = get_file_list(os.path.join(DATASET_ROOT, MASK_DIR_NAME), ".png")

    print(f"1. SỐ LƯỢNG FILE:")
    print(f"   - Ảnh Vệ tinh (.tif): {len(all_images)}")
    print(f"   - Nhãn (Mask .png):   {len(all_masks)}")

    if len(all_images) == 0:
        print("   -> CẢNH BÁO: Không tìm thấy ảnh .tif nào! Kiểm tra lại đường dẫn.")
        return

    # 2. Kiểm tra Đặc tính Ảnh Vệ tinh (Mẫu 50 ảnh)
    print(f"\n2. KIỂM TRA ĐẶC TÍNH ẢNH GỐC (Đọc bằng tifffile):")
    sample_size = min(50, len(all_images))
    samples = random.sample(all_images, sample_size)

    shapes = []
    dtypes = []

    for img_path in samples:
        try:
            # Đọc bằng Tifffile để lấy đủ 10 kênh float64
            img = tifffile.imread(img_path)
            shapes.append(img.shape)
            dtypes.append(img.dtype)
        except Exception as e:
            print(f"Lỗi đọc {img_path}: {e}")

    shape_counts = Counter(shapes)
    print(f"   - Kích thước (H, W, Channels):")
    for shape, count in shape_counts.items():
        print(f"     + {shape}: {count} file (PHẢI CÓ 10 KÊNH)")

    print(f"   - Kiểu dữ liệu: {set(dtypes)} (Chuẩn khoa học là float64/float32)")

    # 3. Phân tích Mask (Tìm số Class và Trọng số)
    if len(all_masks) > 0:
        print(f"\n3. PHÂN TÍCH MASK & TÍNH TOÁN TRỌNG SỐ (CLASS WEIGHTS):")
        mask_samples = random.sample(all_masks, min(500, len(all_masks)))

        total_pixels = 0
        class_counts = Counter()

        for mask_path in tqdm(mask_samples, desc="   Đang đếm Pixel", leave=False):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue

            unique, counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                class_counts[u] += c
            total_pixels += mask.size

        sorted_classes = sorted(class_counts.keys())
        print(f"   - Các Level (Class) tìm thấy: {sorted_classes}")
        print(f"   -> SỐ LỚP CẦN CẤU HÌNH (num_classes): {len(sorted_classes)}")

        # In tỷ lệ phần trăm pixel để xem độ mất cân bằng
        print("\n   - Tỷ lệ phân bố dữ liệu:")
        for cls in sorted_classes:
            pct = (class_counts[cls] / total_pixels) * 100
            print(f"     + Level {cls}: {pct:.2f}%")

    # 4. Trực quan hóa (Lấy 3 kênh RGB từ 10 kênh để vẽ)
    print(f"\n4. ĐANG VẼ MẪU KIỂM TRA (Lưu file 'dataset_preview.png')...")
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))

    viz_samples = random.sample(range(len(all_images)), min(3, len(all_images)))

    for i, idx in enumerate(viz_samples):
        img_path = all_images[idx]
        filename = os.path.basename(img_path)

        # Tìm mask PNG tương ứng
        mask_name = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(DATASET_ROOT, MASK_DIR_NAME, mask_name)

        # 1. Đọc ảnh TIF 10 kênh
        img_10ch = tifffile.imread(img_path)

        # 2. Lấy 3 kênh RGB (0, 1, 2) và chuẩn hóa về [0, 1] để Matplotlib vẽ được
        rgb = img_10ch[:, :, 0:3]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        # 3. Đọc Mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(rgb.shape[:2])

        # Vẽ Ảnh RGB
        axs[i, 0].imshow(rgb)
        axs[i, 0].set_title(f"Multi-spectral (RGB channels only)\n{filename}", fontsize=9)
        axs[i, 0].axis('off')

        # Vẽ Mask [SỬA ĐỂ HIỂN THỊ ĐỦ 6 CLASS]
        axs[i, 1].imshow(mask, cmap=labels_cmap, vmin=-0.5, vmax=5.5)
        axs[i, 1].set_title(f"Mask (Max Level: {mask.max()})", fontsize=9)
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("dataset_preview.png", dpi=200)
    print(" Đã lưu ảnh kiểm tra: dataset_preview.png")


if __name__ == "__main__":
    analyze_dataset()
