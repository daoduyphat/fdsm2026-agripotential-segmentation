import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm


def analyze_dataset_precise(mask_dir, save_path="bao_cao_chi_tiet.png"):
    """
    PHÂN TÍCH DATASET SIÊU CHI TIẾT (BẢN CUỐI CÙNG)
    - Quét 100% pixel (không resize, không sai số)
    - Phân tích đa chiều: Số lượng Pixel, Tần suất xuất hiện trên ảnh, Trọng số.
    - Tự động gộp Class 5 vào Class 4.
    """

    # 1. TIM KIEM TAT CA CAC FILE MASK
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    mask_files = []
    for ext in extensions:
        mask_files.extend(glob(os.path.join(mask_dir, ext)))

    if len(mask_files) == 0:
        print(f"[LOI] Khong tim thay file mask nao tai {mask_dir}")
        return

    total_images = len(mask_files)
    print(f"[THONG TIN] Tim thay {total_images} file mask.")
    print("[THONG TIN] Bat dau quet toan bo tung pixel (100% data)...")

    # Khoi tao bo dem
    global_pixel_counts = np.zeros(5, dtype=np.int64)
    image_presence_counts = np.zeros(5, dtype=np.int64)  # So luong anh co chua class nay


    lut = np.full(256, 255, dtype=np.uint8)
    lut[0] = 0  # Nen (Background)
    lut[1] = 1  # Lop 1
    lut[2] = 2  # Lop 2
    lut[3] = 3  # Lop 3
    lut[4] = 4  # Lop 4
    lut[5] = 4  # Lop 5 => Gop ve Lop 4

    # 3. QUET DU LIEU CHI TIET
    for mask_path in tqdm(mask_files, desc="Dang phan tich"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Map pixel
        mapped = lut[mask]
        mapped = mapped[mapped != 255]  # Bo qua pixel 255

        # Dem pixel trong anh hien tai
        unique, counts = np.unique(mapped, return_counts=True)

        # Cap nhat tong pixel va tan suat xuat hien tren anh
        for u, c in zip(unique, counts):
            class_id = int(u)
            global_pixel_counts[class_id] += int(c)
            image_presence_counts[class_id] += 1  # Anh nay co chua class nay

    # 4. TINH TOAN THONG KE
    num_classes = 5
    class_ids = list(range(num_classes))
    class_names = ["Nen (Bg)", "Lop 1", "Lop 2", "Lop 3", "Lop 4"]

    total_pixels = np.sum(global_pixel_counts)
    if total_pixels == 0:
        print("[LOI] Khong co pixel hop le nao duoc tim thay.")
        return

    # Tinh toan trong so (Inverse Class Frequency)
    weights = []
    for count in global_pixel_counts:
        if count > 0:
            w = total_pixels / (num_classes * count)
        else:
            w = 1.0  # Gia tri an toan neu class khong ton tai
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # 5. IN BAO CAO RA MAN HINH
    print("\n" + "=" * 85)
    print("                BAO CAO PHAN TICH DU LIEU CHI TIET (5 CLASS)")
    print("=" * 85)
    print(f"Tong so anh quet  : {total_images}")
    print(f"Tong so pixel quet: {total_pixels:,}")
    print("-" * 85)
    print(
        f"{'ID':<3} | {'Ten Lop':<12} | {'So luong Pixel':<18} | {'% Pixel':<8} | {'% So Anh Xuat Hien':<18} | {'Trong so'}")
    print("-" * 85)

    for i in range(num_classes):
        name = class_names[i]
        pix_count = global_pixel_counts[i]
        img_count = image_presence_counts[i]
        w = weights[i]

        pix_percent = (pix_count / total_pixels) * 100
        img_percent = (img_count / total_images) * 100

        print(
            f"{i:<3} | {name:<12} | {pix_count:<18,} | {pix_percent:5.2f}%   | {img_count:<5} anh ({img_percent:5.1f}%) | {w:.4f}")

    print("=" * 85)
    print("\n[Hanh dong] Copy dong nay vao file train.py de can bang du lieu:")
    print(f"class_weights = {weights_tensor}")
    print("=" * 85)

    # 6. VE BIEU DO CHI TIET (3 BIEU DO)
    try:
        # Thiet lap font va mau sac
        plt.rcParams.update({'font.size': 10})
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = ['gray', 'crimson', 'forestgreen', 'royalblue', 'saddlebrown']

        # Bieu do 1: Tong so luong Pixel (Log scale)
        bars1 = axes[0].bar(class_names, global_pixel_counts, color=colors, edgecolor='black')
        axes[0].set_yscale('log')
        axes[0].set_title("1. Tong So Luong Pixel (Log Scale)")
        axes[0].set_ylabel("So luong Pixel")
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)
        # Hien thi phan tram tren dinh cot
        for i, bar in enumerate(bars1):
            pct = (global_pixel_counts[i] / total_pixels) * 100
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%",
                         ha='center', va='bottom', fontweight='bold')

        # Bieu do 2: Tan suat xuat hien tren Anh
        image_presence_percentages = (image_presence_counts / total_images) * 100
        bars2 = axes[1].bar(class_names, image_presence_percentages, color=colors, edgecolor='black', alpha=0.8)
        axes[1].set_ylim(0, 105)
        axes[1].set_title("2. Ti Le Xuat Hien Tren Anh (%)")
        axes[1].set_ylabel("% So luong Anh")
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)
        for i, bar in enumerate(bars2):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{image_presence_percentages[i]:.1f}%", ha='center', fontweight='bold')

        # Bieu do 3: Trong so huan luyen (Weights)
        bars3 = axes[2].bar(class_names, weights, color=colors, edgecolor='black', hatch='//')
        axes[2].set_title("3. Trong So Huan Luyen (Class Weights)")
        axes[2].set_ylabel("Gia tri Trong so")
        axes[2].grid(axis='y', linestyle='--', alpha=0.5)
        for i, bar in enumerate(bars3):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{weights[i]:.2f}", ha='center', va='bottom', fontweight='bold')

        plt.suptitle("BAO CAO PHAN TICH DU LIEU (LAND COVER 5 CLASSES)", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[THONG TIN] Da luu bieu do vao file: {save_path}")

    except Exception as e:
        print(f"[LOI] Khong the ve bieu do: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="dataset/train")
    parser.add_argument("--output", type=str, default="bao_cao_chi_tiet.png")
    args = parser.parse_args()

    # Tu dong tim thu muc masks
    mask_dir = os.path.join(args.data_path, "masks")

    if not os.path.exists(mask_dir):
        if os.path.exists(args.data_path):
            mask_dir = args.data_path
        else:
            print("[LOI] Khong the xac dinh thu muc chua masks.")
            exit(1)

    analyze_dataset_precise(mask_dir, save_path=args.output)