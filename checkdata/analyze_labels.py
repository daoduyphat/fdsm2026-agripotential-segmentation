import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

# [IMPORTANT] Force Matplotlib to run in the background to prevent freezing on Windows
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_dataset_precise(mask_dir, save_path="detailed_report_5_classes_bfpr.png"):
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    mask_files = []
    for ext in extensions:
        mask_files.extend(glob(os.path.join(mask_dir, ext)))

    if len(mask_files) == 0:
        print(f"[ERROR] No mask files found in {mask_dir}")
        return

    total_images = len(mask_files)
    print(f"Found {total_images} mask files.")
    print("Starting full pixel-wise analysis...")

    # [CHANGED] 6 -> 5
    global_pixel_counts = np.zeros(5, dtype=np.int64)
    image_presence_counts = np.zeros(5, dtype=np.int64)

    # 3. DETAILED DATA SCANNING (USING ULTRA-FAST BINCOUNT ALGORITHM)
    for mask_path in tqdm(mask_files, desc="Analyzing"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Exclude error pixels (255) if any
        valid_mask = mask[mask != 255]

        if valid_mask.size > 0:
            # [CHANGED] minlength=6 -> minlength=5
            counts = np.bincount(valid_mask.ravel(), minlength=5)
            # Just in case there are values >= 5, truncate them to avoid index out of bounds
            global_pixel_counts += counts[:5]

            # Check which classes are present in the image (ignore classes >= 5)
            present_classes = np.unique(valid_mask)
            valid_classes = present_classes[present_classes < 5]
            image_presence_counts[valid_classes] += 1

    # 4. STATISTICAL CALCULATIONS
    # [CHANGED] 6 -> 5 and removed "Very High"
    num_classes = 5
    class_names = ["Background", "Very Low", "Low", "Medium", "High"]

    total_pixels = np.sum(global_pixel_counts)
    if total_pixels == 0:
        return

    # Calculate weights (Inverse Class Frequency)
    weights = []
    for count in global_pixel_counts:
        if count > 0:
            w = total_pixels / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # 5. PRINT REPORT TO CONSOLE
    print("\n" + "=" * 90)
    print("                DETAILED DATASET ANALYSIS REPORT (5 CLASSES - AFTER PRUNING)")
    print("=" * 90)
    print(f"Total images scanned: {total_images}")
    print(f"Total pixels scanned: {total_pixels:,}")
    print("-" * 90)
    print(
        f"{'ID':<3} | {'Class Name':<12} | {'Pixel Count':<18} | {'Pixel %':<8} | {'Image Presence %':<18} | {'Weight'}")
    print("-" * 90)

    for i in range(num_classes):
        name = class_names[i]
        pix_count = global_pixel_counts[i]
        img_count = image_presence_counts[i]
        w = weights[i]
        pix_percent = (pix_count / total_pixels) * 100
        img_percent = (img_count / total_images) * 100
        print(
            f"{i:<3} | {name:<12} | {pix_count:<18,} | {pix_percent:5.2f}%   | {img_count:<5} imgs ({img_percent:5.1f}%) | {w:.4f}")

    print("=" * 90)
    print("\n[Action] Copy this line into your train.py to balance the dataset:")
    print(f"class_weights = {weights_tensor}")
    print("=" * 90)

    # 6. PLOT DETAILED CHARTS (FIXED FREEZING AND MARGIN ISSUES)
    try:
        plt.rcParams.update({'font.size': 11})
        # Increase figure size to prevent text overlap
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        # [CHANGED] Removed the 6th color
        colors = ['black', '#8B0000', '#FFA500', '#FFFF00', '#32CD32']

        bars1 = axes[0].bar(class_names, global_pixel_counts, color=colors, edgecolor='black')
        axes[0].set_yscale('log')
        axes[0].set_title("1. Total Pixel Count (Log Scale)")
        axes[0].set_ylabel("Pixel Count")
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)

        image_presence_percentages = (image_presence_counts / total_images) * 100
        bars2 = axes[1].bar(class_names, image_presence_percentages, color=colors, edgecolor='black', alpha=0.8)
        axes[1].set_ylim(0, 105)
        axes[1].set_title("2. Image Presence Rate (%)")
        axes[1].set_ylabel("% of Images")
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)

        bars3 = axes[2].bar(class_names, weights, color=colors, edgecolor='black', hatch='//')
        axes[2].set_title("3. Training Weights (Class Weights)")
        axes[2].set_ylabel("Weight Value")
        axes[2].grid(axis='y', linestyle='--', alpha=0.5)

        plt.suptitle("DATASET ANALYSIS REPORT (AGRICULTURAL POTENTIAL - 5 CLASSES)", fontsize=18, fontweight='bold',
                     y=0.98)
        plt.tight_layout(pad=2.0)  # Fix margin issues
        plt.savefig(save_path, dpi=300)
        plt.close(fig)  # Free up RAM immediately
        print(f"\n[INFO] Chart saved to: {save_path}")

    except Exception as e:
        print(f"[ERROR] Cannot plot charts: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=r"E:\datdai\dataset_final\train")
    parser.add_argument("--output", type=str, default="detailed_report_5_classes_atpr.png")
    args = parser.parse_args()

    mask_dir = os.path.join(args.data_path, "masks")
    if not os.path.exists(mask_dir):
        mask_dir = args.data_path

    analyze_dataset_precise(mask_dir, save_path=args.output)
