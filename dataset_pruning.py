import os
import glob
import cv2
import numpy as np
import shutil
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# PATH CONFIGURATION
# =========================
DATASET_ROOT = "dataset_final"
TRASH_DIR = "dataset_final_TRASH"  # Directory for pruned images


def analyze_mask(mask_path):
    """Read the mask and calculate the percentage of agricultural land (Foreground)"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return mask_path, -1

    total_pixels = mask.size
    fg_mask = (mask > 0) & (mask != 255)
    fg_percent = (np.sum(fg_mask) / total_pixels) * 100

    return mask_path, fg_percent


def plot_analysis(good, pure_bg, low_info, total_trash, total_files):
    """Visualize the scanning results with charts"""
    plt.figure(figsize=(14, 6))

    # Chart 1: Quality Distribution (Pie Chart - FLAT DESIGN)
    plt.subplot(1, 2, 1)
    labels = ['Good Info (>=0.5%)', 'Pure Background (0%)', 'Low Info (<0.5%)']
    sizes = [good, pure_bg, low_info]
    colors = ['#2ca02c', '#d62728', '#ff7f0e']

    # [MODIFIED] shadow=False for a modern flat design
    plt.pie(sizes, explode=(0.05, 0, 0), labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
    plt.title('Dataset Quality Distribution', fontweight='bold')

    # Chart 2: Size Comparison (Bar Chart)
    plt.subplot(1, 2, 2)
    categories = ['Original Dataset', 'After Pruning']
    counts = [total_files, total_files - total_trash]

    bars = plt.bar(categories, counts, color=['#1f77b4', '#9467bd'], width=0.5)
    plt.ylabel('Number of Images')
    plt.title('Dataset Size: Before vs After', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (total_files * 0.01), f'{int(yval):,}', ha='center',
                 fontweight='bold')

    plt.tight_layout()
    chart_path = "dataset_analysis.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    return chart_path


def main():
    print("============================================================")
    print("[PHASE 1] DATASET QUALITY ANALYSIS")
    print("============================================================")

    # 1. Find and classify images
    mask_files = []
    for split in ['train', 'val', 'test']:
        mask_files.extend(glob.glob(os.path.join(DATASET_ROOT, split, "masks", "*.png")))

    total_files = len(mask_files)
    print(f"[INFO] Found {total_files} mask files. Starting multi-threaded scan...")

    good_files = []
    pure_bg_files = []
    low_info_files = []
    error_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_file = {executor.submit(analyze_mask, m): m for m in mask_files}

        for future in tqdm(as_completed(future_to_file), total=total_files, desc="Analyzing pixels"):
            m_path, fg_pct = future.result()
            if fg_pct == -1:
                error_count += 1
            elif fg_pct == 0:
                pure_bg_files.append(m_path)
            elif fg_pct < 0.5:
                low_info_files.append(m_path)
            else:
                good_files.append(m_path)

    # Keep 10% of pure background images to prevent overfitting
    random.seed(42)
    random.shuffle(pure_bg_files)
    num_bg_to_keep = int(len(pure_bg_files) * 0.1)

    trash_masks = pure_bg_files[num_bg_to_keep:] + low_info_files
    total_trash = len(trash_masks)

    # 2. Print report and plot charts
    chart_path = plot_analysis(len(good_files), len(pure_bg_files), len(low_info_files), total_trash, total_files)

    print("\n==================================================")
    print("[REPORT] ANALYSIS RESULTS")
    print("==================================================")
    print(f"Total images scanned: {total_files:,}")
    print(f"[+] Good Images (Keep)       : {len(good_files):,} images")
    print(f"[-] Pure Background (0%)     : {len(pure_bg_files):,} images")
    print(f"[-] Low Info (<0.5%)         : {len(low_info_files):,} images")
    print("==================================================")
    print(f"[RECOMMENDATION] Can prune {total_trash:,} redundant image pairs (TIF+PNG).")
    print(f"[INFO] Visual chart exported to: {chart_path}")
    print("==================================================")

    if total_trash == 0:
        print("[INFO] Dataset is already clean. No pruning required. Exiting.")
        return

    # 3. User confirmation
    print("\n[PHASE 2] PRUNING EXECUTION")
    confirm = input(f"[CONFIRM] Do you want to MOVE {total_trash} image pairs to TRASH? (y/n): ")

    if confirm.lower() != 'y':
        print("[INFO] Pruning operation cancelled.")
        return

    # 4. Execute file moving
    os.makedirs(TRASH_DIR, exist_ok=True)
    moved_count = 0

    for mask_path in tqdm(trash_masks, desc="Moving files"):
        tif_path = mask_path.replace("masks", "images").replace(".png", ".tif")

        # Replicate directory structure in TRASH
        rel_mask = os.path.relpath(mask_path, DATASET_ROOT)
        rel_tif = os.path.relpath(tif_path, DATASET_ROOT)

        dest_mask = os.path.join(TRASH_DIR, rel_mask)
        dest_tif = os.path.join(TRASH_DIR, rel_tif)

        os.makedirs(os.path.dirname(dest_mask), exist_ok=True)
        os.makedirs(os.path.dirname(dest_tif), exist_ok=True)

        # Move both Mask and Original Image
        if os.path.exists(mask_path): shutil.move(mask_path, dest_mask)
        if os.path.exists(tif_path): shutil.move(tif_path, dest_tif)
        moved_count += 1

    print("\n============================================================")
    print(f"[SUCCESS] OPERATION COMPLETE! Successfully moved {moved_count} image pairs.")
    print(f"[INFO] The dataset is now optimized. Trash files are located at: {TRASH_DIR}")
    print("============================================================")


if __name__ == "__main__":
    main()
