import argparse
import glob
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def analyze_mask(mask_path: str):
    """Read the mask and calculate the percentage of agricultural land (foreground)."""
    try:
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
    except Exception:
        return mask_path, -1

    total_pixels = mask.size
    fg_mask = (mask > 0) & (mask != 255)
    fg_percent = (np.sum(fg_mask) / total_pixels) * 100

    return mask_path, fg_percent


def plot_analysis(good: int, pure_bg: int, low_info: int, total_trash: int, total_files: int, chart_path: str) -> str:
    """Visualize the scanning results with summary charts."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    labels = ["Good Info (>=0.5%)", "Pure Background (0%)", "Low Info (<0.5%)"]
    sizes = [good, pure_bg, low_info]
    colors = ["#2ca02c", "#d62728", "#ff7f0e"]

    plt.pie(sizes, explode=(0.05, 0, 0), labels=labels, colors=colors, autopct="%1.1f%%", shadow=False, startangle=140)
    plt.title("Dataset Quality Distribution", fontweight="bold")

    plt.subplot(1, 2, 2)
    categories = ["Original Dataset", "After Pruning"]
    counts = [total_files, total_files - total_trash]

    bars = plt.bar(categories, counts, color=["#1f77b4", "#9467bd"], width=0.5)
    plt.ylabel("Number of Images")
    plt.title("Dataset Size: Before vs After", fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (total_files * 0.01),
            f"{int(yval):,}",
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    return chart_path


def main(dataset_root: str, trash_dir: str, chart_path: str, auto_confirm: bool = False) -> None:
    print("=" * 60)
    print("[PHASE 1] DATASET QUALITY ANALYSIS")
    print("=" * 60)

    mask_files = []
    for split in ["train", "val", "test"]:
        mask_files.extend(glob.glob(os.path.join(dataset_root, split, "masks", "*.png")))

    total_files = len(mask_files)
    if total_files == 0:
        print(f"[ERROR] No mask files found under {dataset_root}")
        return

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

    saved_chart = plot_analysis(len(good_files), len(pure_bg_files), len(low_info_files), total_trash, total_files, chart_path)

    print("\n" + "=" * 50)
    print("[REPORT] ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Total images scanned: {total_files:,}")
    if error_count:
        print(f"[WARN] Unreadable masks      : {error_count:,}")
    print(f"[+] Good Images (Keep)       : {len(good_files):,} images")
    print(f"[-] Pure Background (0%)     : {len(pure_bg_files):,} images")
    print(f"[-] Low Info (<0.5%)         : {len(low_info_files):,} images")
    print("=" * 50)
    print(f"[RECOMMENDATION] Can prune {total_trash:,} redundant image pairs (TIF+PNG).")
    print(f"[INFO] Visual chart exported to: {saved_chart}")
    print("=" * 50)

    if total_trash == 0:
        print("[INFO] Dataset is already clean. No pruning required. Exiting.")
        return

    print("\n[PHASE 2] PRUNING EXECUTION")
    if not auto_confirm:
        confirm = input(f"[CONFIRM] Do you want to MOVE {total_trash} image pairs to TRASH? (y/n): ")
        if confirm.lower() != "y":
            print("[INFO] Pruning operation cancelled.")
            return

    os.makedirs(trash_dir, exist_ok=True)
    moved_count = 0

    for mask_path in tqdm(trash_masks, desc="Moving files"):
        tif_path = mask_path.replace("masks", "images").replace(".png", ".tif")

        rel_mask = os.path.relpath(mask_path, dataset_root)
        rel_tif = os.path.relpath(tif_path, dataset_root)

        dest_mask = os.path.join(trash_dir, rel_mask)
        dest_tif = os.path.join(trash_dir, rel_tif)

        os.makedirs(os.path.dirname(dest_mask), exist_ok=True)
        os.makedirs(os.path.dirname(dest_tif), exist_ok=True)

        if os.path.exists(mask_path):
            shutil.move(mask_path, dest_mask)
        if os.path.exists(tif_path):
            shutil.move(tif_path, dest_tif)
        moved_count += 1

    print("\n" + "=" * 60)
    print(f"[SUCCESS] OPERATION COMPLETE! Successfully moved {moved_count} image pairs.")
    print(f"[INFO] The dataset is now optimized. Trash files are located at: {trash_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune low-foreground (<0.5%) and excess pure-background patches.")
    parser.add_argument("--data-root", type=str, default="dataset_final", help="Dataset root containing train/val/test.")
    parser.add_argument("--trash-dir", type=str, default="dataset_final_TRASH", help="Destination for pruned pairs.")
    parser.add_argument("--chart-path", type=str, default="dataset_analysis.png", help="Output path for summary chart.")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation prompt.")
    args = parser.parse_args()

    main(args.data_root, args.trash_dir, args.chart_path, auto_confirm=args.yes)
