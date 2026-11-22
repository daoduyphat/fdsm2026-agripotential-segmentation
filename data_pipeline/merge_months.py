import argparse
import os
import shutil

from tqdm import tqdm


def merge_data(source_root: str, dest_root: str) -> None:
    """Merge monthly 10-band TIF images and PNG masks into unified split directories."""
    if not os.path.exists(source_root):
        print(f"[ERROR] Source directory not found: {source_root}")
        return

    splits = ["train", "val", "test"]

    print("Merging data across months (10-band TIF + PNG masks)...")
    print(f"  Source: {source_root}")
    print(f"  Dest:   {dest_root}")

    for split in splits:
        split_src = os.path.join(source_root, split)
        if not os.path.exists(split_src):
            continue

        print(f"\n--- Processing split: {split.upper()} ---")

        dest_img_dir = os.path.join(dest_root, split, "images")
        dest_mask_dir = os.path.join(dest_root, split, "masks")
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_mask_dir, exist_ok=True)

        month_folders = [
            d for d in sorted(os.listdir(split_src)) if os.path.isdir(os.path.join(split_src, d))
        ]

        for month in month_folders:
            month_path = os.path.join(split_src, month)
            src_img_dir = os.path.join(month_path, "images")
            src_mask_dir = os.path.join(month_path, "masks")

            if not os.path.exists(src_img_dir):
                continue

            files = os.listdir(src_img_dir)

            for filename in tqdm(files, desc=f"  Merging {month}", leave=False):
                if not filename.lower().endswith(".tif"):
                    continue

                new_filename = f"{month}_{filename}"
                shutil.copy2(os.path.join(src_img_dir, filename), os.path.join(dest_img_dir, new_filename))

                stem = os.path.splitext(filename)[0]
                old_mask_path = os.path.join(src_mask_dir, f"{stem}.png")
                if not os.path.exists(old_mask_path):
                    old_mask_path = os.path.join(src_mask_dir, filename)

                if os.path.exists(old_mask_path):
                    new_mask_filename = f"{month}_{stem}.png"
                    shutil.copy2(old_mask_path, os.path.join(dest_mask_dir, new_mask_filename))

    print(f"\n[DONE] Merged multispectral dataset saved to: {dest_root}")
    print(f"  {dest_root}/train/images (*.tif)")
    print(f"  {dest_root}/train/masks  (*.png)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge monthly Sentinel-2 TIF/PNG folders per split.")
    parser.add_argument("--source-root", type=str, default="dataset", help="Root directory with monthly folders.")
    parser.add_argument("--dest-root", type=str, default="dataset_final", help="Destination merged dataset root.")
    args = parser.parse_args()

    merge_data(args.source_root, args.dest_root)
