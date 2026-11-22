import argparse
import os

import cv2
import h5py
import numpy as np
import tifffile
from tqdm import tqdm


def extract_multispectral(h5_path: str, output_root: str, label_type: str = "field") -> None:
    """Extract 10-band Sentinel-2 patches (TIF) and 5-class masks (PNG) from an HDF5 file."""
    if not os.path.exists(h5_path):
        print(f"[ERROR] HDF5 file not found: {h5_path}")
        return

    print(f"Reading HDF5 file: {h5_path}")
    print(f"Extracting 10-band TIF patches to: {output_root}\n")

    os.makedirs(output_root, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        for split in ["train", "val", "test"]:
            if split not in f:
                continue

            print(f"--- Processing split: {split.upper()} ---")

            sentinel_grp = f[split]["sentinel2"]
            labels_grp = f[split]["labels"]

            if label_type not in labels_grp:
                print(f"[WARN] Label '{label_type}' not found in {split}. Skipping.")
                continue

            mask_ds = labels_grp[label_type]
            month_keys = sorted(list(sentinel_grp.keys()))

            for month in month_keys:
                print(f"  Exporting 10-band TIF: {month}...")

                images_ds = sentinel_grp[month]
                total_imgs = images_ds.shape[0]

                save_img_dir = os.path.join(output_root, split, month, "images")
                save_mask_dir = os.path.join(output_root, split, month, "masks")
                os.makedirs(save_img_dir, exist_ok=True)
                os.makedirs(save_mask_dir, exist_ok=True)

                for i in tqdm(range(total_imgs), desc="  Processing", leave=False):
                    try:
                        # Transpose (10, 128, 128) -> (128, 128, 10)
                        img_raw = images_ds[i]
                        img_out = np.transpose(img_raw, (1, 2, 0))

                        # Convert one-hot mask (128, 128, 5) -> class index (128, 128)
                        mask_raw = mask_ds[i]
                        mask_out = np.argmax(mask_raw, axis=-1).astype(np.uint8)

                        filename = f"patch_{i:05d}"
                        tifffile.imwrite(os.path.join(save_img_dir, f"{filename}.tif"), img_out)
                        cv2.imwrite(os.path.join(save_mask_dir, f"{filename}.png"), mask_out)
                    except Exception as exc:
                        print(f"[WARN] Failed {split}/{month}/patch_{i:05d}: {exc}")

    print(f"\n[DONE] Successfully exported 10-band TIF dataset to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export 10-band Sentinel-2 TIFs and 5-class PNG masks from HDF5.")
    parser.add_argument("--h5-path", type=str, default="dataset.h5", help="Path to source HDF5 dataset.")
    parser.add_argument("--output-root", type=str, default="dataset", help="Output directory for extracted splits.")
    parser.add_argument("--label-type", type=str, default="field", help="Label key inside HDF5 labels group.")
    args = parser.parse_args()

    extract_multispectral(args.h5_path, args.output_root, args.label_type)
