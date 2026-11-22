import argparse
import os
import random


def rebalance_deduplicate(data_root: str, output_dir: str) -> None:
    """Deduplicate samples across splits and generate a 70/20/10 train/val/test split."""
    print("=== STARTING DATASET DEDUPLICATION AND 70/20/10 SPLIT ===")

    # Keep first occurrence by split priority: train -> val -> test
    unique_samples = {}
    source_folders = ["train", "val", "test"]
    duplicates_count = 0

    for folder in source_folders:
        img_dir = os.path.join(data_root, folder, "images")
        mask_dir = os.path.join(data_root, folder, "masks")

        if not os.path.exists(img_dir):
            continue

        files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
        print(f"[SCAN] Scanning '{folder}'... Found {len(files)} files.")

        for f_name in files:
            base_name = f_name.replace(".tif", "")
            mask_path = os.path.join(mask_dir, f"{base_name}.png")
            if os.path.exists(mask_path):
                if base_name not in unique_samples:
                    unique_samples[base_name] = folder
                else:
                    duplicates_count += 1

    final_list = list(unique_samples.items())

    print("\n>>> SCAN SUMMARY:")
    print(f"  - Duplicate files removed : {duplicates_count}")
    print(f"  - Total unique valid pairs: {len(final_list)}")

    if len(final_list) == 0:
        print("[ERROR] No valid image/mask pairs found.")
        return

    print("\n[INFO] Shuffling and splitting 70-20-10...")
    random.seed(42)
    random.shuffle(final_list)

    total = len(final_list)
    n_train = int(total * 0.7)
    n_val = int(total * 0.2)

    train_set = final_list[:n_train]
    val_set = final_list[n_train : n_train + n_val]
    test_set = final_list[n_train + n_val :]

    print(f"  - Train: {len(train_set)}")
    print(f"  - Val  : {len(val_set)}")
    print(f"  - Test : {len(test_set)}")

    os.makedirs(output_dir, exist_ok=True)

    def write_file(fname: str, data) -> None:
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            for name, source in data:
                f.write(f"{name},{source}\n")
        print(f"  -> Wrote: {path}")

    write_file("train.txt", train_set)
    write_file("val.txt", val_set)
    write_file("test.txt", test_set)

    print("\n=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate patches and generate 70/20/10 split index files.")
    parser.add_argument("--data-root", type=str, default="dataset_final", help="Dataset root with train/val/test.")
    parser.add_argument("--output-dir", type=str, default="dataset_final", help="Output directory for split *.txt files.")
    args = parser.parse_args()

    rebalance_deduplicate(args.data_root, args.output_dir)
