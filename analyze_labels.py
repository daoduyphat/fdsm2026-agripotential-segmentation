import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from preprocess_data import LandCoverDataset


def analyze_dataset(dataset, save_path="label_statistics.png"):
    label_counter = Counter()

    # Duyệt qua toàn bộ dataset
    for i in tqdm(range(len(dataset)), desc="Analyzing dataset"):
        _, mask = dataset[i]
        unique, counts = torch.unique(mask.long(), return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist()):
            label_counter[u] += c

    classes = sorted(label_counter.keys())
    counts = [label_counter[c] for c in classes]
    total_pixels = sum(counts)

    weights = [1 - (c / total_pixels) for c in counts]
    percentages = [(c / total_pixels) * 100 for c in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("-" * 30)
    print(f"Classes found: {classes}")
    print(f"Pixel counts: {counts}")
    print(f"Class percentages: {[f'{p:.2f}%' for p in percentages]}")
    print(f"Calculated weights: {weights_tensor}")
    print("-" * 30)

    # Vẽ biểu đồ
    try:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, counts, color='skyblue', edgecolor='black')

        plt.xlabel('Class ID')
        plt.ylabel('Pixel Count')
        plt.title('Class Distribution in Dataset')
        plt.xticks(classes)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Hiển thị phần trăm
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{pct:.1f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Chart saved to {save_path}")
    except Exception as e:
        print(f"Error saving chart: {e}")

    return weights_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="dataset/train")
    parser.add_argument("--output", type=str, default="label_statistics.png")
    parser.add_argument("--img-size", type=int, nargs=2, default=(512, 512))
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Path '{args.data_path}' does not exist.")
        exit(1)

    img_dir = os.path.join(args.data_path, "images")
    mask_dir = os.path.join(args.data_path, "masks")

    dataset = LandCoverDataset(
        img_dir,
        mask_dir,
        img_size=tuple(args.img_size),
        augment=False
    )

    analyze_dataset(dataset, args.output)