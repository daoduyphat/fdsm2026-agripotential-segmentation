import os
import glob
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec

# =========================
# CONFIGURATION
# =========================
DATASET_PATH = "dataset_final/train/images"
OUTPUT_FIGURE = "Fig1_Sentinel_Process_Clean.png"
GRID_SIZE = 3 # Lưới 3x3

def get_rgb_from_tif(tif_path):
    img = tiff.imread(tif_path)
    rgb = img[:, :, :3].astype(np.float32)
    # Cân bằng sáng cho ảnh rõ nét
    for c in range(3):
        min_val = np.percentile(rgb[:, :, c], 1)
        max_val = np.percentile(rgb[:, :, c], 99)
        rgb[:, :, c] = np.clip((rgb[:, :, c] - min_val) / (max_val - min_val), 0, 1)
    return rgb

def main():
    print("[INFO] Đang kết xuất đồ họa Hình 1 (Bản Tối giản)...")

    all_tifs = glob.glob(os.path.join(DATASET_PATH, "*.tif"))

    if len(all_tifs) < GRID_SIZE**2:
        print("[ERROR] Không đủ ảnh!")
        return

    np.random.seed(42)
    selected_tifs = np.random.choice(all_tifs, GRID_SIZE**2, replace=False)

    # Ghép ảnh
    patch_h, patch_w, _ = get_rgb_from_tif(selected_tifs[0]).shape
    big_h, big_w = patch_h * GRID_SIZE, patch_w * GRID_SIZE
    big_img = np.zeros((big_h, big_w, 3))
    patches_list = []

    idx = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            img = get_rgb_from_tif(selected_tifs[idx])
            patches_list.append(img)
            big_img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w, :] = img
            idx += 1

    # =========================
    # BẮT ĐẦU VẼ ĐỒ HỌA (CLEAN)
    # =========================
    fig = plt.figure(figsize=(15, 7.5), facecolor='white')
    gs = gridspec.GridSpec(3, 5, width_ratios=[1.3, 0.7, 0.05, 0.8, 0.8])

    # 1. ẢNH LỚN & LƯỚI
    ax_big = fig.add_subplot(gs[:, 0:2])
    ax_big.imshow(big_img)
    ax_big.axis('off')
    ax_big.set_title("Sentinel-2 Satellite Acquisition\n(Multi-spectral Image Grid)",
                     fontsize=15, fontweight='bold', pad=15, color='#2c3e50')

    # Vẽ lưới mờ
    for i in range(1, GRID_SIZE):
        ax_big.axhline(i * patch_h, color='white', linewidth=1, linestyle='-', alpha=0.5)
        ax_big.axvline(i * patch_w, color='white', linewidth=1, linestyle='-', alpha=0.5)

    # Khoanh vùng các Patch được chọn
    highlight_indices = [2, 4, 8]
    colors = ['#FF4500', '#00BFFF', '#8A2BE2'] # Cam, Xanh lơ, Tím

    for i, idx in enumerate(highlight_indices):
        r, c = divmod(idx, GRID_SIZE)
        rect = patches.Rectangle((c * patch_w, r * patch_h), patch_w, patch_h,
                                 linewidth=3.5, edgecolor=colors[i], facecolor='none', zorder=10)
        ax_big.add_patch(rect)


    # 2. BA ẢNH NHỎ (PATCHES) BÊN PHẢI
    ax_patches = []
    titles = ["Patch A (High Potential)", "Patch B (Mixed Cover)", "Patch C (Background)"]

    for i, idx in enumerate(highlight_indices):
        ax_small = fig.add_subplot(gs[i, 3])
        ax_small.imshow(patches_list[idx])
        ax_small.set_xticks([]); ax_small.set_yticks([])
        ax_patches.append(ax_small)

        for spine in ax_small.spines.values():
            spine.set_edgecolor(colors[i])
            spine.set_linewidth(3)

        ax_small.text(1.1, 0.5, titles[i], transform=ax_small.transAxes,
                      fontsize=12, fontweight='bold', color=colors[i], va='center')

    ax_patches[0].set_title("Extracted High-Res Patches\nfor Deep Learning",
                            fontsize=14, fontweight='bold', pad=25, color='#2c3e50')

    # 3. MŨI TÊN KẾT NỐI (CONNECTION ARROWS)
    for i, idx in enumerate(highlight_indices):
        r, c = divmod(idx, GRID_SIZE)
        xyA = ((c + 1) * patch_w, (r + 0.5) * patch_h)
        xyB = (0, patch_h / 2)

        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                              axesA=ax_big, axesB=ax_patches[i],
                              arrowstyle="-|>", shrinkA=5, shrinkB=5,
                              mutation_scale=20, color=colors[i], linewidth=2.5,
                              connectionstyle="angle3,angleA=10,angleB=-90", zorder=20)
        ax_big.add_artist(con)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)

    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Đã tạo thành công ảnh Đồ họa (Clean Version): {OUTPUT_FIGURE}")

if __name__ == "__main__":
    main()