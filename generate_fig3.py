import os
import glob
import numpy as np
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
IMG_DIR = "dataset_final/train/images"
OUTPUT_FIGURE = "Fig3_Segmentation_LandCover.png"


def get_rgb_from_tif(tif_path):
    img = tiff.imread(tif_path)
    # Lấy 3 kênh RGB và chuẩn hóa hiển thị
    rgb = img[:, :, :3].astype(np.float32)
    for c in range(3):
        min_val = np.percentile(rgb[:, :, c], 1)
        max_val = np.percentile(rgb[:, :, c], 99)
        rgb[:, :, c] = np.clip((rgb[:, :, c] - min_val) / (max_val - min_val), 0, 1)
    return rgb


def main():
    print("==================================================")
    print("[INFO] Đang khởi chạy chiến dịch tìm ảnh Mask đẹp nhất...")
    print("==================================================")

    image_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.tif")))

    # [ĐÃ SỬA LỖI Ở ĐÂY: XUỐNG DÒNG LỆNH RETURN]
    if len(image_files) == 0:
        print("[ERROR] Không tìm thấy ảnh trong dataset!")
        return

    # --- THUẬT TOÁN TÌM ẢNH TỐT NHẤT ---
    best_img_path = None
    best_mask_path = None
    max_classes_found = 0

    # Quét qua toàn bộ ảnh để tìm ứng viên tốt nhất
    print(f"[INFO] Đang quét {len(image_files)} ảnh. Vui lòng đợi...")
    for img_path in tqdm(image_files, desc="Searching best image"):
        mask_path = img_path.replace("images", "masks").replace(".tif", ".png")
        if os.path.exists(mask_path):
            # Đọc mask (grayscale)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue

            # Đếm số class duy nhất xuất hiện trong ảnh này
            unique_classes = np.unique(mask)
            num_classes = len(unique_classes)

            # Cập nhật nếu tìm thấy ảnh nhiều class hơn
            if num_classes > max_classes_found:
                max_classes_found = num_classes
                best_img_path = img_path
                best_mask_path = mask_path

            # Mẹo tối ưu: Nếu tìm thấy ảnh có đủ 5-6 class thì dừng luôn cho nhanh
            if max_classes_found >= 5:
                break

    if best_img_path is None:
        print("[ERROR] Không thể tìm thấy cặp ảnh/mask hợp lệ nào!")
        return

    print(f"\n[RESULT] Đã chọn được ảnh: {os.path.basename(best_img_path)}")
    print(f"[RESULT] Số lượng Class xuất hiện trong ảnh này: {max_classes_found} class")
    print("==================================================")

    # =========================
    # VẼ ĐỒ HỌA CHUẨN PAPER
    # =========================
    img_rgb = get_rgb_from_tif(best_img_path)
    mask = cv2.imread(best_mask_path, cv2.IMREAD_GRAYSCALE)

    # Bảng màu 6 class chuẩn lý thuyết
    cmap = mcolors.ListedColormap(['#000000', '#d62728', '#ff7f0e', '#ffff00', '#2ca02c', '#006400'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Tăng chiều cao một chút để nhét thanh màu
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))

    # Panel A: Ảnh RGB
    axes[0].imshow(img_rgb)
    axes[0].set_title("(A) Input: Sentinel-2 RGB Composite", fontsize=14, fontweight='bold', pad=15)
    axes[0].axis('off')

    # Panel B: Ảnh Mask
    im = axes[1].imshow(mask, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title("(B) Target: Theoretical 6-Class Mask", fontsize=14, fontweight='bold', pad=15)
    axes[1].axis('off')

    # Thêm thanh chú thích màu (Colorbar) bên dưới
    cbar_ax = fig.add_axes([0.55, 0.05, 0.35, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=[0, 1, 2, 3, 4, 5])
    cbar.ax.set_xticklabels(['Back', 'Lvl1', 'Lvl2', 'Lvl3', 'Lvl4', 'Lvl5'], fontsize=10, fontweight='bold')
    cbar.set_label('Agricultural Potential Levels', fontsize=11, labelpad=5)

    # Bo viền đậm cho 2 ảnh chính
    for ax in axes:
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=3, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    # Điều chỉnh layout để chừa chỗ cho colorbar
    plt.subplots_adjust(bottom=0.15, wspace=0.1)
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Đã tạo thành công ảnh Hình 3 đẹp nhất tại: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()