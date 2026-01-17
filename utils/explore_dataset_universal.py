import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import glob
import random

# --- CẤU HÌNH ---
# Thay đường dẫn tới dataset MỚI của bạn ở đây
DATASET_ROOT = r"D:\git\agrivison\dataset\DATASET_MERGED"

# Các folder con chứa ảnh và mask (sửa nếu tên folder của bạn khác)
IMG_DIR_NAME = "images"
MASK_DIR_NAME = "masks"


def get_file_list(folder_path):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    files = []
    for ext in extensions:
        # Tìm kiếm đệ quy
        files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
    return sorted(files)


def analyze_dataset():
    if not os.path.exists(DATASET_ROOT):
        print(f"Lỗi: Không tìm thấy thư mục {DATASET_ROOT}")
        return

    print(f"--- ĐANG PHÂN TÍCH DATASET TẠI: {DATASET_ROOT} ---\n")

    # 1. Quét file
    all_images = get_file_list(os.path.join(DATASET_ROOT, "**", IMG_DIR_NAME))
    all_masks = get_file_list(os.path.join(DATASET_ROOT, "**", MASK_DIR_NAME))

    print(f"1. SỐ LƯỢNG FILE:")
    print(f"   - Ảnh tìm thấy: {len(all_images)}")
    print(f"   - Mask tìm thấy: {len(all_masks)}")

    if len(all_images) == 0:
        print("   -> CẢNH BÁO: Không tìm thấy ảnh nào! Kiểm tra lại đường dẫn/tên folder.")
        return

    # 2. Kiểm tra kích thước và kênh màu (trên mẫu ngẫu nhiên)
    print(f"\n2. KIỂM TRA ĐẶC TÍNH ẢNH (Mẫu ngẫu nhiên 50 ảnh):")
    sample_size = min(50, len(all_images))
    samples = random.sample(all_images, sample_size)

    shapes = []
    dtypes = []

    for img_path in samples:
        try:
            # Đọc nguyên bản (không convert màu) để xem gốc là gì
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                shapes.append(img.shape)
                dtypes.append(img.dtype)
        except Exception:
            pass

    # Thống kê Shape
    shape_counts = Counter(shapes)
    print(f"   - Các kích thước tìm thấy (H, W, C):")
    for shape, count in shape_counts.items():
        print(f"     + {shape}: {count} file")

    # Thống kê kiểu dữ liệu
    print(f"   - Kiểu dữ liệu (dtype): {set(dtypes)}")

    # 3. Phân tích Mask (Quan trọng để xác định num_classes)
    if len(all_masks) > 0:
        print(f"\n3. PHÂN TÍCH MASK (Để tìm số lớp - Num Classes):")
        mask_samples = random.sample(all_masks, sample_size)
        unique_values_total = set()

        for mask_path in mask_samples:
            # Mask thường là grayscale
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None: continue

            # Nếu mask có 3 kênh (RGB) -> Cảnh báo
            if len(mask.shape) > 2 and mask.shape[2] > 1:
                print(f"   -> CẢNH BÁO: Mask '{os.path.basename(mask_path)}' có 3 kênh màu! Cần chuyển về 1 kênh.")
                # Thử lấy kênh đầu tiên
                vals = np.unique(mask[:, :, 0])
            else:
                vals = np.unique(mask)

            unique_values_total.update(vals.tolist())

        sorted_values = sorted(list(unique_values_total))
        print(f"   - Các giá trị pixel tìm thấy trong Mask: {sorted_values}")
        print(f"   -> DỰ ĐOÁN SỐ LỚP (num_classes): {len(sorted_values)}")
        if len(sorted_values) > 50:
            print("      (Cảnh báo: Quá nhiều giá trị. Có thể mask chưa được chuẩn hóa về 0,1,2... hoặc là ảnh RGB)")

    # 4. Trực quan hóa (Vẽ 3 cặp ảnh - mask xem thử)
    print(f"\n4. ĐANG VẼ MẪU KIỂM TRA (Lưu file 'dataset_preview.png')...")
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))

    viz_samples = random.sample(range(len(all_images)), min(3, len(all_images)))

    for i, idx in enumerate(viz_samples):
        img_path = all_images[idx]
        # Giả định mask có cùng tên file (hoặc xử lý đổi đuôi nếu cần)
        # Logic tìm mask tương ứng:
        filename = os.path.basename(img_path)
        img_dir = os.path.dirname(img_path)

        # Thử tìm mask ở thư mục masks song song
        # .../train/images/abc.jpg -> .../train/masks/abc.png
        possible_mask_dir = img_dir.replace(IMG_DIR_NAME, MASK_DIR_NAME)

        # Thử đuôi .png trước (phổ biến cho mask)
        mask_name = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(possible_mask_dir, mask_name)

        if not os.path.exists(mask_path):
            # Thử đuôi gốc
            mask_path = os.path.join(possible_mask_dir, filename)

        # Đọc ảnh
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Đọc mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Nhân màu lên cho dễ nhìn nếu giá trị nhỏ (0,1,2)
            if mask.max() < 20:
                mask_vis = mask * 50
            else:
                mask_vis = mask
        else:
            mask_vis = np.zeros(img.shape[:2])  # Mask đen nếu không tìm thấy
            print(f"   -> Không tìm thấy mask cho: {filename}")

        axs[i, 0].imshow(img)
        axs[i, 0].set_title(f"Image: {filename}")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mask_vis, cmap='jet')
        axs[i, 1].set_title(f"Mask (Max val: {mask.max() if os.path.exists(mask_path) else 0})")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("dataset_preview.png")
    print("✅ Đã lưu ảnh kiểm tra: dataset_preview.png")


if __name__ == "__main__":
    analyze_dataset()