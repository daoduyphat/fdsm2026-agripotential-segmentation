import h5py
import os
import cv2
import numpy as np
import tifffile  # [MỚI] Thư viện chuyên dụng lưu ảnh viễn thám đa phổ
from tqdm import tqdm

# --- CẤU HÌNH ---
H5_PATH = r"E:\datdai\dataset.h5"
# [SỬA] Đổi tên folder đầu ra để phân biệt với bản JPG cũ
OUTPUT_ROOT = r"E:\datdai\dataset"
TARGET_LABEL_TYPE = "field"


def extract_multispectral_to_D():
    if not os.path.exists(H5_PATH):
        print(f"Lỗi: Không tìm thấy file {H5_PATH}")
        return

    print(f" Đang đọc file H5: {H5_PATH}")
    print(f" Dữ liệu (10-KÊNH TIF) sẽ được bung ra tại: {OUTPUT_ROOT}\n")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with h5py.File(H5_PATH, 'r') as f:
        for split in ['train', 'val', 'test']:
            if split not in f: continue

            print(f"--- ĐANG XỬ LÝ TẬP: {split.upper()} ---")

            sentinel_grp = f[split]['sentinel2']
            labels_grp = f[split]['labels']

            if TARGET_LABEL_TYPE not in labels_grp:
                print(f" Không tìm thấy nhãn '{TARGET_LABEL_TYPE}'. Bỏ qua.")
                continue

            mask_ds = labels_grp[TARGET_LABEL_TYPE]
            month_keys = sorted(list(sentinel_grp.keys()))

            for month in month_keys:
                print(f"    Đang xuất 10 kênh TIF: {month}...")

                images_ds = sentinel_grp[month]
                total_imgs = images_ds.shape[0]

                save_img_dir = os.path.join(OUTPUT_ROOT, split, month, "images")
                save_mask_dir = os.path.join(OUTPUT_ROOT, split, month, "masks")

                os.makedirs(save_img_dir, exist_ok=True)
                os.makedirs(save_mask_dir, exist_ok=True)

                # Loop xuất ảnh
                for i in tqdm(range(total_imgs), desc="   Processing", leave=False):
                    try:
                        # ==========================================================
                        # 1. XỬ LÝ ẢNH INPUT (GIỮ NGUYÊN 10 KÊNH, FLOAT64)
                        # ==========================================================
                        img_raw = images_ds[i]  # Shape gốc: (10, 128, 128)

                        # Chuyển shape về (128, 128, 10) chuẩn hiển thị (Height, Width, Channels)
                        img_out = np.transpose(img_raw, (1, 2, 0))

                        # ==========================================================
                        # 2. XỬ LÝ MASK (GIỮ NGUYÊN PNG 1 KÊNH, UINT8)
                        # ==========================================================
                        mask_raw = mask_ds[i]
                        # Chuyển từ One-hot (128,128,5) về Index (128,128)
                        mask_out = np.argmax(mask_raw, axis=-1).astype(np.uint8)

                        # ==========================================================
                        # 3. LƯU FILE
                        # ==========================================================
                        filename = f"patch_{i:05d}"

                        # Dùng Tifffile để lưu 10 kênh, không nén, giữ nguyên độ chính xác
                        tifffile.imwrite(os.path.join(save_img_dir, f"{filename}.tif"), img_out)

                        # Dùng OpenCV lưu mask PNG thông thường
                        cv2.imwrite(os.path.join(save_mask_dir, f"{filename}.png"), mask_out)

                    except Exception as e:
                        pass

    print(f"\n HOÀN TẤT! Đã xuất thành công ảnh TIF 10 kênh. Kiểm tra ổ D: {OUTPUT_ROOT}")


if __name__ == "__main__":
    extract_multispectral_to_D()
