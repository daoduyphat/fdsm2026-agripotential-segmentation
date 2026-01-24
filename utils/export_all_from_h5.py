import h5py
import os
import cv2
import numpy as np
from tqdm import tqdm

# --- CẤU HÌNH ---
H5_PATH = r"D:\git\agrivison\dataset\dataset.h5"
OUTPUT_ROOT = r"D:\git\agrivison\dataset\DATASET_FULL_EXTRACTED"
TARGET_LABEL_TYPE = "field"


def normalize_rgb(img_10ch):
    # 1. Lấy 3 kênh RGB (Kênh 0, 1, 2)
    # Shape đầu vào: (10, 128, 128) -> Lấy (3, 128, 128)
    rgb = img_10ch[0:3]

    # 2. Chuyển về (128, 128, 3)
    rgb = np.transpose(rgb, (1, 2, 0))

    # 3. Chuẩn hóa Min-Max (Vẫn thực hiện trên float64 để chính xác)
    max_val = rgb.max()
    min_val = rgb.min()

    if max_val - min_val > 1e-5:  # Tránh chia cho 0
        rgb = (rgb - min_val) / (max_val - min_val)
    else:
        rgb = np.zeros_like(rgb)  # Nếu ảnh đen xì hoặc lỗi thì trả về đen

    # 4. Chuyển sang uint8 (0-255) -> ĐÂY LÀ BƯỚC QUAN TRỌNG
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # 5. Convert RGB -> BGR (Lúc này OpenCV chịu nhận vì là uint8)
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    return bgr


def export_everything_to_D():
    if not os.path.exists(H5_PATH):
        print(f"Lỗi: Không tìm thấy file {H5_PATH}")
        return

    print(f" Đang đọc file H5: {H5_PATH}")
    print(f" Dữ liệu sẽ được bung ra tại: {OUTPUT_ROOT}\n")

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
                print(f"    Đang lưu: {month}...")

                images_ds = sentinel_grp[month]
                total_imgs = images_ds.shape[0]

                save_img_dir = os.path.join(OUTPUT_ROOT, split, month, "images")
                save_mask_dir = os.path.join(OUTPUT_ROOT, split, month, "masks")

                os.makedirs(save_img_dir, exist_ok=True)
                os.makedirs(save_mask_dir, exist_ok=True)

                # Loop xuất ảnh
                for i in tqdm(range(total_imgs), desc="   Processing", leave=False):
                    try:
                        # 1. Ảnh vệ tinh -> JPG (Đã fix lỗi CV_64F)
                        img_raw = images_ds[i]
                        img_out = normalize_rgb(img_raw)

                        # 2. Mask -> PNG
                        mask_raw = mask_ds[i]
                        # One-hot to Index
                        mask_out = np.argmax(mask_raw, axis=-1).astype(np.uint8)

                        # 3. Lưu file
                        filename = f"patch_{i:05d}"
                        cv2.imwrite(os.path.join(save_img_dir, f"{filename}.jpg"), img_out)
                        cv2.imwrite(os.path.join(save_mask_dir, f"{filename}.png"), mask_out)
                    except Exception as e:
                        pass

    print(f"\n HOÀN TẤT! Đã fix lỗi float64. Kiểm tra ổ D: {OUTPUT_ROOT}")


if __name__ == "__main__":
    export_everything_to_D()