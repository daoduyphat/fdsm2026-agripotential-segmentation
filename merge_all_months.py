import os
import shutil
from tqdm import tqdm

# --- CẤU HÌNH ---
# Folder cũ đang chứa các tháng riêng lẻ
SOURCE_ROOT = r"D:\DATASET_FULL_EXTRACTED"

# Folder mới sẽ chứa tất cả ảnh gộp lại
DEST_ROOT = r"D:\DATASET_MERGED"


def merge_data():
    if not os.path.exists(SOURCE_ROOT):
        print(f" Lỗi: Không tìm thấy folder nguồn: {SOURCE_ROOT}")
        return

    # Các tập cần xử lý
    splits = ['train', 'val', 'test']

    print(f" BẮT ĐẦU GỘP DỮ LIỆU TỪ 11 THÁNG...")
    print(f" Nguồn: {SOURCE_ROOT}")
    print(f" Đích:  {DEST_ROOT}")

    for split in splits:
        split_src = os.path.join(SOURCE_ROOT, split)
        if not os.path.exists(split_src):
            continue

        print(f"\n--- ĐANG XỬ LÝ TẬP: {split.upper()} ---")

        # Tạo thư mục đích: D:\DATASET_MERGED\train\images
        dest_img_dir = os.path.join(DEST_ROOT, split, "images")
        dest_mask_dir = os.path.join(DEST_ROOT, split, "masks")
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_mask_dir, exist_ok=True)

        # Duyệt qua các folder tháng (01_january, 02_february...)
        month_folders = [d for d in os.listdir(split_src) if os.path.isdir(os.path.join(split_src, d))]

        for month in month_folders:
            month_path = os.path.join(split_src, month)

            # Đường dẫn folder ảnh/mask con
            src_img_dir = os.path.join(month_path, "images")
            src_mask_dir = os.path.join(month_path, "masks")

            if not os.path.exists(src_img_dir): continue

            # Lấy danh sách ảnh
            files = os.listdir(src_img_dir)

            # Copy và đổi tên
            for filename in tqdm(files, desc=f"   Gộp {month}", leave=False):
                if not filename.lower().endswith(('.jpg', '.png')): continue

                # --- XỬ LÝ ẢNH ---
                # Tên mới: 01_january_img_00001.jpg
                new_filename = f"{month}_{filename}"

                old_img_path = os.path.join(src_img_dir, filename)
                new_img_path = os.path.join(dest_img_dir, new_filename)

                shutil.copy2(old_img_path, new_img_path)

                # --- XỬ LÝ MASK ---
                # Mask tương ứng phải là .png
                mask_name = os.path.splitext(filename)[0] + ".png"
                old_mask_path = os.path.join(src_mask_dir, mask_name)

                # Nếu không tìm thấy mask .png thì thử tìm chính nó (đề phòng)
                if not os.path.exists(old_mask_path):
                    old_mask_path = os.path.join(src_mask_dir, filename)

                if os.path.exists(old_mask_path):
                    # Tên mask mới cũng phải có tháng để khớp với ảnh
                    # 01_january_img_00001.png
                    new_mask_filename = f"{month}_{os.path.splitext(filename)[0]}.png"
                    new_mask_path = os.path.join(dest_mask_dir, new_mask_filename)

                    shutil.copy2(old_mask_path, new_mask_path)

    print(f"\nHOÀN TẤT! Bạn đã có bộ dữ liệu khổng lồ tại: {DEST_ROOT}")
    print("Cấu trúc mới:")
    print(f"   {DEST_ROOT}/train/images (Chứa tất cả các tháng)")
    print(f"   {DEST_ROOT}/train/masks")


if __name__ == "__main__":
    merge_data()