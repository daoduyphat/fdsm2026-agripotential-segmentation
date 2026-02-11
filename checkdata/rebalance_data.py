import os
import random

# ==========================================
# CẤU HÌNH
# ==========================================
DATA_ROOT = "dataset_final"
OUTPUT_DIR = "dataset_final"


def rebalance_deduplicate():
    print("=== BẮT ĐẦU TÁI CẤU TRÚC VÀ KHỬ TRÙNG LẶP (V2) ===")

    # Dùng Dictionary để khử trùng: Key = Tên file, Value = Nguồn
    # Nếu tên file lặp lại, nó sẽ không được thêm mới
    unique_samples = {}

    # Thứ tự ưu tiên: Nếu ảnh có cả ở train và val, sẽ ưu tiên giữ bản ở 'train'
    source_folders = ["train", "val", "test"]

    duplicates_count = 0

    for folder in source_folders:
        img_dir = os.path.join(DATA_ROOT, folder, "images")
        mask_dir = os.path.join(DATA_ROOT, folder, "masks")

        if not os.path.exists(img_dir):
            continue

        # Quét file
        files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
        print(f"[SCAN] Quét folder '{folder}'... Tìm thấy {len(files)} file.")

        for f_name in files:
            base_name = f_name.replace(".tif", "")

            # Kiểm tra mask tồn tại
            mask_path = os.path.join(mask_dir, f"{base_name}.png")
            if os.path.exists(mask_path):
                # LOGIC KHỬ TRÙNG QUAN TRỌNG:
                if base_name not in unique_samples:
                    # Nếu chưa có -> Thêm vào
                    unique_samples[base_name] = folder
                else:
                    # Nếu đã có rồi -> Bỏ qua (Đây là file trùng lặp)
                    duplicates_count += 1

    # Chuyển đổi về dạng List để chia
    # List các tuple: [('abc', 'train'), ('xyz', 'val'), ...]
    final_list = list(unique_samples.items())

    print(f"\n>>> KẾT QUẢ QUÉT:")
    print(f"  - Số file trùng lặp bị loại bỏ: {duplicates_count}")
    print(f"  - Tổng số file duy nhất hợp lệ  : {len(final_list)}")

    if len(final_list) == 0:
        print("[LỖI] Không tìm thấy dữ liệu.")
        return

    # Chia lại tỷ lệ 70-20-10
    print("\n[INFO] Đang xáo trộn và chia lại 70-20-10...")
    random.seed(42)
    random.shuffle(final_list)

    total = len(final_list)
    n_train = int(total * 0.7)
    n_val = int(total * 0.2)

    train_set = final_list[:n_train]
    val_set = final_list[n_train: n_train + n_val]
    test_set = final_list[n_train + n_val:]

    print(f"  - Train: {len(train_set)}")
    print(f"  - Val  : {len(val_set)}")
    print(f"  - Test : {len(test_set)}")

    # Ghi file
    def write_file(fname, data):
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w") as f:
            for name, source in data:
                f.write(f"{name},{source}\n")
        print(f"  -> Đã ghi: {path}")

    write_file("train.txt", train_set)
    write_file("val.txt", val_set)
    write_file("test.txt", test_set)

    print("\n=== HOÀN TẤT V2. HÃY CHẠY LẠI VERIFY_DATA ĐỂ KIỂM TRA ===")


if __name__ == "__main__":
    rebalance_deduplicate()