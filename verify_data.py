import os
import sys

# ==========================================
# CẤU HÌNH
# ==========================================
DATA_ROOT = "dataset_final"  # Tên folder dữ liệu gốc
REQUIRED_FILES = ["train.txt", "val.txt", "test.txt"]


# Màu sắc cho Terminal đẹp hơn
class Colors:
    OK = '\033[92m'  # Xanh lá
    FAIL = '\033[91m'  # Đỏ
    WARN = '\033[93m'  # Vàng
    RESET = '\033[0m'  # Trắng


def check_dataset_health():
    print(f"=== BẮT ĐẦU KIỂM TRA DỮ LIỆU TẠI: '{DATA_ROOT}' ===\n")

    # 1. KIỂM TRA SỰ TỒN TẠI CỦA FILE TXT
    # ---------------------------------------------------------
    missing_txt = []
    split_data = {}  # Lưu nội dung của từng tập để so sánh sau này

    for fname in REQUIRED_FILES:
        fpath = os.path.join(DATA_ROOT, fname)
        if not os.path.exists(fpath):
            print(f"{Colors.FAIL}[MISSING] Không tìm thấy file: {fname}{Colors.RESET}")
            missing_txt.append(fname)
        else:
            with open(fpath, "r") as f:
                # Đọc và loại bỏ dòng trống
                lines = [line.strip() for line in f.read().splitlines() if line.strip()]
                split_data[fname] = lines
            print(f"{Colors.OK}[OK] Đã đọc {fname}: {len(lines)} dòng.{Colors.RESET}")

    if missing_txt:
        print(f"\n{Colors.FAIL}--> DỪNG KIỂM TRA: Thiếu file danh sách!{Colors.RESET}")
        return

    # 2. KIỂM TRA ĐỊNH DẠNG VÀ FILE VẬT LÝ
    # ---------------------------------------------------------
    print("\n--- KIỂM TRA FILE VẬT LÝ (PHYSICAL CHECK) ---")

    total_physical_missing = 0
    all_filenames = {}  # Dùng để check trùng lặp (Data Leakage)

    for fname, lines in split_data.items():
        subset_name = fname.replace(".txt", "").upper()
        print(f"Đang quét tập {subset_name}...")

        subset_filenames = set()
        missing_count = 0

        for line_idx, line in enumerate(lines):
            # Check format "tên_file,nguồn"
            if "," not in line:
                print(
                    f"  {Colors.FAIL}[FORMAT ERROR] Dòng {line_idx + 1} trong {fname} sai định dạng: '{line}'{Colors.RESET}")
                continue

            name, source = line.split(",")

            # Lưu tên file để check trùng lặp ở bước 3
            subset_filenames.add(name)

            # Check file ảnh và mask trên ổ cứng
            img_path = os.path.join(DATA_ROOT, source, "images", f"{name}.tif")
            mask_path = os.path.join(DATA_ROOT, source, "masks", f"{name}.png")

            if not os.path.exists(img_path):
                print(f"  {Colors.FAIL}[MISSING IMG] Không thấy ảnh: {img_path}{Colors.RESET}")
                missing_count += 1
            if not os.path.exists(mask_path):
                print(f"  {Colors.FAIL}[MISSING MASK] Không thấy mask: {mask_path}{Colors.RESET}")
                missing_count += 1

        all_filenames[subset_name] = subset_filenames

        if missing_count == 0:
            print(f"  -> {Colors.OK}Tất cả file ảnh/mask đều tồn tại.{Colors.RESET}")
        else:
            print(f"  -> {Colors.FAIL}CẢNH BÁO: Thiếu {missing_count} file vật lý!{Colors.RESET}")
            total_physical_missing += missing_count

    # 3. KIỂM TRA RÒ RỈ DỮ LIỆU (DATA LEAKAGE)
    # ---------------------------------------------------------
    print("\n--- KIỂM TRA RÒ RỈ DỮ LIỆU (DATA LEAKAGE) ---")
    train_set = all_filenames["TRAIN"]
    val_set = all_filenames["VAL"]
    test_set = all_filenames["TEST"]

    leak_tv = train_set.intersection(val_set)
    leak_vt = val_set.intersection(test_set)
    leak_tt = train_set.intersection(test_set)

    has_leak = False
    if leak_tv:
        print(f"{Colors.FAIL}[LEAK] Có {len(leak_tv)} ảnh vừa ở TRAIN vừa ở VAL!{Colors.RESET}")
        has_leak = True
    if leak_vt:
        print(f"{Colors.FAIL}[LEAK] Có {len(leak_vt)} ảnh vừa ở VAL vừa ở TEST!{Colors.RESET}")
        has_leak = True
    if leak_tt:
        print(f"{Colors.FAIL}[LEAK] Có {len(leak_tt)} ảnh vừa ở TRAIN vừa ở TEST!{Colors.RESET}")
        has_leak = True

    if not has_leak:
        print(f"{Colors.OK}[OK] Tuyệt vời! Dữ liệu được phân chia độc lập hoàn toàn.{Colors.RESET}")

    # 4. THỐNG KÊ TỔNG QUÁT (STATISTICS)
    # ---------------------------------------------------------
    print("\n--- BÁO CÁO TỔNG HỢP ---")
    n_train = len(train_set)
    n_val = len(val_set)
    n_test = len(test_set)
    total = n_train + n_val + n_test

    if total == 0:
        print("Không có dữ liệu để thống kê.")
        return

    print(f"{'TẬP':<10} | {'SỐ LƯỢNG':<10} | {'TỶ LỆ THỰC TẾ':<15} | {'MỤC TIÊU (70-20-10)'}")
    print("-" * 65)
    print(f"{'TRAIN':<10} | {n_train:<10} | {n_train / total * 100:.2f}%{'':<9} | ~70%")
    print(f"{'VAL':<10} | {n_val:<10} | {n_val / total * 100:.2f}%{'':<9} | ~20%")
    print(f"{'TEST':<10} | {n_test:<10} | {n_test / total * 100:.2f}%{'':<9} | ~10%")
    print("-" * 65)
    print(f"{'TỔNG':<10} | {total:<10} | 100.00%")

    # KẾT LUẬN CUỐI CÙNG
    print("\n=== KẾT LUẬN ===")
    if total_physical_missing == 0 and not has_leak and total > 0:
        print(f"{Colors.OK} DỮ LIỆU CHUẨN 100%. SẴN SÀNG ĐỂ TRAIN!{Colors.RESET}")
    else:
        print(f"{Colors.FAIL} DỮ LIỆU CÓ VẤN ĐỀ. VUI LÒNG KIỂM TRA LẠI CÁC LỖI Ở TRÊN.{Colors.RESET}")


if __name__ == "__main__":
    check_dataset_health()