import os
import cv2
import numpy as np
import tifffile
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.colors

# =========================
# GLOBAL CONFIG
# =========================
NUM_CLASSES = 5

labels_cmap = matplotlib.colors.ListedColormap([
    "#000000",  # 0 Background
    "#8B0000",  # 1 Very Low
    "#FFA500",  # 2 Low
    "#FFFF00",  # 3 Average
    "#32CD32"  # 4 High
])


# =========================
# 1. DATASET CLASS (ĐÃ SỬA LỖI ĐỌC SOURCE)
# =========================
class LandCoverDataset(Dataset):
    def __init__(self, data_root, mode="train", augment=True):
        """
        data_root: Đường dẫn chứa ẢNH GỐC (Train/Val/Test)
        mode: 'train', 'val', hoặc 'test'
        """
        self.data_root = data_root
        self.mode = mode

        # 1. Đọc file txt
        txt_path = os.path.join(data_root, f"{mode}.txt")

        # Fallback: Nếu không thấy trong data_root, thử tìm ở thư mục hiện tại
        if not os.path.exists(txt_path):
            txt_path = f"{mode}.txt"

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file danh sách {mode}.txt tại {data_root} hoặc thư mục gốc.")

        print(f"[INIT] Đang đọc danh sách từ: {txt_path}")
        with open(txt_path, "r") as f:
            lines = f.read().splitlines()

        # ============================================================
        # [QUAN TRỌNG] PARSE DANH SÁCH & KIỂM TRA TỒN TẠI
        # ============================================================
        self.samples = []
        missing_count = 0
        last_missing_info = None

        print(f"[CHECK] Đang quét {len(lines)} dòng trong danh sách tập {mode}...")

        for line in lines:
            line = line.strip()
            if not line: continue

            # [FIX] Xử lý format "tên_file,nguồn" (do script rebalance tạo ra)
            if "," in line:
                name, source_folder = line.split(",")
            else:
                # Format cũ (fallback nếu file txt chưa chuẩn)
                name = line
                source_folder = mode

                # Tạo đường dẫn vật lý dựa trên SOURCE FOLDER
            img_path = os.path.join(self.data_root, source_folder, "images", f"{name}.tif")
            mask_path = os.path.join(self.data_root, source_folder, "masks", f"{name}.png")

            # Kiểm tra tồn tại
            if os.path.exists(img_path) and os.path.exists(mask_path):
                # Lưu lại cả tên và nguồn gốc để dùng trong __getitem__
                self.samples.append((name, source_folder))
            else:
                missing_count += 1
                last_missing_info = f"{name} (in folder {source_folder})"

        if missing_count > 0:
            print(f"[WARNING] Đã loại bỏ {missing_count} file bị thiếu ảnh/mask thực tế.")
            print(f"          (Ví dụ file thiếu: {last_missing_info})")

        print(f"[INFO] Final load tập {mode.upper()}: {len(self.samples)} mẫu hợp lệ.")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Lỗi: Không load được mẫu nào! Vui lòng kiểm tra lại đường dẫn trong file .txt hoặc data_root.")

        # Augmentation
        if augment and mode == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Lấy thông tin từ danh sách đã parse (bao gồm cả nguồn gốc)
        name, source_folder = self.samples[idx]

        # Xây dựng đường dẫn dựa trên nguồn gốc thực tế (source_folder)
        # VD: File thuộc tập VAL nhưng nằm vật lý ở folder TRAIN -> code sẽ tìm trong TRAIN
        img_path = os.path.join(self.data_root, source_folder, "images", f"{name}.tif")
        mask_path = os.path.join(self.data_root, source_folder, "masks", f"{name}.png")

        try:
            img = tifffile.imread(img_path).astype(np.float32)
        except Exception:
            # Fallback an toàn (trả về ảnh đen thay vì crash)
            print(f"[ERR] Lỗi đọc file: {img_path}")
            img = np.zeros((128, 128, 10), dtype=np.float32)

        min_v, max_v = img.min(), img.max()
        if max_v > min_v:
            img = (img - min_v) / (max_v - min_v)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((128, 128), dtype=np.uint8)

        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()


# =========================
# 2. HÀM KIỂM TRA (CHECKER)
# =========================
def check_existing_splits(data_root):
    # Chỉ check file txt vì file ảnh giờ nằm rải rác
    all_ok = True
    print(f"\n--- KIỂM TRA FILE LIST TẠI: '{data_root}' ---")
    for f in ["train.txt", "val.txt", "test.txt"]:
        fpath = os.path.join(data_root, f)
        if os.path.exists(fpath):
            # Đếm số dòng
            with open(fpath, 'r') as file:
                lines = len(file.readlines())
            print(f"   {f:<10}: OK ({lines} dòng)")
        else:
            print(f"   {f:<10}: KHÔNG TÌM THẤY")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    ROOT = "dataset_final"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[DEBUG] Script đang chạy tại: {current_dir}")

    if check_existing_splits(ROOT):
        print("\n--> File list ổn. Đang test load mẫu Val...")
        try:
            # Test với tập Val (tập dễ lỗi nhất do rebalance)
            ds = LandCoverDataset(ROOT, mode="val", augment=False)
            if len(ds) > 0:
                img, mask = ds[0]
                print("[SUCCESS] Test load mẫu thành công!")
                print(f"  - Image Shape: {img.shape}")
                print(f"  - Mask Shape : {mask.shape}")
                print(f"  - Unique Labels: {np.unique(mask.numpy()).tolist()}")
        except Exception as e:
            print(f"[ERROR] Lỗi khi đọc mẫu: {e}")
    else:
        print("\n[CẢNH BÁO] Kiểm tra lại tên thư mục hoặc vị trí đặt file code!")