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
# 1. DATASET CLASS
# =========================
class LandCoverDataset(Dataset):
    def __init__(self, data_root, mode="train", augment=True):
        self.data_root = data_root
        self.mode = mode

        # 1. Dinh nghia duong dan
        txt_path = os.path.join(data_root, f"{mode}.txt")
        self.image_dir = os.path.join(data_root, mode, "images")
        self.mask_dir = os.path.join(data_root, mode, "masks")

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file danh sách {txt_path}")

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy thư mục ảnh {self.image_dir}")

        # 2. Doc danh sach tu file TXT
        with open(txt_path, "r") as f:
            raw_names = f.read().splitlines()

        # ============================================================
        # [QUAN TRỌNG] BỘ LỌC KIỂM TRA TỒN TẠI (VALIDATION CHECK)
        # ============================================================
        self.names = []
        missing_count = 0
        last_missing_name = None  # [FIX] Khởi tạo biến để lưu tên file lỗi

        print(f"[INIT] Đang kiểm tra {len(raw_names)} file trong tập {mode}...")

        for name in raw_names:
            img_p = os.path.join(self.image_dir, f"{name}.tif")
            mask_p = os.path.join(self.mask_dir, f"{name}.png")

            # Chi them vao danh sach neu CA 2 FILE (anh va mask) deu ton tai
            if os.path.exists(img_p) and os.path.exists(mask_p):
                self.names.append(name)
            else:
                missing_count += 1
                last_missing_name = name  # [FIX] Lưu lại tên file bị thiếu tại đây

        # In bao cao neu co file thieu
        if missing_count > 0:
            print(f"[WARNING] Đã loại bỏ {missing_count} file có trong .txt nhưng thiếu ảnh/mask thực tế!")
            # [FIX] Dùng biến last_missing_name thay vì name
            print(f"          (Ví dụ file thiếu: {last_missing_name})")

        print(f"[INFO] Final load tập {mode.upper()}: {len(self.names)} mẫu hợp lệ.")
        # ============================================================

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
        return len(self.names)

    def __getitem__(self, idx):
        base = self.names[idx]
        img_path = os.path.join(self.image_dir, f"{base}.tif")
        mask_path = os.path.join(self.mask_dir, f"{base}.png")

        try:
            img = tifffile.imread(img_path).astype(np.float32)
        except Exception:
            print(f"[ERROR] File lỗi format: {img_path}")
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
    required_txt = ["train.txt", "val.txt", "test.txt"]
    print(f"\n--- KIỂM TRA DỮ LIỆU TẠI: '{data_root}' ---")
    all_ok = True
    for fname in required_txt:
        fpath = os.path.join(data_root, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                count = len(f.read().splitlines())
            print(f"   {fname:<10} : OK ({count} mẫu)")
        else:
            print(f"   {fname:<10} : KHÔNG TÌM THẤY")
            all_ok = False

    for folder in ["train", "val", "test"]:
        dpath = os.path.join(data_root, folder)
        if os.path.exists(dpath):
            print(f"   Folder {folder:<5}: OK")
        else:
            print(f"   Folder {folder:<5}: KHÔNG TÌM THẤY")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    ROOT = "dataset_final"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[DEBUG] Script đang chạy tại: {current_dir}")

    if check_existing_splits(ROOT):
        print("\n--> Cấu trúc dữ liệu chuẩn. Đang thử load mẫu...")
        try:
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