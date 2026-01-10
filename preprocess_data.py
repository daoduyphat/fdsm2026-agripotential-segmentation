import os
import cv2
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class LandCoverDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=(512, 512), augment=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        # Lấy danh sách file ảnh (hỗ trợ jpg, png, tif)
        self.files = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])

        # Định nghĩa phép biến đổi (Augmentation) chuẩn
        if self.augment:
            self.transform = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_file = self.files[index]
        img_path = os.path.join(self.image_dir, img_file)

        # --- XỬ LÝ TÌM FILE MASK ---
        # Lấy tên file không đuôi (vd: image_01)
        filename_no_ext = os.path.splitext(img_file)[0]

        # Ưu tiên 1: Tìm mask trùng tên đuôi .png (vd: image_01.png)
        mask_path = os.path.join(self.label_dir, filename_no_ext + ".png")

        # Ưu tiên 2: Nếu không thấy, tìm mask có đuôi _m.png (vd: image_01_m.png)
        if not os.path.exists(mask_path):
            mask_path_m = os.path.join(self.label_dir, filename_no_ext + "_m.png")
            if os.path.exists(mask_path_m):
                mask_path = mask_path_m
        # ---------------------------

        # Đọc ảnh
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Đọc mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Nếu mask vẫn None (do lỗi file hoặc đường dẫn), tạo mask đen để không crash (nhưng in cảnh báo)
        if mask is None:
            # print(f"Warning: Mask not found for {img_file}. Creating empty mask.")
            mask = np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)

        # Áp dụng Augmentation
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].long()  # Mask phải là kiểu Long cho CrossEntropy

        return image, mask