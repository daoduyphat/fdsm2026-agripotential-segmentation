import os
import random
import torch
import numpy as np
import SimpleITK as sitk
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter


class FetalHeadMHADataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=(256, 256), augment=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        self.files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".mha") and f in os.listdir(label_dir)
        ])
        if len(self.files) == 0:
            raise RuntimeError(f"Không tìm thấy file .mha trùng giữa {image_dir} và {label_dir}")

    def __len__(self):
        return len(self.files)

    def read_mha(self, path):
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            arr = arr[0]
        return arr.astype(np.float32)

    def crop_nonzero(self, img, mask):
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return img, mask
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    def normalize(self, img):
        imin, imax = np.min(img), np.max(img)
        if imax - imin < 1e-6:
            return np.zeros_like(img)
        return (img - imin) / (imax - imin)

    def random_augment(self, image, mask):
        # Xoay ±30°
        angle = random.uniform(-30, 30)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Lật ngang/dọc ngẫu nhiên
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.3:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Zoom ngẫu nhiên
        scale = random.uniform(0.9, 1.1)
        h, w = image.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        image = TF.resize(image, (new_h, new_w))
        mask = TF.resize(mask, (new_h, new_w))
        image = TF.center_crop(image, (h, w))
        mask = TF.center_crop(mask, (h, w))

        # Làm mờ nhẹ hoặc thêm nhiễu
        if random.random() < 0.5:
            arr = image.squeeze().numpy()
            if random.random() < 0.5:
                arr = gaussian_filter(arr, sigma=random.uniform(0.5, 1.5))
            else:
                noise = np.random.normal(0, 0.02, arr.shape)
                arr = np.clip(arr + noise, 0, 1)
            image = torch.from_numpy(arr).unsqueeze(0).float()

        # Điều chỉnh sáng / tương phản
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        return image, mask

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = self.read_mha(os.path.join(self.image_dir, fname))
        mask = self.read_mha(os.path.join(self.label_dir, fname))

        # Cắt viền đen
        img, mask = self.crop_nonzero(img, mask)

        # Giảm nhiễu Gaussian
        img = gaussian_filter(img, sigma=1.0)

        # Chuẩn hóa
        img = self.normalize(img)
        mask = (mask > 0).astype(np.float32)

        # Resize và chuyển tensor
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        img = TF.resize(img, self.img_size)
        mask = TF.resize(mask, self.img_size)

        # Augment
        if self.augment:
            img, mask = self.random_augment(img, mask)

        return img.float(), mask.float()

    @staticmethod
    def compute_class_weights(dataset):
        total_pixels = 0
        pos_pixels = 0
        for i in range(len(dataset)):
            _, mask = dataset[i]
            pos_pixels += mask.sum().item()
            total_pixels += mask.numel()
        neg_pixels = total_pixels - pos_pixels
        pos_weight = neg_pixels / (pos_pixels + 1e-6)
        return torch.tensor(pos_weight, dtype=torch.float32)
