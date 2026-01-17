import os
import cv2
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class LandCoverDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=(128, 128), augment=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir} or {label_dir}")

        self.files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))
        ])

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

        filename_no_ext = os.path.splitext(img_file)[0]

        mask_candidates = [
            os.path.join(self.label_dir, filename_no_ext + ".png"),
            os.path.join(self.label_dir, filename_no_ext + "_m.png")
        ]

        mask_path = None
        for path in mask_candidates:
            if os.path.exists(path):
                mask_path = path
                break

        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image: {img_file}")

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].long()

        return image, mask