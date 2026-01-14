import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from graupp import GRAUpp
from preprocess_data import LandCoverDataset

CLASS_COLORS = np.array([
    [255, 255, 0],  # 0: Agriculture/Bareland (Vàng)
    [0, 0, 255],  # 1: Building
    [0, 255, 0],  # 2: Woodland
    [0, 255, 255],  # 3: Water
    [128, 128, 128]  # 4: Road
], dtype=np.uint8)


def compute_metrics(preds, targets, num_classes=5, smooth=1e-6):
    pred_cls = torch.argmax(preds, dim=1)

    correct_pixels = torch.eq(pred_cls, targets).float().sum()
    total_pixels = torch.numel(targets)
    accuracy = correct_pixels / total_pixels

    iou_list = []
    for cls in range(num_classes):
        pred_inds = torch.eq(pred_cls, cls)
        target_inds = torch.eq(targets, cls)

        intersection = (pred_inds & target_inds).float().sum()
        union = pred_inds.float().sum() + target_inds.float().sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        iou_list.append(iou.item())

    return accuracy.item(), np.mean(iou_list)


def calculate_area_km2(pred_mask, class_id, res_meter):
    pixel_count = np.sum(pred_mask == class_id)
    area_m2 = pixel_count * (res_meter ** 2)
    area_km2 = area_m2 / 1_000_000.0
    return area_km2


def visualize_result(img_tensor, mask_tensor, pred_mask, save_path):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    gt_color = CLASS_COLORS[mask_tensor.cpu().numpy()]
    pred_color = CLASS_COLORS[pred_mask]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(gt_color)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(pred_color)
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def test_model(model_path, test_dir, output_dir, res_meter=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GRAUpp(num_classes=5, input_channels=3, deep_supervision=True, attention='cbam')

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_ds = LandCoverDataset(
        os.path.join(test_dir, "images"),
        os.path.join(test_dir, "masks"),
        img_size=(128, 128),
        augment=False
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"Total test images: {len(test_ds)}")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(output_dir, f"test_result_{timestamp}")
    img_save_dir = os.path.join(save_dir, "images")
    os.makedirs(img_save_dir, exist_ok=True)

    results = []
    total_acc = []
    total_iou = []

    print(f"Running prediction (Resolution: {res_meter} m/px)...")

    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(test_loader)):
            img = img.to(device)
            mask = mask.to(device)

            output = model(img)

            acc, iou = compute_metrics(output, mask)
            total_acc.append(acc)
            total_iou.append(iou)

            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            file_name = test_ds.files[i]

            # CHỈ TÍNH DIỆN TÍCH CLASS 0 (VÀNG)
            agri_area = calculate_area_km2(pred_mask, class_id=0, res_meter=res_meter)

            row_data = {
                "File": file_name,
                "Accuracy": round(acc, 4),
                "IoU": round(iou, 4),
                "Agri_Area_km2": round(agri_area, 6)  # Chỉ xuất cột này
            }
            results.append(row_data)

            if i < 20:
                visualize_result(
                    img[0],
                    mask[0],
                    pred_mask,
                    os.path.join(img_save_dir, f"pred_{file_name}")
                )

    mean_acc = np.mean(total_acc)
    mean_iou = np.mean(total_iou)

    print("-" * 30)
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Mean IoU:      {mean_iou:.4f}")
    print("-" * 30)

    df = pd.DataFrame(results)

    total_row = {
        "File": "TOTAL / MEAN",
        "Accuracy": mean_acc,
        "IoU": mean_iou,
        "Agri_Area_km2": df["Agri_Area_km2"].sum()
    }

    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    excel_path = os.path.join(save_dir, "Test_Report_Agri_Only.xlsx")
    df.to_excel(excel_path, index=False)

    print(f"Report saved to: {excel_path}")
    print(f"Images saved to: {img_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="outputs_landcover/GRAUpp_best.pth")
    parser.add_argument("--test-dir", type=str, default="dataset_split/test")
    parser.add_argument("--output", type=str, default="outputs_test")
    parser.add_argument("--resolution", type=float, default=0.5)

    args = parser.parse_args()

    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found at {args.test_dir}")
        exit()

    test_model(args.model_path, args.test_dir, args.output, args.resolution)