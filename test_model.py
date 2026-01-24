import os
import argparse
import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Import models
from graupp import GRAUpp
from preprocess_data import LandCoverDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CLASS_COLORS = np.array([
    [0, 0, 0],  # Class 0: Background
    [255, 0, 0],  # Class 1: Very Low (Đỏ)
    [255, 165, 0],  # Class 2: Low (Cam)
    [255, 255, 0],  # Class 3: Average (Vàng)
    [0, 255, 0],  # Class 4: High (Xanh lá)
], dtype=np.uint8)


def compute_metrics(preds, targets, num_classes=5, smooth=1e-6):
    pred_cls = torch.argmax(preds, dim=1)

    correct_pixels = torch.eq(pred_cls, targets).float().sum()
    total_pixels = torch.numel(targets)
    accuracy = correct_pixels / total_pixels

    iou_list = []
    # Lưu ý: Thường ta quan tâm IoU của các lớp Agri hơn là lớp Background
    for cls in range(num_classes):
        pred_inds = torch.eq(pred_cls, cls)
        target_inds = torch.eq(targets, cls)

        intersection = (pred_inds & target_inds).float().sum()
        union = pred_inds.float().sum() + target_inds.float().sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        iou_list.append(iou.item())

    return accuracy.item(), np.mean(iou_list)


def calculate_total_agri_area_km2(pred_mask, res_meter):
    """
    Tính tổng diện tích đất nông nghiệp (Tổng các pixel có class > 0).
    Class 0 là Background/Unlabeled -> Bỏ qua.
    """
    # Đếm tất cả pixel KHÔNG PHẢI là 0
    pixel_count = np.sum(pred_mask > 0)

    area_m2 = pixel_count * (res_meter ** 2)
    area_km2 = area_m2 / 1_000_000.0
    return area_km2


def visualize_result(img_tensor, mask_tensor, pred_mask, save_path):
    # Denormalize ảnh input
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # Map màu
    gt_color = CLASS_COLORS[mask_tensor.cpu().numpy()]
    pred_color = CLASS_COLORS[pred_mask]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[0].axis('off')

    axs[1].imshow(gt_color)
    axs[1].set_title("Ground Truth\n(Black=Background)")
    axs[1].axis('off')

    axs[2].imshow(pred_color)
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def test_model(model_path, test_dir, output_dir, res_meter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model definition
    model = GRAUpp(num_classes=5, input_channels=3, deep_supervision=True, attention='cbam')

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model weights from: {model_path}")
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
            if isinstance(output, list):
                output = output[-1]

            acc, iou = compute_metrics(output, mask)
            total_acc.append(acc)
            total_iou.append(iou)

            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            try:
                file_name = test_ds.valid_files[i]
            except:
                file_name = f"image_{i:04d}.jpg"

            agri_area = calculate_total_agri_area_km2(pred_mask, res_meter=res_meter)

            row_data = {
                "File": file_name,
                "Accuracy": round(acc, 4),
                "IoU": round(iou, 4),
                "Total_Agri_Area_km2": round(agri_area, 6)
            }
            results.append(row_data)

            if i < 20:
                visualize_result(
                    img[0],
                    mask[0],
                    pred_mask,
                    os.path.join(img_save_dir, f"pred_{file_name}.png")
                )

    mean_acc = np.mean(total_acc)
    mean_iou = np.mean(total_iou)
    total_agri_area_all = sum(d['Total_Agri_Area_km2'] for d in results)

    print("-" * 30)
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Mean IoU:      {mean_iou:.4f}")
    print(f"Total Agricultural Area Detected: {total_agri_area_all:.4f} km2")
    print("-" * 30)

    df = pd.DataFrame(results)

    # Dòng tổng kết
    total_row = {
        "File": "TOTAL / MEAN",
        "Accuracy": mean_acc,
        "IoU": mean_iou,
        "Total_Agri_Area_km2": total_agri_area_all
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    excel_path = os.path.join(save_dir, "Test_Report_Agri_Total.xlsx")
    df.to_excel(excel_path, index=False)

    print(f"Report saved to: {excel_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="outputs_landcover/GRAUpp_best.pth")
    parser.add_argument("--test-dir", type=str, default="dataset/test")
    parser.add_argument("--output", type=str, default="outputs_test")
    parser.add_argument("--resolution", type=float, default=5.0)  # 5m/px

    args = parser.parse_args()

    # Auto-find model logic (giữ nguyên)
    if not os.path.exists(args.model_path) and os.path.exists("outputs_landcover"):
        runs = sorted([d for d in os.listdir("outputs_landcover") if d.startswith("run_")])
        if runs:
            latest_model = os.path.join("outputs_landcover", runs[-1], "GRAUpp_best.pth")
            if os.path.exists(latest_model):
                args.model_path = latest_model
                print(f"-> Auto-detected model: {args.model_path}")

    if not os.path.exists(args.test_dir):
        print(f"Error: Test dir {args.test_dir} not found.")
        exit()

    test_model(args.model_path, args.test_dir, args.output, args.resolution)