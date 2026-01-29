import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from graupp import GRAUpp
from preprocess_data import LandCoverDataset

# =========================
# GLOBAL CONFIG
# =========================
NUM_CLASSES = 5
IGNORE_INDEX = 255

CLASS_NAMES = ["Background", "Level 1", "Level 2", "Level 3", "Level 4"]

PIXEL_SIZE_METERS = 5
PIXEL_AREA_KM2 = (PIXEL_SIZE_METERS ** 2) / 1_000_000

COLOR_MAP = np.array([
    [0, 0, 0],  # 0: Đen
    [139, 0, 0],  # 1: Đỏ sậm
    [255, 165, 0],  # 2: Cam
    [255, 255, 0],  # 3: Vàng
    [50, 205, 50]  # 4: Xanh lá
], dtype=np.uint8)


# =========================
# METRICS & AREA
# =========================
def compute_metrics_per_image(pred, mask):
    valid_mask = mask != IGNORE_INDEX

    # Accuracy
    correct = np.sum(pred[valid_mask] == mask[valid_mask])
    total = np.sum(valid_mask)
    acc = correct / total if total > 0 else 0.0

    # IoU
    ious = []
    for cls in range(NUM_CLASSES):
        tp = np.logical_and(pred == cls, mask == cls).sum()
        fp = np.logical_and(pred == cls, mask != cls).sum()
        fn = np.logical_and(pred != cls, mask == cls).sum()
        denom = tp + fp + fn
        ious.append(np.nan if denom == 0 else tp / denom)

    return acc, np.nanmean(ious), ious


def compute_area(mask):
    areas = {}
    valid = mask != IGNORE_INDEX
    for cls in range(NUM_CLASSES):
        areas[f"L{cls}"] = (
                np.logical_and(mask == cls, valid).sum() * PIXEL_AREA_KM2
        )
    return areas


# =========================
# VISUALIZATION (4 CỘT)
# =========================
def save_test_visual(img, gt, pred, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Ảnh gốc (Lấy 3 kênh RGB từ ảnh 10 kênh)
    img_rgb = np.clip(img[:3].transpose(1, 2, 0), 0, 1)
    axs[0].imshow(img_rgb)
    axs[0].set_title("Input (RGB)")

    # 2. Nhãn Gốc (GT)
    axs[1].imshow(COLOR_MAP[gt])
    axs[1].set_title("Ground Truth")

    # 3. Nhãn Dự Đoán (Pred)
    axs[2].imshow(COLOR_MAP[pred])
    axs[2].set_title("Prediction")

    # 4. OVERLAY (GT + Pred)
    gt_rgb = COLOR_MAP[gt] / 255.0
    pred_rgb = COLOR_MAP[pred] / 255.0
    overlay = 0.5 * gt_rgb + 0.5 * pred_rgb
    axs[3].imshow(np.clip(overlay, 0, 1))
    axs[3].set_title("Overlay (GT + Pred)")

    for j in range(4):
        axs[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================
# TESTING LOOP
# =========================
def test_model(model_path, test_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on device: {device}")

    # 1. KHỞI TẠO MÔ HÌNH (10 Kênh giống train.py)
    model = GRAUpp(num_classes=NUM_CLASSES, input_channels=10).to(device)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    print(f"[INFO] Loading weights: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. KHỞI TẠO DATALOADER (Dùng chung class với train.py)
    test_ds = LandCoverDataset(test_dir, mode="test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"[INFO] Total test images: {len(test_ds)}")

    # Tạo thư mục lưu kết quả
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"test_run_{timestamp}")
    img_save_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(img_save_dir, exist_ok=True)

    results = []

    # Biến cộng dồn cho Bảng tổng kết
    total_area_gt = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    total_area_pr = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    all_preds, all_gts = [], []

    print("[INFO] Bắt đầu Kiểm thử (Inference)...")

    with torch.no_grad():
        for i, (img, mask) in enumerate(tqdm(test_loader)):
            img = img.to(device)
            mask = mask.to(device)

            out = model(img)
            if isinstance(out, list):
                out = out[-1]

            pred_tensor = torch.argmax(out, dim=1)

            # Đẩy về CPU để xử lý số liệu
            pred = pred_tensor.squeeze().cpu().numpy()
            gt = mask.squeeze().cpu().numpy()
            img_np = img.squeeze().cpu().numpy()

            all_preds.append(pred)
            all_gts.append(gt)

            # Tính Area
            area_gt = compute_area(gt)
            area_pr = compute_area(pred)

            # Cộng dồn vào tổng
            for k in area_gt:
                total_area_gt[k] += area_gt[k]
                total_area_pr[k] += area_pr[k]

            # Tính Metrics từng ảnh
            acc, miou, _ = compute_metrics_per_image(pred, gt)

            # Lấy tên file gốc (nếu có lưu trong Dataset)
            try:
                file_name = test_ds.names[i]
            except:
                file_name = f"image_{i:04d}"

            # Lưu vào danh sách báo cáo
            row_data = {
                "File_Name": file_name,
                "Accuracy": acc,
                "mIoU": miou
            }
            for cls in range(NUM_CLASSES):
                row_data[f"GT_Area_L{cls}_km2"] = area_gt[f"L{cls}"]
                row_data[f"Pred_Area_L{cls}_km2"] = area_pr[f"L{cls}"]
                row_data[f"Diff_L{cls}"] = area_pr[f"L{cls}"] - area_gt[f"L{cls}"]
            results.append(row_data)

            # Vẽ ảnh (Giới hạn lưu 50 ảnh đầu để tránh đầy ổ cứng)
            if i < 50:
                save_test_visual(
                    img_np, gt, pred,
                    os.path.join(img_save_dir, f"{file_name}_result.png")
                )

    # 3. TÍNH CHỈ SỐ TOÀN CỤC (GLOBAL)
    all_preds = np.stack(all_preds)
    all_gts = np.stack(all_gts)
    global_ious, global_miou = compute_iou_global(all_preds, all_gts)

    valid_mask = all_gts != IGNORE_INDEX
    global_acc = np.mean(all_preds[valid_mask] == all_gts[valid_mask])

    # 4. IN BẢNG BÁO CÁO TỔNG KẾT
    print("\n" + "=" * 70)
    print(f"               KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print(f"             (Tổng số ảnh: {len(test_ds)} - Tỉ lệ Pixel: {PIXEL_SIZE_METERS}m)")
    print("=" * 70)
    print(f"Global Accuracy : {global_acc:.2%}")
    print(f"Global mIoU     : {global_miou:.4f}")
    print("-" * 70)
    print(f"{'Class':<12} | {'IoU':<8} | {'Pred (km2)':<12} | {'GT (km2)':<12} | {'Diff'}")
    print("-" * 70)

    for i, cls in enumerate(CLASS_NAMES):
        iou_str = "N/A" if np.isnan(global_ious[i]) else f"{global_ious[i]:.4f}"
        ap = total_area_pr[f"L{i}"]
        ag = total_area_gt[f"L{i}"]
        diff = ap - ag
        print(f"{cls:<12} | {iou_str:<8} | {ap:<12.4f} | {ag:<12.4f} | {diff:+.4f}")
    print("=" * 70 + "\n")

    # 5. LƯU EXCEL
    df = pd.DataFrame(results)
    excel_path = os.path.join(save_dir, "Test_Report_Detailed.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"[DONE] Đã xuất báo cáo Excel và {min(50, len(test_ds))} ảnh minh họa tại: {save_dir}")


def compute_iou_global(preds, targets):
    # Hàm tính IoU toàn cục dùng lại từ train.py
    ious = []
    for cls in range(NUM_CLASSES):
        tp = np.logical_and(preds == cls, targets == cls).sum()
        fp = np.logical_and(preds == cls, targets != cls).sum()
        fn = np.logical_and(preds != cls, targets == cls).sum()
        denom = tp + fp + fn
        ious.append(np.nan if denom == 0 else tp / denom)
    return ious, np.nanmean(ious)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TRỎ VÀO FILE BEST MODEL TỪ THƯ MỤC OUTPUTS CỦA BẠN
    parser.add_argument("--model-path", default="outputs/YYYYMMDD_HHMMSS/best_model.pth")
    parser.add_argument("--test-dir", default="dataset_final/test")  # Thư mục chứa test.txt
    parser.add_argument("--output", default="outputs_test")
    args = parser.parse_args()

    # Nếu chưa truyền đường dẫn Model, tự động tìm model mới nhất
    if args.model_path == "outputs/YYYYMMDD_HHMMSS/best_model.pth" and os.path.exists("outputs"):
        runs = sorted([d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))])
        if runs:
            latest_model = os.path.join("outputs", runs[-1], "best_model.pth")
            if os.path.exists(latest_model):
                args.model_path = latest_model
                print(f"[INFO] Tự động chọn Model mới nhất: {latest_model}")

    test_model(args.model_path, args.test_dir, args.output)
