import os
import argparse
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# === MODULES CỦA BẠN ===
# Đảm bảo file graupp.py và preprocess_data.py nằm cùng thư mục
try:
    from graupp import GRAUpp
    from preprocess_data import LandCoverDataset
except ImportError:
    raise ImportError("Không tìm thấy module 'graupp' hoặc 'preprocess_data'. Hãy kiểm tra lại thư mục làm việc.")


# ==========================================
# CẤU HÌNH & HẰNG SỐ (CONFIGURATION)
# ==========================================
class Config:
    NUM_CLASSES = 5
    IGNORE_INDEX = 255
    CLASS_NAMES = ["Background", "Level 1", "Level 2", "Level 3", "Level 4"]

    # Màu sắc chuẩn (R, G, B)
    COLOR_MAP = np.array([
        [0, 0, 0],  # 0: Background (Đen)
        [139, 0, 0],  # 1: Level 1 (Đỏ đậm)
        [255, 165, 0],  # 2: Level 2 (Cam)
        [255, 255, 0],  # 3: Level 3 (Vàng)
        [50, 205, 50]  # 4: Level 4 (Xanh lá)
    ], dtype=np.uint8)

    PIXEL_SIZE_METERS = 5
    PIXEL_AREA_KM2 = (PIXEL_SIZE_METERS ** 2) / 1_000_000


warnings.filterwarnings("ignore")


# ==========================================
# HỆ THỐNG LOGGING (GHI NHẬT KÝ)
# ==========================================
def setup_logger(output_dir):
    log_file = os.path.join(output_dir, "test_log.txt")

    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ==========================================
# CÔNG CỤ TÍNH TOÁN METRIC (METRIC TRACKER)
# ==========================================
class MetricTracker:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        self.total_samples = 0

    def update(self, preds, targets):
        """Cập nhật Confusion Matrix từ batch dự đoán."""
        preds = preds.flatten()
        targets = targets.flatten()

        # Loại bỏ các pixel IGNORE_INDEX
        valid_mask = (targets != Config.IGNORE_INDEX)
        preds = preds[valid_mask]
        targets = targets[valid_mask]

        if len(targets) == 0:
            return

        # Tính Confusion Matrix nhanh bằng bincount
        # Mẹo: CM[gt, pred] += 1
        x = preds + self.num_classes * targets
        bincount_2d = np.bincount(x.astype(np.int64), minlength=self.num_classes ** 2)

        if bincount_2d.size == self.num_classes ** 2:
            conf = bincount_2d.reshape((self.num_classes, self.num_classes))
            self.confusion_matrix += conf

    def compute(self):
        """Tính toán các chỉ số IoU, Accuracy từ Confusion Matrix."""
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        # IoU per class
        with np.errstate(divide='ignore', invalid='ignore'):
            iou_class = tp / (tp + fp + fn)
        iou_class = np.nan_to_num(iou_class)  # Thay NaN bằng 0

        # Mean IoU
        miou = np.mean(iou_class)

        # Global Accuracy
        total_pixels = self.confusion_matrix.sum()
        acc = tp.sum() / total_pixels if total_pixels > 0 else 0.0

        return {
            "mIoU": miou,
            "Accuracy": acc,
            "IoU_Class": iou_class,
            "Confusion_Matrix": self.confusion_matrix
        }


# ==========================================
# CORE EVALUATOR CLASS
# ==========================================
class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tạo thư mục output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(args.output, f"TEST_RUN_{timestamp}")
        self.vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

        self.logger = setup_logger(self.output_dir)
        self.logger.info(f"Output Directory: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")

        # Khởi tạo Metric Tracker
        self.tracker = MetricTracker(Config.NUM_CLASSES)

    def load_model(self):
        self.logger.info(f"Loading model from: {self.args.model_path}")

        # Khởi tạo model với số input channel từ tham số
        model = GRAUpp(num_classes=Config.NUM_CLASSES, input_channels=self.args.input_channels)

        if not os.path.isfile(self.args.model_path):
            self.logger.error(f"Model file not found: {self.args.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.args.model_path}")

        try:
            checkpoint = torch.load(self.args.model_path, map_location=self.device)
            model.load_state_dict(checkpoint, strict=True)
            model.to(self.device)
            model.eval()
            self.logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise e

    def compute_area(self, mask):
        areas = {}
        valid = mask != Config.IGNORE_INDEX
        for cls in range(Config.NUM_CLASSES):
            pixel_count = np.logical_and(mask == cls, valid).sum()
            areas[f"L{cls}"] = pixel_count * Config.PIXEL_AREA_KM2
        return areas

    def visualize(self, img_tensor, gt_mask, pred_mask, filename):
        """Vẽ và lưu ảnh kết quả trực quan."""
        # Chuyển img từ (C, H, W) -> (H, W, C) và lấy 3 kênh đầu RGB
        img_np = img_tensor.cpu().numpy()
        img_rgb = np.clip(img_np[:3].transpose(1, 2, 0), 0, 1)

        gt_color = Config.COLOR_MAP[gt_mask]
        pred_color = Config.COLOR_MAP[pred_mask]

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        # 1. Input Image
        axs[0].imshow(img_rgb)
        axs[0].set_title("Input (RGB)", fontsize=12, fontweight='bold')

        # 2. Ground Truth
        axs[1].imshow(gt_color)
        axs[1].set_title("Ground Truth", fontsize=12, fontweight='bold')

        # 3. Prediction
        axs[2].imshow(pred_color)
        axs[2].set_title("Prediction", fontsize=12, fontweight='bold')

        # 4. Overlay (Input + Pred)
        # Pha trộn ảnh gốc và dự đoán với độ trong suốt alpha=0.6
        axs[3].imshow(img_rgb)
        axs[3].imshow(pred_color, alpha=0.6)
        axs[3].set_title("Overlay (RGB + Pred)", fontsize=12, fontweight='bold')

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, f"{filename}_vis.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def run(self):
        # 1. Load Data
        self.logger.info("Initializing Dataset...")
        try:
            test_dataset = LandCoverDataset(self.args.data_root, mode="test", augment=False)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
        except Exception as e:
            self.logger.error(f"Dataset Error: {e}")
            return

        self.logger.info(f"Total Test Images: {len(test_dataset)}")

        # 2. Load Model
        model = self.load_model()

        # 3. Inference Loop
        detailed_results = []

        self.logger.info("Starting Inference...")

        with torch.no_grad():
            for i, (img, mask) in enumerate(tqdm(test_loader, desc="Testing")):
                img = img.to(self.device)
                mask = mask.to(self.device)  # GT Mask

                # Forward
                outputs = model(img)
                if isinstance(outputs, list):
                    outputs = outputs[-1]

                pred_mask = torch.argmax(outputs, dim=1)  # (1, H, W)

                # Update Metrics Tracker (Toàn cục)
                self.tracker.update(pred_mask.cpu().numpy(), mask.cpu().numpy())

                # Xử lý từng ảnh để tính Area và lưu báo cáo chi tiết
                pred_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
                mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)

                # Tính diện tích
                area_gt = self.compute_area(mask_np)
                area_pr = self.compute_area(pred_np)

                # Tên file
                try:
                    file_name = test_dataset.samples[i][0]
                except:
                    file_name = f"image_{i:04d}"

                # Lưu vào list chi tiết
                row = {"File_Name": file_name}
                # Tính chỉ số riêng cho ảnh này (Local metrics)
                local_tracker = MetricTracker(Config.NUM_CLASSES)
                local_tracker.update(pred_np, mask_np)
                local_metrics = local_tracker.compute()

                row["Accuracy"] = local_metrics["Accuracy"]
                row["mIoU"] = local_metrics["mIoU"]

                for cls_idx in range(Config.NUM_CLASSES):
                    gt_area = area_gt[f"L{cls_idx}"]
                    pr_area = area_pr[f"L{cls_idx}"]
                    row[f"GT_Area_L{cls_idx}"] = gt_area
                    row[f"Pred_Area_L{cls_idx}"] = pr_area
                    row[f"Diff_L{cls_idx}"] = pr_area - gt_area

                detailed_results.append(row)

                # Visualization (Chỉ lưu N ảnh đầu tiên)
                if i < self.args.vis_limit:
                    self.visualize(img.squeeze(), mask_np, pred_np, file_name)

        # 4. Final Report
        global_metrics = self.tracker.compute()

        self.logger.info("-" * 60)
        self.logger.info(f"FINAL RESULTS")
        self.logger.info("-" * 60)
        self.logger.info(f"Global Accuracy : {global_metrics['Accuracy']:.2%}")
        self.logger.info(f"Global mIoU     : {global_metrics['mIoU']:.4f}")
        self.logger.info("-" * 60)

        # In bảng IoU từng lớp
        print(f"\n{'Class Name':<15} | {'IoU':<10}")
        print("-" * 30)
        for idx, name in enumerate(Config.CLASS_NAMES):
            iou = global_metrics['IoU_Class'][idx]
            print(f"{name:<15} | {iou:.4f}")
            self.logger.info(f"Class {name:<10}: IoU = {iou:.4f}")
        print("-" * 30 + "\n")

        # 5. Export Excel
        self.logger.info("Exporting report to Excel...")

        # DataFrame Chi tiết
        df_detailed = pd.DataFrame(detailed_results)

        # DataFrame Tổng hợp
        summary_data = {
            "Metric": ["Global Accuracy", "Global mIoU"],
            "Value": [global_metrics["Accuracy"], global_metrics["mIoU"]]
        }
        for idx, name in enumerate(Config.CLASS_NAMES):
            summary_data["Metric"].append(f"IoU_{name}")
            summary_data["Value"].append(global_metrics['IoU_Class'][idx])

        df_summary = pd.DataFrame(summary_data)

        excel_path = os.path.join(self.output_dir, "Evaluation_Report.xlsx")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_detailed.to_excel(writer, sheet_name='Detailed_Report', index=False)

            # Auto-adjust columns width (Optional formatting)
            for sheet in writer.sheets.values():
                sheet.set_column(0, 20, 15)

        self.logger.info(f"Report saved at: {excel_path}")
        self.logger.info("Testing Completed Successfully.")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Land Cover Testing Script")

    # Path Arguments
    parser.add_argument("--model-path", type=str, default="best_model.pth",
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data-root", type=str, default="dataset_final",
                        help="Root directory of the dataset")
    parser.add_argument("--output", type=str, default="test_results_pro",
                        help="Directory to save results")

    # Model Arguments
    parser.add_argument("--input-channels", type=int, default=10,
                        help="Number of input channels for the model (Default: 10)")

    # Visualization Arguments
    parser.add_argument("--vis-limit", type=int, default=50,
                        help="Number of images to visualize and save (Default: 50)")

    args = parser.parse_args()

    # Run Evaluation
    try:
        evaluator = Evaluator(args)
        evaluator.run()
    except KeyboardInterrupt:
        print("\n[INFO] Testing interrupted by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")