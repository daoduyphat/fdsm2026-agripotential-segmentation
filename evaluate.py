import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline.dataset import AgriPotentialDataset
from metrics import CLASS_NAMES, NUM_CLASSES, compute_metrics, update_confusion_matrix
from models import build_model

PIXEL_AREA_KM2 = 25.0 / 1e6  # 5m x 5m = 25 m^2 per pixel


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation models on AgriPotential test set.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "unetpp", "deeplabv3", "ghost_unetpp", "convnext_upernet"])
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data-root", type=str, default="dataset_final")
    parser.add_argument("--split-file", type=str, default="test.txt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-xlsx", type=str, default="evaluation_report.xlsx")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AgriPotentialDataset(args.data_root, args.split_file, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.model, input_channels=12, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    conf_mat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

    for images, masks in tqdm(loader, desc="Evaluating"):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[-1]
        preds = outputs.argmax(dim=1)
        conf_mat = update_confusion_matrix(conf_mat, preds, masks)

    iou, miou, oa = compute_metrics(conf_mat)

    gt_pixels = conf_mat.sum(dim=1).cpu().numpy()
    pred_pixels = conf_mat.sum(dim=0).cpu().numpy()

    rows = []
    for i, name in enumerate(CLASS_NAMES):
        gt_area = gt_pixels[i] * PIXEL_AREA_KM2
        pred_area = pred_pixels[i] * PIXEL_AREA_KM2
        rows.append({
            "Class": name,
            "IoU": round(float(iou[i]), 4),
            "Pred Area (km2)": round(float(pred_area), 2),
            "GT Area (km2)": round(float(gt_area), 2),
            "Diff (km2)": round(float(pred_area - gt_area), 2),
        })

    df = pd.DataFrame(rows)
    print(f"\nModel: {args.model} | mIoU: {miou:.4f} | OA: {oa:.4f}\n")
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(args.output_xlsx) or ".", exist_ok=True)
    df.to_excel(args.output_xlsx, index=False)


if __name__ == "__main__":
    main()
