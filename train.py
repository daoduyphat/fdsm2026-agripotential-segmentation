import os
import argparse
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from graupp import GRAUpp
from preprocess_data import LandCoverDataset


# --- LOSS FUNCTIONS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=4.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets):
        num_classes = preds.shape[1]
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        TP = (preds * targets_one_hot).sum(dim=(2, 3))
        FP = (preds * (1 - targets_one_hot)).sum(dim=(2, 3))
        FN = ((1 - preds) * targets_one_hot).sum(dim=(2, 3))

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky_index.mean()


# --- METRICS & UTILS ---
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


def calculate_agri_area_km2(preds, res_meter):
    pred_cls = torch.argmax(preds, dim=1)
    pixel_count = torch.sum(pred_cls == 0).item()
    area_m2 = pixel_count * (res_meter ** 2)
    return area_m2 / 1_000_000.0


def generate_qualitative(net, loader, dev, save_dir, num_samples=10):
    os.makedirs(save_dir, exist_ok=True)
    net.eval()
    count = 0
    colors = np.array([
        [255, 255, 0],  # Class 0: Agri
        [0, 0, 255],  # Class 1: Water
        [0, 255, 0],  # Class 2: Woodland
        [0, 255, 255],  # Class 3: Building
        [128, 128, 128]  # Class 4: Road
    ], dtype=np.uint8)

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(dev), masks.to(dev)
            with autocast():
                outputs = net(imgs)

            preds = outputs[-1] if isinstance(outputs, list) else outputs
            preds = torch.argmax(preds, dim=1)

            for i in range(imgs.size(0)):
                if count >= num_samples: return
                inp = imgs[i].permute(1, 2, 0).cpu().numpy()
                gt = masks[i].cpu().numpy()
                pr = preds[i].cpu().float().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(inp)
                axs[0].set_title("Input")
                axs[0].axis("off")

                axs[1].imshow(colors[gt])
                axs[1].set_title("Ground Truth")
                axs[1].axis("off")

                axs[2].imshow(colors[pr.astype(np.uint8)])
                axs[2].set_title("Prediction")
                axs[2].axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"sample_{count:03d}.png"), dpi=150)
                plt.close(fig)
                count += 1


def valid_model(net, loader, criterion_focal, criterion_tversky, dev, res_meter):
    net.eval()
    val_loss = 0.0
    accs, ious = [], []
    total_agri_km2 = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(dev), masks.to(dev)
            with autocast():
                preds = net(imgs)
                if isinstance(preds, list):
                    preds = preds[-1]

                loss = 0.2 * criterion_focal(preds, masks) + 0.8 * criterion_tversky(preds, masks)

            val_loss += loss.item()
            acc, iou = compute_metrics(preds.float(), masks)
            accs.append(acc)
            ious.append(iou)
            total_agri_km2 += calculate_agri_area_km2(preds, res_meter)

    return val_loss / len(loader), np.mean(accs), np.mean(ious), total_agri_km2


def train_model(net, loader_train, loader_val, dev, epochs, lr, output_dir, resolution, model_name="model", patience=5):
    net.to(dev)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience)
    scaler = GradScaler()

    # Weights calculated from analyze_labels.py
    # Class 0: 0.33, Class 1: 1.55, Class 2: 2.38, Class 3: 1.99, Class 4: 2.29
    class_weights = torch.tensor([0.33, 1.55, 2.38, 1.99, 2.29]).to(dev)

    criterion_focal = FocalLoss(alpha=class_weights, gamma=4.0)
    criterion_tversky = TverskyLoss(alpha=0.3, beta=0.7)

    accumulation_steps = 4
    best_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "accuracy": [], "iou": [], "agri_area_km2": []}

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_txt = os.path.join(log_dir, f"{model_name}_training_log.txt")

    with open(log_txt, "w") as f:
        f.write("Epoch\tTrainLoss\tValLoss\tAccuracy\tIoU\tAgri_Area_km2\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        net.train()
        train_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(loader_train, desc=f"{model_name} Epoch {epoch}/{epochs}")

        for i, (imgs, masks) in enumerate(loop):
            imgs, masks = imgs.to(dev), masks.to(dev)

            with autocast():
                preds = net(imgs)
                if isinstance(preds, list):
                    loss_val = sum([0.2 * criterion_focal(p, masks) + 0.8 * criterion_tversky(p, masks) for p in preds])
                else:
                    loss_val = 0.2 * criterion_focal(preds, masks) + 0.8 * criterion_tversky(preds, masks)

                loss_val = loss_val / accumulation_steps

            scaler.scale(loss_val).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_loss = loss_val.item() * accumulation_steps
            train_loss += current_loss
            loop.set_postfix(loss=current_loss)

        avg_train_loss = train_loss / len(loader_train)
        val_loss, val_acc, val_iou, val_agri_km2 = valid_model(
            net, loader_val, criterion_focal, criterion_tversky, dev, resolution
        )

        print(
            f"Ep {epoch} | T.Loss {avg_train_loss:.4f} | V.Loss {val_loss:.4f} | Acc {val_acc:.4f} | IoU {val_iou:.4f} | Agri: {val_agri_km2:.4f} km2")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(val_acc)
        history["iou"].append(val_iou)
        history["agri_area_km2"].append(val_agri_km2)

        with open(log_txt, "a") as f:
            f.write(
                f"{epoch}\t{avg_train_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\t{val_iou:.4f}\t{val_agri_km2:.4f}\n")

        scheduler.step(val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(net.state_dict(), os.path.join(output_dir, f"{model_name}_best.pth"))
            print(f"Save Best IoU: {best_iou:.4f}")
            generate_qualitative(net, loader_val, dev, os.path.join(output_dir, "qualitative"))

    history["params"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    history["time"] = time.time() - start_time

    df_log = pd.DataFrame(
        {k: v for k, v in history.items() if k in ["train_loss", "val_loss", "accuracy", "iou", "agri_area_km2"]})
    df_log.insert(0, "Epoch", range(1, len(history["train_loss"]) + 1))
    df_log.to_excel(os.path.join(log_dir, f"{model_name}_training_log.xlsx"), index=False)

    return history


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default="dataset")
    p.add_argument("--output", type=str, default="outputs_landcover")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--img-size", type=int, nargs=2, default=(128, 128))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--resolution", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' not found.")
        exit(1)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    train_dataset = LandCoverDataset(
        os.path.join(train_dir, "images"),
        os.path.join(train_dir, "masks"),
        img_size=tuple(args.img_size),
        augment=True
    )
    val_dataset = LandCoverDataset(
        os.path.join(val_dir, "images"),
        os.path.join(val_dir, "masks"),
        img_size=tuple(args.img_size),
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("=== Model: GRAU++ (GhostResAttentionUNetPP) ===")
    model = GRAUpp(num_classes=5, input_channels=3, pretrained=False, deep_supervision=True, attention='cbam')

    history = train_model(
        model, train_loader, val_loader, device,
        args.epochs, args.lr, run_dir, args.resolution,
        "GRAUpp", args.patience
    )

    df = pd.DataFrame({k: v for k, v in history.items() if k in ["train_loss", "val_loss", "accuracy", "iou"]})
    df['Epoch'] = range(1, len(df) + 1)

    plt.figure(figsize=(16, 4))
    metrics_to_plot = ["train_loss", "val_loss", "iou"]
    for i, col in enumerate(metrics_to_plot):
        if col in df.columns:
            plt.subplot(1, 3, i + 1)
            plt.plot(df["Epoch"], df[col], marker='o')
            plt.title(col)
            plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_metrics.png"))

    summary_df = pd.DataFrame([{
        "Model": "GRAUpp",
        "Trainable_Params": history["params"],
        "Best_IoU": max(history["iou"]),
        "Time_Train(s)": round(history["time"], 2),
        "Epochs": args.epochs,
        "Batch_Size_Simulated": args.batch_size * 4
    }])

    with pd.ExcelWriter(os.path.join(run_dir, "summary&results.xlsx")) as writer:
        df.to_excel(writer, sheet_name="History", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print("Train finished.")