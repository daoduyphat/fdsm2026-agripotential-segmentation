import os
import argparse
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from graupp import GRAUpp

# IMPORT DATASET
from preprocess_data import LandCoverDataset

# GLOBAL CONFIG
NUM_CLASSES = 5
IGNORE_INDEX = 255

CLASS_NAMES = ["Background", "Level 1", "Level 2", "Level 3", "Level 4"]

PIXEL_SIZE_M = 5
PIXEL_AREA_KM2 = (PIXEL_SIZE_M ** 2) / 1e6

COLOR_MAP = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0]
], dtype=np.uint8)


# LOSS
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        targets_fix = targets.clone()
        targets_fix[targets == IGNORE_INDEX] = 0

        onehot = F.one_hot(targets_fix, NUM_CLASSES).permute(0, 3, 1, 2).float()
        valid = (targets != IGNORE_INDEX).float().unsqueeze(1)

        preds = preds * valid
        onehot = onehot * valid

        inter = (preds * onehot).sum((2, 3))
        union = preds.sum((2, 3)) + onehot.sum((2, 3))

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class JointLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return 0.5 * self.ce(preds, targets) + 0.5 * self.dice(preds, targets)


# METRICS
def compute_iou_global(preds, targets):
    ious = []
    for cls in range(NUM_CLASSES):
        tp = np.logical_and(preds == cls, targets == cls).sum()
        fp = np.logical_and(preds == cls, targets != cls).sum()
        fn = np.logical_and(preds != cls, targets == cls).sum()
        denom = tp + fp + fn
        ious.append(np.nan if denom == 0 else tp / denom)
    return ious, np.nanmean(ious)


def compute_area(preds):
    return {cls: np.sum(preds == cls) * PIXEL_AREA_KM2 for cls in range(NUM_CLASSES)}


# VISUALIZATION
def save_confusion(preds, targets, path):
    valid = targets != IGNORE_INDEX
    cm = confusion_matrix(targets[valid], preds[valid], labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(cm, CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_qualitative(imgs, gts, preds, path, n=4):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    for i in range(n):
        img = imgs[i].cpu().numpy() * std + mean
        img = np.clip(img.transpose(1, 2, 0), 0, 1)

        axs[i, 0].imshow(img)
        axs[i, 1].imshow(COLOR_MAP[gts[i]])
        axs[i, 2].imshow(COLOR_MAP[preds[i]])

        axs[i, 0].set_title("Input")
        axs[i, 1].set_title("GT")
        axs[i, 2].set_title("Prediction")
        for j in range(3):
            axs[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# VALIDATION
def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all, targets_all = [], []

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            if isinstance(outputs, list):
                outputs = outputs[-1]

            loss = criterion(outputs, masks)
            losses.append(loss.item())

            preds = torch.argmax(outputs, 1).cpu().numpy()
            preds_all.append(preds)
            targets_all.append(masks.cpu().numpy())

            if i == 0:
                save_imgs = imgs
                save_gts = masks
                save_preds = preds

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    ious, miou = compute_iou_global(preds_all, targets_all)
    acc = (preds_all == targets_all)[targets_all != IGNORE_INDEX].mean()

    return (
        np.mean(losses),
        acc,
        miou,
        ious,
        preds_all,
        targets_all,
        save_imgs,
        save_gts,
        save_preds
    )


# TRAIN
def train(model, train_loader, val_loader, args, device, out_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=3)
    scaler = GradScaler()

    weights = torch.tensor([0.3340, 1.5467, 2.3826, 1.9852, 2.2913]).to(device)
    criterion = JointLoss(weights)

    best_miou = 0
    best_epoch = 0
    early = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(imgs)
                if isinstance(outputs, list):
                    w = torch.linspace(0.1, 1.0, len(outputs)).to(device)
                    w = w / w.sum()
                    loss = sum(wi * criterion(o, masks) for wi, o in zip(w, outputs))
                else:
                    loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        val_loss, acc, miou, ious, preds, targets, s_i, s_g, s_p = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(miou)
        areas = compute_area(preds)

        # >>> ADDED: LOG FULL PER-CLASS
        print(f"\nEpoch {epoch} | mIoU: {miou:.4f} | Acc: {acc:.2%}")
        print("Per-class IoU & Area:")
        for i, cls in enumerate(CLASS_NAMES):
            iou_str = "N/A" if np.isnan(ious[i]) else f"{ious[i]:.4f}"
            print(f"  - {cls:<10} | IoU: {iou_str:<7} | Area: {areas[i]:.3f} km²")

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "val_loss": val_loss,
            "accuracy": acc,
            "mIoU": miou,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"IoU_{CLASS_NAMES[i]}": ious[i] for i in range(NUM_CLASSES)},
            **{f"Area_{CLASS_NAMES[i]}": areas[i] for i in range(NUM_CLASSES)},
            "Is_Best": ""
        })

        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            early = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            save_confusion(preds, targets, os.path.join(out_dir, "confusion_matrix.png"))
            save_qualitative(s_i, s_g, s_p, os.path.join(out_dir, "qualitative.png"))
        else:
            early += 1
            if early >= args.patience:
                break

    torch.save(model.state_dict(), os.path.join(out_dir, "last_model.pth"))

    for h in history:
        if h["epoch"] == best_epoch:
            h["Is_Best"] = "BEST"

    pd.DataFrame(history).to_excel(
        os.path.join(out_dir, "training_history.xlsx"), index=False
    )


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", default="dataset_validated")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(args.output, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    train_ds = LandCoverDataset(args.processed_path, mode="train", augment=True)
    val_ds = LandCoverDataset(args.processed_path, mode="val", augment=False)

    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = GRAUpp(num_classes=NUM_CLASSES, input_channels=3).to(device)

    train(model, train_loader, val_loader, args, device, out_dir)
    print("[INFO] TRAINING FINISHED")
