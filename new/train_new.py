import os
import argparse
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_new import GRAUppAdvanced, LovaszSoftmaxLoss
from preprocess_new import create_landcover_loaders, check_existing_splits

NUM_CLASSES = 5
IGNORE_INDEX = 255
INPUT_CHANNELS = 12

CLASS_NAMES = ["Background", "Level 1", "Level 2", "Level 3", "Level 4"]

PIXEL_SIZE_METERS = 5
PIXEL_AREA_KM2 = (PIXEL_SIZE_METERS ** 2) / 1_000_000

COLOR_MAP = np.array([
    [0, 0, 0],
    [139, 0, 0],
    [255, 165, 0],
    [255, 255, 0],
    [50, 205, 50]
], dtype=np.uint8)

warnings.filterwarnings("ignore")


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_iou_global(preds, targets):
    ious = []
    for cls in range(NUM_CLASSES):
        tp = np.logical_and(preds == cls, targets == cls).sum()
        denom = np.logical_or(preds == cls, targets == cls).sum()
        ious.append(tp / denom if denom > 0 else np.nan)
    return ious, np.nanmean(ious)


def compute_area_from_mask(mask):
    areas = {}
    valid = mask != IGNORE_INDEX
    for cls in range(NUM_CLASSES):
        pixel_count = np.logical_and(mask == cls, valid).sum()
        areas[f"L{cls}"] = pixel_count * PIXEL_AREA_KM2
    return areas


def save_qualitative(imgs, gts, preds, base_path, n=5):
    num_images = min(n, imgs.shape[0])
    imgs = imgs.cpu().numpy()
    gts = gts.cpu().numpy()

    for i in range(num_images):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        img_rgb = np.clip(imgs[i, :3].transpose(1, 2, 0), 0, 1)
        axs[0].imshow(img_rgb)
        axs[0].set_title(f"Input Sample {i + 1}")
        axs[0].axis("off")

        axs[1].imshow(COLOR_MAP[gts[i]])
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(COLOR_MAP[preds[i]])
        axs[2].set_title("Prediction")
        axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(base_path.replace(".png", f"_{i + 1}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_graphs(df, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='.')
    ax[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='.')
    ax[0].set_title("Loss Curves")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(df["epoch"], df["mIoU"], label="Validation mIoU", color='green', marker='.')
    ax[1].set_title("mIoU Progression")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for cls in range(NUM_CLASSES):
        plt.plot(df["epoch"], df[f"GT_L{cls}"], label=f"GT {CLASS_NAMES[cls]}", linestyle='-')
        plt.plot(df["epoch"], df[f"Pred_L{cls}"], label=f"Pred {CLASS_NAMES[cls]}", linestyle='--')
    plt.title("Area Estimation: Ground Truth vs Prediction")
    plt.ylabel("Area (km2)")
    plt.xlabel("Epoch")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "area_estimation.png"), dpi=200)
    plt.close()


def save_logs(history, best_record, out_dir):
    df = pd.DataFrame(history)
    with pd.ExcelWriter(os.path.join(out_dir, "training_log.xlsx"), engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Full Log", index=False)
        if best_record:
            pd.DataFrame([best_record]).to_excel(writer, sheet_name="Best Model", index=False)
    plot_graphs(df, out_dir)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        probs = torch.softmax(preds, dim=1)
        targets_fix = targets.clone()
        targets_fix[targets == IGNORE_INDEX] = 0

        onehot = F.one_hot(targets_fix, NUM_CLASSES).permute(0, 3, 1, 2).float()
        valid = torch.ne(targets, IGNORE_INDEX).float().unsqueeze(1)

        inter = ((probs * valid) * (onehot * valid)).sum((2, 3))
        union = (probs * valid).sum((2, 3)) + (onehot * valid).sum((2, 3))

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class AdvancedCombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(gamma=2.0, ignore_index=IGNORE_INDEX)
        self.dice = DiceLoss()
        self.lovasz = LovaszSoftmaxLoss(ignore_index=IGNORE_INDEX)

    def forward(self, preds, targets):
        loss_focal = self.focal(preds, targets)
        loss_dice = self.dice(preds, targets)
        loss_lovasz = self.lovasz(torch.softmax(preds, dim=1), targets)
        return 0.25 * loss_focal + 0.75 * loss_dice + 0.5 * loss_lovasz


def predict_with_tta(model, imgs):
    out1 = torch.softmax(model(imgs), dim=1)
    out2 = torch.flip(torch.softmax(model(torch.flip(imgs, [3])), dim=1), [3])
    out3 = torch.flip(torch.softmax(model(torch.flip(imgs, [2])), dim=1), [2])
    return (out1 + out2 + out3) / 3.0


def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all, targets_all = [], []
    area_gt = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    area_pr = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    vis_data = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):
            imgs, masks = batch["image"].to(device), batch["mask"].to(device)

            out_avg = predict_with_tta(model, imgs)

            out_raw = model(imgs)
            if isinstance(out_raw, list): out_raw = out_raw[-1]
            losses.append(criterion(out_raw, masks).item())

            preds = torch.argmax(out_avg, 1).cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_all.append(preds)
            targets_all.append(masks_np)

            for b in range(preds.shape[0]):
                gt_areas = compute_area_from_mask(masks_np[b])
                pr_areas = compute_area_from_mask(preds[b])
                for k in gt_areas:
                    area_gt[k] += gt_areas[k]
                    area_pr[k] += pr_areas[k]

            if i == 0:
                vis_data = (imgs[:5].cpu(), masks[:5].cpu(), preds[:5])

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    ious, miou = compute_iou_global(preds_all, targets_all)
    valid = targets_all != IGNORE_INDEX
    acc = np.mean(preds_all[valid] == targets_all[valid])

    return np.mean(losses), acc, miou, ious, area_gt, area_pr, vis_data


def train(model, train_loader, val_loader, args, device, out_dir):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MODEL: GRAUpp Advanced (ConvNeXt-Tiny + PSA)")
    print(f"PARAMS: {param_count:,}")
    print(f"DEVICE: {device}")
    print(f"OUTPUT: {out_dir}")
    print(f"CONFIG: LR={args.lr} | BS={args.batch_size} | Epochs={args.epochs}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    scaler = GradScaler()
    criterion = AdvancedCombinedLoss()

    history = []
    best_miou = -1.0
    best_record = None

    # Early Stopping Variables
    early_stop_patience = 7
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            imgs, masks = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outs = model(imgs)
                if isinstance(outs, list):
                    weights = [1.0, 0.5, 0.4, 0.3]
                    loss = sum(w * criterion(o, masks) for w, o in zip(weights, outs))
                else:
                    loss = criterion(outs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        val_loss, acc, miou, ious, area_gt, area_pr, vis = validate(model, val_loader, criterion, device)
        avg_train_loss = train_loss / len(train_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f} | Acc: {acc:.2%} | LR: {current_lr:.1e}")
        print("-" * 75)
        print(f"{'Class Name':<15} | {'IoU':<10} | {'Pred (km2)':<12} | {'GT (km2)':<12} | {'Diff'}")
        print("-" * 75)
        for i, name in enumerate(CLASS_NAMES):
            iou_val = f"{ious[i]:.4f}" if not np.isnan(ious[i]) else "N/A"
            diff = area_pr[f"L{i}"] - area_gt[f"L{i}"]
            print(
                f"{name:<15} | {iou_val:<10} | {area_pr[f'L{i}']:.4f}       | {area_gt[f'L{i}']:.4f}       | {diff:+.4f}")
        print("-" * 75 + "\n")

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "mIoU": miou,
            "accuracy": acc,
            "lr": current_lr,
            **{f"IoU_{name}": ious[i] for i, name in enumerate(CLASS_NAMES)},
            **{f"GT_{name}": area_gt[f"L{i}"] for i, name in enumerate(CLASS_NAMES)},
            **{f"Pred_{name}": area_pr[f"L{i}"] for i, name in enumerate(CLASS_NAMES)}
        }
        history.append(record)

        # Logic Early Stopping & Save Best
        if miou > best_miou:
            best_miou = miou
            best_record = record
            early_stop_counter = 0  # Reset counter khi có kết quả tốt hơn

            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            if vis:
                save_qualitative(vis[0], vis[1], vis[2], os.path.join(out_dir, "best_vis.png"))
            print(f"New Best mIoU: {miou:.4f} -> Saved best_model.pth\n")
        else:
            early_stop_counter += 1
            print(f"No improvement. EarlyStopping Counter: {early_stop_counter}/{early_stop_patience}\n")

        torch.save(model.state_dict(), os.path.join(out_dir, "last_model.pth"))
        save_logs(history, best_record, out_dir)

        # Check Early Stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early Stopping triggered at Epoch {epoch}. Best mIoU: {best_miou:.4f}")
            break


def main():
    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", default="dataset_final")
    parser.add_argument("--output", default="final_result_advanced")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if not check_existing_splits(args.processed_path):
        print("Dataset missing. Please check '--processed-path'.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.output, f"RUN_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader = create_landcover_loaders(
        args.processed_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = GRAUppAdvanced(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS).to(device)
    train(model, train_loader, val_loader, args, device, out_dir)


if __name__ == "__main__":
    main()