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
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from thop import profile
except ImportError:
    profile = None

from resnet50 import GRAUpp
from preprocess_new import create_landcover_loaders, check_existing_splits

NUM_CLASSES = 5
IGNORE_INDEX = 255
INPUT_CHANNELS = 12
CLASS_NAMES = ["Background", "Level 1", "Level 2", "Level 3", "Level 4"]

PIXEL_SIZE_METERS = 5
PIXEL_AREA_KM2 = (PIXEL_SIZE_METERS ** 2) / 1_000_000

FINAL_CLASS_WEIGHTS = [0.3680, 1.3597, 2.0944, 1.7449, 2.0141]

warnings.filterwarnings("ignore")

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_model_complexity(model, input_size, device):
    if profile is None:
        return 0, 0
    model_eval = model.eval()
    dummy_input = torch.randn(input_size).to(device)
    try:
        macs, params = profile(model_eval, inputs=(dummy_input,), verbose=False)
        return params, macs * 2
    except:
        return 0, 0

def compute_iou_global(preds, targets):
    ious = []
    valid = targets != IGNORE_INDEX
    for cls in range(NUM_CLASSES):
        p = (preds == cls) & valid
        t = (targets == cls) & valid
        tp = np.logical_and(p, t).sum()
        denom = np.logical_or(p, t).sum()
        ious.append(tp / denom if denom > 0 else np.nan)
    return ious, np.nanmean(ious)

def compute_area_from_mask(mask):
    areas = {}
    valid = mask != IGNORE_INDEX
    for cls in range(NUM_CLASSES):
        pixel_count = np.logical_and(mask == cls, valid).sum()
        areas[f"L{cls}"] = pixel_count * PIXEL_AREA_KM2
    return areas

class JointLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weight).float() if weight else None,
                                      ignore_index=IGNORE_INDEX)

    def forward(self, preds, targets):
        probs = F.softmax(preds, dim=1)
        t_onehot = F.one_hot(targets.clamp(0, NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2).float()
        valid = (targets != IGNORE_INDEX).float().unsqueeze(1)
        inter = (probs * t_onehot * valid).sum((2, 3))
        union = (probs * valid).sum((2, 3)) + (t_onehot * valid).sum((2, 3))
        dice = 1 - (2 * inter + 1e-6) / (union + 1e-6)
        return 0.5 * self.ce(preds, targets) + 0.5 * dice.mean()

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all, targets_all = [], []
    area_gt = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    area_pr = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    vis_data = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Validating", leave=False, ncols=100)):
            if isinstance(batch, dict):
                imgs, masks = batch["image"].to(device), batch["mask"].to(device)
            else:
                imgs, masks = batch[0].to(device), batch[1].to(device)

            out = model(imgs)
            loss = criterion(out, masks)
            losses.append(loss.item())

            preds = torch.argmax(out, 1).cpu().numpy()
            masks_np = masks.cpu().numpy()

            preds_all.append(preds)
            targets_all.append(masks_np)

            for b in range(preds.shape[0]):
                gt = compute_area_from_mask(masks_np[b])
                pr = compute_area_from_mask(preds[b])
                for k in gt:
                    area_gt[k] += gt[k]
                    area_pr[k] += pr[k]

            if i == 0:
                vis_data = (imgs[:5].cpu(), masks[:5].cpu(), preds[:5])

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    ious, miou = compute_iou_global(preds_all, targets_all)
    valid = targets_all != IGNORE_INDEX
    acc = np.mean(preds_all[valid] == targets_all[valid])

    return np.mean(losses), acc, miou, ious, area_gt, area_pr, vis_data

def save_qualitative(imgs, gts, preds, base_path, n=5):
    num_images = min(n, imgs.shape[0])
    imgs = imgs.cpu().numpy()
    gts = gts.cpu().numpy()
    for i in range(num_images):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        img_rgb = np.clip(imgs[i, :3].transpose(1, 2, 0), 0, 1)
        axs[0].imshow(img_rgb)
        axs[0].set_title(f"Input {i + 1}")
        axs[0].axis("off")
        axs[1].imshow(gts[i], vmin=0, vmax=NUM_CLASSES - 1)
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")
        axs[2].imshow(preds[i], vmin=0, vmax=NUM_CLASSES - 1)
        axs[2].set_title("Prediction")
        axs[2].axis("off")
        plt.tight_layout()
        plt.savefig(base_path.replace(".png", f"_{i + 1}.png"), dpi=150)
        plt.close(fig)

def plot_graphs(df, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(df["epoch"], df["train_loss"], label="Train")
    ax[0].plot(df["epoch"], df["val_loss"], label="Val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(df["epoch"], df["mIoU"], label="mIoU", color='green')
    ax[1].set_title("mIoU")
    ax[1].legend()
    plt.savefig(os.path.join(out_dir, "metrics.png"))
    plt.close()

def save_logs(history, best_record, out_dir):
    df = pd.DataFrame(history)
    excel_path = os.path.join(out_dir, "training_log.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Training Log", index=False)
        if best_record:
            pd.DataFrame([best_record]).to_excel(writer, sheet_name="Best Epoch", index=False)
    plot_graphs(df, out_dir)

def train(model, train_loader, val_loader, args, device, out_dir):
    params_num, flops_num = compute_model_complexity(model, (1, INPUT_CHANNELS, 128, 128), device)
    flops_g = flops_num / 1e9

    print(f"MODEL: GRAUpp (ResNet50 Backbone)")
    print(f"PARAMS: {params_num:,}")
    print(f"FLOPs : {flops_g:.2f} G")
    print(f"DEVICE: {device}")
    print(f"OUTPUT: {out_dir}")
    print(f"CONFIG: LR={args.lr} | BS={args.batch_size} | Epochs={args.epochs}")
    print("-" * 75 + "\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler()
    criterion = JointLoss(weight=FINAL_CLASS_WEIGHTS).to(device)

    history = []
    best_miou = -1.0
    best_record = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", leave=True, ncols=100)
        for batch in pbar:
            if isinstance(batch, dict):
                imgs, masks = batch["image"].to(device), batch["mask"].to(device)
            else:
                imgs, masks = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            with autocast():
                out = model(imgs)
                if isinstance(out, list):
                    loss = criterion(out[-1], masks) + 0.4 * sum(criterion(o, masks) for o in out[:-1])
                else:
                    loss = criterion(out, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        val_loss, acc, miou, ious, area_gt, area_pr, vis = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"\nEpoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f} | Acc: {acc:.2%} | LR: {current_lr:.1e}")
        print("-" * 75)
        print(f"{'Class Name':<15} | {'IoU':<10} | {'Pred (km2)':<12} | {'GT (km2)':<12} | {'Diff'}")
        print("-" * 75)

        for i, name in enumerate(CLASS_NAMES):
            iou_val = f"{ious[i]:.4f}" if not np.isnan(ious[i]) else "N/A"
            ap, ag = area_pr[f"L{i}"], area_gt[f"L{i}"]
            diff = ap - ag
            print(f"{name:<15} | {iou_val:<10} | {ap:<12.4f} | {ag:<12.4f} | {diff:+.4f}")

        print("-" * 75)

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "mIoU": miou,
            "accuracy": acc,
            "lr": current_lr
        }
        for i, name in enumerate(CLASS_NAMES):
            record[f"IoU_{name}"] = ious[i]

        history.append(record)

        if miou > best_miou:
            best_miou = miou
            best_record = record.copy()
            best_record["Params"] = f"{params_num:,}"
            best_record["FLOPs (G)"] = f"{flops_g:.2f}"

            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            print(f"New Best mIoU: {miou:.4f} -> Saved best_model.pth\n")

            if vis:
                save_qualitative(vis[0], vis[1], vis[2], os.path.join(out_dir, "best_vis_result.png"))
        else:
            print()

        torch.save(model.state_dict(), os.path.join(out_dir, "last_model.pth"))
        save_logs(history, best_record, out_dir)

def main():
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", default="dataset_final")
    parser.add_argument("--output", default="final_result_graupp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    if not check_existing_splits(args.processed_path):
        print("Dataset missing.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.output, f"RUN_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader = create_landcover_loaders(
        args.processed_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = GRAUpp(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS, pretrained=True, deep_supervision=True, attention='eca').to(device)
    train(model, train_loader, val_loader, args, device, out_dir)

if __name__ == "__main__":
    main()