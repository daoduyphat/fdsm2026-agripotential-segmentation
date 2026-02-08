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
from torch.cuda.amp import GradScaler

from unet import UNet
from preprocess_data import create_landcover_loaders, check_existing_splits

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

FINAL_CLASS_WEIGHTS = [0.3680, 1.3597, 2.0944, 1.7449, 2.0141]

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha).float()
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        if self.alpha is not None and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()

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

        probs = probs * valid
        onehot = onehot * valid

        inter = (probs * onehot).sum((2, 3))
        union = probs.sum((2, 3)) + onehot.sum((2, 3))

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class JointLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.focal = FocalLoss(alpha=weight, gamma=2.0, ignore_index=IGNORE_INDEX)
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return 0.5 * self.focal(preds, targets) + 0.5 * self.dice(preds, targets)

def compute_iou_global(preds, targets):
    ious = []
    for cls in range(NUM_CLASSES):
        tp = np.logical_and(preds == cls, targets == cls).sum()
        fp = np.logical_and(preds == cls, targets != cls).sum()
        fn = np.logical_and(preds != cls, targets == cls).sum()
        denom = tp + fp + fn
        ious.append(np.nan if denom == 0 else tp / denom)
    return ious, np.nanmean(ious)

def compute_area_from_mask(mask):
    areas = {}
    valid = mask != IGNORE_INDEX
    for cls in range(NUM_CLASSES):
        areas[f"L{cls}"] = (
                np.logical_and(mask == cls, valid).sum() * PIXEL_AREA_KM2
        )
    return areas

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

        axs[1].imshow(COLOR_MAP[gts[i]])
        axs[1].set_title(f"GT {i + 1}")
        axs[1].axis("off")

        axs[2].imshow(COLOR_MAP[preds[i]])
        axs[2].set_title(f"Pred {i + 1}")
        axs[2].axis("off")

        plt.tight_layout()
        save_name = base_path.replace(".png", f"_sample_{i + 1}.png")
        plt.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)

def plot_graphs(df, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='.')
    ax[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='.')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training & Validation Loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(df["epoch"], df["mIoU"], label="mIoU", color='green', marker='.')
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("mIoU")
    ax[1].set_title("Validation mIoU")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for cls in range(NUM_CLASSES):
        plt.plot(df["epoch"], df[f"GT_L{cls}"], label=f"GT {CLASS_NAMES[cls]}", linestyle='-')
        plt.plot(df["epoch"], df[f"Pred_L{cls}"], label=f"Pred {CLASS_NAMES[cls]}", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Area (km^2)")
    plt.title("Area Estimation: GT vs Pred")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "area_val_curve.png"), dpi=300)
    plt.close()

def save_logs(history, best_record, out_dir):
    df_history = pd.DataFrame(history)
    excel_path = os.path.join(out_dir, "train_log.xlsx")

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_history.to_excel(writer, sheet_name="Training Log", index=False)
        if best_record is not None:
            pd.DataFrame([best_record]).to_excel(writer, sheet_name="Best Epoch", index=False)

    plot_graphs(df_history, out_dir)

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all, targets_all = [], []
    area_gt = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    area_pr = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    vis = None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            out = model(imgs)
            if isinstance(out, list): out = out[-1]

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
                vis = (imgs[:5].cpu(), masks[:5].cpu(), preds[:5])

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    ious, miou = compute_iou_global(preds_all, targets_all)
    valid_mask = targets_all != IGNORE_INDEX
    acc = np.mean(preds_all[valid_mask] == targets_all[valid_mask])

    return np.mean(losses), acc, miou, ious, area_gt, area_pr, vis

def train(model, train_loader, val_loader, args, device, out_dir):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("----------------------------------------")
    print(f"MODEL: UNet")
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print("----------------------------------------")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scaler = GradScaler()
    weights = torch.tensor(FINAL_CLASS_WEIGHTS, device=device).float()
    criterion = JointLoss(weights)

    history = []
    best_miou = -1.0
    best_epoch_record = None

    print(f"\n[INFO] Config: Adam | LR: FIXED 0.0005 | Early Stopping: DISABLED")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                out = model(imgs)
                if isinstance(out, list):
                    w = torch.linspace(0.1, 1.0, len(out), device=device)
                    w = w / w.sum()
                    loss = sum(wi * criterion(o, masks) for wi, o in zip(w, out))
                else:
                    loss = criterion(out, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        val_loss, acc, miou, ious, area_gt, area_pr, vis = validate(model, val_loader, criterion, device)

        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        if np.isnan(avg_train_loss) or np.isnan(val_loss):
            print("\n[WARNING] Loss is NaN! Training might be unstable.")

        if np.isnan(miou):
            miou = 0.0
            print("\n[WARNING] mIoU is NaN. Force set to 0.0 to prevent crash.")

        print(
            f"\nEpoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f} | Acc: {acc:.2%} | LR: {current_lr:.1e}")
        print(f"{'Class':<12} | {'IoU':<8} | {'Area Pred':<12} | {'Area GT':<12} | {'Diff'}")
        print("-" * 65)
        for i, cls in enumerate(CLASS_NAMES):
            iou_str = "N/A" if np.isnan(ious[i]) else f"{ious[i]:.4f}"
            ap, ag = area_pr[f"L{i}"], area_gt[f"L{i}"]
            print(f"{cls:<12} | {iou_str:<8} | {ap:<12.4f} | {ag:<12.4f} | {ap - ag:+.4f}")
        print("\n")

        current_record = {
            "epoch": epoch,
            "model_params": total_params,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "mIoU": miou,
            **{f"IoU_{CLASS_NAMES[i]}": ious[i] for i in range(NUM_CLASSES)},
            **{f"GT_L{i}": area_gt[f"L{i}"] for i in range(NUM_CLASSES)},
            **{f"Pred_L{i}": area_pr[f"L{i}"] for i in range(NUM_CLASSES)}
        }
        history.append(current_record)

        if miou > best_miou:
            best_miou = miou
            best_epoch_record = current_record

            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))

            if vis:
                save_qualitative(
                    vis[0], vis[1], vis[2],
                    os.path.join(out_dir, "best_vis_result.png")
                )
            print(f"   >>> [SAVE] Best Model Updated (Epoch {epoch}) - mIoU: {miou:.4f}")

        try:
            torch.save(model.state_dict(), os.path.join(out_dir, "last_model.pth"))
        except Exception as e:
            print(f"[ERROR] Không thể lưu last_model.pth: {e}")

        save_logs(history, best_epoch_record, out_dir)

def main():
    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", default="dataset_final")
    parser.add_argument("--output", default="final_result")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    print(f"\n[AUTO-RUN] Checking data structure...")
    if not check_existing_splits(args.processed_path):
        print("\n[CRITICAL ERROR] Data check failed!")
        return

    print("[AUTO-RUN] Data valid. Starting training...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device}")

    out_dir = os.path.join(args.output, f"FINAL_RUN_UNET_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output Folder: {out_dir}")

    g = torch.Generator()
    g.manual_seed(0)

    train_loader, val_loader = create_landcover_loaders(
        args.processed_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = UNet(INPUT_CHANNELS, NUM_CLASSES).to(device)

    train(model, train_loader, val_loader, args, device, out_dir)
    print(f"[DONE] Finished. Results at: {out_dir}")

if __name__ == "__main__":
    main()