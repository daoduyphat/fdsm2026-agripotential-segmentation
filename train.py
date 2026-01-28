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
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Import các module của bạn
from graupp import GRAUpp
from preprocess_data import LandCoverDataset

# =========================
# GLOBAL CONFIG
# =========================
NUM_CLASSES = 5
IGNORE_INDEX = 255

CLASS_NAMES = ["Background", "Level 1", "Level 2", "Level 3", "Level 4"]

# Cấu hình tính diện tích
PIXEL_SIZE_METERS = 5
PIXEL_AREA_KM2 = (PIXEL_SIZE_METERS ** 2) / 1_000_000

COLOR_MAP = np.array([
    [0, 0, 0],  # 0: Background
    [139, 0, 0],  # 1: Very Low
    [255, 165, 0],  # 2: Low
    [255, 255, 0],  # 3: Medium
    [50, 205, 50]  # 4: High
], dtype=np.uint8)

FINAL_CLASS_WEIGHTS = [0.3680, 1.3597, 2.0944, 1.7449, 2.0141]


# =========================
# REPRODUCIBILITY SETUP (Khóa Random)
# =========================
def set_seed(seed=0):
    """Thiết lập hạt giống cố định cho toàn bộ hệ thống"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    """Khóa random cho từng worker trong DataLoader"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================
# LOSS FUNCTIONS
# =========================
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
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return 0.5 * self.ce(preds, targets) + 0.5 * self.dice(preds, targets)


# =========================
# METRICS & UTILS
# =========================
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

        # Input RGB
        img_rgb = np.clip(imgs[i, :3].transpose(1, 2, 0), 0, 1)
        axs[0].imshow(img_rgb)
        axs[0].set_title(f"Input (Sample {i + 1})")
        axs[0].axis("off")

        # Ground Truth
        axs[1].imshow(COLOR_MAP[gts[i]])
        axs[1].set_title(f"Ground Truth (Sample {i + 1})")
        axs[1].axis("off")

        # Prediction
        axs[2].imshow(COLOR_MAP[preds[i]])
        axs[2].set_title(f"Prediction (Sample {i + 1})")
        axs[2].axis("off")

        plt.tight_layout()
        save_name = base_path.replace(".png", f"_sample_{i + 1}.png")
        plt.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)


def plot_graphs(df, out_dir):
    # 1. Loss & mIoU Curve
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(df["epoch"], df["train_loss"], label="Train Loss")
    ax[0].plot(df["epoch"], df["val_loss"], label="Val Loss")
    ax[0].set_title("Loss Curve")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(df["epoch"], df["mIoU"], label="mIoU", color='green')
    ax[1].set_title("Validation mIoU Curve")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=300)
    plt.close()

    # 2. Area Curve
    plt.figure(figsize=(10, 6))
    for cls in range(NUM_CLASSES):
        plt.plot(df["epoch"], df[f"GT_L{cls}"], label=f"GT {CLASS_NAMES[cls]}", linestyle='-')
        plt.plot(df["epoch"], df[f"Pred_L{cls}"], label=f"Pred {CLASS_NAMES[cls]}", linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Area (km²)")
    plt.title("Agricultural Area Estimation: GT vs Prediction")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "area_val_curve.png"), dpi=300)
    plt.close()


# =========================
# VALIDATION LOOP
# =========================
def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all, targets_all = [], []
    area_gt = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    area_pr = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    vis = None

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(device), masks.to(device)

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
                # Lưu lại batch đầu tiên để vẽ hình
                vis = (imgs[:5].cpu(), masks[:5].cpu(), preds[:5])

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    ious, miou = compute_iou_global(preds_all, targets_all)
    valid_mask = targets_all != IGNORE_INDEX
    acc = np.mean(preds_all[valid_mask] == targets_all[valid_mask])

    return np.mean(losses), acc, miou, ious, area_gt, area_pr, vis


# =========================
# TRAINING LOOP
# =========================
def train(model, train_loader, val_loader, args, device, out_dir):
    # [BEST CONFIGURATION] Hardcode: Adam + LR 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.1
    )
    scaler = GradScaler()

    weights = torch.tensor(FINAL_CLASS_WEIGHTS, device=device).float()
    criterion = JointLoss(weights)

    history = []
    best_miou = -1
    early_trigger = 0
    PATIENCE = 5  # Dừng sớm nếu không cải thiện trong 5 epoch

    print(f"\n[INFO] Bắt đầu huấn luyện với Top 1 Grid Search:")
    print(f"       Optimizer: Adam | LR: 0.0005 | Seed: 0")
    print(f"       Class Weights: {FINAL_CLASS_WEIGHTS}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast():
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

        val_loss, acc, miou, ious, area_gt, area_pr, vis = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(miou)
        current_lr = optimizer.param_groups[0]['lr']

        # LOGGING RA MÀN HÌNH
        print(f"\nEpoch {epoch} | mIoU: {miou:.4f} | Acc: {acc:.2%} | LR: {current_lr:.1e}")
        print(f"{'Class':<12} | {'IoU':<8} | {'Area Pred':<12} | {'Area GT':<12} | {'Diff'}")
        print("-" * 65)
        for i, cls in enumerate(CLASS_NAMES):
            iou_str = "N/A" if np.isnan(ious[i]) else f"{ious[i]:.4f}"
            ap, ag = area_pr[f"L{i}"], area_gt[f"L{i}"]
            print(f"{cls:<12} | {iou_str:<8} | {ap:<12.4f} | {ag:<12.4f} | {ap - ag:+.4f}")
        print("\n")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss,
            "accuracy": acc,
            "mIoU": miou,
            **{f"GT_L{i}": area_gt[f"L{i}"] for i in range(NUM_CLASSES)},
            **{f"Pred_L{i}": area_pr[f"L{i}"] for i in range(NUM_CLASSES)}
        })

        if miou > best_miou:
            best_miou = miou
            early_trigger = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))

            # Lưu ảnh kết quả tốt nhất
            imgs, gts, preds = vis
            save_qualitative(imgs, gts, preds, os.path.join(out_dir, f"best_epoch_{epoch}.png"))
        else:
            early_trigger += 1
            if early_trigger >= PATIENCE:
                print(f"[INFO] Early Stopping tại Epoch {epoch}. Best mIoU: {best_miou:.4f}")
                break

    # LƯU KẾT QUẢ CUỐI CÙNG RA EXCEL
    df = pd.DataFrame(history)
    with pd.ExcelWriter(os.path.join(out_dir, "train_log.xlsx"), engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
        best_idx = df["mIoU"].idxmax()
        writer.sheets["Sheet1"].set_row(best_idx + 1, None, writer.book.add_format({"bg_color": "#C6EFCE"}))

    plot_graphs(df, out_dir)


# =========================
# MAIN
# =========================
def main():
    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", default="dataset_final")
    parser.add_argument("--output", default="final_result_production")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Thiết bị: {device}")

    # Tạo thư mục lưu trữ timestamp
    out_dir = os.path.join(args.output, f"FINAL_RUN_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(0)

    train_ds = LandCoverDataset(args.processed_path, "train", True)
    val_ds = LandCoverDataset(args.processed_path, "val", False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker, generator=g  # Khóa random worker
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker, generator=g  # Khóa random worker
    )

    model = GRAUpp(num_classes=NUM_CLASSES, input_channels=10).to(device)

    # Bắt đầu huấn luyện
    train(model, train_loader, val_loader, args, device, out_dir)

    print(f"[DONE] Hoàn tất huấn luyện. Kết quả lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
