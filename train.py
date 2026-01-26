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
    [0, 0, 0],       # 0: Black (Background)
    [139, 0, 0],     # 1: Dark Red
    [255, 165, 0],   # 2: Orange
    [255, 255, 0],   # 3: Yellow
    [50, 205, 50]    # 4: Lime Green
], dtype=np.uint8)


# =========================
# SEED FIXER
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# LOSS
# =========================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        probs = torch.softmax(preds, dim=1)

        targets_fix = targets.clone()
        targets_fix[targets == IGNORE_INDEX] = 0

        onehot = F.one_hot(
            targets_fix, NUM_CLASSES
        ).permute(0, 3, 1, 2).float()

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
        self.ce = nn.CrossEntropyLoss(
            weight=weight, ignore_index=IGNORE_INDEX
        )
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return 0.5 * self.ce(preds, targets) + 0.5 * self.dice(preds, targets)


# =========================
# METRICS & AREA
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


# =========================
# VISUALIZATION
# =========================
def save_qualitative(imgs, gts, preds, base_path, n=5):
    num_images = min(n, imgs.shape[0])

    imgs = imgs.cpu().numpy()
    gts = gts.cpu().numpy()

    for i in range(num_images):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        img_rgb = np.clip(imgs[i, :3].transpose(1, 2, 0), 0, 1)
        axs[0].imshow(img_rgb)
        axs[0].set_title(f"Input (Sample {i+1})")
        axs[0].axis("off")

        axs[1].imshow(COLOR_MAP[gts[i]])
        axs[1].set_title(f"Ground Truth (Sample {i+1})")
        axs[1].axis("off")

        axs[2].imshow(COLOR_MAP[preds[i]])
        axs[2].set_title(f"Prediction (Sample {i+1})")
        axs[2].axis("off")

        plt.tight_layout()
        save_name = base_path.replace(".png", f"_sample_{i+1}.png")
        plt.savefig(save_name, dpi=200, bbox_inches='tight')
        plt.close(fig)


def plot_learning_curves(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(df["epoch"], df["train_loss"], label="Train")
    ax[0].plot(df["epoch"], df["val_loss"], label="Val")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(df["epoch"], df["mIoU"], label="mIoU")
    ax[1].set_title("mIoU")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_area_curve(df, path):
    plt.figure(figsize=(10, 6))
    for cls in range(NUM_CLASSES):
        plt.plot(df["epoch"], df[f"GT_L{cls}"], label=f"GT {CLASS_NAMES[cls]}")
        plt.plot(df["epoch"], df[f"Pred_L{cls}"], "--", label=f"Pred {CLASS_NAMES[cls]}")

    plt.xlabel("Epoch")
    plt.ylabel("Area (km²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# =========================
# VALIDATE
# =========================
def validate(model, loader, criterion, device):
    model.eval()

    losses = []
    preds_all, targets_all = [], []

    area_gt = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    area_pr = {f"L{i}": 0 for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(device), masks.to(device)

            out = model(imgs)
            if isinstance(out, list):
                out = out[-1]

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


# =========================
# TRAIN
# =========================
def train(model, train_loader, val_loader, args, device, out_dir, opt_name="Adam", lr=5e-4):
    if opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif opt_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer {opt_name} not supported.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3
    )
    scaler = GradScaler()

    weights = torch.tensor(
        [0.27, 1.28, 1.98, 1.65, 1.90], device=device
    )
    criterion = JointLoss(weights)

    history = []
    best_miou = -1
    early_trigger = 0
    PATIENCE = 5

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

        print(f"Epoch {epoch} | mIoU: {miou:.4f} | Acc: {acc:.2%}")

        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss,
            "accuracy": acc,
            "mIoU": miou,
        }

        for i in range(NUM_CLASSES):
            row[f"GT_L{i}"] = area_gt[f"L{i}"]
            row[f"Pred_L{i}"] = area_pr[f"L{i}"]

        history.append(row)

        if miou > best_miou:
            best_miou = miou
            early_trigger = 0

            torch.save(
                model.state_dict(),
                os.path.join(out_dir, "best_model.pth")
            )

            imgs, gts, preds = vis
            save_qualitative(
                imgs, gts, preds,
                os.path.join(out_dir, f"best_epoch_{epoch}.png")
            )
        else:
            early_trigger += 1
            if early_trigger >= PATIENCE:
                print(f"[INFO] Early Stopping triggered at Epoch {epoch}.")
                break

    df = pd.DataFrame(history)

    with pd.ExcelWriter(
        os.path.join(out_dir, "train_log.xlsx"),
        engine="xlsxwriter"
    ) as writer:
        df.to_excel(writer, index=False)
        best = df["mIoU"].idxmax()
        writer.sheets["Sheet1"].set_row(
            best + 1, None,
            writer.book.add_format({"bg_color": "#C6EFCE"})
        )

    plot_learning_curves(df, os.path.join(out_dir, "learning_curves.png"))
    plot_area_curve(df, os.path.join(out_dir, "area_val_curve.png"))

    return best_miou


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", default="dataset_final/train")
    parser.add_argument("--output", default="outputs_grid_search")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Starting Grid Search (80 configurations) on {device}...")

    train_ds = LandCoverDataset(args.processed_path, "train", True)
    val_ds = LandCoverDataset(args.processed_path, "val", False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    optimizers = ["Adam", "AdamW", "SGD", "RMSprop"]
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
    seeds = [42, 3407, 2026, 1234, 0]

    total_trials = len(optimizers) * len(learning_rates) * len(seeds)
    grid_results = []
    best_overall_miou = -1
    best_config = None

    timestamp = time.strftime("%Y%m%d_%H%M")
    base_out_dir = os.path.join(args.output, f"MEGA_GRID_{timestamp}")
    os.makedirs(base_out_dir, exist_ok=True)

    trial = 1

    for opt in optimizers:
        for lr in learning_rates:
            for seed in seeds:
                print("-" * 60)
                print(f"TRIAL {trial}/{total_trials}: Opt={opt} | LR={lr} | Seed={seed}")
                print("-" * 60)

                set_seed(seed)
                model = GRAUpp(num_classes=NUM_CLASSES, input_channels=10).to(device)

                trial_dir = os.path.join(base_out_dir, f"{opt}_lr_{lr}_seed_{seed}")
                os.makedirs(trial_dir, exist_ok=True)

                best_miou = train(
                    model, train_loader, val_loader, args, device, trial_dir, opt_name=opt, lr=lr
                )

                grid_results.append({
                    "Optimizer": opt,
                    "Learning_Rate": lr,
                    "Seed": seed,
                    "Best_mIoU": best_miou
                })

                if best_miou > best_overall_miou:
                    best_overall_miou = best_miou
                    best_config = (opt, lr, seed)

                del model
                torch.cuda.empty_cache()
                trial += 1

    print("=" * 60)
    print(f"BEST CONFIGURATION: Opt={best_config[0]} | LR={best_config[1]} | Seed={best_config[2]}")
    print(f"ACHIEVED mIoU: {best_overall_miou:.4f}")
    print("=" * 60)

    df_grid = pd.DataFrame(grid_results)
    df_grid.sort_values(by="Best_mIoU", ascending=False, inplace=True)
    df_grid.to_excel(os.path.join(base_out_dir, "Grid_Search_Summary.xlsx"), index=False)
    print(f"[DONE] Full report saved to: {base_out_dir}")


if __name__ == "__main__":
    main()