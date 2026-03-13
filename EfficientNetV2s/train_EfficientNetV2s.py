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

from EfficientNetV2s import GRAUppAdvanced
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
    imgs_np = imgs.cpu().numpy()
    gts_np = gts.cpu().numpy()

    for i in range(num_images):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        img_rgb = np.clip(imgs_np[i, :3].transpose(1, 2, 0), 0, 1)
        axs[0].imshow(img_rgb)
        axs[0].set_title(f"Input Sample {i + 1}")
        axs[0].axis("off")

        axs[1].imshow(COLOR_MAP[gts_np[i]])
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(COLOR_MAP[preds[i]])
        axs[2].set_title("Prediction")
        axs[2].axis("off")

        plt.tight_layout()
        save_name = base_path.replace(".png", f"_{i + 1}.png")
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_graphs(df, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    ax[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='.')
    ax[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='.')
    ax[0].set_title("Loss Curves")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(df["epoch"], df["mIoU"], label="Validation mIoU", color='green', marker='.')
    ax[1].set_title("mIoU Progression")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("mIoU")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for cls in range(NUM_CLASSES):
        # Kiểm tra key tồn tại trước khi plot để tránh KeyError
        if f"GT_L{cls}" in df.columns:
            plt.plot(df["epoch"], df[f"GT_L{cls}"], label=f"GT {CLASS_NAMES[cls]}", linestyle='-')
            plt.plot(df["epoch"], df[f"Pred_L{cls}"], label=f"Pred {CLASS_NAMES[cls]}", linestyle='--')
    plt.title("Area Estimation: Ground Truth vs Prediction")
    plt.xlabel("Epoch")
    plt.ylabel("Area (km2)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "area_val_curve.png"), dpi=200)
    plt.close()


def save_logs(history, best_record, out_dir):
    df_history = pd.DataFrame(history)
    excel_path = os.path.join(out_dir, "train_log.xlsx")

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_history.to_excel(writer, sheet_name="Training Log", index=False)
        if best_record is not None:
            pd.DataFrame([best_record]).to_excel(writer, sheet_name="Best Epoch", index=False)

    plot_graphs(df_history, out_dir)


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
        super(DiceLoss, self).__init__()
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
        super(JointLoss, self).__init__()
        self.focal = FocalLoss(alpha=weight, gamma=2.0, ignore_index=IGNORE_INDEX)
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return 0.5 * self.focal(preds, targets) + 0.5 * self.dice(preds, targets)


def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all, targets_all = [], []
    area_gt = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    area_pr = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    vis_data = None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):
            if isinstance(batch, dict):
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
            else:
                imgs, masks = batch
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
                vis_data = (imgs[:5].cpu(), masks[:5].cpu(), preds[:5])

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    ious, miou = compute_iou_global(preds_all, targets_all)
    valid_mask = targets_all != IGNORE_INDEX
    acc = np.mean(preds_all[valid_mask] == targets_all[valid_mask])

    return np.mean(losses), acc, miou, ious, area_gt, area_pr, vis_data

def set_trainable(module, trainable):
    for p in module.parameters():
        p.requires_grad = trainable


def configure_encoder_by_epoch(model, epoch):
    blocks = model.encoder.blocks

    # Freeze toàn bộ
    set_trainable(model.encoder, False)

    # Epoch 1–10 block 5 + decoder
    if epoch <= 10:
        set_trainable(blocks[5], True)

    # Epoch 11–25 unfreeze block4
    elif epoch <= 25:
        set_trainable(blocks[5], True)
        set_trainable(blocks[4], True)

    # Epoch 26+ unfreeze all
    else:
        set_trainable(model.encoder, True)

def build_optimizer(model, base_lr, epoch):
    backbone_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "encoder" in name:
            backbone_params.append(param)
        else:
            decoder_params.append(param)

    # LR backbone theo stage
    if epoch <= 10:
        backbone_lr = 0.1 * base_lr      # chỉ block5 train
    elif epoch <= 25:
        backbone_lr = 0.05 * base_lr     # block4 + block5
    else:
        backbone_lr = 0.1 * base_lr      # full unfreeze

    return torch.optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": decoder_params, "lr": base_lr}
    ])

def train(model, train_loader, val_loader, args, device, out_dir):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MODEL: EfficientNetV2S")
    print(f"PARAMS: {param_count:,}")
    print(f"DEVICE: {device}")
    print(f"OUTPUT: {out_dir}")
    print(f"CONFIG: LR={args.lr} | BS={args.batch_size} | Epochs={args.epochs}")

    scaler = GradScaler()
    weights = torch.tensor(FINAL_CLASS_WEIGHTS, device=device).float()
    criterion = JointLoss(weights)

    history = []
    best_miou = -1.0
    best_epoch_record = None

    # Early Stopping Config
    early_stop_patience = 7
    early_stop_counter = 0

    optimizer = None
    scheduler = None

    for epoch in range(1, args.epochs + 1):

        configure_encoder_by_epoch(model, epoch)

        # Chỉ rebuild optimizer khi stage thay đổi
        if epoch in [1, 11, 26]:
            optimizer = build_optimizer(model, args.lr, epoch)

            # Scheduler tạo 1 lần mỗi stage
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )

        model.train()

        # Giữ BN encoder ở eval khi chưa unfreeze toàn bộ
        if epoch <= 25:
            model.encoder.eval()

        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            if isinstance(batch, dict):
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
            else:
                imgs, masks = batch
                imgs, masks = imgs.to(device), masks.to(device)

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
            pbar.set_postfix({'loss': loss.item()})

        val_loss, acc, miou, ious, area_gt, area_pr, vis = validate(model, val_loader, criterion, device)
        avg_train_loss = train_loss / len(train_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[-1]['lr']

        if np.isnan(avg_train_loss) or np.isnan(val_loss):
            print("Warning: Loss is NaN")
        if np.isnan(miou):
            miou = 0.0
            print("Warning: mIoU is NaN")

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

        # [FIX] Dùng key GT_L{i} thay vì tên class để khớp với plot_graphs
        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "mIoU": miou,
            "accuracy": acc,
            "lr": current_lr,
            **{f"IoU_{name}": ious[i] for i, name in enumerate(CLASS_NAMES)},
            **{f"GT_L{i}": area_gt[f"L{i}"] for i in range(NUM_CLASSES)},
            **{f"Pred_L{i}": area_pr[f"L{i}"] for i in range(NUM_CLASSES)}
        }
        history.append(record)

        # Early Stopping & Save Model
        if miou > best_miou:
            best_miou = miou
            best_epoch_record = record
            early_stop_counter = 0  # Reset
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            if vis:
                save_qualitative(vis[0], vis[1], vis[2], os.path.join(out_dir, "best_vis_result.png"))
            print(f"New Best mIoU: {miou:.4f} -> Saved best_model.pth\n")
        else:
            early_stop_counter += 1
            print(f"No improvement. EarlyStopping Counter: {early_stop_counter}/{early_stop_patience}\n")

        try:
            torch.save(model.state_dict(), os.path.join(out_dir, "last_model.pth"))
        except Exception:
            pass

        save_logs(history, best_epoch_record, out_dir)

        if early_stop_counter >= early_stop_patience:
            print(f"Early Stopping triggered at Epoch {epoch}. Best mIoU: {best_miou:.4f}")
            break


def main():
    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", default="dataset_final")
    parser.add_argument("--output", default="result_EfficientNetV2S")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    if not check_existing_splits(args.processed_path):
        print("Dataset missing or invalid structure.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(args.output, f"RUN_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader, val_loader = create_landcover_loaders(
        args.processed_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = GRAUppAdvanced(
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        deep_supervision=True,
        pretrained=True
    ).to(device)


    train(model, train_loader, val_loader, args, device, out_dir)
    print(f"Finished. Results saved at: {out_dir}")


if __name__ == "__main__":
    main()