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
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from thop import profile
except ImportError:
    profile = None

# Đổi import sang model Ghost mới
from cvn2_tiny import GhostGRAUpp_ConvNeXtV2 
from preprocess_data import LandCoverDataset, check_existing_splits
from torch.utils.data import DataLoader

# GLOBAL CONFIG
NUM_CLASSES = 5
IGNORE_INDEX = 255
INPUT_CHANNELS = 12
CLASS_NAMES = ["Background", "Level 1", "Level 2", "Level 3", "Level 4"]

PIXEL_SIZE_METERS = 5
PIXEL_AREA_KM2 = (PIXEL_SIZE_METERS ** 2) / 1_000_000
FINAL_CLASS_WEIGHTS = [0.3680, 1.3597, 2.0944, 1.7449, 2.0141]

# Kỹ thuật huấn luyện
WARMUP_EPOCHS = 5
BACKBONE_LR = 5e-5
EARLY_STOP_PATIENCE = 15 # Tăng nhẹ patience cho bản Ghost vì nó nhẹ và ổn định hơn

warnings.filterwarnings("ignore")

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def compute_model_complexity(model, input_size, device):
    if profile is None: return 0, 0
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        return params, macs * 2
    except: return 0, 0

def compute_iou_global(preds, targets):
    ious = []
    valid = targets != IGNORE_INDEX
    for cls in range(NUM_CLASSES):
        p, t = (preds == cls) & valid, (targets == cls) & valid
        tp = np.logical_and(p, t).sum()
        denom = np.logical_or(p, t).sum()
        ious.append(tp / denom if denom > 0 else np.nan)
    return ious, np.nanmean(ious)

def compute_area_from_mask(mask):
    areas = {}
    valid = mask != IGNORE_INDEX
    for cls in range(NUM_CLASSES):
        areas[f"L{cls}"] = np.logical_and(mask == cls, valid).sum() * PIXEL_AREA_KM2
    return areas

class JointLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weight).float() if weight else None, ignore_index=IGNORE_INDEX)
    def forward(self, preds, targets):
        return self.ce(preds, targets)

def validate(model, loader, criterion, device):
    model.eval()
    losses, preds_all, targets_all = [], [], []
    area_gt = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    area_pr = {f"L{i}": 0 for i in range(NUM_CLASSES)}
    vis_data = None
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(loader, desc="Validating", leave=False, ncols=100)):
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
                gt, pr = compute_area_from_mask(masks_np[b]), compute_area_from_mask(preds[b])
                for k in gt:
                    area_gt[k], area_pr[k] = area_gt[k] + gt[k], area_pr[k] + pr[k]
            if i == 0: vis_data = (imgs[:5].cpu(), masks[:5].cpu(), preds[:5])
    preds_all, targets_all = np.concatenate(preds_all), np.concatenate(targets_all)
    ious, miou = compute_iou_global(preds_all, targets_all)
    acc = np.mean(preds_all[targets_all != IGNORE_INDEX] == targets_all[targets_all != IGNORE_INDEX])
    return np.mean(losses), acc, miou, ious, area_gt, area_pr, vis_data

def save_qualitative(imgs, gts, preds, base_path, n=5):
    num_images = min(n, imgs.shape[0])
    imgs, gts = imgs.numpy(), gts.numpy()
    for i in range(num_images):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(np.clip(imgs[i, :3].transpose(1, 2, 0), 0, 1)); axs[0].axis("off")
        axs[1].imshow(gts[i], vmin=0, vmax=NUM_CLASSES-1); axs[1].axis("off")
        axs[2].imshow(preds[i], vmin=0, vmax=NUM_CLASSES-1); axs[2].axis("off")
        plt.tight_layout()
        plt.savefig(base_path.replace(".png", f"_{i+1}.png"), dpi=150); plt.close(fig)

def train(model, train_loader, val_loader, args, device, out_dir):
    params_num, flops_num = compute_model_complexity(model, (1, INPUT_CHANNELS, 128, 128), device)
    flops_g = flops_num / 1e9
    
    backbone_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if "encoder" not in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 0},
        {'params': decoder_params, 'lr': args.lr}
    ], weight_decay=0.01)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    criterion = JointLoss(weight=FINAL_CLASS_WEIGHTS).to(device)

    print(f"MODEL : Ghost-GRAUpp + ConvNeXt V2 Tiny\nPARAMS: {params_num:,}\nFLOPs : {flops_g:.2f} G\nCONFIG: LR={args.lr} | BS={args.batch_size}\n" + "-"*75)

    best_miou, history = -1.0, []
    best_record = {}
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        if epoch <= WARMUP_EPOCHS:
            for p in backbone_params: p.requires_grad = False
            status = "Backbone FROZEN"
        else:
            for p in backbone_params: p.requires_grad = True
            optimizer.param_groups[0]['lr'] = BACKBONE_LR
            status = "Backbone UNFROZEN"

        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", ncols=100)
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(imgs)
                loss = sum(criterion(o, masks) for o in out)/len(out) if isinstance(out, list) else criterion(out, masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'state': status[:6]})

        val_loss, acc, miou, ious, area_gt, area_pr, vis = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {miou:.4f} | Acc: {acc:.2%} | LR: {optimizer.param_groups[-1]['lr']:.1e}")
        print("-" * 75 + f"\n{'Class Name':<15} | {'IoU':<10} | {'Pred (km2)':<12} | {'GT (km2)':<12} | {'Diff'}\n" + "-" * 75)
        
        current_record = {
            "epoch": epoch, 
            "train_loss": avg_train_loss, 
            "val_loss": val_loss, 
            "mIoU": miou, 
            "accuracy": acc
        }
        
        for i, name in enumerate(CLASS_NAMES):
            iou_v = f"{ious[i]:.4f}" if not np.isnan(ious[i]) else "N/A"
            ap, ag = area_pr[f"L{i}"], area_gt[f"L{i}"]
            current_record[f"IoU_{name}"] = ious[i]
            print(f"{name:<15} | {iou_v:<10} | {ap:<12.4f} | {ag:<12.4f} | {ap-ag:+.4f}")
        print("-" * 75)

        if miou > best_miou:
            best_miou = miou
            patience_counter = 0
            
            best_record = current_record.copy()
            best_record["Params"] = f"{params_num:,}"
            best_record["FLOPs (G)"] = f"{flops_g:.2f}"
            
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            print(f"New Best mIoU: {miou:.4f} -> Saved best_model.pth\n")
            if vis: save_qualitative(vis[0], vis[1], vis[2], os.path.join(out_dir, "best_vis.png"))
        else:
            patience_counter += 1
            print(f"No improvement in mIoU for {patience_counter} epochs.\n")

        history.append(current_record)
        
        with pd.ExcelWriter(os.path.join(out_dir, "training_log.xlsx"), engine="xlsxwriter") as writer:
            pd.DataFrame(history).to_excel(writer, sheet_name="Training Log", index=False)
            if best_record:
                pd.DataFrame([best_record]).to_excel(writer, sheet_name="Best Epoch", index=False)

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs. Training finished.")
            break

def main():
    set_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", default="dataset_final")
    parser.add_argument("--output", default="final_result_ghost")
    parser.add_argument("--epochs", type=int, default=150); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    if not check_existing_splits(args.processed_path): return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.output, f"GHOST_RUN_{time.strftime('%Y%m%d_%H%M%S')}"); os.makedirs(out_dir, exist_ok=True)

    train_ds = LandCoverDataset(args.processed_path, "train", True)
    val_ds = LandCoverDataset(args.processed_path, "val", False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker)

    # Khởi tạo mô hình Ghost mới
    model = GhostGRAUpp_ConvNeXtV2(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS, pretrained=True).to(device)
    train(model, train_loader, val_loader, args, device, out_dir)

if __name__ == "__main__": main()