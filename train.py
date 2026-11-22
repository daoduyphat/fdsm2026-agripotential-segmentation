import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data_pipeline.dataset import create_dataloaders
from metrics import CLASS_WEIGHTS, IGNORE_INDEX, NUM_CLASSES, compute_metrics, update_confusion_matrix
from models import build_model


class JointLoss(nn.Module):
    """Equal-weighted (0.5 / 0.5) Weighted Cross-Entropy and Multi-class Dice Loss."""

    def __init__(self, weights, ignore_index=255):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]

        ce_loss = F.cross_entropy(logits, targets, weight=self.weights, ignore_index=self.ignore_index)

        valid_mask = targets != self.ignore_index
        targets_clean = torch.where(valid_mask, targets, torch.zeros_like(targets))
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets_clean, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1).float()

        probs = probs * valid_mask
        one_hot = one_hot * valid_mask
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6)).mean()

        return 0.5 * ce_loss + 0.5 * dice_loss


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    conf_mat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[-1]
        total_loss += criterion(outputs, masks).item()

        preds = outputs.argmax(dim=1)
        conf_mat = update_confusion_matrix(conf_mat, preds, masks)

    _, miou, oa = compute_metrics(conf_mat)
    return total_loss / max(len(loader), 1), miou, oa


def main():
    parser = argparse.ArgumentParser(description="Train segmentation models on AgriPotential.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "unetpp", "deeplabv3", "ghost_unetpp", "convnext_upernet"])
    parser.add_argument("--data-root", type=str, default="dataset_final")
    parser.add_argument("--split-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        args.data_root, args.split_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = build_model(args.model, input_channels=12, num_classes=NUM_CLASSES).to(device)
    criterion = JointLoss(CLASS_WEIGHTS, ignore_index=IGNORE_INDEX).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_miou = 0.0
    patience_counter = 0
    ckpt_path = os.path.join(args.output_dir, f"{args.model}_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        val_loss, val_miou, val_oa = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | mIoU: {val_miou:.4f} | OA: {val_oa:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print(f"Early stopping triggered at epoch {epoch}. Best mIoU: {best_miou:.4f}")
                break


if __name__ == "__main__":
    main()
