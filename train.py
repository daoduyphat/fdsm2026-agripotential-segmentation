import os
import random
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torchsummary import summary
import SimpleITK as sitk
import scipy.ndimage as ndimage
import torch.nn.functional as F

# Model
from graupp import GRAUpp


# ========================= DATASET ===============================
class FetalHeadDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=(256, 256), normalize=True, augment=False):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.files = sorted([f for f in os.listdir(self.image_dir) if f in os.listdir(self.label_dir)])
        if len(self.files) == 0:
            raise RuntimeError(f"không khớp giữa {image_dir} và {label_dir}")
        self.img_size = img_size
        self.normalize = normalize
        self.augment = augment
        self.resize = T.Resize(img_size)

    def __len__(self):
        return len(self.files)

    def load_mha(self, path):
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        if array.ndim == 3:
            array = array[0]
        return array

    def crop_nonzero(self, image, mask):
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return image, mask
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        image = image[y_min:y_max + 1, x_min:x_max + 1]
        mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        return image, mask

    def normalize_image(self, image):
        imin, imax = np.min(image), np.max(image)
        if imax - imin == 0:
            return np.zeros_like(image)
        return (image - imin) / (imax - imin)

    def augment_image(self, image, mask):
        angle = random.uniform(-30, 30)
        scale = random.uniform(0.9, 1.1)
        brightness = random.uniform(0.8, 1.2)
        image = ndimage.rotate(image, angle, reshape=False, order=1)
        mask = ndimage.rotate(mask, angle, reshape=False, order=0)
        image = ndimage.zoom(image, scale, order=1)
        mask = ndimage.zoom(mask, scale, order=0)
        if random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        if random.random() < 0.3:
            image = ndimage.gaussian_filter(image, sigma=random.uniform(0.5, 1.0))
        image = np.clip(image * brightness, 0, 1)
        return image, mask

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = self.image_dir / fname
        lbl_path = self.label_dir / fname

        image = self.load_mha(img_path)
        label = self.load_mha(lbl_path)

        image = ndimage.gaussian_filter(image, sigma=0.5)
        image, label = self.crop_nonzero(image, label)
        image = self.normalize_image(image)

        if self.augment:
            image, label = self.augment_image(image, label)

        image = Image.fromarray((image * 255).astype(np.uint8))
        label = Image.fromarray((label > 0.5).astype(np.uint8) * 255)

        image = self.resize(image)
        label = self.resize(label)

        image = T.ToTensor()(image)
        label = T.ToTensor()(label)
        label = (label > 0.5).float()
        return image.float(), label.float()


# ========================= METRICS ===============================
def dice_coef(preds, targets, smooth=1e-6):
    if isinstance(preds, list):
        preds = preds[-1]
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (denom + smooth)
    return dice.mean().item()


def iou_score(preds, targets, smooth=1e-6):
    if isinstance(preds, list):
        preds = preds[-1]
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()


# ========================= LOSS ===============================
class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, bce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, preds, targets):
        if isinstance(preds, list):
            preds = preds[-1]
        bce_loss = self.bce(preds, targets)
        preds_sig = torch.sigmoid(preds)
        inter = (preds_sig * targets).sum(dim=(1, 2, 3))
        dice_loss = 1 - (2 * inter + self.smooth) / (
                preds_sig.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss.mean()


# ========================= GRAD-CAM++ (fixed) ===============================
class GradCAMpp:
    """
    Lightweight Grad-CAM++ implementation. Register on a conv layer module object.
    NOTE: hooks must NOT detach outputs/gradients else backward won't work.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Save references to tensors (do NOT detach here)
        def forward_hook(module, input, output):
            self.activations = output  # tensor requiring grad for backward path
        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] is gradient w.r.t. the forward output
            self.gradients = grad_out[0]
        # register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_mask=None):
        """
        input_tensor: 1xCxHxW (requires grad enabled outside)
        returns: cam_up numpy array shape (1,1,H,W) or (H,W) depending caller
        """
        # ensure grads are zeroed
        self.model.zero_grad()
        outputs = self.model(input_tensor)  # forward with grad enabled
        output = outputs[-1] if isinstance(outputs, list) else outputs  # 1x1xH'xW'
        if target_mask is not None:
            score = torch.sum(output * target_mask.to(output.device))
        else:
            score = torch.sum(output)
        # backward to populate gradients (hooks will capture grads & activations)
        score.backward(retain_graph=True)

        grads = self.gradients  # BxCxHxW
        acts = self.activations  # BxCxHxW

        # compute Grad-CAM++ (on CPU/with no grad)
        with torch.no_grad():
            # weights per channel
            weights = torch.mean(grads ** 2, dim=(2, 3), keepdim=True)  # BxCx1x1
            numerator = torch.sum(weights * acts, dim=(2, 3))
            denominator = torch.sum(weights * (acts ** 2), dim=(2, 3)) * 2.0 + 1e-8
            alpha = numerator / denominator  # BxC
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)  # BxCx1x1
            weights_final = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)  # Bx1x1x1
            cam = torch.sum(weights_final * acts, dim=1, keepdim=True)  # Bx1xHxW
            cam = F.relu(cam)
            cam_min = cam.view(cam.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
            cam_max = cam.view(cam.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            cam_up = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam_up.cpu().numpy()  # B x 1 x H x W


# ========================= QUALITATIVE & HEATMAP SAVER (combined files) ===============================
def _make_overlay_rgb(inp, gt_mask, pred_mask):
    """
    inp: HxW float [0,1]
    gt_mask, pred_mask: HxW binary {0,1}
    returns RGB uint8 overlay where:
     - background is grayscale inp,
     - GT highlighted green, Pred highlighted red, overlap yellow.
    """
    inp_uint8 = (np.clip(inp, 0, 1) * 255).astype(np.uint8)
    h, w = inp_uint8.shape
    base_rgb = np.stack([inp_uint8, inp_uint8, inp_uint8], axis=-1).astype(np.float32)
    color_mask = np.zeros((h, w, 3), dtype=np.float32)
    # GT -> green
    color_mask[..., 1] = gt_mask * 255.0
    # Pred -> red
    color_mask[..., 0] = pred_mask * 255.0
    # blend: alpha where mask present
    mask_any = np.clip(gt_mask + pred_mask, 0, 1)
    alpha = (0.7 * mask_any)[..., None]  # how strongly color appears
    overlay = (alpha * color_mask + (1 - alpha) * base_rgb).astype(np.uint8)
    return overlay


def generate_qualitative(model, val_loader, device, save_dir, num_samples=10):
    """
    For each sample create ONE image (2x2 grid):
     [ input | gt ]
     [ pred  | overlay ]
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = outputs[-1] if isinstance(outputs, list) else outputs
            preds_sig = torch.sigmoid(preds)
            preds_bin = (preds_sig > 0.5).float()

            for i in range(imgs.size(0)):
                if count >= num_samples:
                    return
                inp = imgs[i, 0].cpu().numpy()  # HxW in [0,1]
                gt = masks[i, 0].cpu().numpy()  # HxW {0,1}
                pr = preds_bin[i, 0].cpu().numpy()  # HxW {0,1}

                # overlay rgb (grayscale background + colored masks)
                overlay_rgb = _make_overlay_rgb(inp, gt, pr)

                # create figure 2x2
                fig, axs = plt.subplots(2, 2, figsize=(6, 6))
                axs[0, 0].imshow(inp, cmap="gray")
                axs[0, 0].set_title("Input")
                axs[0, 0].axis("off")

                axs[0, 1].imshow(gt, cmap="gray")
                axs[0, 1].set_title("GT")
                axs[0, 1].axis("off")

                axs[1, 0].imshow(pr, cmap="gray")
                axs[1, 0].set_title("Pred")
                axs[1, 0].axis("off")

                axs[1, 1].imshow(overlay_rgb)
                axs[1, 1].set_title("Overlay (GT green / Pred red)")
                axs[1, 1].axis("off")

                outpath = os.path.join(save_dir, f"sample_{count:03d}_qualitative.png")
                plt.tight_layout()
                plt.savefig(outpath, dpi=150)
                plt.close(fig)

                count += 1


def get_target_layer(model):
    """
    Find last Conv2d in encoder.layer4 or layer3, else conv1.
    """
    if hasattr(model, "encoder"):
        enc = model.encoder
        for candidate in ["layer4", "layer3"]:
            if hasattr(enc, candidate):
                layer = getattr(enc, candidate)
                # search reversed modules for Conv2d
                for m in reversed(list(layer.modules())):
                    if isinstance(m, nn.Conv2d):
                        return m
        # fallback: conv1 inside encoder.conv1 sequential
        if hasattr(enc, "conv1"):
            seq = enc.conv1
            for m in seq:
                if isinstance(m, nn.Conv2d):
                    return m
    return None


def generate_heatmaps(model, val_loader, device, save_dir, num_samples=10):
    """
    For each sample create ONE 2x3 figure:
    Row1: Input | GT | Pred
    Row2: CAM raw (jet) | CAM overlay (jet on input) | Combined overlay (mask + cam)
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    target_layer = get_target_layer(model)
    if target_layer is None:
        print("Không tìm thấy layer phù hợp cho Grad-CAM. Bỏ qua heatmap.")
        return

    gradcam = GradCAMpp(model, target_layer)
    count = 0

    # we need gradients for grad-cam -> enable grad during forward/backward
    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        for i in range(imgs.size(0)):
            if count >= num_samples:
                return

            img = imgs[i:i + 1]  # 1xCxHxW
            mask = masks[i:i + 1]

            # compute prediction (no-grad ok)
            with torch.no_grad():
                outputs = model(img)
                pred = outputs[-1] if isinstance(outputs, list) else outputs
                pred_sig = torch.sigmoid(pred)
                pred_bin = (pred_sig > 0.5).float()

            # run grad-cam with gradient enabled
            with torch.enable_grad():
                cam = gradcam.generate(img, target_mask=None)  # Bx1xHxW as numpy

            cam_map = cam[0, 0]  # HxW in [0,1]
            inp = img[0, 0].detach().cpu().numpy()
            gt = mask[0, 0].detach().cpu().numpy()
            pr = pred_bin[0, 0].detach().cpu().numpy()

            # color maps
            cmap = plt.get_cmap("jet")
            cam_rgb = (cmap(cam_map)[:, :, :3] * 255).astype(np.uint8)
            inp_uint8 = (np.clip(inp, 0, 1) * 255).astype(np.uint8)
            inp_rgb = np.stack([inp_uint8] * 3, axis=-1)

            # overlay cam on input
            cam_overlay = (0.6 * cam_rgb.astype(np.float32) + 0.4 * inp_rgb.astype(np.float32)).astype(np.uint8)

            # combined overlay: also show mask edges (red/green) on top of cam_overlay
            combined = cam_overlay.copy().astype(np.uint8)
            # add colored masks: GT green, Pred red (where present, override lightly)
            combined = combined.astype(np.float32)
            combined[..., 1] = np.clip(combined[..., 1] + gt * 100, 0, 255)  # boost green where GT
            combined[..., 0] = np.clip(combined[..., 0] + pr * 100, 0, 255)  # boost red where Pred
            combined = combined.astype(np.uint8)

            # make the 2x3 figure
            fig, axs = plt.subplots(2, 3, figsize=(12, 8))

            axs[0, 0].imshow(inp, cmap="gray"); axs[0, 0].set_title("Input"); axs[0, 0].axis("off")
            axs[0, 1].imshow(gt, cmap="gray"); axs[0, 1].set_title("GT"); axs[0, 1].axis("off")
            axs[0, 2].imshow(pr, cmap="gray"); axs[0, 2].set_title("Pred"); axs[0, 2].axis("off")

            axs[1, 0].imshow(cam_map, cmap="jet"); axs[1, 0].set_title("CAM Raw (jet)"); axs[1, 0].axis("off")
            axs[1, 1].imshow(cam_overlay); axs[1, 1].set_title("CAM on Input"); axs[1, 1].axis("off")
            axs[1, 2].imshow(combined); axs[1, 2].set_title("CAM + Masks"); axs[1, 2].axis("off")

            outpath = os.path.join(save_dir, f"sample_{count:03d}_heatmap.png")
            plt.tight_layout()
            plt.savefig(outpath, dpi=150)
            plt.close(fig)

            count += 1


# ========================= VALIDATION ===============================
def valid_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    dices = []
    ious = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks).item()
            val_loss += loss
            dices.append(dice_coef(preds, masks))
            ious.append(iou_score(preds, masks))

    return val_loss / len(val_loader), float(np.mean(dices)), float(np.mean(ious))


# ========================= TRAINING ===============================
def train_model(model, train_loader, val_loader, device, epochs, lr, output_dir,
                model_name="model", patience=5, pos_weight=None):

    model.to(device)

    try:
        summary(model, (1, 256, 256))
    except Exception:
        summary(model, (3, 256, 256))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số trainable của {model_name}: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = WeightedBCEDiceLoss(pos_weight=pos_weight)
    best_dice = 0.0

    history = {"train_loss": [], "val_loss": [], "dice": [], "iou": [], "params": total_params}
    start_time = time.time()

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_txt = os.path.join(log_dir, f"{model_name}_training_log.txt")

    with open(log_txt, "w", encoding="utf-8") as f:
        f.write("Epoch\tTrainLoss\tValLoss\tDice\tIoU\n")

    # ----------------------------
    # MAIN TRAIN LOOP
    # ----------------------------
    for epoch in range(1, epochs + 1):

        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"{model_name} Epoch {epoch}/{epochs}")

        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_dice, val_iou = valid_model(model, val_loader, criterion, device)

        print(f"{model_name} Epoch {epoch}/{epochs} | "
              f"Train Loss {avg_train_loss:.4f} | Val Loss {val_loss:.4f} | "
              f"Dice {val_dice:.4f} | IoU {val_iou:.4f}")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["dice"].append(val_dice)
        history["iou"].append(val_iou)

        # Save training log
        with open(log_txt, "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{avg_train_loss:.4f}\t{val_loss:.4f}\t{val_dice:.4f}\t{val_iou:.4f}\n")

        scheduler.step(val_dice)

        # ----------------------------------------
        # SAVE BEST MODEL + QUALITATIVE IMAGES + HEATMAPS
        # ----------------------------------------
        if val_dice > best_dice:
            best_dice = val_dice

            best_path = os.path.join(output_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), best_path)

            print(f"🎉 Lưu BEST MODEL: {model_name} (Dice={best_dice:.4f})")

            # === SAVE 10 QUALITATIVE SAMPLES ===
            qual_dir = os.path.join(output_dir, "qualitative")
            print("   → Tạo ảnh qualitative (10 mẫu)...")
            generate_qualitative(model, val_loader, device, qual_dir, num_samples=10)

            # === SAVE 10 HEATMAPS ===
            heat_dir = os.path.join(output_dir, "heatmaps")
            print("   → Tạo heatmap Grad-CAM++ (10 mẫu)...")
            generate_heatmaps(model, val_loader, device, heat_dir, num_samples=10)

    # END TRAIN LOOP
    total_time = time.time() - start_time
    history["time"] = total_time

    # Save training history to excel
    df_log = pd.DataFrame({
        "Epoch": range(1, len(history["train_loss"]) + 1),
        "Train Loss": history["train_loss"],
        "Val Loss": history["val_loss"],
        "Dice": history["dice"],
        "IoU": history["iou"]
    })

    df_log.to_excel(os.path.join(log_dir, f"{model_name}_training_log.xlsx"), index=False)
    print(f"Đã lưu log huấn luyện vào {log_dir}")

    return history


# ========================= MAIN ===============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="outputs")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--img-size", type=int, nargs=2, default=(256, 256))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", device)

    train_dataset = FetalHeadDataset(
        "dataset/train/image_mha", "dataset/train/label_mha",
        img_size=tuple(args.img_size), augment=True
    )
    val_dataset = FetalHeadDataset(
        "dataset/val/image_mha", "dataset/val/label_mha",
        img_size=tuple(args.img_size), augment=False
    )

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Thư mục lưu kết quả: {run_dir}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # ---------------- LABEL STATS ----------------
    print("Thống kê nhãn trong tập huấn luyện...")
    all_labels = []

    for _, masks in tqdm(train_loader, desc="Đếm nhãn"):
        masks = masks.view(-1)
        all_labels.append(masks)

    all_labels = torch.cat(all_labels)
    unique, counts = torch.unique(all_labels, return_counts=True)

    print("Các lớp:", unique.tolist())
    print("Số lượng:", counts.tolist())

    label_stats = pd.DataFrame({"class": unique.tolist(), "count": counts.tolist()})
    label_stats.to_excel(os.path.join(run_dir, "label_statistics.xlsx"), index=False)

    txt_path = os.path.join(run_dir, "label_statistics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Thống kê nhãn trong tập huấn luyện\n")
        f.write(f"Các lớp: {unique.tolist()}\n")
        f.write(f"Số lượng: {counts.tolist()}\n")

    # Weight for BCE
    class_weights = 1.0 / (counts.float() + 1e-6)
    class_weights = class_weights / class_weights.sum()
    pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else torch.tensor(1.0)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    print("Trọng số lớp (pos_weight):", pos_weight.item())

    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"Trọng số lớp (pos_weight): {pos_weight.item()}\n")

    # ---------------- TRAIN MODEL ----------------
    histories = {}

    print("=== Model: GRAU++ (GhostResAttentionUNetPP) ===")
    histories["GRAUpp"] = train_model(
        GRAUpp(num_classes=1, input_channels=1, pretrained=False, deep_supervision=True, attention=None),
        train_loader, val_loader, device,
        args.epochs, args.lr, run_dir,
        "GRAUpp",
        args.patience,
        pos_weight=pos_weight
    )

    # ---------------- SAVE METRICS PLOTS ----------------
    max_len = max(len(h["train_loss"]) for h in histories.values())
    df = pd.DataFrame({"Epoch": list(range(1, max_len + 1))})

    for model_name, h in histories.items():
        def pad(lst): return lst + [np.nan] * (max_len - len(lst))

        df[f"{model_name}_Train_Loss"] = pad(h["train_loss"])
        df[f"{model_name}_Val_Loss"] = pad(h["val_loss"])
        df[f"{model_name}_Dice"] = pad(h["dice"])
        df[f"{model_name}_IoU"] = pad(h["iou"])

    plt.figure(figsize=(18, 5))
    for i, metric in enumerate(["Train_Loss", "Val_Loss", "Dice", "IoU"]):
        plt.subplot(1, 4, i + 1)
        for model_name in histories.keys():
            col = f"{model_name}_{metric}"
            plt.plot(df["Epoch"], df[col], label=model_name, marker="o")
        plt.xlabel("Epoch")
        plt.title(metric)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "all_metrics.png"), dpi=150)
    plt.close()

    # ---------------- SUMMARY ----------------
    summary_rows = []
    for model_name, h in histories.items():
        best_epoch = int(np.nanargmax(h["dice"])) + 1
        best_dice = np.nanmax(h["dice"])
        best_iou = np.nanmax(h["iou"])
        summary_rows.append({
            "Model": model_name,
            "Trainable_Params": h["params"],
            "Best_Epoch": best_epoch,
            "Best_Dice": best_dice,
            "Best_IoU": best_iou,
            "Time_Train(s)": round(h["time"], 2),
            "LR": args.lr,
            "Batch_Size": args.batch_size,
            "Epochs": args.epochs,
            "Img_Size": tuple(args.img_size),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(run_dir, "summary&results.xlsx")

    with pd.ExcelWriter(summary_path) as writer:
        df.to_excel(writer, sheet_name="Training_History", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"đã lưu toàn bộ kết quả vào: {summary_path}")
    print("train xong.")
