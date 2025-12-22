import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
from torch.utils.data import DataLoader
from models.graupp import GRAUpp
from preprocess_data import FetalHeadMHADataset
from analyze_labels import analyze_dataset
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F


# -------------------- Metrics --------------------
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
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


# -------------------- Grad-CAM++ --------------------
class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_mask=None):
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        out = outputs[-1] if isinstance(outputs, list) else outputs

        if target_mask is not None:
            score = torch.sum(out * target_mask.to(out.device))
        else:
            score = torch.sum(out)

        score.backward(retain_graph=True)

        grads = self.gradients      # B x C x h x w
        acts = self.activations     # B x C x h x w

        with torch.no_grad():
            weights = torch.mean(grads ** 2, dim=(2, 3), keepdim=True)
            numerator = torch.sum(weights * acts, dim=(2, 3))
            denominator = torch.sum(weights * (acts ** 2), dim=(2, 3)) * 2.0 + 1e-8
            alpha = numerator / denominator
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)

            w = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)
            cam = torch.sum(w * acts, dim=1, keepdim=True)
            cam = F.relu(cam)

            cam_min = cam.view(cam.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
            cam_max = cam.view(cam.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            cam_up = F.interpolate(
                cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
            )

        return cam_up.cpu().numpy()


def get_target_layer(model):
    if hasattr(model, "encoder"):
        enc = model.encoder

        for block in ["layer4", "layer3"]:
            if hasattr(enc, block):
                blk = getattr(enc, block)
                for m in reversed(list(blk.modules())):
                    if isinstance(m, torch.nn.Conv2d):
                        return m

        if hasattr(enc, "conv1"):
            for m in enc.conv1:
                if isinstance(m, torch.nn.Conv2d):
                    return m

    return None


# -------------------- Saving helpers --------------------
def save_mha(array_uint8, out_path):
    img = sitk.GetImageFromArray(array_uint8)
    sitk.WriteImage(img, out_path)


# -------------------- MAIN TEST FUNCTION --------------------
def test_model(model_path, image_dir, label_dir, output_dir,
               batch_size=4, img_size=(256, 256), save_heatmaps=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", device)
    print(f"Đang load model từ: {model_path}")

    model = GRAUpp(num_classes=1, input_channels=1, pretrained=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataset = FetalHeadMHADataset(image_dir, label_dir, img_size=img_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Tổng số ảnh test: {len(dataset)}")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, f"run_test_{timestamp}")
    pred_dir = os.path.join(run_dir, "predictions")
    qual_dir = os.path.join(run_dir, "qualitative")
    heat_dir = os.path.join(run_dir, "heatmaps")

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(qual_dir, exist_ok=True)
    os.makedirs(heat_dir, exist_ok=True)

    dice_list, iou_list = [], []
    per_image_metrics = []

    gradcam = None
    if save_heatmaps:
        cam_target = get_target_layer(model)
        if cam_target is not None:
            gradcam = GradCAMpp(model, cam_target)
        else:
            print("Không tìm thấy layer Grad-CAM++.")

    loop = tqdm(dataloader, desc="Đang test model")

    for batch_idx, (imgs, masks) in enumerate(loop):
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.no_grad():
            preds = model(imgs)

        batch_dice = dice_coef(preds, masks)
        batch_iou = iou_score(preds, masks)

        dice_list.append(batch_dice)
        iou_list.append(batch_iou)
        loop.set_postfix(dice=batch_dice, iou=batch_iou)

        if isinstance(preds, list):
            preds = preds[-1]

        preds_bin = (torch.sigmoid(preds) > 0.5).float().cpu().numpy()
        imgs_np = imgs.cpu().numpy()
        masks_np = masks.cpu().numpy()

        B = preds_bin.shape[0]

        for j in range(B):
            idx = batch_idx * batch_size + j
            if idx >= len(dataset):
                break

            fname = dataset.files[idx]
            fname_stem = Path(fname).stem

            # ---------------- SAVING MHA ----------------
            save_mha(preds_bin[j, 0].astype(np.uint8),
                     os.path.join(pred_dir, fname))

            per_image_metrics.append({
                "Image": fname,
                "Dice": float(batch_dice),
                "IoU": float(batch_iou)
            })

            inp = imgs_np[j, 0]
            gt = masks_np[j, 0]
            pr = preds_bin[j, 0]

            # ------------------- QUALITATIVE (2×2) -------------------
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))

            axs[0, 0].imshow(inp, cmap="gray"); axs[0, 0].set_title("Input")
            axs[0, 1].imshow(gt, cmap="gray"); axs[0, 1].set_title("Ground Truth")
            axs[1, 0].imshow(pr, cmap="gray"); axs[1, 0].set_title("Prediction")

            overlay = np.zeros((inp.shape[0], inp.shape[1], 3), dtype=np.uint8)
            overlay[..., 1] = (gt * 255).astype(np.uint8)
            overlay[..., 0] = (pr * 255).astype(np.uint8)
            axs[1, 1].imshow(overlay); axs[1, 1].set_title("Overlay")

            for ax in axs.flatten():
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(qual_dir, f"{fname_stem}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

            # ------------------- HEATMAPS (1×3) -------------------
            if gradcam is not None:
                img_tensor = imgs[j:j+1].detach().clone().to(device)
                img_tensor.requires_grad_(True)

                model.zero_grad()
                cam = gradcam.generate(img_tensor)
                cam_map = cam[0, 0]

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                axs[0].imshow(inp, cmap="gray")
                axs[0].set_title("Input")

                axs[1].imshow(cam_map, cmap="jet")
                axs[1].set_title("CAM Raw")

                inp_uint8 = (np.clip(inp, 0, 1) * 255).astype(np.uint8)
                heat = (plt.get_cmap("jet")(cam_map)[:, :, :3] * 255).astype(np.uint8)
                overlay_hm = (0.6 * heat + 0.4 * np.stack([inp_uint8]*3, axis=-1)).astype(np.uint8)
                axs[2].imshow(overlay_hm)
                axs[2].set_title("CAM Overlay")

                for ax in axs:
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(heat_dir, f"{fname_stem}_heat.png"),
                            dpi=150, bbox_inches="tight")
                plt.close()

    # ---------------- SUMMARY ----------------
    mean_dice = np.mean(dice_list)
    mean_iou = np.mean(iou_list)

    print("\n========== KẾT QUẢ TEST ==========")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU : {mean_iou:.4f}")
    print("=================================")

    df_metrics = pd.DataFrame(per_image_metrics)
    df_metrics.to_excel(os.path.join(run_dir, "per_image_metrics.xlsx"), index=False)

    analyze_dataset(dataset, save_path=os.path.join(run_dir, "test_label_statistics.xlsx"))

    df_summary = pd.DataFrame([{
        "Model_Path": str(model_path),
        "Total_Images": len(dataset),
        "Mean_Dice": mean_dice,
        "Mean_IoU": mean_iou,
        "Batch_Size": batch_size,
        "Img_Size": img_size
    }])

    df_summary.to_excel(os.path.join(run_dir, "summary&results.xlsx"), index=False)

    print(f"Kết quả đã lưu trong: {run_dir}")
    return mean_dice, mean_iou


# -------------------- MAIN --------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / "outputs"

    runs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )
    if len(runs) == 0:
        raise FileNotFoundError("Không tìm thấy thư mục run_* trong outputs/")
    latest_run = runs[-1]

    model_path = latest_run / "GRAUpp_best.pth"
    if not model_path.exists():
        alt = latest_run / "GhostResAttentionUNetPP_best.pth"
        if alt.exists():
            model_path = alt
        else:
            raise FileNotFoundError(f"Không tìm thấy file model trong {latest_run}")

    image_dir = r"E:\pycharmfetalheadsegmentation\dataset\test\image_mha"
    label_dir = r"E:\pycharmfetalheadsegmentation\dataset\test\label_mha"

    output_test_dir = base_dir / "outputs_test"
    output_test_dir.mkdir(exist_ok=True)

    test_model(
        str(model_path),
        str(image_dir),
        str(label_dir),
        str(output_test_dir),
        batch_size=4,
        img_size=(256, 256),
        save_heatmaps=True
    )
