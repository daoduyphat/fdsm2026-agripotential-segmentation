import torch

NUM_CLASSES = 5
IGNORE_INDEX = 255
CLASS_NAMES = [
    "Background",
    "Level 1 (Very Low)",
    "Level 2 (Low)",
    "Level 3 (Average)",
    "Level 4 (High)",
]
CLASS_WEIGHTS = [0.3680, 1.3597, 2.0944, 1.7449, 2.0141]


def update_confusion_matrix(conf_mat: torch.Tensor, preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Accumulate confusion matrix counts for valid (non-ignored) pixels."""
    valid = masks != IGNORE_INDEX
    k = (masks[valid] * NUM_CLASSES + preds[valid]).to(torch.int64)
    return conf_mat + torch.bincount(k, minlength=NUM_CLASSES * NUM_CLASSES).view(NUM_CLASSES, NUM_CLASSES)


def compute_metrics(conf_mat: torch.Tensor):
    """Compute per-class IoU, mIoU, and Overall Accuracy (OA) from a confusion matrix."""
    tp = torch.diag(conf_mat).float()
    fp = conf_mat.sum(dim=0).float() - tp
    fn = conf_mat.sum(dim=1).float() - tp
    iou = tp / (tp + fp + fn + 1e-8)
    miou = iou.mean().item()
    oa = (tp.sum() / (conf_mat.sum().float() + 1e-8)).item()
    return iou.cpu().numpy(), miou, oa
