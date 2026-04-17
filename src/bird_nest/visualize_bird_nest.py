"""
Evaluation & visualization script for bird nest detection (Mask R-CNN).

Generates the following plots:
  1. loss_curves.png             – Training / validation loss over epochs
  2. pr_curve_bbox_ap50.png      – Precision-Recall curve (bbox, IoU ≥ 0.50)
  3. pr_curve_segm_ap50.png      – Precision-Recall curve (mask, IoU ≥ 0.50)
  4. iou_hist_bbox.png           – IoU distribution histogram (bbox)
  5. iou_hist_mask.png           – IoU distribution histogram (mask)
  6. threshold_sweep_bbox.png    – Precision / Recall / F1 vs confidence threshold (bbox)
"""

import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
import torchvision.transforms.v2 as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_iou
from tqdm.auto import tqdm
from PIL import Image

from windows_utils import COCODataset, tuple_batch, create_polygon_mask
from cjm_pytorch_utils.core import get_torch_device, set_seed, move_data_to_device
from cjm_pil_utils.core import resize_img
from cjm_torchvision_tfms.core import ResizeMax, PadSquare

# ──────────────────────────────────────────────
# 0. Paths & setup
# ──────────────────────────────────────────────
repo_root = Path(__file__).resolve().parent.parent.parent
dataset_dir = repo_root / "data" / "active" / "bird_nest" / "coco"
coco_json_path = dataset_dir / "dataset.json"
image_dir = dataset_dir

output_dir = repo_root / "experiments" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device(get_torch_device())
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# 1. Load COCO annotations & class names
# ──────────────────────────────────────────────
with open(coco_json_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

class_names = ["background"] + [cat["name"] for cat in coco_data["categories"]]
class_to_idx = {c: i for i, c in enumerate(class_names)}
print(f"Classes: {class_names}")

# ──────────────────────────────────────────────
# 2. Find the best checkpoint & history
# ──────────────────────────────────────────────
project_dir = repo_root / "experiments" / "pytorch-mask-r-cnn-bird-nest"
run_dirs = sorted([p for p in project_dir.glob("*") if p.is_dir()], reverse=True)

checkpoint_path = None
history_path = None
for rd in run_dirs:
    pths = list(rd.glob("*.pth"))
    hist = rd / "history.jsonl"
    if pths and hist.exists() and hist.stat().st_size > 0:
        checkpoint_path = pths[0]
        history_path = hist
        break

if checkpoint_path is None:
    raise FileNotFoundError("No checkpoint with history found under " + str(project_dir))

print(f"Checkpoint : {checkpoint_path}")
print(f"History    : {history_path}")

# ──────────────────────────────────────────────
# 3. Load model
# ──────────────────────────────────────────────
num_classes = len(class_names)
model = maskrcnn_resnet50_fpn_v2(weights=None)
in_features_box = model.roi_heads.box_predictor.cls_score.in_features
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes=num_classes)
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced=dim_reduced,
                                                    num_classes=num_classes)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded ✓")

# ──────────────────────────────────────────────
# 4. Build validation dataset (deterministic split – same seed as training)
# ──────────────────────────────────────────────
train_sz = 512
set_seed(1234)
all_img_ids = [img["id"] for img in coco_data["images"]]
random.shuffle(all_img_ids)
train_split = int(len(all_img_ids) * 0.8)
val_img_ids = all_img_ids[train_split:]

resize_max = ResizeMax(max_sz=train_sz)
pad_square = PadSquare(shift=False, fill=0)

val_tfms = transforms.Compose([
    resize_max, pad_square,
    transforms.Resize([train_sz] * 2, antialias=True),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.SanitizeBoundingBoxes(),
])

val_dataset = COCODataset(coco_json_path, image_dir, class_to_idx, val_tfms, img_ids=val_img_ids)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                          collate_fn=tuple_batch, num_workers=0)
print(f"Validation images: {len(val_dataset)}")


# ──────────────────────────────────────────────
# Helper: compute mask IoU
# ──────────────────────────────────────────────
def mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Compute IoU between two boolean masks (H, W)."""
    pred = pred_mask.bool().flatten()
    gt   = gt_mask.bool().flatten()
    intersection = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    return (intersection / union).item() if union > 0 else 0.0


# ──────────────────────────────────────────────
# 5. Run inference and collect per-detection results
# ──────────────────────────────────────────────
print("Running inference on validation set …")

all_scores      = []   # confidence of every prediction
all_bbox_ious   = []   # best IoU (bbox) of every prediction with GT
all_mask_ious   = []   # best IoU (mask) of every prediction with GT
# Arrays for multiple IoU thresholds: 0.50, 0.75, 0.90
all_tp_bbox_50  = []
all_tp_bbox_75  = []
all_tp_bbox_90  = []
all_tp_mask_50  = []
all_tp_mask_75  = []
all_tp_mask_90  = []
total_gt_boxes  = 0    # total ground-truth objects

with torch.no_grad():
    for inputs, targets in tqdm(val_loader, desc="Eval"):
        imgs = torch.stack(inputs).to(device)
        outputs = model(imgs)

        for out, tgt in zip(outputs, targets):
            gt_boxes  = tgt["boxes"].to("cpu")
            gt_masks  = tgt["masks"].to("cpu")
            gt_labels = tgt["labels"].to("cpu")

            # skip images with 0 GT (negative samples contribute 0 GT)
            n_gt = gt_boxes.shape[0]
            total_gt_boxes += n_gt

            pred_boxes  = out["boxes"].cpu()
            pred_scores = out["scores"].cpu()
            pred_masks  = (out["masks"].cpu() >= 0.5).squeeze(1)  # (N, H, W)

            # Only keep foreground predictions (label > 0)
            pred_labels = out["labels"].cpu()
            fg_mask = pred_labels > 0
            pred_boxes  = pred_boxes[fg_mask]
            pred_scores = pred_scores[fg_mask]
            pred_masks  = pred_masks[fg_mask]

            if pred_boxes.shape[0] == 0:
                continue

            if n_gt == 0:
                # All predictions are FP
                for s in pred_scores.tolist():
                    all_scores.append(s)
                    all_bbox_ious.append(0.0)
                    all_mask_ious.append(0.0)
                    all_tp_bbox_50.append(0)
                    all_tp_bbox_75.append(0)
                    all_tp_bbox_90.append(0)
                    all_tp_mask_50.append(0)
                    all_tp_mask_75.append(0)
                    all_tp_mask_90.append(0)
                continue

            # Compute bbox IoU matrix (N_pred × N_gt)
            iou_matrix = box_iou(pred_boxes, gt_boxes)

            # Compute mask IoU matrix
            mask_iou_matrix = torch.zeros(pred_masks.shape[0], n_gt)
            for pi in range(pred_masks.shape[0]):
                for gi in range(n_gt):
                    mask_iou_matrix[pi, gi] = mask_iou(pred_masks[pi], gt_masks[gi])

            # Sort predictions by score descending for greedy matching
            order = pred_scores.argsort(descending=True)
            
            # We must do greedy matching INDEPENDENTLY per IoU threshold
            matched_bbox_50 = set(); matched_bbox_75 = set(); matched_bbox_90 = set()
            matched_mask_50 = set(); matched_mask_75 = set(); matched_mask_90 = set()

            for idx in order:
                idx = idx.item()
                score = pred_scores[idx].item()
                best_bbox_iou, best_bbox_gi = iou_matrix[idx].max(0)
                best_mask_iou_val, best_mask_gi = mask_iou_matrix[idx].max(0)

                best_bbox_iou = best_bbox_iou.item()
                best_bbox_gi  = best_bbox_gi.item()
                best_mask_iou_val = best_mask_iou_val.item()
                best_mask_gi  = best_mask_gi.item()

                # Evaluate TP for BBoxes
                tp_b_50 = 0
                if best_bbox_iou >= 0.50 and best_bbox_gi not in matched_bbox_50:
                    tp_b_50 = 1; matched_bbox_50.add(best_bbox_gi)
                tp_b_75 = 0
                if best_bbox_iou >= 0.75 and best_bbox_gi not in matched_bbox_75:
                    tp_b_75 = 1; matched_bbox_75.add(best_bbox_gi)
                tp_b_90 = 0
                if best_bbox_iou >= 0.90 and best_bbox_gi not in matched_bbox_90:
                    tp_b_90 = 1; matched_bbox_90.add(best_bbox_gi)

                # Evaluate TP for Masks
                tp_m_50 = 0
                if best_mask_iou_val >= 0.50 and best_mask_gi not in matched_mask_50:
                    tp_m_50 = 1; matched_mask_50.add(best_mask_gi)
                tp_m_75 = 0
                if best_mask_iou_val >= 0.75 and best_mask_gi not in matched_mask_75:
                    tp_m_75 = 1; matched_mask_75.add(best_mask_gi)
                tp_m_90 = 0
                if best_mask_iou_val >= 0.90 and best_mask_gi not in matched_mask_90:
                    tp_m_90 = 1; matched_mask_90.add(best_mask_gi)

                all_scores.append(score)
                all_bbox_ious.append(best_bbox_iou)
                all_mask_ious.append(best_mask_iou_val)
                all_tp_bbox_50.append(tp_b_50); all_tp_bbox_75.append(tp_b_75); all_tp_bbox_90.append(tp_b_90)
                all_tp_mask_50.append(tp_m_50); all_tp_mask_75.append(tp_m_75); all_tp_mask_90.append(tp_m_90)

all_scores    = np.array(all_scores)
all_bbox_ious = np.array(all_bbox_ious)
all_mask_ious = np.array(all_mask_ious)

all_tp_bbox_50 = np.array(all_tp_bbox_50)
all_tp_bbox_75 = np.array(all_tp_bbox_75)
all_tp_bbox_90 = np.array(all_tp_bbox_90)
all_tp_mask_50 = np.array(all_tp_mask_50)
all_tp_mask_75 = np.array(all_tp_mask_75)
all_tp_mask_90 = np.array(all_tp_mask_90)

# For threshold sweep, use 0.50 criterion as default
all_tp_bbox = all_tp_bbox_50

print(f"Total predictions: {len(all_scores)}, Total GT boxes: {total_gt_boxes}")

# ──────────────────────────────────────────────
# Helper: compute PR curve from sorted TP array
# ──────────────────────────────────────────────
def compute_pr_curve(scores, tp_array, total_gt):
    """Return (precision, recall, ap) sorted by decreasing score."""
    if len(scores) == 0 or total_gt == 0:
        return np.array([0.0]), np.array([0.0]), 0.0

    order = np.argsort(-scores)
    tp_sorted = tp_array[order]

    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1 - tp_sorted)

    precision = cum_tp / (cum_tp + cum_fp + 1e-12)
    recall    = cum_tp / (total_gt + 1e-12)

    # Prepend sentinel (recall=0, precision=1) for proper AP integration
    recall    = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    # Make precision monotonically decreasing (standard COCO interpolation)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute AP as area under the interpolated PR curve
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])

    return precision, recall, ap


# ──────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})
MAIN_COLOR  = "#2563EB"
SECOND_COLOR = "#F97316"
THIRD_COLOR  = "#10B981"
FILL_ALPHA   = 0.25

# ══════════════════════════════════════════════
# PLOT 1 – Loss Curves
# ══════════════════════════════════════════════
print("Plotting loss_curves …")
with open(history_path, "r", encoding="utf-8") as f:
    history = [json.loads(line) for line in f if line.strip()]

hist_df = pd.DataFrame(history)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(hist_df["epoch"], hist_df["train_loss"], color=MAIN_COLOR,
        marker="o", markersize=4, linewidth=2, label="Train Loss")
ax.plot(hist_df["epoch"], hist_df["valid_loss"], color=SECOND_COLOR,
        marker="s", markersize=4, linewidth=2, label="Valid Loss")
best_epoch = hist_df.loc[hist_df["valid_loss"].idxmin()]
ax.axvline(best_epoch["epoch"], color="#EF4444", linestyle="--", linewidth=1,
           label=f'Best (epoch {int(best_epoch["epoch"])}, val={best_epoch["valid_loss"]:.4f})')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training & Validation Loss Curves")
ax.legend(frameon=True)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "loss_curves.png")
plt.close(fig)
print(f"  ✓ {output_dir / 'loss_curves.png'}")

# ══════════════════════════════════════════════
# PLOT 2 – PR Curve (BBox: AP50, AP75, AP90)
# ══════════════════════════════════════════════
print("Plotting pr_curve_bbox …")
prec_b_50, rec_b_50, ap50_bbox = compute_pr_curve(all_scores, all_tp_bbox_50, total_gt_boxes)
prec_b_75, rec_b_75, ap75_bbox = compute_pr_curve(all_scores, all_tp_bbox_75, total_gt_boxes)
prec_b_90, rec_b_90, ap90_bbox = compute_pr_curve(all_scores, all_tp_bbox_90, total_gt_boxes)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_b_50, prec_b_50, color=MAIN_COLOR, linewidth=2, label=f"IoU 0.50 (AP = {ap50_bbox:.3f})")
ax.fill_between(rec_b_50, prec_b_50, alpha=0.15, color=MAIN_COLOR)
ax.plot(rec_b_75, prec_b_75, color=SECOND_COLOR, linewidth=2, label=f"IoU 0.75 (AP = {ap75_bbox:.3f})")
ax.plot(rec_b_90, prec_b_90, color=THIRD_COLOR, linewidth=2, linestyle="--", label=f"IoU 0.90 (AP = {ap90_bbox:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("PR Curve – Bounding Box")
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])
ax.legend(frameon=True, loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "pr_curve_bbox.png")
plt.close(fig)
print(f"  ✓ {output_dir / 'pr_curve_bbox.png'}")

# ══════════════════════════════════════════════
# PLOT 3 – PR Curve (Segm: AP50, AP75, AP90)
# ══════════════════════════════════════════════
print("Plotting pr_curve_segm …")
prec_m_50, rec_m_50, ap50_mask = compute_pr_curve(all_scores, all_tp_mask_50, total_gt_boxes)
prec_m_75, rec_m_75, ap75_mask = compute_pr_curve(all_scores, all_tp_mask_75, total_gt_boxes)
prec_m_90, rec_m_90, ap90_mask = compute_pr_curve(all_scores, all_tp_mask_90, total_gt_boxes)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_m_50, prec_m_50, color=MAIN_COLOR, linewidth=2, label=f"IoU 0.50 (AP = {ap50_mask:.3f})")
ax.fill_between(rec_m_50, prec_m_50, alpha=0.15, color=MAIN_COLOR)
ax.plot(rec_m_75, prec_m_75, color=SECOND_COLOR, linewidth=2, label=f"IoU 0.75 (AP = {ap75_mask:.3f})")
ax.plot(rec_m_90, prec_m_90, color=THIRD_COLOR, linewidth=2, linestyle="--", label=f"IoU 0.90 (AP = {ap90_mask:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("PR Curve – Segmentation Mask")
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])
ax.legend(frameon=True, loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "pr_curve_segm.png")
plt.close(fig)
print(f"  ✓ {output_dir / 'pr_curve_segm.png'}")

# ══════════════════════════════════════════════
# PLOT 4 – IoU Histogram (BBox)
# ══════════════════════════════════════════════
print("Plotting iou_hist_bbox …")
fig, ax = plt.subplots(figsize=(7, 4.5))
if len(all_bbox_ious) > 0:
    ax.hist(all_bbox_ious, bins=30, range=(0, 1), color=MAIN_COLOR, edgecolor="white",
            alpha=0.8, label="BBox IoU")
    mean_iou = all_bbox_ious.mean()
    ax.axvline(mean_iou, color="#EF4444", linestyle="--", linewidth=1.5,
               label=f"Mean IoU = {mean_iou:.3f}")
    ax.axvline(0.5, color="#F59E0B", linestyle=":", linewidth=1.5,
               label="IoU = 0.50 threshold")
ax.set_xlabel("IoU")
ax.set_ylabel("Count")
ax.set_title("IoU Distribution – Bounding Boxes")
ax.legend(frameon=True)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "iou_hist_bbox.png")
plt.close(fig)
print(f"  ✓ {output_dir / 'iou_hist_bbox.png'}")

# ══════════════════════════════════════════════
# PLOT 5 – IoU Histogram (Mask)
# ══════════════════════════════════════════════
print("Plotting iou_hist_mask …")
fig, ax = plt.subplots(figsize=(7, 4.5))
if len(all_mask_ious) > 0:
    ax.hist(all_mask_ious, bins=30, range=(0, 1), color=THIRD_COLOR, edgecolor="white",
            alpha=0.8, label="Mask IoU")
    mean_iou = all_mask_ious.mean()
    ax.axvline(mean_iou, color="#EF4444", linestyle="--", linewidth=1.5,
               label=f"Mean IoU = {mean_iou:.3f}")
    ax.axvline(0.5, color="#F59E0B", linestyle=":", linewidth=1.5,
               label="IoU = 0.50 threshold")
ax.set_xlabel("IoU")
ax.set_ylabel("Count")
ax.set_title("IoU Distribution – Segmentation Masks")
ax.legend(frameon=True)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "iou_hist_mask.png")
plt.close(fig)
print(f"  ✓ {output_dir / 'iou_hist_mask.png'}")

# ══════════════════════════════════════════════
# PLOT 6 – Threshold Sweep (BBox)
# ══════════════════════════════════════════════
print("Plotting threshold_sweep_bbox …")
thresholds = np.arange(0.05, 1.0, 0.025)
sweep_precision = []
sweep_recall    = []
sweep_f1        = []

for thr in thresholds:
    mask = all_scores >= thr
    if mask.sum() == 0:
        sweep_precision.append(0.0)
        sweep_recall.append(0.0)
        sweep_f1.append(0.0)
        continue
    tp = all_tp_bbox[mask].sum()
    fp = mask.sum() - tp
    fn = total_gt_boxes - tp

    p = tp / (tp + fp + 1e-12)
    r = tp / (tp + fn + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)
    sweep_precision.append(p)
    sweep_recall.append(r)
    sweep_f1.append(f1)

sweep_precision = np.array(sweep_precision)
sweep_recall    = np.array(sweep_recall)
sweep_f1        = np.array(sweep_f1)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(thresholds, sweep_precision, color=MAIN_COLOR, linewidth=2, label="Precision")
ax.plot(thresholds, sweep_recall,    color=SECOND_COLOR, linewidth=2, label="Recall")
ax.plot(thresholds, sweep_f1,        color=THIRD_COLOR, linewidth=2, label="F1")

best_f1_idx = sweep_f1.argmax()
best_thr = thresholds[best_f1_idx]
ax.axvline(best_thr, color="#EF4444", linestyle="--", linewidth=1.2,
           label=f"Best F1={sweep_f1[best_f1_idx]:.3f} @ thr={best_thr:.2f}")

ax.set_xlabel("Confidence Threshold")
ax.set_ylabel("Metric Value")
ax.set_title("Precision / Recall / F1 vs Confidence Threshold (BBox)")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
ax.legend(frameon=True, loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "threshold_sweep_bbox.png")
plt.close(fig)
print(f"  ✓ {output_dir / 'threshold_sweep_bbox.png'}")

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("All plots saved to:", output_dir)
print(f"  AP@50 (bbox): {ap50_bbox:.4f}")
print(f"  AP@50 (mask): {ap50_mask:.4f}")
print(f"  Best F1 threshold (bbox): {best_thr:.2f} (F1={sweep_f1[best_f1_idx]:.4f})")
print("=" * 50)
