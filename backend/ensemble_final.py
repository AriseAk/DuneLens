import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp

# =============================================================================
# CONFIG
# =============================================================================

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_H     = 640
IMG_W     = 640
N_CLASSES = 11          # Both models now output 11 classes

DATA_DIR   = "./Offroad_Segmentation_testImages/Offroad_Segmentation_testImages"
image_dir  = os.path.join(DATA_DIR, "Color_Images")
mask_dir   = os.path.join(DATA_DIR, "Segmentation")
OUTPUT_DIR = "./ensemble_predictions_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# CLASS INFO
# =============================================================================

CLASS_NAMES = [
    "Background",     # 0
    "Trees",          # 1
    "Lush Bushes",    # 2
    "Dry Grass",      # 3
    "Dry Bushes",     # 4
    "Ground Clutter", # 5
    "Flowers",        # 6  ← was missing in old B2
    "Logs",           # 7
    "Rocks",          # 8
    "Landscape",      # 9
    "Sky",            # 10
]

# Per-class ensemble weight for B2 (CNN gets 1 - this value).
# B2 (transformer) is stronger on large uniform regions (sky, landscape, trees).
# CNN (U-Net) is stronger on fine-grained small rare classes (flowers, logs, rocks).
# These are starting values — tune after checking per-class IoU from both models.
B2_CLASS_WEIGHTS = torch.tensor([
    0.60,  # Background     — B2 slightly better on flat regions
    0.65,  # Trees          — B2 good at large canopy
    0.60,  # Lush Bushes
    0.60,  # Dry Grass
    0.55,  # Dry Bushes
    0.50,  # Ground Clutter — roughly equal
    0.35,  # Flowers        — CNN better (small, rare class)
    0.35,  # Logs           — CNN better (thin elongated structures)
    0.40,  # Rocks          — CNN better (texture-heavy)
    0.65,  # Landscape      — B2 better (large uniform)
    0.70,  # Sky            — B2 clearly better (large smooth region)
], dtype=torch.float32)  # shape: (11,)

CNN_CLASS_WEIGHTS = 1.0 - B2_CLASS_WEIGHTS  # shape: (11,)

# =============================================================================
# VALUE MAP — 11 classes
# =============================================================================

VALUE_MAP = {
    0:     0,
    100:   1,
    200:   2,
    300:   3,
    500:   4,
    550:   5,
    600:   6,   # Flowers
    700:   7,
    800:   8,
    7100:  9,
    10000: 10,
}

# =============================================================================
# TRANSFORM
# Desert dataset statistics — same for both models
# =============================================================================

transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=[0.452, 0.431, 0.376], std=[0.218, 0.213, 0.207]),
    ToTensorV2()
])

# =============================================================================
# MASK REMAP
# =============================================================================

def remap_mask(mask):
    if len(mask.shape) == 2:
        raw = mask.astype(np.int32)
    else:
        raw = (mask[:, :, 0].astype(np.int32)
               + mask[:, :, 1].astype(np.int32) * 256)

    mapped = np.zeros(raw.shape, dtype=np.uint8)
    for raw_val, cls in VALUE_MAP.items():
        mapped[raw == raw_val] = cls
    return mapped

# =============================================================================
# IoU — per-class and mean
# =============================================================================

def compute_iou(pred, gt, n_classes=N_CLASSES):
    pred = pred.flatten()
    gt   = gt.flatten()
    ious = []
    for c in range(n_classes):
        inter = np.logical_and(pred == c, gt == c).sum()
        union = np.logical_or(pred  == c, gt == c).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


def compute_per_class_iou(pred, gt, n_classes=N_CLASSES):
    pred = pred.flatten()
    gt   = gt.flatten()
    ious = {}
    for c in range(n_classes):
        inter = np.logical_and(pred == c, gt == c).sum()
        union = np.logical_or(pred  == c, gt == c).sum()
        ious[CLASS_NAMES[c]] = (inter / union) if union > 0 else float("nan")
    return ious

# =============================================================================
# LOAD SEGFORMER B2 — 11 classes
# =============================================================================

print("Loading SegFormer B2 (11 classes)...")

model_b2 = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2",
    num_labels=N_CLASSES,
    ignore_mismatched_sizes=True,
)
ckpt_b2 = torch.load("best_model_final.pth", map_location=DEVICE)
model_b2.load_state_dict(ckpt_b2["model_state_dict"])
model_b2.to(DEVICE).eval()
print(f"  B2 loaded  (epoch {ckpt_b2.get('epoch','?')}, "
      f"val IoU={ckpt_b2.get('val_iou', '?')})")

# =============================================================================
# LOAD CNN (U-Net ResNet34) — 11 classes
# =============================================================================

print("Loading U-Net CNN (11 classes)...")

cnn = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=N_CLASSES,   # 11 — no slicing needed anymore
    activation=None,
)
ckpt_cnn = torch.load("best_unet_model.pth", map_location=DEVICE, weights_only=False)
cnn.load_state_dict(ckpt_cnn["model_state_dict"])
cnn.to(DEVICE).eval()
print(f"  CNN loaded (epoch {ckpt_cnn.get('epoch','?')}, "
      f"val IoU={ckpt_cnn.get('val_iou', '?')})")

# =============================================================================
# PREDICT HELPERS
# =============================================================================

def predict_b2(img_tensor):
    """SegFormer forward + upsample to (IMG_H, IMG_W)."""
    logits = model_b2(pixel_values=img_tensor).logits       # (1, 11, H/4, W/4)
    return F.interpolate(logits, size=(IMG_H, IMG_W),
                         mode="bilinear", align_corners=False)  # (1, 11, H, W)


def predict_cnn(img_tensor):
    """U-Net forward — already full resolution."""
    logits = cnn(img_tensor)                                # (1, 11, H, W)
    return F.interpolate(logits, size=(IMG_H, IMG_W),
                         mode="bilinear", align_corners=False)


# =============================================================================
# TTA — horizontal flip only (fast, reliable)
# =============================================================================

def tta(predict_fn, img_tensor):
    """Average original + horizontally flipped prediction."""
    l1 = predict_fn(img_tensor)
    l2 = predict_fn(torch.flip(img_tensor, dims=[3]))
    l2 = torch.flip(l2, dims=[3])
    return (l1 + l2) / 2.0


# =============================================================================
# CLASS-AWARE ENSEMBLE
#
# Instead of a single scalar blend (0.75*B2 + 0.25*CNN) which treats all
# classes equally, we apply per-class weights so the transformer dominates
# on large regions and the CNN dominates on small rare classes.
#
# Shape math:
#   p_b2, p_cnn  : (1, 11, H, W)  — softmax probabilities
#   weights_b2   : (1, 11,  1,  1) — broadcast over spatial dims
#   final        : (1, 11, H, W)  — weighted sum (sums to 1 per pixel)
# =============================================================================

def class_aware_ensemble(p_b2, p_cnn):
    w_b2  = B2_CLASS_WEIGHTS.to(p_b2.device).view(1, N_CLASSES, 1, 1)
    w_cnn = CNN_CLASS_WEIGHTS.to(p_b2.device).view(1, N_CLASSES, 1, 1)
    return w_b2 * p_b2 + w_cnn * p_cnn   # (1, 11, H, W)


# =============================================================================
# INFERENCE LOOP
# =============================================================================

ids = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
print(f"\nImages: {len(ids)}")
print(f"Running class-aware ensemble (B2 + CNN, 11 classes, TTA)...\n")

iou_scores       = []
per_class_totals = {name: [] for name in CLASS_NAMES}

with torch.no_grad():
    for name in tqdm(ids):

        # ── Load & preprocess ────────────────────────────────────────────────
        img = cv2.imread(os.path.join(image_dir, name))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

        # ── TTA predictions from both models ─────────────────────────────────
        logits_b2  = tta(predict_b2,  img_tensor)   # (1, 11, H, W)
        logits_cnn = tta(predict_cnn, img_tensor)   # (1, 11, H, W)

        # ── Convert to probabilities ─────────────────────────────────────────
        p_b2  = torch.softmax(logits_b2,  dim=1)
        p_cnn = torch.softmax(logits_cnn, dim=1)

        # ── Class-aware weighted ensemble ────────────────────────────────────
        final = class_aware_ensemble(p_b2, p_cnn)   # (1, 11, H, W)

        # ── Argmax → prediction map ──────────────────────────────────────────
        pred = torch.argmax(final, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # ── Save prediction ──────────────────────────────────────────────────
        cv2.imwrite(os.path.join(OUTPUT_DIR, name), pred)

        # ── Evaluate against GT if available ─────────────────────────────────
        mask_path = os.path.join(mask_dir, name)
        if os.path.exists(mask_path):
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            gt = remap_mask(mask_raw)
            gt = cv2.resize(gt, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

            iou = compute_iou(pred, gt)
            iou_scores.append(iou)

            pc = compute_per_class_iou(pred, gt)
            for cls_name, val in pc.items():
                if not np.isnan(val):
                    per_class_totals[cls_name].append(val)

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 55)
print("  ENSEMBLE RESULTS")
print("=" * 55)

if iou_scores:
    print(f"\n  Mean IoU : {np.mean(iou_scores):.4f}")
    print(f"  Std      : {np.std(iou_scores):.4f}")
    print(f"  Min      : {np.min(iou_scores):.4f}")
    print(f"  Max      : {np.max(iou_scores):.4f}")

    print("\n  Per-class IoU:")
    print(f"  {'Class':<18} {'IoU':>6}")
    print("  " + "-" * 26)
    for cls_name, vals in per_class_totals.items():
        if vals:
            print(f"  {cls_name:<18} {np.mean(vals):>6.4f}")
        else:
            print(f"  {cls_name:<18}    N/A  (not in test set)")

print(f"\n  Predictions saved to: {OUTPUT_DIR}")
print("=" * 55)
print("Done")