"""
Confusion Matrix Generator for SegFormer B2
Run this in your project environment where the model checkpoint is available.

Usage:
    python generate_confusion_matrix.py

Output:
    confusion_matrix.png  — ready to paste into the report
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from PIL import Image

# ── CONFIG — edit these paths ──────────────────────────────────────────────
VAL_DIR    = "./Offroad_Segmentation_Training_Dataset/val"   # val folder
MODEL_PATH = "model/best_model/best_model_final.pth"                          # your B2 checkpoint
OUTPUT     = "confusion_matrix.png"

IMG_H, IMG_W = 640, 640
N_CLASSES    = 11
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Background",
    "Trees",
    "Lush Bushes",
    "Dry Grass",
    "Dry Bushes",
    "Gnd Clutter",
    "Flowers",
    "Logs",
    "Rocks",
    "Landscape",
    "Sky",
]

VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10,
}

transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def remap_mask(mask_array):
    out = np.zeros_like(mask_array, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        out[mask_array == raw] = cls
    return out


# ── Load model ─────────────────────────────────────────────────────────────
print("Loading SegFormer B2 ...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2", num_labels=N_CLASSES, ignore_mismatched_sizes=True
)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()
print(f"  Loaded  (epoch {ckpt.get('epoch','?')}, val_iou={ckpt.get('val_iou','?')})")

# ── Accumulate confusion matrix over val set ───────────────────────────────
image_dir = os.path.join(VAL_DIR, "Color_Images")
mask_dir  = os.path.join(VAL_DIR, "Segmentation")
ids       = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
print(f"Val images: {len(ids)}")

conf_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

with torch.no_grad():
    for name in tqdm(ids, desc="Evaluating"):
        img  = np.array(Image.open(os.path.join(image_dir, name)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(mask_dir,  name)))
        gt   = remap_mask(mask)

        inp    = transform(image=img)["image"].unsqueeze(0).to(DEVICE)
        logits = model(pixel_values=inp).logits
        logits = F.interpolate(logits, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
        pred   = logits.argmax(dim=1).squeeze().cpu().numpy()

        # resize GT to match
        from PIL import Image as PILImage
        gt_img = PILImage.fromarray(gt)
        gt     = np.array(gt_img.resize((IMG_W, IMG_H), PILImage.NEAREST))

        # flatten and accumulate
        flat_pred = pred.flatten()
        flat_gt   = gt.flatten()
        for p, g in zip(flat_pred, flat_gt):
            conf_matrix[g, p] += 1

# ── Normalise rows (recall per class) ─────────────────────────────────────
row_sums = conf_matrix.sum(axis=1, keepdims=True).clip(min=1)
cm_norm  = conf_matrix / row_sums   # values 0–1

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))
fig.patch.set_facecolor("#FEFCF8")
ax.set_facecolor("#FEFCF8")

im = ax.imshow(cm_norm, interpolation="nearest", cmap="YlOrBr", vmin=0, vmax=1)

# colour bar
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("Recall (row-normalised)", fontsize=11, color="#333333")
cbar.ax.tick_params(labelsize=9, colors="#555555")

# ticks
ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=9, color="#333333")
ax.set_yticklabels(CLASS_NAMES, fontsize=9, color="#333333")

# cell text
thresh = cm_norm.max() / 2.0
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        val = cm_norm[i, j]
        txt = f"{val:.2f}" if val > 0.01 else ""
        color = "white" if val > thresh else "#333333"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                color=color, fontweight="bold" if i == j else "normal")

# labels & title
ax.set_xlabel("Predicted Class", fontsize=12, color="#333333", labelpad=10)
ax.set_ylabel("Ground Truth Class", fontsize=12, color="#333333", labelpad=10)
ax.set_title("SegFormer B2 — Confusion Matrix (Val Set, Row-Normalised)",
             fontsize=13, color="#7A4F1A", fontweight="bold", pad=14)

# gold diagonal highlight
for i in range(N_CLASSES):
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                 fill=False, edgecolor="#C9913E", linewidth=1.8))

plt.tight_layout()
plt.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nSaved → {OUTPUT}")

# ── Per-class stats ────────────────────────────────────────────────────────
print("\nPer-class Recall (diagonal):")
print(f"  {'Class':<18} {'Recall':>8}  {'Support':>9}")
print("  " + "-"*38)
for i in range(N_CLASSES):
    recall  = cm_norm[i, i]
    support = conf_matrix[i].sum()
    print(f"  {CLASS_NAMES[i]:<18} {recall:>8.3f}  {support:>9,}")