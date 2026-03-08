"""
segformer_train.py
==================
SegFormer-B2 fine-tuning for offroad / desert semantic segmentation.

Usage
-----
  python segformer_train.py            # train
  python segformer_train.py --eval     # evaluate on test set only
"""

import os
import time
import statistics
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────────────

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES     = 11
IMG_H         = 640
IMG_W         = 640
BATCH_SIZE    = 2
ACCUM_STEPS   = 4        # effective batch = BATCH_SIZE * ACCUM_STEPS
N_EPOCHS      = 20
WARMUP_EPOCHS = 3        # encoder frozen for first N epochs
LR            = 6e-5
WEIGHT_DECAY  = 1e-4
FOCAL_GAMMA   = 2.0

# Desert dataset colour statistics
MEAN = [0.452, 0.431, 0.376]
STD  = [0.218, 0.213, 0.207]

# Raw pixel value → class index
VALUE_MAP = {
    0:     0,   # Background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}

# Paths — update to your local directories
TRAIN_DIR  = "../Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train"
VAL_DIR    = "../Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val"
TEST_DIR   = "../Offroad_Segmentation_testImages/Offroad_Segmentation_testImages"
OUTPUT_DIR = "../train_stats"
CKPT_PATH  = "best_model_final.pth"

# Inference benchmark settings
TARGET_MS      = 50.0
WARMUP_RUNS    = 10
BENCHMARK_RUNS = 50


# ── Mask Utilities ─────────────────────────────────────────────────────────────

def remap_mask(mask_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a raw BGR mask image into a single-channel uint8 class-index mask.

    Raw masks encode their value across two channels:
        raw_value = B_channel + G_channel * 256
    This handles values up to 65 535, covering all label values including
    Sky (10 000).
    """
    if mask_bgr.ndim == 3:
        raw = (mask_bgr[:, :, 0].astype(np.int32)
               + mask_bgr[:, :, 1].astype(np.int32) * 256)
    else:
        raw = mask_bgr.astype(np.int32)

    out = np.zeros(raw.shape, dtype=np.uint8)
    for raw_val, cls_idx in VALUE_MAP.items():
        out[raw == raw_val] = cls_idx
    return out


# ── Albumentations Transforms ──────────────────────────────────────────────────

def build_train_transform() -> A.Compose:
    """Aggressive augmentation pipeline for desert / offroad UGV imagery."""
    return A.Compose([
        # Spatial
        A.Resize(IMG_H, IMG_W, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101, p=0.4,
        ),
        # Photometric — desert lighting variation
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        ], p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        # Sensor / atmospheric noise
        A.GaussNoise(p=0.4),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        A.RandomFog(p=0.15),
        # Motion blur (moving vehicle)
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        # Normalise & convert to tensor
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def build_val_transform() -> A.Compose:
    """Deterministic pipeline: resize + normalise only."""
    return A.Compose([
        A.Resize(IMG_H, IMG_W, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ── Dataset ────────────────────────────────────────────────────────────────────

class OffroadSegDataset(Dataset):
    """
    Offroad segmentation dataset.

    Expected directory layout::

        data_dir/
            Color_Images/   ← RGB images (.png)
            Segmentation/   ← raw label images (.png, same filenames)
    """

    def __init__(self, data_dir: str, transform: A.Compose = None):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir  = os.path.join(data_dir, "Segmentation")
        self.transform = transform

        self.ids = sorted(f for f in os.listdir(self.image_dir) if f.endswith(".png"))
        assert len(self.ids) > 0, f"No .png images found in {self.image_dir}"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        # Load image
        img = cv2.imread(os.path.join(self.image_dir, name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask — preserve both channels to recover raw label values
        mask_raw = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_UNCHANGED)
        mask = remap_mask(mask_raw)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"]

        return img, mask.long()


# ── Loss Functions ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class Focal Loss — down-weights easy pixels, focuses on hard ones."""

    def __init__(self, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, ignore_index=self.ignore_index, reduction="none")
        p_t   = torch.exp(-ce)
        loss  = ((1 - p_t) ** self.gamma) * ce
        valid = targets != self.ignore_index
        return loss[valid].mean()


class DiceLoss(nn.Module):
    """Soft Dice Loss — maximises IoU for small / rare classes (logs, rocks)."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls   = logits.shape[1]
        probs   = F.softmax(logits, dim=1)                      # (B, C, H, W)
        valid   = (targets != self.ignore_index).unsqueeze(1)
        t_clamp = targets.clone()
        t_clamp[targets == self.ignore_index] = 0

        oh    = F.one_hot(t_clamp, n_cls).permute(0, 3, 1, 2).float() * valid
        probs = probs * valid

        dims  = (0, 2, 3)
        inter = (probs * oh).sum(dim=dims)
        card  = (probs + oh).sum(dim=dims)
        dice  = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()


class FocalDiceLoss(nn.Module):
    """Focal + Dice combined loss."""

    def __init__(self, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, ignore_index=ignore_index)
        self.dice  = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal(logits, targets) + self.dice(logits, targets)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int = N_CLASSES,
    smooth: float = 1e-6,
):
    """
    Returns ``(mean_iou, mean_dice, pixel_accuracy)`` as Python floats.
    Classes absent from the batch are excluded from the IoU mean (NaN-safe).
    """
    preds = logits.argmax(dim=1).view(-1)
    tgts  = targets.view(-1)

    iou_list, dice_list = [], []
    for c in range(n_classes):
        p     = preds == c
        t     = tgts  == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        iou_list.append(
            (inter / (union + smooth)).item() if union > 0 else float("nan")
        )
        denom = p.sum().float() + t.sum().float()
        dice_list.append(((2 * inter + smooth) / (denom + smooth)).item())

    pixel_acc = (preds == tgts).float().mean().item()
    return float(np.nanmean(iou_list)), float(np.mean(dice_list)), pixel_acc


# ── Model Helpers ──────────────────────────────────────────────────────────────

def build_model() -> SegformerForSemanticSegmentation:
    """Load SegFormer-B2 with a fresh N_CLASSES decode head."""
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=N_CLASSES,
        ignore_mismatched_sizes=True,
    )
    return model.to(DEVICE)


def freeze_encoder(model: SegformerForSemanticSegmentation):
    for p in model.segformer.parameters():
        p.requires_grad = False
    print("Encoder frozen  (decoder-only warm-up)")


def unfreeze_encoder(model: SegformerForSemanticSegmentation):
    for p in model.segformer.parameters():
        p.requires_grad = True
    print("Encoder unfrozen (end-to-end fine-tuning)")


# ── Training / Validation Loops ────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, accum_steps):
    model.train()
    losses, ious, dices, accs = [], [], [], []
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  Train", leave=False)
    for step, (imgs, masks) in enumerate(pbar, 1):
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            raw_logits = model(pixel_values=imgs).logits       # (B, C, H/4, W/4)
            logits = F.interpolate(
                raw_logits, size=(IMG_H, IMG_W),
                mode="bilinear", align_corners=False,
            )                                                  # (B, C, H, W)
            loss = criterion(logits, masks) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            iou, dice, acc = compute_metrics(logits.detach(), masks)

        losses.append(loss.item() * accum_steps)
        ious.append(iou); dices.append(dice); accs.append(acc)
        pbar.set_postfix(loss=f"{np.mean(losses):.3f}", iou=f"{np.nanmean(ious):.3f}")

    return (
        float(np.mean(losses)),
        float(np.nanmean(ious)),
        float(np.mean(dices)),
        float(np.mean(accs)),
    )


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    losses, ious, dices, accs = [], [], [], []

    pbar = tqdm(loader, desc="    Val", leave=False)
    for imgs, masks in pbar:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            raw_logits = model(pixel_values=imgs).logits
            logits = F.interpolate(
                raw_logits, size=(IMG_H, IMG_W),
                mode="bilinear", align_corners=False,
            )
            loss = criterion(logits, masks)

        iou, dice, acc = compute_metrics(logits, masks)
        losses.append(loss.item())
        ious.append(iou); dices.append(dice); accs.append(acc)
        pbar.set_postfix(iou=f"{np.nanmean(ious):.3f}")

    return (
        float(np.mean(losses)),
        float(np.nanmean(ious)),
        float(np.mean(dices)),
        float(np.mean(accs)),
    )


# ── Plotting & Logging ─────────────────────────────────────────────────────────

def save_training_plots(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("SegFormer Training — Offroad Segmentation", fontsize=14)

    pairs = [
        ("train_loss",  "val_loss",  "Loss",           axes[0, 0]),
        ("train_iou",   "val_iou",   "Mean IoU",       axes[0, 1]),
        ("train_dice",  "val_dice",  "Dice Score",     axes[1, 0]),
        ("train_acc",   "val_acc",   "Pixel Accuracy", axes[1, 1]),
    ]
    for tr_key, vl_key, title, ax in pairs:
        ax.plot(epochs, history[tr_key], label="train", marker="o", markersize=3)
        ax.plot(epochs, history[vl_key], label="val",   marker="s", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "all_metrics.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")


def save_metrics_txt(history: dict, best_iou: float):
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    n = len(history["train_loss"])

    with open(metrics_path, "w") as f:
        f.write("SEGFORMER TRAINING RESULTS\n" + "=" * 65 + "\n\n")
        f.write(f"Best Val IoU  : {best_iou:.4f}\n")
        f.write(
            f"Best Val Dice : {max(history['val_dice']):.4f}  "
            f"(Epoch {history['val_dice'].index(max(history['val_dice'])) + 1})\n"
        )
        f.write(
            f"Best Val Acc  : {max(history['val_acc']):.4f}  "
            f"(Epoch {history['val_acc'].index(max(history['val_acc'])) + 1})\n"
        )
        f.write(
            f"Lowest Val Loss: {min(history['val_loss']):.4f}  "
            f"(Epoch {history['val_loss'].index(min(history['val_loss'])) + 1})\n\n"
        )

        header = (
            f"{'Ep':>3} {'TrLoss':>8} {'VLoss':>8} {'TrIoU':>7} {'VIoU':>7} "
            f"{'TrDice':>7} {'VDice':>7} {'TrAcc':>7} {'VAcc':>7}\n"
        )
        f.write(header)
        f.write("-" * len(header) + "\n")
        for i in range(n):
            f.write(
                f"{i+1:>3} "
                f"{history['train_loss'][i]:>8.4f} {history['val_loss'][i]:>8.4f} "
                f"{history['train_iou'][i]:>7.4f} {history['val_iou'][i]:>7.4f} "
                f"{history['train_dice'][i]:>7.4f} {history['val_dice'][i]:>7.4f} "
                f"{history['train_acc'][i]:>7.4f} {history['val_acc'][i]:>7.4f}\n"
            )

    print(f"Metrics saved to: {metrics_path}")


# ── Inference Benchmark ────────────────────────────────────────────────────────

def run_latency_benchmark(model):
    """Time 50 forward passes on a dummy input and report latency statistics."""
    model.eval()
    dummy = torch.randn(1, 3, IMG_H, IMG_W, device=DEVICE)

    print(f"Warming up ({WARMUP_RUNS} passes)…")
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(pixel_values=dummy)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    print(f"Benchmarking ({BENCHMARK_RUNS} passes)…")
    latencies = []

    if DEVICE.type == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(BENCHMARK_RUNS):
                start_ev.record()
                model(pixel_values=dummy)
                end_ev.record()
                torch.cuda.synchronize()
                latencies.append(start_ev.elapsed_time(end_ev))
    else:
        with torch.no_grad():
            for _ in range(BENCHMARK_RUNS):
                t0 = time.perf_counter()
                model(pixel_values=dummy)
                latencies.append((time.perf_counter() - t0) * 1000)

    mean_ms = statistics.mean(latencies)
    std_ms  = statistics.stdev(latencies)
    p90_ms  = sorted(latencies)[int(0.90 * BENCHMARK_RUNS)]

    print(f"\n── Latency Report ─────────────────────────────────")
    print(f"  Mean   : {mean_ms:.2f} ms")
    print(f"  Std    : {std_ms:.2f} ms")
    print(f"  Min    : {min(latencies):.2f} ms")
    print(f"  Max    : {max(latencies):.2f} ms")
    print(f"  P90    : {p90_ms:.2f} ms")
    print(f"  Target : {TARGET_MS} ms")

    if mean_ms <= TARGET_MS:
        print(f"\n  ✓ PASS — {mean_ms:.2f}ms ≤ {TARGET_MS}ms  →  Ready for submission!")
    else:
        print(f"\n  ✗ FAIL — {mean_ms:.2f}ms > {TARGET_MS}ms  →  Consider optimisation.")


# ── Test-set Evaluation ────────────────────────────────────────────────────────

def compute_iou_numpy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-image mean IoU computed in NumPy (used during test evaluation)."""
    pred = pred.flatten()
    gt   = gt.flatten()
    ious = []
    for c in range(N_CLASSES):
        inter = np.logical_and(pred == c, gt == c).sum()
        union = np.logical_or(pred == c,  gt == c).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def evaluate_test_set(model):
    """Run inference on TEST_DIR and report mean IoU."""
    image_dir = os.path.join(TEST_DIR, "Color_Images")
    mask_dir  = os.path.join(TEST_DIR, "Segmentation")
    transform = build_val_transform()

    ids = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
    print(f"Found {len(ids)} test images")

    model.eval()
    iou_scores   = []
    debug_printed = False

    with torch.no_grad():
        for name in tqdm(ids):
            img = cv2.imread(os.path.join(image_dir, name))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)
            logits = model(pixel_values=tensor).logits
            logits = F.interpolate(logits, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
            pred   = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

            mask_path = os.path.join(mask_dir, name)
            if os.path.exists(mask_path):
                mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                gt = remap_mask(mask_raw)
                gt = cv2.resize(gt, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

                if not debug_printed:
                    print(f"\nDEBUG — Pred classes: {np.unique(pred)}, GT classes: {np.unique(gt)}")
                    debug_printed = True

                iou_scores.append(compute_iou_numpy(pred, gt))

    if iou_scores:
        print(f"\nTest Mean IoU: {np.mean(iou_scores):.4f}")
    else:
        print("\nNo IoU scores computed (no matching masks found).")


# ── Main Training Loop ─────────────────────────────────────────────────────────

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Print device info
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Datasets & loaders
    train_ds = OffroadSegDataset(TRAIN_DIR, transform=build_train_transform())
    val_ds   = OffroadSegDataset(VAL_DIR,   transform=build_val_transform())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")

    # Model
    model = build_model()
    freeze_encoder(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable now   : {trainable:,}  (decoder only during warm-up)")

    # Loss, optimiser, scheduler, AMP scaler
    criterion = FocalDiceLoss(gamma=FOCAL_GAMMA).to(DEVICE)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # Resume from checkpoint if available
    history    = {k: [] for k in ["train_loss", "val_loss", "train_iou", "val_iou",
                                   "train_dice", "val_dice", "train_acc", "val_acc"]}
    start_epoch = 1
    best_iou    = 0.0

    if os.path.exists(CKPT_PATH):
        print("Checkpoint found. Loading…")
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt["val_iou"]
        print(f"Resuming from epoch {start_epoch}, best IoU so far: {best_iou:.4f}")

    # Training
    print("=" * 65)
    print("Starting training")
    print(f"  Epochs          : {N_EPOCHS}")
    print(f"  Warm-up epochs  : {WARMUP_EPOCHS}")
    print(f"  Batch size      : {BATCH_SIZE}  (effective {BATCH_SIZE * ACCUM_STEPS})")
    print("=" * 65)

    for epoch in range(start_epoch, N_EPOCHS + 1):

        # Unfreeze encoder after warm-up and rebuild optimiser with layer-wise LR
        if epoch == max(start_epoch, WARMUP_EPOCHS + 1):
            unfreeze_encoder(model)
            optimizer = optim.AdamW([
                {"params": model.segformer.parameters(),  "lr": LR * 0.1},
                {"params": model.decode_head.parameters(), "lr": LR},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=N_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6,
            )
            scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

        print(f"\nEpoch {epoch}/{N_EPOCHS}")

        tr_loss, tr_iou, tr_dice, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, ACCUM_STEPS,
        )
        vl_loss, vl_iou, vl_dice, vl_acc = validate(model, val_loader, criterion)

        scheduler.step()

        # Record history
        for k, v in zip(history.keys(),
                        [tr_loss, vl_loss, tr_iou, vl_iou, tr_dice, vl_dice, tr_acc, vl_acc]):
            history[k].append(v)

        print(f"  Train  loss={tr_loss:.4f}  IoU={tr_iou:.4f}  Dice={tr_dice:.4f}  Acc={tr_acc:.4f}")
        print(f"  Val    loss={vl_loss:.4f}  IoU={vl_iou:.4f}  Dice={vl_dice:.4f}  Acc={vl_acc:.4f}")

        # Save best checkpoint
        if vl_iou > best_iou:
            best_iou = vl_iou
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_iou": vl_iou},
                       CKPT_PATH)
            print(f"✓ New best IoU {best_iou:.4f} — saved to {CKPT_PATH}")

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")

    # Post-training: plots, metrics, latency benchmark
    save_training_plots(history)
    save_metrics_txt(history, best_iou)

    print("\n── Latency Benchmark ──────────────────────────────")
    run_latency_benchmark(model)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer offroad segmentation")
    parser.add_argument("--eval", action="store_true",
                        help="Skip training; evaluate best checkpoint on the test set")
    args = parser.parse_args()

    if args.eval:
        model = build_model()
        ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint (epoch {ckpt['epoch']}, Val IoU={ckpt['val_iou']:.4f})")
        evaluate_test_set(model)
    else:
        train()