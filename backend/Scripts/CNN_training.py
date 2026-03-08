import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

plt.switch_backend('Agg')


TRAIN_DIR  = "../Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train"
VAL_DIR    = "../Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val"
OUTPUT_DIR = "../train_stats"

# =============================================================================
# Hyperparameters
# =============================================================================
BATCH_SIZE = 8      # U-Net is light enough for batch size 8
IMG_SIZE   = 256    # fast and sufficient for this task
N_EPOCHS   = 40
LR         = 1e-3
N_CLASSES  = 11

# =============================================================================
# Class Mapping
# =============================================================================
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
    10000: 10   # Sky
}

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass',
    'Dry Bushes', 'Ground Clutter', 'Flowers', 'Logs',
    'Rocks', 'Landscape', 'Sky'
]


def convert_mask(mask_array):
    """Convert raw pixel values to class IDs."""
    new_mask = np.zeros_like(mask_array, dtype=np.uint8)
    for raw_val, class_id in VALUE_MAP.items():
        new_mask[mask_array == raw_val] = class_id
    return new_mask


# =============================================================================
# Dataset
# =============================================================================

class SegDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir  = os.path.join(data_dir, 'Segmentation')
        self.ids       = sorted(os.listdir(self.image_dir))
        self.augment   = augment

        self.train_aug = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.4),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
            A.GaussianBlur(p=0.2),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        self.val_aug = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name      = self.ids[idx]
        image     = np.array(Image.open(os.path.join(self.image_dir, name)).convert("RGB"))
        mask      = np.array(Image.open(os.path.join(self.mask_dir,  name)))
        mask      = convert_mask(mask)

        aug       = self.train_aug if self.augment else self.val_aug
        result    = aug(image=image, mask=mask)
        return result['image'], result['mask'].long()


# =============================================================================
# CNN Model: U-Net with ResNet34 backbone
# - Encoder: ResNet34 (pretrained on ImageNet) extracts features
# - Decoder: Upsamples features back to full resolution
# =============================================================================

def build_unet(num_classes):
    """
    U-Net with ResNet34 encoder.
    ResNet34 is a pure CNN — no transformers, no attention, just convolutions.
    Fast, reliable, and excellent for segmentation.
    """
    model = smp.Unet(
        encoder_name    = "resnet34",     # CNN backbone
        encoder_weights = "imagenet",     # pretrained weights
        in_channels     = 3,              # RGB input
        classes         = num_classes,    # output classes
        activation      = None            # raw logits (we apply softmax in loss)
    )
    return model


# =============================================================================
# Combined Loss: CrossEntropy + Dice
# CrossEntropy: good for overall accuracy
# Dice Loss: good for small/rare classes like Flowers, Logs
# =============================================================================

class CombinedLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ce_loss   = nn.CrossEntropyLoss()
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)

    def forward(self, outputs, masks):
        return self.ce_loss(outputs, masks) + self.dice_loss(outputs, masks)


# =============================================================================
# Metrics
# =============================================================================

def compute_iou(pred_logits, target, num_classes=N_CLASSES):
    pred          = torch.argmax(pred_logits, dim=1).cpu().numpy().flatten()
    target        = target.cpu().numpy().flatten()
    iou_per_class = []

    for cls in range(num_classes):
        pred_c   = pred == cls
        target_c = target == cls
        inter    = np.logical_and(pred_c, target_c).sum()
        union    = np.logical_or(pred_c,  target_c).sum()
        if union == 0:
            iou_per_class.append(np.nan)
        else:
            iou_per_class.append(inter / union)

    return np.nanmean(iou_per_class)


def compute_pixel_acc(pred_logits, target):
    pred = torch.argmax(pred_logits, dim=1)
    return (pred == target).float().mean().item()


def per_class_iou(pred_logits, target, num_classes=N_CLASSES):
    """Returns IoU for each class separately — useful for analysis."""
    pred   = torch.argmax(pred_logits, dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    ious   = {}

    for cls in range(num_classes):
        pred_c   = pred == cls
        target_c = target == cls
        inter    = np.logical_and(pred_c, target_c).sum()
        union    = np.logical_or(pred_c,  target_c).sum()
        if union == 0:
            ious[CLASS_NAMES[cls]] = np.nan
        else:
            ious[CLASS_NAMES[cls]] = inter / union

    return ious


def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, ious, accs = [], [], []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="  Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss    = loss_fn(outputs, masks)
            losses.append(loss.item())
            ious.append(compute_iou(outputs, masks))
            accs.append(compute_pixel_acc(outputs, masks))

    model.train()
    return np.mean(losses), np.nanmean(ious), np.mean(accs)


# =============================================================================
# Save Plots
# =============================================================================

def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('U-Net Training Results', fontsize=16)

    axes[0,0].plot(history['train_loss'], label='Train', color='blue')
    axes[0,0].plot(history['val_loss'],   label='Val',   color='orange')
    axes[0,0].set_title('Loss vs Epoch')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)

    axes[0,1].plot(history['train_iou'], label='Train', color='blue')
    axes[0,1].plot(history['val_iou'],   label='Val',   color='orange')
    axes[0,1].set_title('IoU vs Epoch')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('IoU')
    axes[0,1].legend()
    axes[0,1].grid(True)

    axes[1,0].plot(history['train_acc'], label='Train', color='blue')
    axes[1,0].plot(history['val_acc'],   label='Val',   color='orange')
    axes[1,0].set_title('Pixel Accuracy vs Epoch')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True)

    axes[1,1].plot(history['lr'], color='green')
    axes[1,1].set_title('Learning Rate Schedule')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('LR')
    axes[1,1].grid(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plots → {out_path}")


def save_metrics_txt(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(path, 'w') as f:
        f.write("U-Net TRAINING RESULTS\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Model:    U-Net with ResNet34 backbone\n")
        f.write(f"Epochs:   {N_EPOCHS}\n")
        f.write(f"Img Size: {IMG_SIZE}x{IMG_SIZE}\n\n")
        f.write(f"Best Val IoU:      {max(history['val_iou']):.4f}  "
                f"(Epoch {np.argmax(history['val_iou'])+1})\n")
        f.write(f"Best Val Accuracy: {max(history['val_acc']):.4f}  "
                f"(Epoch {np.argmax(history['val_acc'])+1})\n")
        f.write(f"Lowest Val Loss:   {min(history['val_loss']):.4f}  "
                f"(Epoch {np.argmin(history['val_loss'])+1})\n\n")
        f.write("=" * 65 + "\n")
        header = f"{'Epoch':<8}{'TrLoss':<12}{'VaLoss':<12}{'TrIoU':<12}{'VaIoU':<12}{'TrAcc':<12}{'VaAcc':<12}\n"
        f.write(header)
        f.write("-" * 78 + "\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1:<8}"
                    f"{history['train_loss'][i]:<12.4f}"
                    f"{history['val_loss'][i]:<12.4f}"
                    f"{history['train_iou'][i]:<12.4f}"
                    f"{history['val_iou'][i]:<12.4f}"
                    f"{history['train_acc'][i]:<12.4f}"
                    f"{history['val_acc'][i]:<12.4f}\n")

    print(f"  Saved metrics → {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU found. Training will be slow on CPU.")

    # Datasets
    print("\nLoading datasets...")
    train_ds = SegDataset(TRAIN_DIR, augment=True)
    val_ds   = SegDataset(VAL_DIR,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {N_EPOCHS}")

    # Model
    print("\nBuilding U-Net (ResNet34 CNN backbone)...")
    model      = build_unet(N_CLASSES).to(device)
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable parameters: {n_params:.1f}M")

    # Loss, optimizer, scheduler
    loss_fn   = CombinedLoss(N_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=N_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou':  [], 'val_iou':  [],
        'train_acc':  [], 'val_acc':  [],
        'lr': []
    }

    best_val_iou    = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, 'best_unet_model.pth')

    print(f"\nStarting training...")
    print("=" * 70)

    for epoch in range(N_EPOCHS):
        model.train()
        train_losses, train_ious, train_accs = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2}/{N_EPOCHS} [Train]", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            train_ious.append(compute_iou(outputs.detach(), masks))
            train_accs.append(compute_pixel_acc(outputs.detach(), masks))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        val_loss, val_iou, val_acc = evaluate(model, val_loader, loss_fn, device)
        current_lr = scheduler.get_last_lr()[0]

        # Store history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_loss)
        history['train_iou'].append(np.nanmean(train_ious))
        history['val_iou'].append(val_iou)
        history['train_acc'].append(np.mean(train_accs))
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Save best model
        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            torch.save({
                'epoch':      epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_iou':    val_iou,
                'val_acc':    val_acc,
            }, best_model_path)

        print(f"Epoch {epoch+1:2}/{N_EPOCHS} | "
              f"Loss: {np.mean(train_losses):.4f}/{val_loss:.4f} | "
              f"IoU: {np.nanmean(train_ious):.4f}/{val_iou:.4f} | "
              f"Acc: {val_acc:.4f}"
              f"{' ← BEST' if is_best else ''}")

    # Final save
    print("\nSaving results...")
    save_plots(history, OUTPUT_DIR)
    save_metrics_txt(history, OUTPUT_DIR)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_unet_model.pth'))

    print(f"\n{'='*50}")
    print(f"  Best Val IoU:  {best_val_iou:.4f}")
    print(f"  Final Val IoU: {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Acc: {history['val_acc'][-1]:.4f}")
    print(f"  Outputs saved: {OUTPUT_DIR}")
    print(f"{'='*50}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()