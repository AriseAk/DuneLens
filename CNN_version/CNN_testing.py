import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm


TEST_DIR   = '/content/drive/MyDrive/EliteHack/Offroad_Segmentation_testImages'
MODEL_PATH = '/content/drive/MyDrive/EliteHack/outputs/best_unet_model.pth'


IMG_SIZE  = 256
N_CLASSES = 11
BATCH_SIZE = 8

# =============================================================================
# Class Info
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
    new_mask = np.zeros_like(mask_array, dtype=np.uint8)
    for raw_val, class_id in VALUE_MAP.items():
        new_mask[mask_array == raw_val] = class_id
    return new_mask


# =============================================================================
# Test Dataset
# =============================================================================

class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir  = os.path.join(data_dir, 'Segmentation')
        self.ids       = sorted(os.listdir(self.image_dir))

        self.transform = A.Compose([
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
        result    = self.transform(image=image, mask=mask)
        return result['image'], result['mask'].long()


# =============================================================================
# Metrics
# =============================================================================

def compute_iou_per_class(pred_logits, target, num_classes=N_CLASSES):
    """Returns IoU for each class separately."""
    pred   = torch.argmax(pred_logits, dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    ious   = []

    for cls in range(num_classes):
        pred_c   = pred == cls
        target_c = target == cls
        inter    = np.logical_and(pred_c, target_c).sum()
        union    = np.logical_or(pred_c,  target_c).sum()

        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)

    return ious


def compute_pixel_acc(pred_logits, target):
    pred = torch.argmax(pred_logits, dim=1)
    return (pred == target).float().mean().item()


# =============================================================================
# Main Test Function
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    print(f"\nLoading test dataset from: {TEST_DIR}")
    test_ds     = TestDataset(TEST_DIR)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test samples: {len(test_ds)}")

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,       # no pretrained, we load our own weights
        in_channels     = 3,
        classes         = N_CLASSES,
        activation      = None
    ).to(device)

    # Load saved weights
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
              f"with Val IoU: {checkpoint.get('val_iou', '?'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully!")

    # Evaluate
    print("\nRunning evaluation on test set...")
    print("=" * 60)

    model.eval()
    all_class_ious = [[] for _ in range(N_CLASSES)]
    all_pixel_accs = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs       = model(images)

            # Per class IoU for this batch
            batch_ious = compute_iou_per_class(outputs, masks)
            for cls in range(N_CLASSES):
                if not np.isnan(batch_ious[cls]):
                    all_class_ious[cls].append(batch_ious[cls])

            all_pixel_accs.append(compute_pixel_acc(outputs, masks))

    # Calculate final scores
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    class_ious = []
    print(f"\nPer-Class IoU:")
    print("-" * 40)
    for cls in range(N_CLASSES):
        if len(all_class_ious[cls]) > 0:
            cls_iou = np.mean(all_class_ious[cls])
            class_ious.append(cls_iou)
            bar = "█" * int(cls_iou * 20)
            print(f"  {CLASS_NAMES[cls]:<20} {cls_iou:.4f}  {bar}")
        else:
            print(f"  {CLASS_NAMES[cls]:<20} N/A (not present in test set)")

    mean_iou    = np.nanmean(class_ious)
    mean_acc    = np.mean(all_pixel_accs)

    print("\n" + "=" * 60)
    print(f"  Mean IoU (mIoU):   {mean_iou:.4f}  ({mean_iou*100:.2f}%)")
    print(f"  Pixel Accuracy:    {mean_acc:.4f}  ({mean_acc*100:.2f}%)")
    print("=" * 60)

    # Performance rating
    if mean_iou >= 0.65:
        rating = "Excellent"
    elif mean_iou >= 0.50:
        rating = "Good"
    elif mean_iou >= 0.35:
        rating = "Moderate"
    else:
        rating = "Needs Improvement"

    print(f"\n  Performance Rating: {rating}")
    print("\nTest complete!")

    return mean_iou


if __name__ == "__main__":
    main()