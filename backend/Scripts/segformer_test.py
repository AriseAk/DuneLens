import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "../best_model_final.pth"
DATA_DIR = "../Offroad_Segmentation_testImages/Offroad_Segmentation_testImages"
OUTPUT_DIR = "../predictions1"

IMG_H = 640
IMG_W = 640
N_CLASSES = 11

os.makedirs(OUTPUT_DIR, exist_ok=True)

VALUE_MAP = {
0:0,
100:1,
200:2,
300:3,
500:4,
550:5,
600:6,
700:7,
800:8,
7100:9,
10000:10
}

def remap_mask(mask):

    if len(mask.shape) == 2:
        raw = mask.astype(np.int32)
    else:
        raw = mask[:,:,0].astype(np.int32) + mask[:,:,1].astype(np.int32)*256

    mapped = np.zeros(raw.shape, dtype=np.uint8)

    for raw_val, cls in VALUE_MAP.items():
        mapped[raw == raw_val] = cls

    return mapped

def compute_iou(pred, gt):

    pred = pred.flatten()
    gt = gt.flatten()

    ious = []

    for c in range(N_CLASSES):

        p = pred == c
        t = gt == c

        inter = np.logical_and(p,t).sum()
        union = np.logical_or(p,t).sum()

        if union == 0:
            continue

        ious.append(inter/union)

    if len(ious)==0:
        return 0

    return np.mean(ious)

transform = A.Compose([
A.Resize(IMG_H, IMG_W),
A.Normalize(mean=[0.452,0.431,0.376],
std=[0.218,0.213,0.207]),
ToTensorV2()
])

print("Loading model...")

model = SegformerForSemanticSegmentation.from_pretrained(
"nvidia/mit-b2",
num_labels=N_CLASSES,
ignore_mismatched_sizes=True
)

checkpoint = torch.load(MODEL_PATH,map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()

print("Model loaded")

image_dir = os.path.join(DATA_DIR,"Color_Images")
mask_dir = os.path.join(DATA_DIR,"Segmentation")

ids = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

print("Found",len(ids),"images")

iou_scores = []
debug_printed = False

with torch.no_grad():

    for name in tqdm(ids):

        img_path = os.path.join(image_dir,name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        data = transform(image=img)
        img_tensor = data["image"].unsqueeze(0).to(DEVICE)

        logits = model(pixel_values=img_tensor).logits

        logits = F.interpolate(
            logits,
            size=(IMG_H,IMG_W),
            mode="bilinear",
            align_corners=False
        )

        pred = torch.argmax(logits,dim=1).squeeze().cpu().numpy().astype(np.uint8)

        mask_path = os.path.join(mask_dir,name)

        if os.path.exists(mask_path):

            mask_raw = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)

            gt = remap_mask(mask_raw)

            gt = cv2.resize(
                gt,
                (IMG_W,IMG_H),
                interpolation=cv2.INTER_NEAREST
            )

            # DEBUG PRINT (only once)
            if not debug_printed:
                print("\nDEBUG CHECK")
                print("Pred classes:", np.unique(pred))
                print("GT classes:", np.unique(gt))
                debug_printed = True

            iou = compute_iou(pred,gt)

            iou_scores.append(iou)

if len(iou_scores)>0:

    print("\nMean IoU:",np.mean(iou_scores))

else:

    print("\nNo IoU computed")

print("\nDone")