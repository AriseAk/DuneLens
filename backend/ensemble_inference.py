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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_H = 640
IMG_W = 640
N_CLASSES = 10

DATA_DIR = "./Offroad_Segmentation_testImages/Offroad_Segmentation_testImages"
image_dir = os.path.join(DATA_DIR,"Color_Images")
mask_dir = os.path.join(DATA_DIR,"Segmentation")

OUTPUT_DIR = "./ensemble_predictions"
os.makedirs(OUTPUT_DIR,exist_ok=True)

# =========================
# TRANSFORM
# =========================

transform = A.Compose([
    A.Resize(IMG_H,IMG_W),
    A.Normalize(mean=[0.452,0.431,0.376],std=[0.218,0.213,0.207]),
    ToTensorV2()
])

# =========================
# MASK REMAP
# =========================

VALUE_MAP = {
0:0,100:1,200:2,300:3,500:4,
550:5,600:6,700:7,800:8,7100:9,10000:10
}

def remap_mask(mask):

    if len(mask.shape)==2:
        raw = mask.astype(np.int32)
    else:
        raw = mask[:,:,0].astype(np.int32) + mask[:,:,1].astype(np.int32)*256

    mapped = np.zeros(raw.shape,dtype=np.uint8)

    for raw_val,cls in VALUE_MAP.items():
        mapped[raw==raw_val] = cls

    return mapped

# =========================
# IOU
# =========================

def compute_iou(pred,gt):

    pred = pred.flatten()
    gt = gt.flatten()

    ious=[]

    for c in range(N_CLASSES):

        p = pred==c
        t = gt==c

        inter = np.logical_and(p,t).sum()
        union = np.logical_or(p,t).sum()

        if union==0:
            continue

        ious.append(inter/union)

    if len(ious)==0:
        return 0

    return np.mean(ious)

# =========================
# LOAD SEGFORMER B2
# =========================

print("Loading SegFormer B2")

model_b2 = SegformerForSemanticSegmentation.from_pretrained(
"nvidia/mit-b2",
num_labels=N_CLASSES,
ignore_mismatched_sizes=True
)

ckpt = torch.load("best_model.pth",map_location=DEVICE)
model_b2.load_state_dict(ckpt["model_state_dict"])

model_b2.to(DEVICE)
model_b2.eval()

# =========================
# LOAD CNN (U-NET)
# =========================

print("Loading CNN")

cnn = smp.Unet(
encoder_name="resnet34",
encoder_weights=None,
in_channels=3,
classes=11,  # original training
activation=None
)

ckpt = torch.load("best_unet_model.pth",map_location=DEVICE,weights_only=False)
cnn.load_state_dict(ckpt["model_state_dict"])

cnn.to(DEVICE)
cnn.eval()

print("Models loaded")

# =========================
# SEGFORMER PREDICT
# =========================

def segformer_predict(model,img):

    logits = model(pixel_values=img).logits

    logits = F.interpolate(
        logits,
        size=(IMG_H,IMG_W),
        mode="bilinear",
        align_corners=False
    )

    return logits

# =========================
# TTA
# =========================

def tta_segformer(model,img):

    logits1 = segformer_predict(model,img)

    flipped = torch.flip(img,dims=[3])
    logits2 = segformer_predict(model,flipped)
    logits2 = torch.flip(logits2,dims=[3])

    return (logits1 + logits2)/2


def tta_cnn(model,img):

    logits1 = model(img)

    flipped = torch.flip(img,dims=[3])
    logits2 = model(flipped)
    logits2 = torch.flip(logits2,dims=[3])

    return (logits1 + logits2)/2

# =========================
# INFERENCE
# =========================

ids = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

print("Images:",len(ids))

iou_scores=[]

with torch.no_grad():

    for name in tqdm(ids):

        img_path = os.path.join(image_dir,name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        data = transform(image=img)
        img_tensor = data["image"].unsqueeze(0).to(DEVICE)

        logits_b2 = tta_segformer(model_b2,img_tensor)

        logits_cnn = tta_cnn(cnn,img_tensor)

        logits_cnn = F.interpolate(
            logits_cnn,
            size=(IMG_H,IMG_W),
            mode="bilinear",
            align_corners=False
        )

        # remove CNN extra class
        logits_cnn = logits_cnn[:,:10,:,:]

        # =========================
        # SOFTMAX ENSEMBLE
        # =========================

        p_b2 = torch.softmax(logits_b2,dim=1)
        p_cnn = torch.softmax(logits_cnn,dim=1)

        final = (
            0.75 * p_b2 +
            0.25 * p_cnn
        )

        pred = torch.argmax(final,dim=1).squeeze().cpu().numpy()

        cv2.imwrite(
            os.path.join(OUTPUT_DIR,name),
            pred.astype(np.uint8)
        )

        mask_path = os.path.join(mask_dir,name)

        if os.path.exists(mask_path):

            mask_raw = cv2.imread(mask_path,cv2.IMREAD_UNCHANGED)

            gt = remap_mask(mask_raw)

            gt = cv2.resize(
                gt,
                (IMG_W,IMG_H),
                interpolation=cv2.INTER_NEAREST
            )

            iou = compute_iou(pred,gt)

            iou_scores.append(iou)

# =========================
# RESULTS
# =========================

if len(iou_scores)>0:
    print("\nMean IoU:",np.mean(iou_scores))

print("\nPredictions saved to:",OUTPUT_DIR)
print("Done")