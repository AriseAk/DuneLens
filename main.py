from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Point to the exact path according to your project structure
MODEL_PATH = "./backend/best_model.pth" 

CLASS_MAP = {
    0: {"name": "Background", "color": [0, 0, 0], "hex": "#000000"},
    1: {"name": "Trees", "color": [44, 160, 44], "hex": "#2ca02c"},
    2: {"name": "Lush Bushes", "color": [152, 223, 138], "hex": "#98df8a"},
    3: {"name": "Dry Grass", "color": [232, 169, 76], "hex": "#e8a94c"},
    4: {"name": "Dry Bushes", "color": [201, 145, 62], "hex": "#c9913e"},
    5: {"name": "Ground Clutter", "color": [140, 86, 75], "hex": "#8c564b"},
    6: {"name": "Logs", "color": [196, 156, 148], "hex": "#c49c94"},
    7: {"name": "Rocks", "color": [127, 127, 127], "hex": "#7f7f7f"},
    8: {"name": "Landscape", "color": [184, 148, 106], "hex": "#b8946a"},
    9: {"name": "Sky", "color": [31, 119, 180], "hex": "#1f77b4"}
}

print("Loading Segformer model...")
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2", num_labels=10, ignore_mismatched_sizes=True
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

class ImageRequest(BaseModel):
    image_data: str 

@app.post("/predict")
async def predict(request: ImageRequest):
    # 1. Decode base64 image
    encoded_data = request.image_data.split(",")[1] if "," in request.image_data else request.image_data
    img_bytes = base64.b64decode(encoded_data)
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    # 2. Run Inference
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False,
    )
    predictions = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    # 3. Create Colored Mask Image
    color_mask = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    for class_idx, class_info in CLASS_MAP.items():
        color_mask[predictions == class_idx] = class_info["color"]
    
    # Blend original image and mask (50% opacity)
    mask_img = Image.fromarray(color_mask)
    blended = Image.blend(image, mask_img, alpha=0.6)
    
    # Convert blended image back to base64 to send to frontend
    buffered = BytesIO()
    blended.save(buffered, format="JPEG")
    mask_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 4. Calculate Distribution
    total_pixels = predictions.size
    unique_classes, counts = np.unique(predictions, return_counts=True)
    
    distribution = []
    for cls_idx, count in zip(unique_classes, counts):
        if cls_idx == 0: continue # Skip Background
        percentage = (count / total_pixels) * 100
        if percentage > 0.5: # Only include features that make up >0.5% of the image
            distribution.append({
                "name": CLASS_MAP[cls_idx]["name"],
                "percentage": float(percentage),
                "color": CLASS_MAP[cls_idx]["hex"]
            })

    # Sort distribution by highest percentage
    distribution = sorted(distribution, key=lambda x: x["percentage"], reverse=True)

    return {
        "originalImage": request.image_data,
        "maskImage": mask_base64,
        "distribution": distribution
    }