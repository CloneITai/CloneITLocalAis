import sys
import os

# === STEP 0: Add segment-anything to path ===
SEGMENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'segment-anything'))
print(f"🛠️ Adding to sys.path: {SEGMENT_PATH}")
sys.path.append(SEGMENT_PATH)

# === STEP 1: Imports ===
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# === CONFIG ===
print("🔧 Setting paths...")
IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'mug.png')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'segment-anything', 'sam_vit_b_01ec64.pth')

# === STEP 2: Load SAM Model ===
print("📦 Loading SAM model...")
sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH).to("cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

# === STEP 3: Load Image ===
print(f"🖼️ Loading image from {IMAGE_PATH}...")
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"❌ Image not found at {IMAGE_PATH}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === STEP 4: Generate Masks ===
print("🧠 Generating segmentation masks (this may take a few seconds)...")
masks = mask_generator.generate(image_rgb)
print(f"✅ Generated {len(masks)} mask(s).")

# === STEP 5: Select Largest Mask ===
print("🎯 Selecting largest mask...")
largest_mask = max(masks, key=lambda x: x['area'])['segmentation']

# === STEP 6: Create Transparent PNG ===
print("🎨 Creating output image with feathered edges...")
output = np.zeros_like(image_rgb)
output[largest_mask] = image_rgb[largest_mask]

# Create blurred alpha channel
alpha = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
alpha[largest_mask] = 255
blurred_alpha = cv2.GaussianBlur(alpha, (21, 21), 0)

# Combine into RGBA
rgba = np.dstack((output, blurred_alpha))
result = Image.fromarray(rgba)
result.save("output_product_feathered.png")

print("✅ Done! Saved as output_product_feathered.png")