import os
import sys

# === STEP 0: Add segment-anything to Python path ===
SEGMENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'segment-anything'))
sys.path.append(SEGMENT_PATH)

# === STEP 1: Imports ===
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def remove_background(image_path: str) -> str:
    """
    Runs Segment Anything to remove the background and returns path to output image.
    """
    print("🔧 Starting background removal pipeline...")
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODEL_PATH = os.path.join(BASE_DIR, 'segment-anything', 'sam_vit_b_01ec64.pth')

    # === STEP 2: Load SAM Model ===
    print("📦 Loading SAM model...")
    sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH).to("cpu")
    mask_generator = SamAutomaticMaskGenerator(sam)

    # === STEP 3: Load Image ===
    print(f"🖼️ Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # === STEP 4: Generate Masks ===
    print("🧠 Generating segmentation masks...")
    masks = mask_generator.generate(image_rgb)
    print(f"✅ {len(masks)} mask(s) generated.")

    # === STEP 5: Select Largest Mask ===
    print("🎯 Selecting largest mask...")
    largest_mask = max(masks, key=lambda x: x['area'])['segmentation']

    # === STEP 6: Create Transparent PNG ===
    print("🎨 Creating output image...")
    output = np.zeros_like(image_rgb)
    output[largest_mask] = image_rgb[largest_mask]

    alpha = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    alpha[largest_mask] = 255
    blurred_alpha = cv2.GaussianBlur(alpha, (21, 21), 0)

    rgba = np.dstack((output, blurred_alpha))
    result = Image.fromarray(rgba)

    output_path = os.path.join(os.path.dirname(image_path), "output_product_feathered.png")
    result.save(output_path)

    print(f"✅ Done! Saved as {output_path}")
    return output_path