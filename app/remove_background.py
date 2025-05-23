import os
import sys

# Add segment-anything to Python path
SEGMENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'segment-anything'))
sys.path.append(SEGMENT_PATH)

import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def log_step(message: str, log: list[str]):
    print(message)
    log.append(message)

def remove_background(image_path: str) -> tuple[str, list[str]]:
    """
    Runs Segment Anything to remove the background and returns (output_path, status_log).
    """
    log = []

    log_step("🔧 Starting background removal pipeline...", log)

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODEL_PATH = os.path.join(BASE_DIR, 'segment-anything', 'sam_vit_b_01ec64.pth')

    log_step("📦 Loading SAM model...", log)
    sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH).to("cpu")
    mask_generator = SamAutomaticMaskGenerator(sam)

    log_step(f"🖼️ Loading image from {image_path}...", log)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    log_step("🧠 Generating segmentation masks...", log)
    masks = mask_generator.generate(image_rgb)
    log_step(f"✅ {len(masks)} mask(s) generated.", log)

    log_step("🎯 Selecting largest mask...", log)
    largest_mask = max(masks, key=lambda x: x['area'])['segmentation']

    log_step("🎨 Creating output image...", log)
    output = np.zeros_like(image_rgb)
    output[largest_mask] = image_rgb[largest_mask]

    alpha = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    alpha[largest_mask] = 255
    blurred_alpha = cv2.GaussianBlur(alpha, (21, 21), 0)

    rgba = np.dstack((output, blurred_alpha))
    result = Image.fromarray(rgba)

    output_path = os.path.join(os.path.dirname(image_path), "output_product_feathered.png")
    result.save(output_path)

    log_step(f"✅ Done! Saved as {output_path}", log)
    return output_path, log