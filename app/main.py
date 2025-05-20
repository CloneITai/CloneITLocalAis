from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import io
import os
import cv2
import numpy as np
from PIL import Image

# === Import the background remover ===
from .remove_background import remove_background

app = FastAPI()

@app.post("/remove-background")
async def api_remove_background(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    contents = await file.read()
    image_np = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image content")

    # Save uploaded image temporarily
    temp_input_path = os.path.join("temp_input.png")
    cv2.imwrite(temp_input_path, image)

    # Call SAM processor
    output_path = remove_background(temp_input_path)

    # Load the result image
    with open(output_path, "rb") as f:
        image_bytes = f.read()

    # Optionally delete temp files
    os.remove(temp_input_path)
    os.remove(output_path)

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)