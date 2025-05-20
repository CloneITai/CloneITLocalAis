from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
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

    try:
        # Call SAM processor
        output_path, status_log = remove_background(temp_input_path)

        # Read the result as base64 if you want to embed in JSON (optional)
        with open(output_path, "rb") as f:
            image_bytes = f.read()

        # Clean up
        os.remove(temp_input_path)
        os.remove(output_path)

        return JSONResponse(content={
            "status": "success",
            "steps": status_log,
            "message": "Background removed successfully",
            "image": image_bytes.hex()  # or base64 if needed
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })