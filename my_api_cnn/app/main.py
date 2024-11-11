# app/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging

from model import build_CNN_AE_model

app = FastAPI(
    title="CNN Model Serving API",
    description="API for serving CNN model predictions",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define class labels (if applicable)
# For segmentation tasks, you might not have discrete classes. Adjust accordingly.
CLASS_LABELS = {
    0: "Class A",
    1: "Class B",
    2: "Class C"
}

# Define the response model
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

# API Key for simple authentication (Optional)
API_KEY = os.getenv("API_KEY")  # Set this environment variable securely

def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        logger.warning(f"Unauthorized access attempt with API_KEY: {x_api_key}")
        raise HTTPException(status_code=403, detail="Could not validate API KEY")
    return x_api_key

# Load the model at startup
@app.on_event("startup")
def load_cnn_model():
    global model
    try:
        model = build_CNN_AE_model(input_shape=(224, 224, 3))  # Adjust input shape if needed
        model.load_weights("best_model.h5")
        model.compile(optimizer='adam', loss='binary_crossentropy')  # Adjust compile parameters if necessary
        logger.info("Model loaded and compiled successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    # Validate the file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(f"Invalid file type received: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG or PNG.")
    
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess the image
        image = image.resize((224, 224))  # Adjust size as per your model's requirement
        image_array = np.array(image) / 255.0  # Normalize if required
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions, axis=1)[0])
        
        # Map class index to class label (if applicable)
        prediction_label = CLASS_LABELS.get(predicted_class, "Unknown")
        
        logger.info(f"Prediction: {prediction_label}, Confidence: {confidence}")
        
        return JSONResponse(content={"prediction": prediction_label, "confidence": confidence})
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

