# app/utils.py

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the uploaded image for prediction.
    
    Args:
        image (PIL.Image.Image): The uploaded image.
        
    Returns:
        np.ndarray: Preprocessed image array.
    """
    # Example preprocessing steps
    image = image.resize((224, 224))  # Resize as per model's requirement
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

