# app/model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2DTranspose, Activation
from tensorflow.keras.applications import EfficientNetB0

def build_CNN_AE_model(input_shape=(224, 224, 3)) -> Model:
    """
    Create CNN autoencoder model.
    
    Returns:
        (Model): Keras model.
    """
    # Load the EfficientNetB0 model architecture without weights and without the top layer
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights=None)

    # Add upsampling and convolutional layers for segmentation
    x = base_model.output
    x = Conv2DTranspose(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(16, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(4, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(1, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = Activation('sigmoid')(x)
    
    return Model(inputs=base_model.input, outputs=x)

