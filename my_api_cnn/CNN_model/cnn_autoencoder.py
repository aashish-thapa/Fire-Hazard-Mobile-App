import re
from typing import Dict, List, Text, Tuple, Callable
import matplotlib.pyplot as plt
from matplotlib import colors

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.python.keras.utils.losses_utils import reduce_weighted_loss
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

# Constants for the data reader
INPUT_FEATURES = [
    'elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph',
    'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask'
]

OUTPUT_FEATURES = ['FireMask']

# Data statistics
# For each variable, the statistics are ordered in the form:
# (min_clip, max_clip, mean, standard deviation)
DATA_STATS = {
    'elevation': (0.0, 3141.0, 657.3003, 649.0147),
    'pdsi': (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    'NDVI': (-9821.0, 9996.0, 5157.625, 2466.6677),
    'pr': (0.0, 44.53038024902344, 1.7398051, 4.482833),
    'sph': (0.0, 1.0, 0.0071658953, 0.0042835088),
    'th': (0.0, 360.0, 190.32976, 72.59854),
    'tmmn': (253.15, 298.9489, 281.08768, 8.982386),
    'tmmx': (253.15, 315.0923, 295.17383, 9.815496),
    'vs': (0.0, 10.02431, 3.8500874, 1.4109988),
    'erc': (0.0, 106.2489166, 37.326267, 20.846027),
    'population': (0.0, 2534.06298828125, 25.531384, 154.72331),
    'PrevFireMask': (-1.0, 1.0, 0.0, 1.0),
    'FireMask': (-1.0, 1.0, 0.0, 1.0)
}

# Data preprocessing functions
def random_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
    num_in_channels: int,
    num_out_channels: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Randomly axis-align crop input and output image tensors."""
    combined = tf.concat([input_img, output_img], axis=2)
    combined = tf.image.random_crop(
        combined,
        [sample_size, sample_size, num_in_channels + num_out_channels]
    )
    input_img = combined[:, :, :num_in_channels]
    output_img = combined[:, :, -num_out_channels:]
    return input_img, output_img

def center_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Center crops input and output image tensors."""
    central_fraction = sample_size / input_img.shape[0]
    input_img = tf.image.central_crop(input_img, central_fraction)
    output_img = tf.image.central_crop(output_img, central_fraction)
    return input_img, output_img

def _get_base_key(key: Text) -> Text:
    """Extracts the base key from the provided key."""
    match = re.match(r'([a-zA-Z]+)', key)
    if match:
        return match.group(1)
    raise ValueError(f'The provided key does not match the expected pattern: {key}')

def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    """Clips and rescales inputs with the stats corresponding to `key`."""
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(f'No data statistics available for the requested key: {key}.')
    min_val, max_val, _, _ = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan(inputs - min_val, max_val - min_val)

def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    """Clips and normalizes inputs with the stats corresponding to `key`."""
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(f'No data statistics available for the requested key: {key}.')
    min_val, max_val, mean, std = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    inputs -= mean
    return tf.math.divide_no_nan(inputs, std)

def _get_features_dict(
    sample_size: int,
    features: List[Text]
) -> Dict[Text, tf.io.FixedLenFeature]:
    """Creates a features dictionary for TensorFlow IO."""
    sample_shape = [sample_size, sample_size]
    features = set(features)
    columns = [
        tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32)
        for _ in features
    ]
    return dict(zip(features, columns))

def _parse_fn(
    example_proto: tf.train.Example,
    data_size: int,
    sample_size: int,
    num_in_channels: int,
    clip_and_normalize: bool,
    clip_and_rescale: bool,
    random_crop: bool,
    center_crop: bool
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reads a serialized example."""
    if random_crop and center_crop:
        raise ValueError('Cannot have both random_crop and center_crop be True')
    input_features, output_features = INPUT_FEATURES, OUTPUT_FEATURES
    feature_names = input_features + output_features
    features_dict = _get_features_dict(data_size, feature_names)
    features = tf.io.parse_single_example(example_proto, features_dict)

    if clip_and_normalize:
        inputs_list = [
            _clip_and_normalize(features.get(key), key) for key in input_features
        ]
    elif clip_and_rescale:
        inputs_list = [
            _clip_and_rescale(features.get(key), key) for key in input_features
        ]
    else:
        inputs_list = [features.get(key) for key in input_features]

    inputs_stacked = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])

    outputs_list = [features.get(key) for key in output_features]
    assert outputs_list, 'outputs_list should not be empty'
    outputs_stacked = tf.stack(outputs_list, axis=0)
    outputs_stacked_shape = outputs_stacked.get_shape().as_list()
    assert len(outputs_stacked.shape) == 3, (
        'outputs_stacked should be rank 3 '
        f'but dimensions of outputs_stacked are {outputs_stacked_shape}'
    )
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])

    if random_crop:
        input_img, output_img = random_crop_input_and_output_images(
            input_img, output_img, sample_size, num_in_channels, 1)
    if center_crop:
        input_img, output_img = center_crop_input_and_output_images(
            input_img, output_img, sample_size)
    return input_img, output_img

def get_dataset(
    file_pattern: Text,
    data_size: int,
    sample_size: int,
    batch_size: int,
    num_in_channels: int,
    compression_type: Text,
    clip_and_normalize: bool,
    clip_and_rescale: bool,
    random_crop: bool,
    center_crop: bool
) -> tf.data.Dataset:
    """Gets the dataset from the file pattern."""
    if clip_and_normalize and clip_and_rescale:
        raise ValueError('Cannot have both normalize and rescale.')
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x: _parse_fn(
            x, data_size, sample_size, num_in_channels, clip_and_normalize,
            clip_and_rescale, random_crop, center_crop),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Set batch size
BATCH_SIZE = 32

# Prepare datasets
train_dataset = get_dataset(
    'next_day_wildfire_spread_train*',
    data_size=64, sample_size=32, batch_size=BATCH_SIZE,
    num_in_channels=12, compression_type=None, clip_and_normalize=True,
    clip_and_rescale=False, random_crop=True, center_crop=False
)

validation_dataset = get_dataset(
    'next_day_wildfire_spread_eval*',
    data_size=64, sample_size=32, batch_size=BATCH_SIZE,
    num_in_channels=12, compression_type=None, clip_and_normalize=True,
    clip_and_rescale=False, random_crop=True, center_crop=False
)

test_dataset = get_dataset(
    'next_day_wildfire_spread_test*',
    data_size=64, sample_size=32, batch_size=BATCH_SIZE,
    num_in_channels=12, compression_type=None, clip_and_normalize=True,
    clip_and_rescale=False, random_crop=True, center_crop=False
)

# Titles for plotting
TITLES = [
    'Elevation', 'Wind\ndirection', 'Wind\nvelocity', 'Min\ntemp', 'Max\ntemp',
    'Humidity', 'Precip', 'Drought', 'Vegetation', 'Population\ndensity',
    'Energy\nrelease\ncomponent', 'Previous\nfire\nmask', 'Fire\nmask'
]

# Function to plot samples from dataset
def plot_samples_from_dataset(dataset: tf.data.Dataset, n_rows: int):
    """Plot 'n_rows' rows of samples from dataset."""
    inputs, labels = next(iter(dataset))
    fig = plt.figure(figsize=(15, 6.5))
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    n_features = 12
    for i in range(n_rows):
        for j in range(n_features + 1):
            plt.subplot(n_rows, n_features + 1, i * (n_features + 1) + j + 1)
            if i == 0:
                plt.title(TITLES[j], fontsize=13)
            if j < n_features - 1:
                plt.imshow(inputs[i, :, :, j], cmap='viridis')
            elif j == n_features - 1:
                plt.imshow(inputs[i, :, :, -1], cmap=CMAP, norm=NORM)
            elif j == n_features:
                plt.imshow(labels[i, :, :, 0], cmap=CMAP, norm=NORM)
            plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot samples
plot_samples_from_dataset(train_dataset, 5)

# Define evaluation metrics
def IoU_metric(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    """Calculation of Intersection over Union (IoU) metric."""
    real_mask = tf.where(real_mask < 0, 0, real_mask)
    intersection = np.logical_and(real_mask, predicted_mask)
    union = np.logical_or(real_mask, predicted_mask)
    if np.sum(union) == 0:
        return 1.0
    return np.sum(intersection) / np.sum(union)

def recall_metric(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    """Calculation of recall metric."""
    real_mask = tf.where(real_mask < 0, 0, real_mask)
    true_positives = np.sum(np.logical_and(real_mask, predicted_mask))
    actual_positives = np.sum(real_mask)
    if actual_positives == 0:
        return 1.0
    return true_positives / actual_positives

def precision_metric(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    """Calculation of precision metric."""
    real_mask = tf.where(real_mask < 0, 0, real_mask)
    true_positives = np.sum(np.logical_and(real_mask, predicted_mask))
    predicted_positives = np.sum(predicted_mask)
    if predicted_positives == 0:
        return 1.0
    return true_positives / predicted_positives

def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Dice loss function calculator.
    
    Args:
        y_true (Tensor): 
        y_pred (Tensor):
    Returns:
        (Tensor): Dice loss for each element of a batch.
    """
    smooth = 1e-6
    y_true_f = K.reshape(y_true, (BATCH_SIZE, -1))
    y_pred_f = K.reshape(y_pred, (BATCH_SIZE, -1))
    intersection = K.sum(y_true_f * y_pred_f, axis=1)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f, axis=1) + K.sum(y_pred_f, axis=1) + smooth)

def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    BCE loss function calculator.
    
    Args:
        y_true (Tensor): 
        y_pred (Tensor):
    Returns:
        (Tensor): Mean BCE Dice loss over a batch.
    """
    
    y_true_f = K.reshape(y_true, (BATCH_SIZE, -1))
    y_pred_f = K.reshape(y_pred, (BATCH_SIZE, -1))
    return reduce_weighted_loss(tf.keras.losses.binary_crossentropy(y_true_f, y_pred_f) + dice_coef(y_true, y_pred))
    
# Evaluation function
def evaluate_model(prediction_function: Callable[[tf.Tensor], tf.Tensor],
                   eval_dataset: tf.data.Dataset) -> Tuple[float, float, float, float]:
    """
    Loads dataset according to file pattern and evaluates model's predictions on it.
    
    Parameters:
        model (Callable[[tf.Tensor], tf.Tensor]): Function for model inference.
        eval_dataset (tf.dataDataset): Dataset for evaluation.
    
    Returns:
        Tuple[float, float, float, float]: IoU score, recall score, precision score and mean loss.
    """
    IoU_measures = []
    recall_measures = []
    precision_measures = []
    losses = []
    
    for inputs, labels in tqdm(eval_dataset):
        # Prediction shape (N, W, H)
        predictions = prediction_function(inputs)
        for i in range(inputs.shape[0]):
            IoU_measures.append(IoU_metric(labels[i, :, :,  0], predictions[i, :, :]))
            recall_measures.append(recall_metric(labels[i, :, :,  0], predictions[i, :, :]))
            precision_measures.append(precision_metric(labels[i, :, :,  0], predictions[i, :, :]))
        labels_cleared = tf.where(labels < 0, 0, labels)
        losses.append(bce_dice_loss(labels_cleared, tf.expand_dims(tf.cast(predictions, tf.float32), axis=-1)))
            
    mean_IoU = np.mean(IoU_measures)
    mean_recall = np.mean(recall_measures)
    mean_precision = np.mean(precision_measures)
    mean_loss = np.mean(losses)
    return mean_IoU, mean_recall, mean_precision, mean_loss
# Import necessary modules for model building
from tensorflow.keras.applications import EfficientNetB0

# Build the model

def build_CNN_AE_model() -> Model:
    """
    Create CNN auto encode model.
    
    Returns:
        (Model): Keras model.
    """
    # Load the EfficientNetB0 model architecture
    base_model = EfficientNetB0(input_shape=(32, 32, 12), include_top=False, weights=None)



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
    x = Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    
    return Model(inputs=base_model.input, outputs=x)

# Create the segmentation model
segmentation_model = build_CNN_AE_model()
segmentation_model.summary()

# Train the model and save the best model locally

# Training function
# Training function
def train_model(model: Model, train_dataset: tf.data.Dataset, epochs:int=10) -> Tuple[List[float], List[float]]:
    """
    Trains a model using train dataset. (Save weights of model with best IoU)
    
    Args:
        model (Model): Model to train.
        train_dataset (Dataset): Training dataset.
        epochs (int): Number of epochs
    Returns:
        Tuple[List[float], List[float]]: Train losses and Validation losses
    """
    loss_fn = bce_dice_loss
    optimizer = tf.keras.optimizers.Adam()
    batch_losses = []
    val_losses = []
    best_IoU = 0.0
    
    for epoch in range(epochs):
        losses = []
        print(f'Epoch {epoch+1}/{epochs}')
        # Iterate through the dataset
        progress = tqdm(train_dataset)
        for images, masks in progress:
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(images, training=True)
                label = tf.where(masks < 0, 0, masks)
                # Compute the loss
                loss = loss_fn(label, predictions)
                losses.append(loss.numpy())
                progress.set_postfix({'batch_loss': loss.numpy()})
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            # Update the model's weights
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Evaluate model
        print("Evaluation...")
        IoU, recall, precision, val_loss = evaluate_model(lambda x: tf.where(model.predict(x) > 0.5, 1, 0)[:,:,:,0], validation_dataset)
        print("Validation set metrics:")
        print(f"Mean IoU: {IoU}\nMean precision: {precision}\nMean recall: {recall}\nValidation loss: {val_loss}\n")
        # Save best model
        if IoU > best_IoU:
            best_IoU = IoU
            model.save_weights("best.h5")
        
        # Print the loss for monitoring
        print(f'Epoch: {epoch}, Train loss: {np.mean(losses)}')
        batch_losses.append(np.mean(losses))
        val_losses.append(val_loss)
    
    print(f"Best model IoU: {best_IoU}")
    return batch_losses, val_losses

# Set reproducability
tf.random.set_seed(1337)

segmentation_model = build_CNN_AE_model()
train_losses, val_losses = train_model(segmentation_model, train_dataset, epochs=15)

# Plot training and validation losses
def plot_train_and_val_losses(train_losses, val_losses):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(train_losses)
    axs[0].set_title("Train Loss")
    axs[1].plot(val_losses)
    axs[1].set_title("Validation Loss")
    plt.show()

plot_train_and_val_losses(train_losses, val_losses)

# Load best model weights
segmentation_model = build_CNN_AE_model()
segmentation_model.load_weights("best.h5")

# Evaluate model on test set
print("Evaluation...")
print("Test set metrics:")
IoU, recall, precision, val_loss = evaluate_model(
    lambda x: tf.where(segmentation_model.predict(x) > 0.5, 1, 0)[:, :, :, 0], test_dataset)
print(f"Mean IoU: {IoU}\nMean precision: {precision}\nMean recall: {recall}\nTest loss: {val_loss}")

import shutil
import os

# Save the entire model after loading the best weights
segmentation_model.save('best_model.h5')

# Optionally, zip the model for easy download
#shutil.make_archive('/kaggle/working/best_model', 'zip', '/kaggle/working', 'best_model.h5')

# If you generated plots during training, zip them as well
plot_files = ['training_loss.png', 'training_auc.png']  # Replace with your actual plot filenames
for plot in plot_files:
    plt.savefig()
#shutil.make_archive('/kaggle/working/training_plots', 'zip', '/kaggle/working', '.')

print("Model and plots have been saved and zipped in the working directory.")


class NaivePredictor:
    """
    Naive predictor that predicts fire only if cell has chance being on fire more than 0.2
    """
    def __init__(self) -> None:
        """
        Initialize model and create frequency matrix
        """
        self.frequency_matrix = tf.zeros((32, 32), dtype=np.float32)
    
    def train(self, train_dataset: tf.data.Dataset) -> None:
        """
        Train by calculating frequency for each cell.
        
        Args:
            train_dataset (Dataset): Dataset to train on.
        """
        for _, labels in tqdm(train_dataset):
            label_batch = labels[:, :, :, 0]
            label_batch = tf.where(label_batch < 0, 0, label_batch)
            self.frequency_matrix = self.frequency_matrix + np.sum(label_batch, axis=0)
        self.frequency_matrix = self.frequency_matrix / np.max(self.frequency_matrix)
        self.frequency_matrix = tf.where(self.frequency_matrix > 0.2, 1, 0)
    
    def predict(self, X: tf.Tensor) -> tf.Tensor:
        """
        Dummy predict function.
        
        Args:
            train_dataset (Dataset): Dataset to train on.
        Returns:
            (Tensor): Predicted fire mask.
        """
        return tf.tile(tf.expand_dims(self.frequency_matrix, axis=0), [X.shape[0],1,1])

naive_predictor = NaivePredictor()
naive_predictor.train(train_dataset)


# Function to display inference results
def show_inference(
    n_rows: int,
    features: tf.Tensor,
    label: tf.Tensor,
    prediction_function: Callable[[tf.Tensor], tf.Tensor]
) -> None:
    """Displays model inference results."""
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    fig = plt.figure(figsize=(15, n_rows * 4))
    prediction = prediction_function(features)
    for i in range(n_rows):
        plt.subplot(n_rows, 3, i * 3 + 1)
        plt.title("Previous day fire")
        plt.imshow(features[i, :, :, -1], cmap=CMAP, norm=NORM)
        plt.axis('off')
        plt.subplot(n_rows, 3, i * 3 + 2)
        plt.title("True next day fire")
        plt.imshow(label[i, :, :, 0], cmap=CMAP, norm=NORM)
        plt.axis('off')
        plt.subplot(n_rows, 3, i * 3 + 3)
        plt.title("Predicted next day fire")
        plt.imshow(prediction[i, :, :])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display inference results
features, labels = next(iter(test_dataset))
show_inference(5, features, labels, lambda x: tf.where(segmentation_model.predict(x) > 0.5, 1, 0)[:, :, :, 0])
