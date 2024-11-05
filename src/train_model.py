import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
import argparse
import time
import traceback
import tensorflow as tf
from keras.datasets import mnist
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config_loader import load_config
from terminal_display import *

# Load configuration from JSON file
try:
    config = load_config()
    model_config = config['model_parameters']
except Exception as e:
    print_text(f"Error loading configuration: {e}")
    traceback.print_exc()
    raise

# Constants and hyperparameters from config
NUM_CLASSES = model_config['NUM_CLASSES']
INPUT_SHAPE = tuple(model_config['INPUT_SHAPE'])
BATCH_SIZE = model_config['BATCH_SIZE']
FILTERS = model_config['FILTERS']
KERNEL_SIZE = tuple(model_config['KERNEL_SIZE'])
POOL_SIZE = tuple(model_config['POOL_SIZE'])
DENSE_UNITS = model_config['DENSE_UNITS']
MAXIMUM_EPOCHS = model_config['MAXIMUM_EPOCHS']
ROTATION_RANGE = model_config['ROTATION_RANGE']
WIDTH_SHIFT_RANGE = model_config['WIDTH_SHIFT_RANGE']
HEIGHT_SHIFT_RANGE = model_config['HEIGHT_SHIFT_RANGE']
ZOOM_RANGE = model_config['ZOOM_RANGE']

def load_and_preprocess_data():
    """Loads and normalizes the MNIST dataset."""
    try:
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        
        # Normalize images to the range [0, 1]
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        
        # Reshape images to the expected input shape (28x28x1)
        train_images = train_images.reshape((train_images.shape[0], *INPUT_SHAPE))
        test_images = test_images.reshape((test_images.shape[0], *INPUT_SHAPE))
        
        return (train_images, train_labels)
    except Exception as e:
        print_text(f"Error in load_and_preprocess_data: {e}")
        traceback.print_exc()
        raise

def get_data_augmentation():
    """Creates an ImageDataGenerator for data augmentation."""
    try:
        return ImageDataGenerator(
            rotation_range=ROTATION_RANGE,
            width_shift_range=WIDTH_SHIFT_RANGE,
            height_shift_range=HEIGHT_SHIFT_RANGE,
            zoom_range=ZOOM_RANGE,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            validation_split=0.2
        )
    except Exception as e:
        print_text(f"Error in get_data_augmentation: {e}")
        traceback.print_exc()
        raise

def build_model(input_shape, num_classes):
    """Builds and compiles a CNN model for image classification."""
    try:
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(FILTERS[0], KERNEL_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(POOL_SIZE),
            layers.Dropout(0.3),
            layers.Conv2D(FILTERS[1], KERNEL_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(POOL_SIZE),
            layers.Dropout(0.3),
            layers.Conv2D(FILTERS[2], KERNEL_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(DENSE_UNITS, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print_text(f"Error in build_model: {e}")
        traceback.print_exc()
        raise

def train_and_save_model(model, train_images, train_labels, epochs, batch_size, save_path='../assets/models/handwritten_digit_model.keras'):
    """Trains the model with data augmentation and saves the trained model to disk."""
    try:
        if epochs is None:
            print_text("Auto training started, it will stop when the model is detected as good...")
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
            callbacks = [early_stopping, reduce_lr]
            epochs = MAXIMUM_EPOCHS
        else:
            print_text(f"Starting training for {epochs} epochs...")
            callbacks = []

        data_generator = get_data_augmentation()

        # Train-validation split with data augmentation
        train_data_gen = data_generator.flow(train_images, train_labels, batch_size=batch_size, subset='training')
        validation_data_gen = data_generator.flow(train_images, train_labels, batch_size=batch_size, subset='validation')

        start_time = time.time()
        model.fit(
            train_data_gen,
            epochs=epochs,
            validation_data=validation_data_gen,
            callbacks=callbacks
        )
        end_time = time.time()
        total_time = end_time - start_time
    except Exception as e:
        print_text(f"Error when training the model: {e}")
        traceback.print_exc()
        raise

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print_text(f"Best model found and saved successfully! Total training time: {total_time:.2f} seconds.")
    except Exception as e:
        print_text(f"Error when saving the model: {e}")
        traceback.print_exc()
        raise

def main():
    try:
        parser = argparse.ArgumentParser(description='Train a CNN model on the MNIST dataset.')
        parser.add_argument('--epochs', '-ep', type=int, help='Number of epochs for training')
        args = parser.parse_args()

        print_title("Handwritten Digit Recognition Model Training")

        (train_images, train_labels) = load_and_preprocess_data()

        model = build_model(INPUT_SHAPE, NUM_CLASSES)

        train_and_save_model(model, train_images, train_labels, args.epochs, BATCH_SIZE)
    except Exception as e:
        print_text(f"Error in main: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
