import os
import warnings
import argparse
from keras.datasets import mnist # type: ignore
from keras import layers, models
import traceback

# Constants and hyperparameters
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
BATCH_SIZE = 64
FILTERS = [32, 64, 64]  # Number of filters in Conv2D layers
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DENSE_UNITS = 64

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
        
        return (train_images, train_labels), (test_images, test_labels)
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {e}")
        traceback.print_exc()
        raise

def build_model(input_shape, num_classes):
    """Builds and compiles a CNN model for image classification."""
    try:
        model = models.Sequential([
            layers.Conv2D(FILTERS[0], KERNEL_SIZE, activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(POOL_SIZE),
            layers.Conv2D(FILTERS[1], KERNEL_SIZE, activation='relu'),
            layers.MaxPooling2D(POOL_SIZE),
            layers.Conv2D(FILTERS[2], KERNEL_SIZE, activation='relu'),
            layers.Flatten(),
            layers.Dense(DENSE_UNITS, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        
        return model
    except Exception as e:
        print(f"Error in build_model: {e}")
        traceback.print_exc()
        raise

def train_and_save_model(model, train_images, train_labels, epochs, batch_size, save_path='handwritten_digit_model.h5'):
    """Trains the model and saves the trained model to disk."""
    try:
        model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    except Exception as e:
        print(f"Error when training the model: {e}")
        traceback.print_exc()
        raise

    try:
        # Save the trained model
        model.save(save_path)
    except Exception as e:
        print(f"Error when saving the model: {e}")
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='Train a CNN model on the MNIST dataset.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs for training')
    args = parser.parse_args()

    try:
        # Load and preprocess the data
        (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

        # Build the CNN model
        model = build_model(INPUT_SHAPE, NUM_CLASSES)

        # Train and save the model
        train_and_save_model(model, train_images, train_labels, args.epochs, BATCH_SIZE)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()