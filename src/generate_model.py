import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import argparse
import time
import traceback
from keras.datasets import mnist # type: ignore
from keras import layers, models
from keras.callbacks import EarlyStopping
from config_loader import load_config

# Load configuration from JSON file
config = load_config()
model_config = config['model_parameters']

# Constants and hyperparameters from config
NUM_CLASSES = model_config['NUM_CLASSES']
INPUT_SHAPE = tuple(model_config['INPUT_SHAPE'])
BATCH_SIZE = model_config['BATCH_SIZE']
FILTERS = model_config['FILTERS']
KERNEL_SIZE = tuple(model_config['KERNEL_SIZE'])
POOL_SIZE = tuple(model_config['POOL_SIZE'])
DENSE_UNITS = model_config['DENSE_UNITS']
MAXIMUM_EPOCHS = model_config['MAXIMUM_EPOCHS']

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
            layers.Input(shape=input_shape),
            layers.Conv2D(FILTERS[0], KERNEL_SIZE, activation='relu'),
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

def train_and_save_model(model, train_images, train_labels, epochs, batch_size, save_path='../assets/models/handwritten_digit_model.keras'):
    """Trains the model and saves the trained model to disk."""
    try:
        if epochs is None:
            print("Auto training started, it will stop when the model is detected as good...")  # Message for auto training
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            callbacks = [early_stopping]
            epochs = MAXIMUM_EPOCHS
        else:
            print(f"Starting training for {epochs} epochs...")  # Message for user-defined epochs
            callbacks = []
        
        start_time = time.time()  # Record start time
        model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks)
        end_time = time.time()  # Record end time
        total_time = end_time - start_time  # Calculate total time
    except Exception as e:
        print(f"Error when training the model: {e}")
        traceback.print_exc()
        raise

    try:
        # Save the trained model in the native Keras format
        model.save(save_path)
        print(f"Best model found and saved successfully! Total training time: {total_time:.2f} seconds.")  # Message after saving the model
    except Exception as e:
        print(f"Error when saving the model: {e}")
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description='Train a CNN model on the MNIST dataset.')
    parser.add_argument('--epochs', '-ep', type=int, help='Number of epochs for training')
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