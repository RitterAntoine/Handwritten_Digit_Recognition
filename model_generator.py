import os
import warnings
from keras.datasets import mnist # type: ignore
from keras import layers, models

# TensorFlow and CUDA configuration
def configure_environment():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA for CPU-only execution
    warnings.filterwarnings("ignore", category=UserWarning, module='keras')  # Suppress Keras warnings

# Constants and hyperparameters
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
EPOCHS = 5
BATCH_SIZE = 64
FILTERS = [32, 64, 64]  # Number of filters in Conv2D layers
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DENSE_UNITS = 64

def load_and_preprocess_data():
    """Loads and normalizes the MNIST dataset."""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Normalize images to the range [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Reshape images to the expected input shape (28x28x1)
    train_images = train_images.reshape((train_images.shape[0], *INPUT_SHAPE))
    test_images = test_images.reshape((test_images.shape[0], *INPUT_SHAPE))
    
    return (train_images, train_labels), (test_images, test_labels)

def build_model(input_shape, num_classes):
    """Builds and compiles a CNN model for image classification."""
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

def train_and_save_model(model, train_images, train_labels, epochs, batch_size, save_path='handwritten_digit_model.h5'):
    """Trains the model and saves the trained model to disk."""
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    
    # Save the trained model
    model.save(save_path)

def main():
    # Configure the environment (suppress TensorFlow/CUDA logs)
    configure_environment()

    # Load and preprocess the data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

    # Build the CNN model
    model = build_model(INPUT_SHAPE, NUM_CLASSES)

    # Train and save the model
    train_and_save_model(model, train_images, train_labels, EPOCHS, BATCH_SIZE)

if __name__ == "__main__":
    main()
