# Import necessary libraries
import os
import warnings
import tensorflow as tf
from keras.datasets import mnist
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # For image processing
import tkinter as tk  # For creating a file dialog
from tkinter import filedialog  # For file dialog functionality

# Suppress TensorFlow and CUDA messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress info and warning messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Suppress Keras warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Initialize variables
num_classes = 10  # Digits from 0 to 9
input_shape = (28, 28, 1)  # Shape of each MNIST image (28x28 grayscale)
epochs = 5  # Number of times to train on the dataset
batch_size = 64  # Number of samples per gradient update

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images by scaling pixel values to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images to have an extra dimension for the CNN (channels = 1 for grayscale)
train_images = train_images.reshape((train_images.shape[0], *input_shape))
test_images = test_images.reshape((test_images.shape[0], *input_shape))

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Output layer with 10 classes for digits 0-9
])

# Compile the model with an optimizer, loss function, and performance metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the training data
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# === Load and process a custom image ===
def load_and_process_image(image_path):
    """Loads an external image, converts it to grayscale, resizes it to 28x28, and normalizes it."""
    img = Image.open(image_path).convert('L')  # Open image and convert to grayscale
    img = img.resize((28, 28))  # Resize the image to 28x28 pixels
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = img_array / 255.0  # Normalize the image to [0, 1] range
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model (1, 28, 28, 1)
    return img_array

# === Function to open file dialog and get image path ===
def open_file_dialog():
    """Opens a file dialog to choose an image."""
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    return file_path

# Get the image path from the user
image_path = open_file_dialog()

if image_path:
    # Load and preprocess the custom image
    custom_image = load_and_process_image(image_path)

    # Display the custom image
    plt.imshow(custom_image.reshape(28, 28), cmap='gray')
    plt.title("Custom Image")
    plt.show()

    # Make a prediction on the custom image
    prediction = model.predict(custom_image)
    predicted_digit = np.argmax(prediction)

    # Print the predicted digit
    print(f"Predicted digit: {predicted_digit}")
else:
    print("No image selected.")
