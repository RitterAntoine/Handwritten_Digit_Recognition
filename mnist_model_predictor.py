# === Code to load the model and use it for predictions ===
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Suppress TensorFlow and CUDA messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Load the trained model
model = models.load_model('handwritten_digit_model.h5')
print("Model loaded successfully.")

# === Function to open file dialog and get folder path ===
def open_folder_dialog():
    """Opens a file dialog to choose a folder."""
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window
    folder_path = filedialog.askdirectory(title="Select a Folder Containing Images")
    return folder_path

# === Load and process a custom image ===
def load_and_process_image(image_path):
    """Loads an external image, converts it to grayscale, resizes it to 28x28, and normalizes it."""
    img = Image.open(image_path).convert('L')  # Open image and convert to grayscale
    img = img.resize((28, 28))  # Resize the image to 28x28 pixels
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = img_array / 255.0  # Normalize the image to [0, 1] range
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model (1, 28, 28, 1)
    return img_array

# Get the folder path from the user
folder_path = open_folder_dialog()

if folder_path:
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print("No image files found in the selected folder.")
    else:
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)

            # Load and preprocess the custom image
            custom_image = load_and_process_image(image_path)

            # Make a prediction on the custom image
            prediction = model.predict(custom_image)
            predicted_digit = np.argmax(prediction)

            # Print the predicted digit
            print(f"Predicted digit for {image_file}: {predicted_digit}")
else:
    print("No folder selected.")