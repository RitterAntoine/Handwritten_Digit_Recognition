import os
import warnings
import numpy as np
import tensorflow as tf
from keras import models
from PIL import Image, ImageDraw
import tkinter as tk

# Suppress TensorFlow and CUDA messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Load the trained model
model = models.load_model('handwritten_digit_model.h5')
print("Model loaded successfully.")

# Function to preprocess the drawn image
def preprocess_image(image):
    """Converts the drawn image to a 28x28 grayscale image and normalizes it."""
    image = image.resize((28, 28)).convert('L')  # Resize and convert to grayscale
    img_array = np.array(image)  # Convert to numpy array
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
    return img_array

# Function to predict the digit from the drawn image
def predict_digit(image):
    """Predicts the digit from the processed image."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)

# Function to handle the drawing on the canvas
def paint(event):
    """Draws on the canvas."""
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
    draw.line([x1, y1, x2, y2], fill='black', width=10)

# Function to clear the canvas
def clear_canvas():
    """Clears the canvas."""
    canvas.delete('all')
    draw.rectangle((0, 0, 200, 200), fill='white')

# Function to handle the prediction
def handle_predict():
    """Handles the prediction and displays the result."""
    digit = predict_digit(image)
    result_label.config(text=f'Predicted Digit: {digit}')

# Set up the main application window
root = tk.Tk()
root.title("Draw a Digit")

# Set up the canvas for drawing
canvas = tk.Canvas(root, width=200, height=200, bg='white')
canvas.pack()

# Set up the image and drawing context
image = Image.new('L', (200, 200), 'white')
draw = ImageDraw.Draw(image)

# Bind the paint function to the canvas
canvas.bind('<B1-Motion>', paint)

# Set up the buttons
clear_button = tk.Button(root, text='Clear', command=clear_canvas)
clear_button.pack(side='left')

predict_button = tk.Button(root, text='Predict', command=handle_predict)
predict_button.pack(side='right')

# Set up the label to display the result
result_label = tk.Label(root, text='Predicted Digit: None')
result_label.pack()

# Start the Tkinter main loop
root.mainloop()