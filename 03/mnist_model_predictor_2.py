import os
import warnings
import numpy as np
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
    return prediction

# Function to handle the drawing on the canvas
def paint(event):
    """Draws on the canvas."""
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=20)
    draw.ellipse([x1, y1, x2, y2], fill='black', width=20)
    if 'lastx' in globals() and 'lasty' in globals():
        canvas.create_line(lastx, lasty, event.x, event.y, fill='black', width=20)
        draw.line([lastx, lasty, event.x, event.y], fill='black', width=20)
    globals()['lastx'], globals()['lasty'] = event.x, event.y

def reset(event):
    """Resets the lastx and lasty variables and makes a prediction."""
    globals().pop('lastx', None)
    globals().pop('lasty', None)
    handle_predict()  # Make a prediction when the user finishes drawing

# Function to clear the canvas
def clear_canvas():
    """Clears the canvas."""
    canvas.delete('all')
    draw.rectangle((0, 0, canvas_width, canvas_height), fill='white')
    result_label.config(text='Predicted Digit: None')  # Reset the prediction label
    probabilities_label.config(text='Probabilities: None')  # Reset the probabilities label

# Function to handle the prediction
def handle_predict():
    """Handles the prediction and displays the result."""
    prediction = predict_digit(image)
    predicted_digit = np.argmax(prediction)
    result_label.config(text=f'Predicted Digit: {predicted_digit}')
    
    # Get probabilities and sort them in descending order
    probabilities = prediction[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probabilities = [(i, probabilities[i]) for i in sorted_indices]
    
    # Display probabilities
    probabilities_text = "Probabilities:\n" + "\n".join([f"{digit}: {prob:.4f}" for digit, prob in sorted_probabilities])
    probabilities_label.config(text=probabilities_text)

# Set up the main application window
root = tk.Tk()
root.title("Draw a Digit")

# Set up the main frame
main_frame = tk.Frame(root)
main_frame.pack()

# Set up the canvas for drawing
canvas_width = 800
canvas_height = 800
canvas = tk.Canvas(main_frame, width=canvas_width, height=canvas_height, bg='white')
canvas.pack(side='left')

# Set up the image and drawing context
image = Image.new('L', (canvas_width, canvas_height), 'white')
draw = ImageDraw.Draw(image)

# Bind the paint function to the canvas
canvas.bind('<B1-Motion>', paint)
canvas.bind('<ButtonRelease-1>', reset)

# Set up the frame for predictions
prediction_frame = tk.Frame(main_frame)
prediction_frame.pack(side='right', padx=20)

# Set up the label to display the result
result_label = tk.Label(prediction_frame, text='Predicted Digit: None', font=('Helvetica', 24))
result_label.pack(pady=10)

# Set up the label to display the probabilities
probabilities_label = tk.Label(prediction_frame, text='Probabilities: None', font=('Helvetica', 16))
probabilities_label.pack(pady=10)

# Set up the clear button
clear_button = tk.Button(prediction_frame, text='Clear', command=clear_canvas, font=('Helvetica', 16), width=10, height=2)
clear_button.pack(pady=10)

# Start the Tkinter main loop
root.mainloop()