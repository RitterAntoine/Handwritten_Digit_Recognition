import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import warnings
import numpy as np
from keras import models
from PIL import Image, ImageDraw
import tkinter as tk
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Load the trained model
model = models.load_model('handwritten_digit_model.h5')
print("Model loaded successfully.")

# Constants
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 800
FONT_LARGE = ('Helvetica', 24)
FONT_MEDIUM = ('Helvetica', 16)

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        # Initialize last coordinates
        self.lastx, self.lasty = None, None
        
        # Set up the main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack()
        
        # Set up the canvas for drawing
        self.canvas = tk.Canvas(self.main_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
        self.canvas.pack(side='left')
        
        # Set up the image and drawing context
        self.image = Image.new('L', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind the paint function to the canvas
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        
        # Set up the frame for predictions
        self.prediction_frame = tk.Frame(self.main_frame)
        self.prediction_frame.pack(side='right', padx=20)
        
        # Set up the label to display the result
        self.result_label = tk.Label(self.prediction_frame, text='Predicted Digit: None', font=FONT_LARGE)
        self.result_label.pack(pady=10)
        
        # Set up the label to display the probabilities
        self.probabilities_label = tk.Label(self.prediction_frame, text='Probabilities: None', font=FONT_MEDIUM)
        self.probabilities_label.pack(pady=10)
        
        # Set up the clear button
        self.clear_button = tk.Button(self.prediction_frame, text='Clear', command=self.clear_canvas, font=FONT_MEDIUM, width=10, height=2)
        self.clear_button.pack(pady=10)
    
    def preprocess_image(self, image):
        # Converts the drawn image to a 28x28 grayscale image and normalizes it.
        image = image.resize((28, 28)).convert('L')  # Resize and convert to grayscale
        img_array = np.array(image)  # Convert to numpy array
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
        return img_array
    
    def predict_digit(self, image):
        # Predicts the digit from the processed image.
        processed_image = self.preprocess_image(image)
        prediction = model.predict(processed_image)
        return prediction
    
    def paint(self, event):
        # Draws on the canvas.
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=20)
        self.draw.ellipse([x1, y1, x2, y2], fill='black', width=20)
        if self.lastx and self.lasty:
            self.canvas.create_line(self.lastx, self.lasty, event.x, event.y, fill='black', width=20)
            self.draw.line([self.lastx, self.lasty, event.x, event.y], fill='black', width=20)
        self.lastx, self.lasty = event.x, event.y
    
    def reset(self, event):
        # Resets the lastx and lasty variables and makes a prediction.
        self.lastx, self.lasty = None, None
        self.handle_predict()  # Make a prediction when the user finishes drawing
    
    def clear_canvas(self):
        # Clears the canvas.
        self.canvas.delete('all')
        self.draw.rectangle((0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), fill='white')
        self.result_label.config(text='Predicted Digit: None')  # Reset the prediction label
        self.probabilities_label.config(text='Probabilities: None')  # Reset the probabilities label
    
    def handle_predict(self):
        # Handles the prediction and displays the result.
        prediction = self.predict_digit(self.image)
        predicted_digit = np.argmax(prediction)
        self.result_label.config(text=f'Predicted Digit: {predicted_digit}')
        
        # Get probabilities and sort them in descending order
        probabilities = prediction[0]
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probabilities = [(i, probabilities[i]) for i in sorted_indices]
        
        # Display probabilities
        probabilities_text = "Probabilities:\n" + "\n".join([f"{digit}: {prob:.4f}" for digit, prob in sorted_probabilities])
        self.probabilities_label.config(text=probabilities_text)

# Set up the main application window
root = tk.Tk()
app = DigitRecognizerApp(root)

# Start the Tkinter main loop
root.mainloop()