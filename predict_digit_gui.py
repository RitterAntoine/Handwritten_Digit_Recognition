import os
# Suppress unnecessary logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import warnings
import numpy as np
from keras import models
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Constants
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 800
FONT_LARGE = ('Helvetica', 24)
FONT_MEDIUM = ('Helvetica', 16)

class DigitRecognizerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Draw a Digit")
        self.root.geometry("1400x950")  # Set the default window size

        self.model = self.load_model()

        # Styling using ttk themes
        style = ttk.Style()
        style.theme_use("clam")  # Use a more modern theme (you can try others like "default", "vista", etc.)

        # Initialize last coordinates
        self.lastx, self.lasty = None, None
        
        # Set up the main frame using grid layout
        self.main_frame = ttk.Frame(root)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20)

        # Set up the canvas frame with a border for styling
        self.canvas_frame = ttk.Frame(self.main_frame, borderwidth=2, relief='groove')
        self.canvas_frame.grid(row=0, column=0, padx=20, pady=20)

        # Set up the canvas for drawing
        self.canvas = tk.Canvas(self.canvas_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
        self.canvas.pack()

        # Set up the image and drawing context
        self.image = Image.new('L', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Bind the paint function to the canvas
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

        # Set up the prediction frame on the right side
        self.prediction_frame = ttk.Frame(self.main_frame)
        self.prediction_frame.grid(row=0, column=1, sticky='n', padx=20)

        # Set up the label to display the result
        self.result_label = ttk.Label(self.prediction_frame, text='Predicted Digit: None', font=FONT_LARGE)
        self.result_label.grid(row=0, column=0, pady=10)

        # Set up the label to display the probabilities
        self.probabilities_label = ttk.Label(self.prediction_frame, text='Probabilities: None', font=FONT_MEDIUM, wraplength=300, justify='left')
        self.probabilities_label.grid(row=1, column=0, pady=10)

        # Set up the clear button with larger size and padding
        self.clear_button = ttk.Button(self.prediction_frame, text='Clear Canvas', command=self.clear_canvas, width=20)
        self.clear_button.grid(row=2, column=0, pady=20, padx=20, ipady=10)

        # Set up an exit button with larger size and padding
        self.exit_button = ttk.Button(self.prediction_frame, text='Exit', command=self.root.quit, width=20)
        self.exit_button.grid(row=3, column=0, pady=20, padx=20, ipady=10)

    def load_model(self):
        """Loads the trained model from a file."""
        try:
            model = models.load_model('handwritten_digit_model.keras')
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            self.root.quit()

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Converts the drawn image to a 28x28 grayscale image and normalizes it."""
        image = image.resize((28, 28)).convert('L')  # Resize and convert to grayscale
        img_array = np.array(image)  # Convert to numpy array
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
        return img_array

    def predict_digit(self, image: Image.Image) -> np.ndarray:
        """Predicts the digit from the processed image."""
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        return prediction

    def paint(self, event: tk.Event):
        """Draws on the canvas."""
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=20)
        self.draw.ellipse([x1, y1, x2, y2], fill='black', width=20)

        if self.lastx and self.lasty:
            self.canvas.create_line(self.lastx, self.lasty, event.x, event.y, fill='black', width=20)
            self.draw.line([self.lastx, self.lasty, event.x, event.y], fill='black', width=20)

        self.lastx, self.lasty = event.x, event.y

    def reset(self, event: tk.Event):
        """Resets the last coordinates and makes a prediction."""
        self.lastx, self.lasty = None, None
        self.handle_predict()  # Make a prediction when the user finishes drawing

    def clear_canvas(self):
        """Clears the canvas."""
        self.canvas.delete('all')
        self.draw.rectangle((0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), fill='white')
        self.result_label.config(text='Predicted Digit: None')  # Reset the prediction label
        self.probabilities_label.config(text='Probabilities: None')  # Reset the probabilities label

    def handle_predict(self):
        """Handles the prediction and displays the result."""
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
