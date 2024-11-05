import os
import warnings
import numpy as np
from keras import models
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGraphicsScene, QGraphicsView, QMessageBox
from PyQt5.QtGui import QBrush, QPen, QImage, QPainter, QPixmap
from PyQt5.QtCore import Qt, QPoint
import absl.logging
from config_loader import load_config

# Suppress unnecessary logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Load configuration from JSON file
config = load_config()
ui_config = config['ui_parameters']

# Constants from config
CANVAS_WIDTH = ui_config['CANVAS_WIDTH']
CANVAS_HEIGHT = ui_config['CANVAS_HEIGHT']
FONT_LARGE = ui_config['FONT_LARGE']
FONT_MEDIUM = ui_config['FONT_MEDIUM']

class DigitRecognizerApp(QWidget):
    def __init__(self):
        """Initialize the DigitRecognizerApp with the main window."""
        super().__init__()
        self.setWindowTitle("Draw a Digit")
        self.setGeometry(100, 100, 1400, 950)
        self.model = self.load_model()
        self.last_point = None
        self.image = Image.new('L', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
        self.draw = ImageDraw.Draw(self.image)

        self.init_ui()

    def init_ui(self):
        """Set up the UI components."""
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Scene for drawing
        self.canvas_scene = QGraphicsScene(self)
        self.canvas_view = QGraphicsView(self.canvas_scene)
        self.canvas_view.setRenderHint(QPainter.Antialiasing)
        self.canvas_view.setFixedSize(CANVAS_WIDTH, CANVAS_HEIGHT)
        layout.addWidget(self.canvas_view)

        # Setup prediction display
        self.prediction_layout = QVBoxLayout()
        layout.addLayout(self.prediction_layout)

        self.result_label = QLabel('Predicted Digit: None')
        self.prediction_layout.addWidget(self.result_label)

        self.probabilities_label = QLabel('Probabilities: None')
        self.prediction_layout.addWidget(self.probabilities_label)

        self.clear_button = QPushButton('Clear Canvas')
        self.clear_button.clicked.connect(self.clear_canvas)
        self.prediction_layout.addWidget(self.clear_button)

        self.exit_button = QPushButton('Exit')
        self.exit_button.clicked.connect(self.close)
        self.prediction_layout.addWidget(self.exit_button)

        # Set up mouse events
        self.canvas_view.setMouseTracking(True)
        self.canvas_view.mousePressEvent = self.start_drawing
        self.canvas_view.mouseMoveEvent = self.draw_on_canvas
        self.canvas_view.mouseReleaseEvent = self.stop_drawing

        # Create a QPixmap to draw on
        self.pixmap = QPixmap(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.pixmap.fill(Qt.white)  # Fill with white to start
        self.canvas_scene.addPixmap(self.pixmap)

    def load_model(self):
        """Loads the trained model from a file."""
        try:
            model = models.load_model('../assets/models/handwritten_digit_model.keras')
            print("Model loaded successfully.")
            return model
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading model: {e}")
            self.close()

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Converts the drawn image to a 28x28 grayscale image and normalizes it."""
        image = image.resize((28, 28)).convert('L')  # Resize and convert to grayscale
        img_array = np.array(image) / 255.0  # Normalize to [0, 1]
        return img_array.reshape(1, 28, 28, 1)  # Reshape for the model

    def predict_digit(self, image: Image.Image) -> np.ndarray:
        """Predicts the digit from the processed image."""
        processed_image = self.preprocess_image(image)
        return self.model.predict(processed_image)

    def start_drawing(self, event):
        """Starts the drawing process on mouse press."""
        self.last_point = QPoint(event.pos())
        self.draw_on_canvas(event)

    def draw_on_canvas(self, event):
        """Draws on the canvas while the mouse is moved."""
        if self.last_point is not None:
            painter = QPainter(self.pixmap)  # Draw on the pixmap
            brush = QBrush(Qt.black)
            painter.setBrush(brush)
            painter.setPen(QPen(Qt.black, 20, Qt.SolidLine))
            current_point = QPoint(event.pos())
            painter.drawLine(self.last_point, current_point)
            self.draw.line((self.last_point.x(), self.last_point.y(), current_point.x(), current_point.y()), fill='black', width=20)
            self.last_point = current_point
            self.canvas_scene.clear()  # Clear the scene before re-adding the pixmap
            self.canvas_scene.addPixmap(self.pixmap)  # Add the updated pixmap to the scene

    def stop_drawing(self, event):
        """Stops the drawing process on mouse release and makes a prediction."""
        self.last_point = None
        self.handle_predict()

    def clear_canvas(self):
        """Clears the canvas and resets predictions."""
        self.pixmap.fill(Qt.white)  # Fill the pixmap with white
        self.canvas_scene.clear()
        self.canvas_scene.addPixmap(self.pixmap)  # Add the clear pixmap back to the scene
        self.result_label.setText('Predicted Digit: None')
        self.probabilities_label.setText('Probabilities: None')

    def handle_predict(self):
        """Handles the prediction and displays the result."""
        prediction = self.predict_digit(self.image)
        predicted_digit = np.argmax(prediction)
        self.result_label.setText(f'Predicted Digit: {predicted_digit}')

        probabilities = prediction[0]
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probabilities = [(i, probabilities[i]) for i in sorted_indices]

        probabilities_text = "Probabilities:\n" + "\n".join([f"{digit}: {prob:.4f}" for digit, prob in sorted_probabilities])
        self.probabilities_label.setText(probabilities_text)

if __name__ == "__main__":
    app = QApplication([])
    digit_recognizer = DigitRecognizerApp()
    digit_recognizer.show()
    app.exec_()
