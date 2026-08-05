import base64
import io
import os

import numpy as np
import onnxruntime
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIRECTORY, 'model', 'mnist_model.onnx')
TEMPLATE_DIRECTORY = os.path.join(BASE_DIRECTORY, 'templates')

MODEL_INPUT_WIDTH = 28
MODEL_INPUT_HEIGHT = 28

app = Flask(__name__, template_folder=TEMPLATE_DIRECTORY)
CORS(app)

print("Loading model...")
try:
    inference_session = onnxruntime.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    model_input_name = inference_session.get_inputs()[0].name
    print(f"Model loaded: {MODEL_PATH}")
except Exception as error:
    print(f"CRITICAL: Unable to load model at path {MODEL_PATH}")
    print(f"Error: {error}")
    inference_session = None
    model_input_name = None


def preprocess_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    image_bytes = base64.b64decode(base64_string)
    # The canvas sends an opaque black background with white strokes, so the
    # alpha channel carries no information and converting to "L" is enough.
    grayscale_image = Image.open(io.BytesIO(image_bytes)).convert("L")
    # BOX resampling averages the source pixels, which is what the previous
    # OpenCV call did with INTER_AREA.
    resized_image = grayscale_image.resize(
        (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), Image.Resampling.BOX,
    )

    normalized_pixels = np.asarray(resized_image, dtype=np.float32) / 255.0
    return normalized_pixels.reshape(1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 1)


@app.route('/')
def index():
    return send_file(os.path.join(TEMPLATE_DIRECTORY, 'index.html'))


@app.route('/predict', methods=['POST'])
def predict():
    if inference_session is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    try:
        request_data = request.get_json()
        if not request_data or 'image' not in request_data:
            return jsonify({'error': 'No image received'}), 400

        input_tensor = preprocess_image(request_data['image'])

        class_probabilities = inference_session.run(None, {model_input_name: input_tensor})[0][0]
        predicted_digit = int(np.argmax(class_probabilities))

        return jsonify({
            'digit': predicted_digit,
            'confidences': [float(probability) for probability in class_probabilities],
        })

    except Exception as error:
        print(f"Error during prediction: {error}")
        return jsonify({'error': str(error)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
