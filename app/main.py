import os
import base64
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'mnist_model.h5')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ CRITICAL: Unable to load model at path {MODEL_PATH}")
    print(f"Error: {e}")
    model = None

def preprocess_image(base64_string):
    """Image processing pipeline: Base64 -> Numpy -> Grayscale -> 28x28 -> Normalize"""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    img_bytes = base64.b64decode(base64_string)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    
    img = cv2.imdecode(img_arr, 1) 
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    processed = resized.astype('float32') / 255.0
    processed = np.expand_dims(processed, axis=0)
    processed = np.expand_dims(processed, axis=-1)
    
    return processed


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image received'}), 400

        input_tensor = preprocess_image(data['image'])
  
        prediction = model.predict(input_tensor, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            'digit': predicted_class,
            'confidence': f"{confidence * 100:.1f}%"
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)