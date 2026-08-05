# Handwritten Digit Recognition (MNIST)

A lightweight, web-based Deep Learning application that recognizes handwritten digits (0-9) in real-time.

**Live Demo**: https://mnist.antoineritter.fr

Designed to be minimalist, responsive, and easy to deploy on ARM architectures (Raspberry Pi) using Docker.

## Features

- **Real-time Prediction**: Draw on the canvas and get instant feedback.
- **Smart Debouncing**: Predictions trigger automatically 1s after you stop drawing.
- **Mobile Friendly**: Fully responsive canvas supporting touch events.
- **Efficient Architecture**:
    - **Backend**: Flask API running the CNN with ONNX Runtime.
    - **Frontend**: Lightweight HTML5 Canvas with dark mode UI.

## Training and inference are separated

The model is trained with TensorFlow, but the web application never imports it.
Training exports the network to ONNX, and the server runs that file with ONNX
Runtime. The deployed image therefore carries no training stack: it went from
2.4 GB to 483 MB, and the two sides can be updated independently.

- `app/` holds everything the server needs, including `app/model/mnist_model.onnx`.
- `training/` holds the training script, its own requirements and the Keras model it produced.

Both models are committed, so the application runs straight after cloning and
the network can still be retrained or fine-tuned from the Keras file.

## Tech Stack

- Language: Python 3.10
- Framework: Flask
- Inference: ONNX Runtime
- Training: TensorFlow (Keras), exported with tf2onnx
- Image processing: Pillow
- Containerization: Docker & Docker Compose
- Server: Gunicorn

## Project Structure

```
.
├── Dockerfile                  # Production image, inference only
├── requirements.txt            # Inference dependencies
├── app/
│   ├── main.py                 # Flask application entry point
│   ├── templates/
│   │   └── index.html          # Frontend UI
│   └── model/
│       └── mnist_model.onnx    # Model served in production
└── training/
    ├── requirements.txt        # Training dependencies (TensorFlow, tf2onnx)
    ├── train_model.py          # Trains the CNN and exports it to ONNX
    └── model/
        └── mnist_model.h5      # Keras model produced by the training script
```

## Getting Started

1. Clone the repository
```
git clone https://github.com/RitterAntoine/Handwritten_Digit_Recognition.git
cd Handwritten_Digit_Recognition
```

2. Create a virtual environment
```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the inference dependencies
```
pip install -r requirements.txt
```

4. Run the application
```
python app/main.py
```
Open your browser at http://localhost:5000

## Retraining the model

Only needed to change the network itself. The script trains on MNIST, writes
the Keras model in `training/model/` and exports the ONNX file the server
reads.

```
pip install -r training/requirements.txt
python training/train_model.py
```
