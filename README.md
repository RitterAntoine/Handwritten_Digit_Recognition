# Handwritten Digit Recognition (MNIST)

A lightweight, web-based Deep Learning application that recognizes handwritten digits (0-9) in real-time.

**Live Demo**: https://mnist.antoineritter.fr

Designed to be minimalist, responsive, and easy to deploy on ARM architectures (Raspberry Pi) using Docker.

## Features

- **Real-time Prediction**: Draw on the canvas and get instant feedback.
- **Smart Debouncing**: Predictions trigger automatically 1s after you stop drawing.
- **Mobile Friendly**: Fully responsive canvas supporting touch events.
- **Efficient Architecture**:
    - **Backend**: Flask API serving a pre-trained CNN model.
    - **Frontend**: Lightweight HTML5 Canvas with dark mode UI.

## Tech Stack

- Language: Python 3.10
- Framework: Flask
- ML Library: TensorFlow (Keras)
- Computer Vision: OpenCV
- Containerization: Docker & Docker Compose
- Server: Gunicorn

## Project Structure

```
.
├── Dockerfile                  # Production image configuration
├── docker-compose.yml          # Container orchestration
└── Handwritten_Digit_Recognition/
    ├── requirements.txt        # Python dependencies
    └── app/
        ├── main.py             # Flask application entry point
        ├── templates/
        │   └── index.html      # Frontend UI
        └── model/
            └── mnist_model.h5  # Pre-trained CNN Model
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

3. Install dependencies
```
pip install -r requirements.txt
```

4. Train the model (if missing)
```
python train_model.py
```

5. Run the application
```
python app/main.py
```
Open your browser at http://localhost:5000
