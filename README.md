# Handwritten Digit Recognition with CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) model to recognize handwritten digits using the MNIST dataset. The project is divided into two parts: the training of the CNN model and a GUI application that allows users to draw digits and get real-time predictions from the trained model.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Digit Recognition GUI](#running-the-digit-recognition-gui)
- [Dependencies](#dependencies)

## Overview

The project uses the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). A CNN model is built and trained on this dataset to classify digits. Once the model is trained, it is saved and used in a graphical user interface (GUI) application where users can draw digits on a canvas, and the application will predict the digit along with the associated probabilities.

### Key Features:

- **CNN Model**: Built using Keras and TensorFlow, the model is trained on the MNIST dataset and achieves high accuracy.
- **Tkinter GUI**: A simple drawing interface allows users to draw digits and get real-time predictions.
- **Real-time Predictions**: After drawing, the model predicts the digit and provides the probabilities for all possible digits (0-9).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/RitterAntoine/Handwritten_Digit_Recognition.git
    cd Handwritten_Digit_Recognition
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the CNN model on the MNIST dataset, run the following script:
```bash
python generate_model.py
```

This will:
- Load the MNIST dataset.
- Normalize the images.
- Build a CNN model using Keras.
- Train the model for 20 epochs with a batch size of 64.
- Save the trained model as `handwritten_digit_model.h5`.

### Running the Digit Recognition GUI

Once the model is trained, you can launch the digit recognition GUI application. This allows you to draw digits and receive predictions.

To start the application, run:
```bash
python predict_digit_gui.py
```

In the application:
- Draw a digit on the canvas.
- The app will predict the digit once you finish drawing.
- The probabilities for each digit (0-9) will be displayed.

You can also clear the canvas and draw a new digit by clicking the "Clear" button.

## Dependencies

The project requires the following Python libraries:

- `tensorflow` (or `keras`) for building and training the CNN model.
- `Pillow` for image manipulation in the GUI.
- `Tkinter` for the graphical user interface.
- `numpy` for numerical operations.

You can install all dependencies by running:
```bash
poetry install
```
