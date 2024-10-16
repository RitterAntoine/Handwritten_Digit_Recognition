# Handwritten Digit Recognition with CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) model to recognize handwritten digits using the MNIST dataset. It features both the training of a CNN model and an interactive graphical user interface (GUI) for real-time digit recognition.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the GUI](#running-the-gui)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [License](#license)
- [Contributing](#contributing)

## Overview

![Handwritten Digit Recognition](https://github.com/user-attachments/assets/63f3afdd-5909-425f-ab37-5db4df777e09)

This project uses the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 28x28 grayscale images representing handwritten digits (0-9). A Convolutional Neural Network (CNN) is built and trained to classify these digits with high accuracy. Once trained, the model is used in a graphical user interface (GUI), allowing users to draw digits and get real-time predictions.

## Key Features
- **Convolutional Neural Network (CNN)**: Built using Keras and TensorFlow, the model achieves high accuracy on the MNIST dataset.
- **Intuitive GUI**: A clean, user-friendly interface built with Tkinter where users can draw digits for real-time predictions.
- **Real-Time Predictions**: After drawing, the app predicts the digit and displays the associated probabilities for each digit (0-9).
- **Customizable Architecture**: The CNN model and GUI can be easily modified for further experimentation or use with different datasets.

## Installation

Follow these steps to set up the project on your local machine.

1. **Clone the repository**:
    ```bash
    git clone https://github.com/RitterAntoine/Handwritten_Digit_Recognition.git
    cd Handwritten_Digit_Recognition
    ```

2. **Install required dependencies using [UV](https://github.com/expo/uv)** (the fast package manager):
    ```bash
    uv install
    ```

3. **Activate the virtual environment** (if using UV):
    ```bash
    uv shell
    ```

   UV automatically creates and manages a virtual environment for your project.

## Usage

### Training the Model

To train the CNN model on the MNIST dataset, run the following command:
```bash
python generate_model.py
```

This script will:

- Load and preprocess the MNIST dataset.
- Build and train a CNN model for 25 epochs (with a batch size of 64).
- Save the trained model as handwritten_digit_model.h5 in the project directory.

### Running the GUI

Once the model is trained, you can launch the GUI application to start drawing digits and receive predictions. Run the following command:

```bash
python predict_digit_gui.py
```

#### How to Use the GUI:

1. Draw a Digit: Click and drag on the canvas to draw a digit.
2. Get Predictions: After releasing the mouse, the application will display the predicted digit and the probability distribution for each digit.
3. Clear the Canvas: Click the "Clear" button to reset the canvas for a new drawing.
4. Exit: Use the "Exit" button to close the application.

## Dependencies

This project uses the following Python libraries:

- TensorFlow/Keras: For building and training the CNN model.
- Pillow: For image processing in the GUI.
- Tkinter: For the graphical user interface.
- Numpy: For numerical operations.
- UV: For managing the project’s dependencies and virtual environments.

To install dependencies, run:

```bash
uv install
```

Project Structure
```
├── generate_model.py        # Script to train the CNN model on the MNIST dataset
├── predict_digit_gui.py     # Script to run the GUI for drawing digits and predicting results
├── handwritten_digit_model.h5 # Trained model (generated after running generate_model.py)
├── requirements.txt         # List of required dependencies
├── README.md                # Project documentation (this file)
└── assets/                  # Directory containing example images or assets for the README
```