# Trains the digit classifier and exports it for the inference service.
#
# The web application never imports TensorFlow: it only reads the ONNX file
# produced here with onnxruntime. That keeps the served image small, and this
# script is the only place where the training stack is needed.
import os
import subprocess
import tempfile

import tensorflow as tf
from tensorflow.keras import layers, models

TRAINING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_PATH = os.path.join(TRAINING_DIRECTORY, 'model', 'mnist_model.h5')
ONNX_MODEL_PATH = os.path.join(TRAINING_DIRECTORY, '..', 'app', 'model', 'mnist_model.onnx')

NUMBER_OF_EPOCHS = 20
BATCH_SIZE = 64
ONNX_OPSET_VERSION = 17


def build_convolutional_model():
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])


def export_to_onnx(trained_model, onnx_output_path):
    # tf2onnx converts a SavedModel directory, not a Keras file, so the model
    # goes through a temporary export first.
    with tempfile.TemporaryDirectory() as saved_model_directory:
        trained_model.export(saved_model_directory)
        subprocess.run(
            [
                'python', '-m', 'tf2onnx.convert',
                '--saved-model', saved_model_directory,
                '--output', onnx_output_path,
                '--opset', str(ONNX_OPSET_VERSION),
            ],
            check=True,
        )


def train_and_save():
    print("Loading MNIST dataset...")
    (training_images, training_labels), _ = tf.keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and add the single color channel the model expects.
    training_images = training_images / 255.0
    training_images = training_images.reshape(-1, 28, 28, 1)

    model = build_convolutional_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    print("Training in progress...")
    model.fit(
        training_images,
        training_labels,
        epochs=NUMBER_OF_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
    )

    os.makedirs(os.path.dirname(KERAS_MODEL_PATH), exist_ok=True)
    model.save(KERAS_MODEL_PATH)
    print(f"Keras model saved at: {KERAS_MODEL_PATH}")

    os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)
    export_to_onnx(model, ONNX_MODEL_PATH)
    print(f"ONNX model saved at: {ONNX_MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
