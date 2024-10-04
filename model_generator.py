# === Code for training and saving the model ===
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import warnings
from keras.datasets import mnist
from keras import layers, models
import tensorflow as tf

# Suppress TensorFlow and CUDA messages
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

# Initialize variables
num_classes = 10
input_shape = (28, 28, 1)
epochs = 20
batch_size = 64

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images to (28, 28, 1)
train_images = train_images.reshape((train_images.shape[0], *input_shape))
test_images = test_images.reshape((test_images.shape[0], *input_shape))

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)

# Save the trained model
model.save('handwritten_digit_model.h5')
print("Model saved as 'handwritten_digit_model.h5'")
