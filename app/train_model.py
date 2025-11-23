import tensorflow as tf
from tensorflow.keras import layers, models
import os

def train_and_save():
    # 1. Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Normalize and reshape (add color channel)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # 3. Create a simple CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Train
    print("Training in progress...")
    model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

    # 5. Save the model in the src folder
    if not os.path.exists('src'):
        os.makedirs('src')
        
    save_path = os.path.join('src', 'mnist_model.h5')
    model.save(save_path)
    print(f"Model saved successfully at: {save_path}")

if __name__ == "__main__":
    train_and_save()