import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt

# Define important variables
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 28, 28, 1  # Image dimensions and channels
NUM_CLASSES = 10  # Number of classes in the dataset
EPOCHS = 10  # Number of epochs for training

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape data to include a single color channel
train_images = train_images.reshape((train_images.shape[0], IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
test_images = test_images.reshape((test_images.shape[0], IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

# Create the sequential model
model = models.Sequential()

# Add a Convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
model.add(layers.MaxPooling2D((2, 2)))  # Add a Max Pooling layer

# Add another convolutional layer and pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the results and add a fully connected layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Output layer with 10 units for the 10 digit classes, with softmax activation
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

# Compile the model with the Adam optimizer and sparse categorical crossentropy loss
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=EPOCHS, 
                    validation_data=(test_images, test_labels))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Plot training & validation accuracy values and save to a file
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy_plot.png')  # Save the plot as an image file

# Clear the current figure before creating the next plot
plt.clf()

# Plot training & validation loss values and save to a file
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss_plot.png')  # Save the plot as an image file