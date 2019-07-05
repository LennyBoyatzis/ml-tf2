import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist

((train_images, train_labels),
 (test_images, test_labels)) = fashion_mnist.load_data()

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot']

# 1. Preprocess the data

# Scale pixel values between 0 and 1 before
# feeding them into the neural network

train_images = train_images / 255.0
test_images = test_images / 255.0

# 2. Build the network

# Basic building block of network is the layer
# Layers extract representations from data fed to them

# Layer 1: Flatten (transforms 2D array of 28x28 pixels to 1D of 28*28 pixels =
# 784 pixels)
# Unstacks the rows of pixels and lines them up in 1D

# Layer 2: Dense or fully connected layer
# First dense layer is 128 nodes

# Layer 3: Dense or fully connected softmax layer
# Last layer is a 10-node softmax that returns an array of
# 10 probability scores that all sum to 1

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 3. Compile the model
# Loss function: measures how accurate the model is during training
# Optimizer: How model is updates based on data it sees and its loss function
# Metrics: Used to monitor training and testing steps

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\n Test accuracy:', test_acc)
