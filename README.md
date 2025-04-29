# Digit Classification with Neural Networks

This project implements a digit classification model using a deep neural network trained on the MNIST dataset. It uses TensorFlow and Keras for model development and training.

## 📊 Dataset

The MNIST dataset is a classic benchmark in machine learning consisting of:
- 60,000 training images
- 10,000 test images
- Each image is a 28x28 grayscale image of a handwritten digit (0–9)

The dataset is directly loaded using `tensorflow.keras.datasets.mnist`, making it easy to use with no external downloads required.

## 🚀 Features

- Loads and pre-processes the MNIST dataset
- Normalizes input pixel values to [0, 1]
- Builds a deep learning model using `tf.keras.Sequential`
- Trains the model with categorical cross-entropy loss
- Evaluates model performance on test data
- Visualizes predictions on sample test images
