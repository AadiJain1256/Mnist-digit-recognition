# MNIST Handwritten Digit Recognition using PyTorch

A Convolutional Neural Network (CNN) built with PyTorch to recognize handwritten digits from the MNIST dataset.

The project includes model training, evaluation, model saving/loading, confusion matrix analysis, and prediction on external handwritten digit images.

## Features

- MNIST handwritten digit classification
- CNN implemented using PyTorch
- GPU/CUDA support
- Model training and evaluation
- Confusion matrix visualization
- Saved model weights
- Independent inference using `predict.py`
- External handwritten digit prediction
- Image preprocessing pipeline for real-world handwritten images

## Dataset

The project uses the MNIST dataset:

- Training images: 60,000
- Test images: 10,000
- Image size: 28 × 28 pixels
- Image type: Grayscale
- Number of classes: 10 (digits 0–9)

## CNN Architecture

```text
Input: 1 × 28 × 28

Conv2d(1, 32, 3)
      ↓
ReLU
      ↓
MaxPool2d(2)
      ↓
Conv2d(32, 64, 3)
      ↓
ReLU
      ↓
MaxPool2d(2)
      ↓
Flatten
      ↓
Linear(1600, 10)
      ↓
Output: 10 classes