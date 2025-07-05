# Handwritten Digit Recognition with CNN

A Convolutional Neural Network implementation for recognizing handwritten digits (0-9) using TensorFlow/Keras and the MNIST dataset.

## Project Overview

- **Accuracy**: 98.8% on test data
- **Model Size**: ~37K parameters
- **Training Time**: ~4 minutes (10 epochs)
- **Dataset**: MNIST (70,000 handwritten digit images)

## Complete Pipeline Flow

```
                                                                                          ┌─────────────────┐
                                                                                          │   MNIST Dataset │
                                                                                          │  70,000 images  │
                                                                                          └────────┬────────┘
                                                                                                   │
                                                                                                   ▼
                                                                                          ┌─────────────────┐     ┌─────────────────┐
                                                                                          │ Training Data   │     │   Test Data     │
                                                                                          │ 60,000 images   │     │ 10,000 images   │
                                                                                          └────────┬────────┘     └─────────────────┘
                                                                                                   │
                                                                                                   ▼
                                                                                          ┌─────────────────────────────┐
                                                                                          │     Train-Val Split         │
                                                                                          │  ┌──────────┬─────────────┐ │
                                                                                          │  │ Training │ Validation  │ │
                                                                                          │  │  42,000  │   18,000    │ │
                                                                                          │  └──────────┴─────────────┘ │
                                                                                          └──────────────┬──────────────┘
                                                                                                         │
                                                                                                         ▼
                                                                                          ┌─────────────────────────────┐
                                                                                          │    Data Preprocessing       │
                                                                                          │  • Normalization (÷255)     │
                                                                                          │  • Reshape (28,28,1)        │
                                                                                          └──────────────┬──────────────┘
                                                                                                         │
                                                                                                         ▼
                                                                                          ┌─────────────────────────────┐
                                                                                          │      Model Building         │
                                                                                          │  ┌─────────────────────┐    │
                                                                                          │  │ Input (28×28×1)     │    │
                                                                                          │  ├─────────────────────┤    │
                                                                                          │  │ Conv2D (8 filters)  │    │
                                                                                          │  ├─────────────────────┤    │
                                                                                          │  │ MaxPooling2D (2×2)  │    │
                                                                                          │  ├─────────────────────┤    │
                                                                                          │  │ Conv2D (16 filters) │    │
                                                                                          │  ├─────────────────────┤    │
                                                                                          │  │ MaxPooling2D (2×2)  │    │
                                                                                          │  ├─────────────────────┤    │
                                                                                          │  │ Flatten             │    │
                                                                                          │  ├─────────────────────┤    │
                                                                                          │  │ Dense (128)         │    │
                                                                                          │  ├─────────────────────┤    │
                                                                                          │  │ Dropout (0.2)       │    │
                                                                                          │  ├─────────────────────┤    │
                                                                                          │  │ Dense (10)          │    │
                                                                                          │  └─────────────────────┘    │
                                                                                          └──────────────┬──────────────┘
                                                                                                         │
                                                                                                         ▼
                                                                                          ┌─────────────────────────────┐
                                                                                          │      Model Training         │
                                                                                          │  • Optimizer: Adam          │
                                                                                          │  • Loss: Sparse CrossEntropy│
                                                                                          │  • Epochs: 10               │
                                                                                          │  • Batch Size: 32           │
                                                                                          └──────────────┬──────────────┘
                                                                                                         │
                                                                                                         ▼
                                                                                          ┌─────────────────────────────┐
                                                                                          │      Model Evaluation       │
                                                                                          │  ┌───────────────────────┐  │
                                                                                          │  │ • Training Curves     │  │
                                                                                          │  │ • Confusion Matrix    │  │
                                                                                          │  │ • Accuracy Metrics    │  │
                                                                                          │  │ • Sample Predictions  │  │
                                                                                          │  └───────────────────────┘  │
                                                                                          └──────────────┬──────────────┘
                                                                                                         │
                                                                                                         ▼
                                                                                          ┌─────────────────────────────┐
                                                                                          │        Final Model          │
                                                                                          │    Test Accuracy: 98.8%     │
                                                                                          └─────────────────────────────┘
```

## Detailed Process Flow

### 1. **Data Loading and Exploration**
- Load 70,000 grayscale images of handwritten digits
- Each image is 28×28 pixels
- Labels range from 0 to 9

### 2. **Data Splitting**
- Original training set (60,000) split into:
  - Training: 42,000 images (70%)
  - Validation: 18,000 images (30%)
- Test set: 10,000 images (kept separate)

### 3. **Data Preprocessing**
- **Normalization**: Scale pixel values from [0,255] to [0,1] range
- **Reshaping**: Add channel dimension for CNN compatibility (28,28) → (28,28,1)

### 4. **Model Architecture**
The CNN architecture consists of:
- **Feature Extraction Layers**:
  - First Conv2D layer with 8 filters detects basic features like edges
  - MaxPooling reduces spatial dimensions by half
  - Second Conv2D layer with 16 filters detects complex patterns
  - Another MaxPooling for further dimension reduction
- **Classification Layers**:
  - Flatten converts 2D features to 1D vector
  - Dense layer with 128 neurons learns feature combinations
  - Dropout prevents overfitting by randomly dropping connections
  - Final Dense layer outputs probabilities for 10 digits

### 5. **Model Training**
- Uses Adam optimizer for adaptive learning rates
- Sparse categorical crossentropy loss for multi-class classification
- Trains for 10 epochs with batch size of 32
- Monitors validation performance to prevent overfitting

### 6. **Model Evaluation**
- **Training Accuracy**: 99.6%
- **Validation Accuracy**: 98.7%
- **Test Accuracy**: 98.8%
- Confusion matrix reveals common misclassifications (4↔9, 3↔5)
- Visualizations show model performance and prediction samples
  ![image](https://github.com/user-attachments/assets/aa456e45-b009-4c22-9aaf-75662aaadb0c)


## Key Results

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| Accuracy | 99.6% | 98.7% | 98.8% |
| Loss | 0.012 | 0.045 | 0.039 |

- Small gap between training and test accuracy indicates good generalization
- Model converges smoothly without overfitting

