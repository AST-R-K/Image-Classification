# 🧠 Image Classification Using CNN on CIFAR-10

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📘 Overview

This project implements an **image classification model** using a **Convolutional Neural Network (CNN)** on the **CIFAR-10 dataset**.  
The goal is to train a deep learning model that can classify images into **10 categories**, such as airplane, automobile, bird, cat, and others.  

The model is built and trained using **TensorFlow/Keras**, with techniques like **data augmentation**, **batch normalization**, and **dropout** to achieve high accuracy and generalization.

---

## 🎯 Problem Statement

The challenge is to classify small RGB images (32×32 pixels) into one of ten object categories.  
Due to visual similarities between certain classes (e.g., cat vs. dog, truck vs. automobile), the task requires the model to effectively learn spatial and hierarchical features directly from the image data.

---

## 📦 Dataset Used

**Dataset:** CIFAR-10  
**Source:** `tensorflow.keras.datasets.cifar10`  
**Classes:** 10 categories  
**Training Samples:** 50,000  
**Testing Samples:** 10,000  
**Image Size:** 32×32×3 (RGB)

| Label | Class Name |
|--------|-------------|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

**Preprocessing Steps**
- Pixel normalization: values scaled to `[0, 1]`
- One-hot encoding of labels
- Data augmentation (rotation, flips, shifts, zooms)

---

## 🧱 Model Architecture

The CNN model was built using **Keras Sequential API** with multiple convolutional and pooling layers, followed by fully connected layers.

| Layer | Type | Description |
|--------|------|-------------|
| 1 | Conv2D | 32 filters, 3×3 kernel, ReLU activation |
| 2 | BatchNormalization | Normalizes feature maps |
| 3 | Conv2D | 32 filters, 3×3 kernel, ReLU activation |
| 4 | MaxPooling2D | 2×2 pooling |
| 5 | Dropout | 25% dropout |
| 6 | Conv2D | 64 filters, 3×3 kernel, ReLU activation |
| 7 | BatchNormalization | Normalizes layer output |
| 8 | Conv2D | 64 filters, 3×3 kernel, ReLU activation |
| 9 | MaxPooling2D | 2×2 pooling |
| 10 | Dropout | 25% dropout |
| 11 | Conv2D | 128 filters, 3×3 kernel, ReLU activation |
| 12 | BatchNormalization | Normalizes layer output |
| 13 | MaxPooling2D | 2×2 pooling |
| 14 | Dropout | 25% dropout |
| 15 | Flatten | Converts 2D feature maps to 1D vector |
| 16 | Dense | 512 neurons, ReLU activation |
| 17 | Dropout | 50% dropout |
| 18 | Dense | 10 neurons, Softmax activation (output layer) |

**Training Configuration**
- **Optimizer:** Adam (lr = 0.001)  
- **Loss:** Categorical Crossentropy  
- **Metric:** Accuracy  
- **Batch Size:** 64  
- **Epochs:** 30  

---

## ⚙️ How to Run

### 1️⃣ Prerequisites
Make sure you have Python and the required libraries installed:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn

