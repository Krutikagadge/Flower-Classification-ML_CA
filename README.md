# Flower-Classification-ML_CA

# 🧠 CNN vs VGG16 for Multi-Class Image Classification

## 📘 Overview

This project compares two convolutional neural network (CNN) architectures — a **custom-built CNN** and a **transfer learning model based on VGG16** — for a **multi-class image classification problem** involving 18 distinct categories.  

The objective is to evaluate whether transfer learning with a pre-trained network (VGG16) provides a measurable performance improvement over a traditional CNN trained from scratch.

### ✨ Key Results
- **Custom CNN Accuracy:** 75%  
- **VGG16 (Fine-tuned) Accuracy:** 87%  
- **Improvement:** +12% accuracy, with significant gains in precision, recall, and F1-score.

---

## 📊 Dataset

### Source
The dataset consists of **18 classes of labeled images**, each resized to **128×128 RGB** format.  
(If the dataset is public, you can mention it here, e.g., *"Dataset derived from Kaggle’s XYZ Dataset."*)

### Size
- **Training Samples:** ~80% of total data  
- **Testing Samples:** ~20% of total data  
- **Image Dimensions:** 128 × 128 × 3  
- **Classes:** 18

### Preprocessing
- Images were normalized to a `[0,1]` range.  
- One-hot encoding was applied to class labels.  
- Data augmentation was performed using:
  - Random rotation (20–30°)
  - Horizontal flipping
  - Zoom and shift transformations  
- This helped mitigate overfitting and improve model generalization.

---

## 🧩 Methods

### 1️⃣ Custom CNN
A custom convolutional model was built from scratch using **TensorFlow/Keras**.  
The architecture included:
| Layer Type | Details |
|-------------|----------|
| Conv2D + ReLU | 64 filters, 3×3 |
| MaxPooling2D | 2×2 |
| Conv2D + ReLU | 128 filters, 3×3 |
| MaxPooling2D | 2×2 |
| Conv2D + ReLU | 128 filters, 3×3 |
| MaxPooling2D | 2×2 |
| Dense + ReLU | 256 units |
| Dropout | 0.5 |
| Dense + Softmax | 18 units (output) |

Learning rate was managed using an **exponential decay schedule** starting at `1e-3`.

---

### 2️⃣ VGG16 (Transfer Learning)
A **pre-trained VGG16 model** (from ImageNet) was used as the base, with the top layers removed.  
Custom dense layers were added for fine-tuning:
| Layer Type | Details |
|-------------|----------|
| Flatten | — |
| Dense + ReLU | 512 units |
| BatchNormalization | — |
| Dropout | 0.5 |
| Dense + ReLU | 256 units |
| BatchNormalization | — |
| Dropout | 0.3 |
| Dense + Softmax | 18 units (output) |

Two-stage training:
1. **Stage 1:** Train only top (custom) layers (10 epochs)  
2. **Stage 2:** Unfreeze and fine-tune the last 4 VGG layers (20 epochs)

Optimizer: **Adam** with learning rate decay (`1e-4` → `1e-5`)

---

### 🧮 Why This Approach?

- **Custom CNN:** Establishes a baseline performance.  
- **VGG16 Transfer Learning:** Leverages pre-trained ImageNet weights for better feature extraction and faster convergence.  
- **Data Augmentation:** Prevents overfitting and improves robustness.

**Alternative architectures considered:** ResNet50, InceptionV3, and MobileNetV2 — but VGG16 was chosen for its balance of simplicity, interpretability, and performance on small datasets.

---

## 🧰 Steps to Run the Code

### Prerequisites
Make sure you have the following installed:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
