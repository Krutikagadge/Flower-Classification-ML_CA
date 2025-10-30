# 🌸 Flower Classification Using CNN & VGG16  

## 📘 Overview  
This project focuses on classifying **17 different flower categories** using deep learning models — a **Custom Convolutional Neural Network (CNN)** and a **Transfer Learning model (VGG16)**.  
The goal is to automatically identify flower species from images with **high accuracy**.

Image classification plays a critical role in **agricultural automation**, **biodiversity tracking**, and **plant identification** applications.  
By leveraging modern CNN architectures, this project achieves strong classification performance and includes a **Streamlit web app** for real-time flower prediction.  

---

## 🌼 Dataset Source  
**Dataset:** [Oxford 17 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/)  
- **Number of Categories:** 17  
- **Total Images:** 1,360 (80 images per category)  
- **Image Size:** 128 × 128 pixels (resized during preprocessing)  
- **Format:** JPEG  

---

## 🧹 Preprocessing Steps  
1. Loaded all images from the `Data/` directory.  
2. Filtered only valid image folders (ignored hidden/system files).  
3. Resized all images to **(128 × 128)** for uniformity.  
4. Normalized pixel values (`/255.0`).  
5. One-hot encoded the labels.  
6. Split into **train/test (80/20)**.  
7. Saved class mapping (`class_names.npy`) for consistent label alignment during inference.  

---

## 🧠 Methods  

### 1️⃣ Custom CNN Model  
**Architecture:**  
- Input Layer: (128×128×3)  
- 3 × Convolution + MaxPooling blocks  
- Flatten Layer  
- Dense (256, ReLU)  
- Dropout (0.5)  
- Output Layer (Softmax for 17 classes)  

**Optimizer:** Adam (Exponential Decay Learning Rate)  
**Loss Function:** Categorical Crossentropy  
**Data Augmentation:** Rotation, Zoom, Horizontal Flip  

---

### 2️⃣ Transfer Learning with VGG16  
**Base Model:** Pretrained **VGG16** (on ImageNet, without top layers)  

**Custom Layers Added:**  
- Flatten  
- Dense (512 → BatchNorm → Dropout)  
- Dense (256 → BatchNorm → Dropout)  
- Dense (17, Softmax)  

**Training Phases:**  
1. Train only custom top layers (frozen base).  
2. Fine-tune last 4 convolutional layers of VGG16 with smaller learning rate.  

**Optimizer:** Adam (learning rate scheduling applied)  

---

## 🧩 Model Comparison  

| Model | Accuracy | Training Time | Comments |
|:------|:----------:|:--------------:|:----------|
| **Custom CNN** | ~87% | Faster | Lightweight baseline, quick convergence |
| **VGG16 (Fine-tuned)** | ~94% | Slower | Strong generalization and robust performance |

---

## 🚀 Steps to Run the Code  

### 🧰 Prerequisites  
Install the required dependencies:  
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn streamlit

