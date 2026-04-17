from weasyprint import HTML
import os

# Content for the README.md based on the posters and HTML data provided
readme_content = """# BloomSense: Deep Learning-Based Flower Identification

BloomSense is a lightweight, deep learning-powered system designed for real-time flower species recognition paired with semantic and cultural insights. By bridging the gap between computer vision and botanical knowledge, the system provides users with not just the name of a flower, but also its symbolic significance.

## 🌸 Project Overview

Manual flower identification often requires expert botanical knowledge. BloomSense automates this process using state-of-the-art Convolutional Neural Networks (CNNs).

- **Accuracy:** 91% across 102 species.
- **Inference Speed:** ~0.5 seconds per image.
- **Dataset:** 8,000+ images (Oxford Flower Dataset).
- **Core Technology:** Transfer Learning with MobileNetV2.

## 🚀 Key Features

- **Automated Recognition:** Instant identification of 102 different flower species.
- **Semantic Insights:** Displays cultural meanings and symbolism (e.g., Rose: Love & Passion).
- **Lightweight Architecture:** Optimized for deployment without heavy GPU requirements.
- **Real-time Processing:** Fast inference suitable for mobile and web applications.
- **Robustness:** Utilizes data augmentation (rotation, zoom, flip) to handle real-world image variations.

## 🛠️ Technology Stack

- **Language:** Python 3.x
- **Frameworks:** TensorFlow / Keras
- **Model Architecture:** MobileNetV2 (Pre-trained on ImageNet)
- **Deployment:** Streamlit (Web Interface)
- **Libraries:** NumPy, Pandas, Matplotlib, OpenCV

## 📊 Model Architecture & Pipeline

The system follows a structured end-to-end pipeline:
1. **Upload:** User uploads an RGB image.
2. **Preprocess:** Image is resized to 224x224 pixels.
3. **MobileNetV2:** Feature extraction using a frozen pre-trained base.
4. **Classify:** A custom classification head with Global Average Pooling and Softmax activation identifies the species.
5. **Meaning:** The system fetches and displays the cultural significance of the identified flower.

## 📈 Performance

| Component | Detail | Value |
| :--- | :--- | :--- |
| Base Model | MobileNetV2 | Transfer Learning |
| Input Size | RGB Resize | 224x224 |
| Training Epochs | Convergence | 20 |
| Accuracy | Species Rate | 91% |

## 💻 Installation & Usage

1. **Clone the repository:**