# Blind Sight: Automated Image Captioning

**Blind Sight** is a mobile application designed to assist visually impaired individuals by providing real-time auditory feedback about their surroundings. By leveraging Deep Learning and a Client-Server architecture, the application captures images, generates descriptive captions, and converts them into speech, enhancing the user's independence and environmental interaction.

## 📖 Table of Contents
* [Introduction](#-introduction)
* [How It Works](#-how-it-works)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Tech Stack](#-tech-stack)
* [Dataset](#-dataset)
* [Installation & Setup](#-installation--setup)
* [Future Roadmap](#-future-roadmap)
* [Contributors](#-contributors)

---

## 📝 Introduction
Navigating the world without visual cues is a significant challenge. **Blind Sight** bridges this gap by acting as a digital visual aid. The user simply points their smartphone camera at a scene, and the application utilizes a Convolutional Neural Network (CNN) combined with a Recurrent Neural Network (RNN) to "see" the image and describe it aloud.

---

## ⚙️ How It Works
The project utilizes an Encoder-Decoder architecture to bridge computer vision and natural language processing.

1.  **Image Capture:** The user captures an image via the Flutter mobile app.
2.  **Transmission:** The image is sent to the backend server via FastAPI.
3.  **Feature Extraction (Encoder):** **Inception v3**, a pre-trained CNN, processes the image to extract high-level feature vectors.
4.  **Caption Generation (Decoder):** These vectors are passed to a **3-layer Stacked LSTM** network to generate a sentence sequence.
5.  **Auditory Output:** The generated text is converted to audio using a Text-to-Speech (TTS) engine.

### Process Flow Diagram
![Process Flow Diagram](path/to/your/flow_chart_image.png)
---

## 🏗 System Architecture
The application follows a strict client-server data flow to ensure low latency and high accuracy. The mobile client handles input/output, while the server handles heavy AI processing.

### Architecture Diagram
![System Architecture Diagram](path/to/your/architecture_image.png)
* **Client:** Flutter App (Camera, TTS).
* **Protocol:** HTTP/HTTPS (GET/POST) via FastAPI.
* **Server:** Python environment hosting PyTorch models (Inception V3 + LSTM).

---

## ✨ Key Features
* **Real-time Assistance:** Immediate capture-to-speech processing.
* **Tactile UI:** Minimalist interface designed specifically for visually impaired users.
* **Deep Learning:** Uses Inception v3 (efficient parameters) and LSTM for context-aware captioning.
* **Scalable Backend:** Built with FastAPI for high-performance asynchronous request handling.

---

## 💻 Tech Stack

### Mobile Application (Frontend)
* **Framework:** Flutter (Dart)
* **Modules:** Camera, HTTP, Text-to-Speech (TTS)

### Backend & AI
* **Language:** Python
* **API Framework:** FastAPI
* **Machine Learning:** PyTorch
* **Models:**
    * **CNN:** Inception v3 (Pre-trained on ImageNet)
    * **RNN:** LSTM (3-Layer Stacked)

---

## 📂 Dataset
The model is trained and evaluated using the **Flickr Dataset Family**:
* **Flickr8k:** Used for initial testing, debugging, and hyperparameter tuning (8,000 images).
* **Flickr30k:** Used for final training to ensure better generalization (30,000 images).

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8+
* Flutter SDK
* Android Studio / VS Code

### 1. Backend Setup
```bash
# Clone the repository
git clone [https://github.com/yourusername/blind-sight.git](https://github.com/yourusername/blind-sight.git)
cd blind-sight/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install torch torchvision fastapi uvicorn pillow

# Run the Server
uvicorn main:app --reload
