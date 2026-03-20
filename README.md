# sign_language_translator
# 🤟 Real-time Sinhala Sign Language Translator

A Deep Learning-based mobile application designed to bridge the communication gap for the deaf and mute community in Sri Lanka. This project uses **Computer Vision** and **Recurrent Neural Networks (LSTM)** to translate Sinhala Sign Language (SSL) into text and speech.

## ✨ Key Features
- **Real-time Detection:** High-speed hand landmark tracking using Google MediaPipe.
- **Sinhala Support:** Focused on translating localized Sinhala Sign Language gestures.
- **AI-Powered:** Uses an LSTM model to understand the sequence of hand movements.
- **Speech Output:** Converts translated text into audible speech for seamless communication.

## 🛠️ Tech Stack
- **AI/ML:** Python, MediaPipe, TensorFlow/Keras
- **Mobile:** Flutter (Dart)
- **Libraries:** OpenCV, NumPy, Scikit-learn
- **Tools:** VS Code, GitHub Desktop

## 🚀 Getting Started
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/Thathsara-Devindi/sign_language_translator.git](https://github.com/Thathsara-Devindi/sign_language_translator.git)


2. **Setup Virtual Environment**
Bash
python -m venv venv
.\venv\Scripts\activate
3. Install Dependencies
Bash
pip install opencv-python mediapipe
4. Run Hand Tracking Test
Bash
python test_hand.py

**📈 Project Roadmap**

[x] Initial Research & Environment Setup

[x] Real-time Hand Landmark Detection (Prototype)

[ ] Data Collection for Sinhala Gestures

[ ] Training the LSTM Model

[ ] Flutter App Integration

[ ] Final Testing & Deployment
