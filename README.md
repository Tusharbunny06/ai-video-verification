# 🎥 AI Video Verification System

This project demonstrates an **end-to-end system for verifying AI-generated videos**
using **generation-time digital provenance** and **deepfake detection**.

---

## 🚩 Problem Statement
With the rapid growth of Generative AI models capable of producing realistic videos,
traditional deepfake detection methods have become unreliable and prone to false positives.
There is a need for a system that can **deterministically verify AI-generated videos**
when provenance is available and gracefully handle uncertainty when it is not.

---

## 💡 Solution Overview
This system follows a two-stage verification approach:

1. **Generation-Time Provenance Verification**
   - Videos generated using Generative AI embed provenance metadata at generation time.
   - Verification is deterministic using cryptographic hashes.

2. **Deepfake Detection (Fallback)**
   - When provenance is missing, a CNN-based deepfake detector is used.
   - Results are probabilistic and marked with low confidence.

---

## 🤖 Where GenAI Is Used
The project includes a **prompt-to-video Generative AI module**:
- A text prompt is converted into an image using **Stable Diffusion**
- The image is animated into a short video
- Provenance is embedded immediately after generation

This demonstrates how AI-generated videos **should be authenticated** in real-world systems.

---

## 🔄 System Pipeline
Text Prompt
↓
GenAI (Text → Image)
↓
Image → Video Animation
↓
Generation-Time Provenance Embedded
↓
Verification System
├─ Provenance Found → AI-Generated
└─ No Provenance → Deepfake Detection
---

## 🧪 How to Run

### 1️⃣ Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

2️⃣Install dependencies
pip install -r requirements.txt

 3️⃣Run the application
streamlit run app.py

Dataset URL : https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset
