# Arabic Sign Language Recognition (ArSL)

This project presents a deep learning solution for Arabic Sign Language (ArSL) recognition using Convolutional Neural Networks (CNNs) . The system efficiently predicts ArSL letters from images, demonstrating the power of CNNs in accessibility-focused applications with Gradio interface. 

## Dataset

- *ArASL_Database_54K_Final*: 54,049 grayscale images, 32 classes.
- *Source*: Mendeley Data.

## Obstacles & Challenges

- *Scarcity of Data*: Limited public datasets cover only letters, not words or sentences.
  - Solution: Used ArSL alphabet dataset as a baseline, leaving room for future word/phrase expansion.
- *Only Using Letters*: Model handles letters only, limiting real-life use.
  - Solution: Built a prototype on letters as a foundation, planning to extend to words/sentences later.
- *Arabic Alphabet Limitation*: Model works only with Arabic signs, not other sign languages.
  - Solution: Focused on Arabic users for high accuracy, with future multi-language support via transfer learning.
- *Choosing Suitable Layers*: Dense layers failed to capture image spatial details.
  - Solution: Adopted CNNs for better feature extraction (shapes, curves), boosting accuracy.
- *Detecting Edges/Corners*: Subtle hand gesture differences were hard to detect.
  - Solution: Used Conv2D filters to automatically detect edges, corners, and textures.

## Future Work

- *Dataset Expansion*: Add diverse data (backgrounds, lighting, videos).
- *Transfer Learning*: Use pre-trained models (e.g., MobileNet) for better generalization.
- *Data Augmentation*: Apply rotations, flips, and noise for robustness.
- *Real-Time Deployment*: Enable real-time detection with TensorFlow Lite and mobile/web apps.
- *Advanced Models*: Explore RNNs, LSTMs, or Vision Transformers for video gestures.
- *User-Centered Apps*: Develop bilingual (ArSL â†” text/speech) systems with personalization.

## Setup

bash
pip install -r requirements.txt
python app.py


## Demo

- Live at: \[(https://huggingface.co/spaces/JoeMax3/Sign_Language_Detection)\]
