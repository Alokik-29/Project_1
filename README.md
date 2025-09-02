🎵 Speech Emotion Recognition with Audio Spectrogram Transformer (AST)


📌 Project Overview

This project focuses on classifying human emotions from speech audio using deep learning.
We fine-tuned a pretrained Audio Spectrogram Transformer (AST) model on the RAVDESS dataset and validated performance on the CREMA-D dataset to check cross-dataset generalization.


🚀 Key Features

Preprocessing: Audio converted to spectrograms (using Librosa & Torchaudio).

Model: Pretrained AST (Audio Spectrogram Transformer) fine-tuned with transfer learning.

Regularization: Early stopping, learning rate scheduler.

Evaluation: Cross-dataset validation on CREMA-D.

Visualization: Training/validation curves + confusion matrix.


📂 Dataset

RAVDESS 🎭 – Training dataset (~1,400 speech samples, 8 emotions).

CREMA-D 🎤 – Used for testing cross-dataset generalization.

Emotion classes:
Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised


🧠 Model Training

Framework: PyTorch
Optimizer: AdamW
Loss: CrossEntropy
Epochs: Early stopping at ~20 epochs
Hardware: Google Colab (GPU T4)


📊 Results


Dataset	Accuracy (%)

RAVDESS (Train/Test)	  84%
CREMA-D (Cross-dataset)	  54%

Loss/accuracy curves and confusion matrix are included in the report.


📌 Insights

The AST model learns emotions well on RAVDESS.
Accuracy drops significantly on CREMA-D, showing dataset bias and lack of generalization.
Future improvements: data augmentation, domain adaptation, larger multi-dataset training.


🛠️ Tech Stack

Python, PyTorch, Torchaudio, Librosa, NumPy, Matplotlib

Google Colab

📜 How to Run
# Install dependencies
pip install torch torchaudio librosa matplotlib

# Clone repo
git clone https://github.com/your-username/speech-emotion-ast.git
cd speech-emotion-ast

# Run training
python train.py


🎯 Future Work

Deploy demo using Gradio / Streamlit.

Train on multiple datasets for better generalization.

Explore self-supervised audio models like Wav2Vec2.0.
