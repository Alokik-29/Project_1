🎵 Speech Emotion Recognition with Audio Spectrogram Transformer (AST)

📌 Project Overview


This project classifies human emotions from speech audio using deep learning.

We fine-tuned a pretrained Audio Spectrogram Transformer (AST) model on the RAVDESS dataset and validated performance on CREMA-D for cross-dataset generalization.

🚀 Key Features


Preprocessing: Audio converted to spectrograms (Librosa & Torchaudio)

Model: Pretrained AST fine-tuned with transfer learning

Regularization: Early stopping, learning rate scheduler, dropout

Evaluation: Cross-dataset validation on CREMA-D

Visualization: Training/validation curves + confusion matrix

📂 Datasets


RAVDESS 🎭 – Training dataset (~1,400 speech samples, 8 emotions)

CREMA-D 🎤 – Testing for cross-dataset generalization

Emotion classes: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

🧠 Model Training


Framework: PyTorch

Optimizer: AdamW

Loss: CrossEntropy

Epochs: Early stopping at ~20 epochs

Hardware: Google Colab (GPU T4)

📊 Results


Dataset	Accuracy (%)

RAVDESS (Train/Test)	91%

CREMA-D (Cross-dataset)	58%

Loss/accuracy curves and confusion matrix are included in the project report.

📌 Insights


AST performs well on RAVDESS but shows dataset bias on CREMA-D

Future improvements: data augmentation, domain adaptation, multi-dataset training

🌐 Live Demo

Try the project live here:

[Hugging Face Space – Audio Emotion Recognition](https://huggingface.co/spaces/alokik29/audio-emotion-gradio)

🛠️ Tech Stack


Python, PyTorch, Torchaudio, Librosa, NumPy, Matplotlib, Google Colab

📜 How to Run Locally


# Install dependencies
pip install torch torchaudio librosa matplotlib gradio

# Clone repo
git clone https://github.com/Alokik-29/Project_1.git
cd Project_1

# Run Gradio demo
python app.py

🎯 Future Work


Deploy demo with Gradio / Streamlit (done ✅)

Train on multiple datasets for better generalization

Explore self-supervised audio models like Wav2Vec2.0
