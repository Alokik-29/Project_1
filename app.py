
import gradio as gr
import torch
import librosa
import numpy as np
import joblib
from pathlib import Path
from transformers import ASTFeatureExtractor
import torch.nn as nn
from transformers import ASTForAudioClassification

# Load label map and feature extractor
emotion_map = joblib.load("emotion_map.pkl")
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Predictor class
class ImprovedEmotionPredictor:
    def __init__(self, model_path, feature_extractor):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor
        self.emotion_labels = list(emotion_map.values())

        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        num_labels = len(emotion_map)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.config.hidden_size, num_labels)
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict_emotion(self, audio_path_or_array):
        if isinstance(audio_path_or_array, (str, Path)):
            y, sr = librosa.load(audio_path_or_array, sr=16000)
        else:
            y = audio_path_or_array
        y = y.astype(np.float32)
        inputs = self.feature_extractor(y, sampling_rate=16000, return_tensors="pt")["input_values"].to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs).logits
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)
        predicted_emotion = self.emotion_labels[predicted_class[0]]
        confidence = probabilities[0][predicted_class[0]].item()
        return f"{predicted_emotion} ({confidence:.2f})"

# Initialize predictor
predictor = ImprovedEmotionPredictor("best_improved_model.pth", feature_extractor)

# Gradio interface
demo = gr.Interface(
    fn=predictor.predict_emotion,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="🎤 AST Audio Emotion Recognition",
    description="Upload a .wav audio clip and detect the emotion from tone of voice."
)

demo.launch()
