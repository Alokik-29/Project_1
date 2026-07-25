import io
import os
import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from huggingface_hub import hf_hub_download

app = FastAPI()

# Download model files from your Hugging Face Space instead of bundling
# them in git (they're too large for a normal git push)
os.makedirs("models", exist_ok=True)

model_path = hf_hub_download(
    repo_id="alokik29/audio-emotion-gradio",
    filename="best_improved_model.pth",
    repo_type="space",
    local_dir="models",
)
emotion_map_path = hf_hub_download(
    repo_id="alokik29/audio-emotion-gradio",
    filename="emotion_map.pkl",
    repo_type="space",
    local_dir="models",
)

# Allow your Vercel frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your Vercel URL once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load everything once at startup ----
emotion_map = joblib.load(emotion_map_path)
emotion_labels = list(emotion_map.values())

feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.config.hidden_size, len(emotion_map))
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


def preprocess_audio(file_bytes: bytes):
    # librosa can load directly from a file-like object, no need to save to disk
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000)
    y = y.astype(np.float32)
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")["input_values"]
    return inputs.to(device)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    input_tensor = preprocess_audio(audio_bytes)

    with torch.no_grad():
        logits = model(input_tensor).logits
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(logits, dim=1)

    emotion = emotion_labels[pred_idx[0]]
    confidence = probs[0][pred_idx[0]].item()

    return {
        "emotion": emotion,
        "confidence": round(confidence, 4)
    }