import torch, os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import io

app = FastAPI(title="Deepfake Detector")

device = torch.device("cpu")
MODEL_ID = "sandesh2233/Deepfakes_detection"

model = ViTForImageClassification.from_pretrained(MODEL_ID)
model.to(device)
model.eval()
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
id2label = {0: "REAL", 1: "FAKE"}

@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return {
        "prediction": id2label[pred],
        "confidence": round(confidence, 4),
        "fake_score": round(probs[0][1].item(), 4),
        "real_score": round(probs[0][0].item(), 4),
        "verdict": "DEEPFAKE DETECTED" if pred == 1 else "AUTHENTIC"
    }
