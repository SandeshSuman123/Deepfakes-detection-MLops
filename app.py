import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import io

app = FastAPI()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(
    "./vit-deepfake/checkpoint-55"
).to(device)

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

model.eval()

id2label = {0: "REAL", 1: "FAKE"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "prediction": id2label[pred],
        "confidence": round(confidence, 4)
    }