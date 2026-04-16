import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import ViTForImageClassification, ViTImageProcessor

# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# LOAD MODEL (YOUR TRAINED MODEL)
# -------------------------------
model_path = "./vit-deepfake/checkpoint-55"
model = ViTForImageClassification.from_pretrained(model_path).to(device)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

model.eval()

# LABELS

id2label = {0: "REAL", 1: "FAKE"}

# -------------------------------
# LOAD IMAGE
# -------------------------------
image_path = "Celeb_V2/test/real/00004_face_281.jpg"
image = Image.open(image_path).convert("RGB")

# -------------------------------
# PREPROCESS
# -------------------------------
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# -------------------------------
# PREDICTION
# -------------------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

label = id2label[pred]

# -------------------------------
# PRINT RESULT
# -------------------------------
print(f"\nPrediction: {label}")
print(f"Confidence: {confidence:.4f}\n")

# -------------------------------
# DRAW RESULT ON IMAGE
# -------------------------------
draw = ImageDraw.Draw(image)

text = f"{label} ({confidence:.2%})"

# Optional: font (fallback if not found)
try:
    font = ImageFont.truetype("arial.ttf", 30)
except:
    font = ImageFont.load_default()

# Draw background rectangle
bbox = draw.textbbox((10, 10), text, font=font)
draw.rectangle(bbox, fill="black")

# Draw text
draw.text((10, 10), text, fill="white", font=font)

# -------------------------------
# SAVE OUTPUT
# -------------------------------
output_path = "result.png"
image.save(output_path)

print(f"Saved result image to: {output_path}")