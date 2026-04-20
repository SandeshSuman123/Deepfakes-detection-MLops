# Deepfakes-detection-MLops
# Deepfake Detection with Vision Transformer (ViT)

> **🚀 Live Demo:** [https://sandesh2233-deepfake-detector.hf.space/](https://sandesh2233-deepfake-detector.hf.space/)
>
> **🤗 Model on HuggingFace Hub:** [sandesh2233/Deepfakes_detection](https://huggingface.co/sandesh2233/Deepfakes_detection/tree/main)

---

## Project Overview

This is an end-to-end **MLOps course project** that fine-tunes a **Vision Transformer (ViT)** model to detect AI-generated deepfake faces. The project covers the full MLOps lifecycle — from data preparation and model training to experiment tracking, containerization, and live deployment on **Hugging Face Spaces**.

The model is trained on the **Celeb-DF v2** dataset and achieves **99.93% AUC** on the test set.

---

##  Results

| Metric | Score |
|--------|-------|
| **Test AUC** | **0.9993** |
| **Test Accuracy** | 89.17% |
| **Test F1 (Fake)** | 0.9023 |
| **False Negatives** | 0 *(zero missed fakes)* |

> All metrics tracked and logged via **MLflow** — experiment: `deepfake-detection`, run: `vit-celebdf-v2`

---

##  Project Structure

```
deepfake_project/

├── train.py            # ViT fine-tuning script (Celeb-DF v2)
├── retrain.py          # Retraining / continual learning script
├── inference.py        # Standalone inference script
├── app.py              # Backend API / application entry point

├── frontend/           # Web UI
│   └── index.html      # HTML interface for inference

├── Dockerfile          # Docker image for deployment
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
```

---

## Model Architecture

- **Base Model:** `google/vit-base-patch16-224` (Vision Transformer)
- **Task:** Binary classification — `real` vs `fake`
- **Input:** 224×224 RGB face images
- **Output:** Probability score (`0` = real, `1` = fake)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch Size (train) | 256 |
| Learning Rate | 2e-4 |
| Warmup | 10% of total steps |
| Weight Decay | 0.01 |
| Precision | FP16 (mixed precision) |
| GPU | NVIDIA RTX PRO 6000 Blackwell (102 GB) |
| Augmentation | Horizontal flip, rotation ±10°, color jitter |

---

##  Dataset

**Celeb-DF v2** — a high-quality deepfake dataset with celebrity face swaps.

```
Train :  2,640 images  (1,320 real / 1,320 fake)
Val   :  1,140 images  (570  real / 570  fake)
Test  :    360 images  (180  real / 180  fake)
```

> Dataset is **not included** in this repo due to size.  
> Download from the [official Celeb-DF v2 source](https://github.com/yuezunli/celeb-deepfakeforensics).

---

##  Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/sandesh2233/deepfake_project.git
cd deepfake_project
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place your Celeb-DF v2 frames in the following structure:

```
data/
  train/real/   train/fake/
  val/real/     val/fake/
  test/real/    test/fake/
```

### 4. Train the model

```bash
python train.py
```

### 5. Run inference on a single image

```bash
python inference.py --image path/to/face.jpg
```
## Docker

Build and run the app locally using Docker:

```bash
# Build the image
docker build -t deepfake-detector .

# Run the container
docker run -p 7860:7860 deepfake-detector
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Experiment Tracking with MLflow

All training runs are tracked in MLflow with hyperparameters, metrics, and model artifacts.

```bash
# Start the MLflow UI (use a free port if 5000 is taken)
mlflow ui --port 5001
```

Open [http://127.0.0.1:5001](http://127.0.0.1:5001) to browse experiments.

**Logged parameters:** model name, dataset, epochs, batch size, learning rate, FP16, image size, augmentation strategy  
**Logged metrics:** `test_accuracy`, `test_auc`, `test_f1_fake`  
**Artifacts:** full model checkpoint saved under `model/`

---

##  Deployment

The model is deployed as a live web app on **Hugging Face Spaces** using a HTML,CSS & JS frontend and **Docker** container.

| Resource | Link |
|----------|------|
| 🚀 Live App | [https://sandesh2233-deepfake-detector.hf.space/](https://sandesh2233-deepfake-detector.hf.space/) |
| 🤗 Model Weights | [sandesh2233/Deepfakes_detection](https://huggingface.co/sandesh2233/Deepfakes_detection/tree/main) |

### Deployment steps used

```bash
# Login to Hugging Face
huggingface-cli login

# Push model from training script (PUSH_TO_HUB = True in train.py)
# OR push Space files manually via the HF web UI / Git
```

The `hf_space/` directory contains all files needed for the Space:
- `app.py` — Gradio interface that loads the model from the Hub
- `Dockerfile` — container definition for the Space runtime
- `requirements.txt` — Space dependencies
- `README.md` — Space card metadata

---

##  MLOps Pipeline Summary

```
Raw Video (Celeb-DF v2)
        │
        ▼
  Frame Extraction & Labeling
        │
        ▼
  Data Augmentation (flip, rotate, color jitter)
        │
        ▼
  ViT Fine-tuning  ◄──── MLflow Tracking
        │                  (params + metrics + artifacts)
        ▼
  Best Model (by AUC on val set)
        │
        ▼
  HuggingFace Hub  ──►  Hugging Face Space (Frontend + Docker)
                                │
                                ▼
                         Live Public Demo
```

---

##  Requirements

```
torch
torchvision
transformers
datasets
scikit-learn
mlflow
gradio
Pillow
numpy
```

Install all with:

```bash
pip install -r requirements.txt
```

---

##  Acknowledgements

- [Celeb-DF v2 Dataset](https://github.com/yuezunli/celeb-deepfakeforensics) — Li et al., 2020
- [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) — Dosovitskiy et al., 2020
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [MLflow](https://mlflow.org/)

---

## 👤 Author

**Sandesh Suman**  
MLOps Course Project  
[HuggingFace Profile](https://huggingface.co/sandesh2233)
