# retrain.py
import os
import math
import mlflow
import torch
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Force single GPU ───────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID  = "sandesh2233/Deepfakes_detection"
DATA_DIR  = "./data"
HF_REPO   = "sandesh2233/Deepfakes_detection"
EPOCHS    = 2
LR        = 5e-5
BATCH     = 16
IMG_SIZE  = 224

ID2LABEL  = {0: "real", 1: "fake"}
LABEL2ID  = {"real": 0, "fake": 1}

print("=" * 52)
print("  AUTOMATED RETRAINING TRIGGERED")
print("  New data detected — starting retrain...")
print("=" * 52)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ── Dataset ────────────────────────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []

        for label_name, label_id in LABEL2ID.items():
            folder = Path(root_dir) / label_name
            if not folder.exists():
                raise FileNotFoundError(
                    f"Folder not found: {folder}\n"
                    f"Make sure data/real/ and data/fake/ both exist."
                )
            imgs = (
                list(folder.glob("*.jpg"))
                + list(folder.glob("*.jpeg"))
                + list(folder.glob("*.png"))
            )
            if not imgs:
                raise ValueError(f"No images found in {folder}")
            self.samples.extend([(str(p), label_id) for p in imgs])

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"Data loaded: {len(self.samples)} images  ({n_real} real, {n_fake} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] Skipping corrupt image '{path}': {e}")
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

        if self.transform:
            img = self.transform(img)

        assert label in (0, 1), f"Invalid label {label} at index {idx}"

        return {
            "pixel_values": img,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ── Transforms ─────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ── Collate ────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels       = torch.stack([b["labels"]       for b in batch])

    assert labels.min() >= 0 and labels.max() <= 1, \
        f"Bad labels in batch: {labels.tolist()}"

    return {"pixel_values": pixel_values, "labels": labels}


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    probs = (
        torch.softmax(torch.tensor(p.predictions, dtype=torch.float32), dim=1)[:, 1]
        .numpy()
    )
    return {
        "accuracy": round(accuracy_score(p.label_ids, preds), 4),
        "auc":      round(roc_auc_score(p.label_ids, probs), 4),
    }


# ── Load model ─────────────────────────────────────────────────────────────────
print(f"\nLoading base model: {MODEL_ID}")
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model     = ViTForImageClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)
model.to(device)
print(f"Model loaded. Labels: {model.config.id2label}")


# ── Load & split dataset (80% train, 20% eval) ─────────────────────────────────
full_data  = DeepfakeDataset(DATA_DIR, transform=train_transform)
val_size   = max(1, int(0.2 * len(full_data)))
train_size = len(full_data) - val_size

train_data, val_data = random_split(
    full_data,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)   # reproducible split
)
print(f"Split  : {train_size} train  |  {val_size} eval")

steps_per_epoch = math.ceil(train_size / BATCH)
total_steps     = steps_per_epoch * EPOCHS
warmup_steps    = max(1, int(0.1 * total_steps))


# ── Training args ──────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir                  = "./vit-deepfake-retrained",
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH,
    per_device_eval_batch_size  = BATCH,
    learning_rate               = LR,
    warmup_steps                = warmup_steps,
    weight_decay                = 0.01,
    fp16                        = torch.cuda.is_available(),
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "auc",
    greater_is_better           = True,
    logging_steps               = 5,
    report_to                   = "none",
    dataloader_num_workers      = 0,
)


# ── MLflow + Training ──────────────────────────────────────────────────────────
mlflow.set_experiment("deepfake-retraining")

with mlflow.start_run(run_name="retrain-v2"):
    mlflow.log_params({
        "base_model":  MODEL_ID,
        "new_samples": len(full_data),
        "train_size":  train_size,
        "val_size":    val_size,
        "epochs":      EPOCHS,
        "lr":          LR,
        "batch_size":  BATCH,
        "trigger":     "new_data_detected",
    })

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_data,
        eval_dataset    = val_data,
        compute_metrics = compute_metrics,
        data_collator   = collate_fn,
    )

    print("\n" + "=" * 52)
    print(f"  Retraining started...")
    print(f"  Total samples : {len(full_data)}")
    print(f"  Train / Eval  : {train_size} / {val_size}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Batch size    : {BATCH}")
    print(f"  Steps/epoch   : {steps_per_epoch}")
    print(f"  Warmup steps  : {warmup_steps}")
    print("=" * 52 + "\n")

    trainer.train()

    print("\nRunning final evaluation...")
    metrics = trainer.evaluate()

    mlflow.log_metrics({
        "eval_accuracy": metrics.get("eval_accuracy", 0),
        "eval_auc":      metrics.get("eval_auc", 0),
    })

    # ── Push to HF Hub ─────────────────────────────────────────────────────
    print("\nPushing retrained model to HuggingFace Hub...")
    try:
        model.push_to_hub(HF_REPO, commit_message="Retrained on new data v2")
        processor.push_to_hub(HF_REPO)
        print(f"Model live at: https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"[WARN] HF push failed: {e}")
        print("Model saved locally at ./vit-deepfake-retrained")

    mlflow.end_run()

    print("\n" + "=" * 52)
    print("  Retraining complete!")
    print(f"  AUC      : {metrics.get('eval_auc', 0):.4f}")
    print(f"  Accuracy : {metrics.get('eval_accuracy', 0):.4f}")
    print("=" * 52)
