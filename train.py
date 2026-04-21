"""
Deepfake Detection — ViT Fine-tuning on Celeb-DF v2
======================================================
Dataset structure expected:
    data/
      train/real/   train/fake/
      val/real/     val/fake/
      test/real/    test/fake/

Run:
    python train.py

"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ─────────────────────────────────────────────
Dataset path
# ─────────────────────────────────────────────
DATA_DIR    = "/home/sandesh/deepfake_project/Celeb_V2"  # folder with train/ val/ test/
MODEL_NAME  = "google/vit-base-patch16-224"
OUTPUT_DIR  = "./vit-deepfake"
HF_REPO     = "sandesh2233/Deepfakes_detection"
PUSH_TO_HUB = True   # set True after `huggingface-cli login`

# RTX PRO 6000 Blackwell (102 GB) optimised settings
BATCH_TRAIN  = 256    
BATCH_EVAL   = 512    
EPOCHS       = 5
LR           = 2e-4
NUM_WORKERS  = 16     
IMG_SIZE     = 224

# Labels
ID2LABEL = {0: "real", 1: "fake"}
LABEL2ID = {"real": 0, "fake": 1}

# Folder to save new uploaded images
NEW_DATA_DIR = "./new_data"
RETRAIN_THRESHOLD = 50  # retrain after 50 new images


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    """
    Reads images from:
        <root_dir>/<split>/real/*.jpg  (label 0)
        <root_dir>/<split>/fake/*.jpg  (label 1)
    """

    def __init__(self, root_dir: str, split: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        for label_name, label_id in LABEL2ID.items():
            folder = Path(root_dir) / split / label_name
            if not folder.exists():
                raise FileNotFoundError(
                    f"Expected folder not found: {folder}\n"
                    f"Check DATA_DIR and that {split}/{label_name}/ exists."
                )
            imgs = (
                list(folder.glob("*.jpg"))
                + list(folder.glob("*.jpeg"))
                + list(folder.glob("*.png"))
            )
            if not imgs:
                print(f"[WARN] No images found in {folder}")
            self.samples.extend([(str(p), label_id) for p in imgs])

        # Validate all labels are 0 or 1 up-front
        bad = [(p, l) for p, l in self.samples if l not in (0, 1)]
        if bad:
            raise ValueError(f"Dataset has {len(bad)} samples with invalid labels: {bad[:5]}")

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(
            f"[{split:5s}] {len(self.samples):5d} images  "
            f"({n_real} real, {n_fake} fake)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]

        # ── FIX 1a: Graceful fallback for corrupt / unreadable images ──────────
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError, Exception) as exc:
            print(f"[WARN] Cannot open '{path}': {exc}  →  using black placeholder.")
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

        if self.transform:
            img = self.transform(img)

        # ── FIX 1b: Hard assert — catches any future label corruption early ────
        assert label in (0, 1), f"Invalid label {label!r} at index {idx} ({path})"

        return {
            "pixel_values": img,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = (
        torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1)[:, 1]
        .numpy()
    )

    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    f1  = f1_score(labels, preds, pos_label=1)
    cm  = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix → TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp}")
    return {
        "accuracy": round(float(acc), 4),
        "auc":      round(float(auc), 4),
        "f1_fake":  round(float(f1),  4),
    }


# ─────────────────────────────────────────────
# COLLATE
# ─────────────────────────────────────────────
def collate_fn(batch: list[dict]) -> dict:
    # ── FIX 2: torch.stack on pre-made tensors — no double-wrapping ────────────
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels       = torch.stack([b["labels"]       for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
        print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ── Datasets ────────────────────────────────────────────────────────────────
    train_ds = DeepfakeDataset(DATA_DIR, "train", transform=train_transform)
    val_ds   = DeepfakeDataset(DATA_DIR, "val",   transform=eval_transform)
    test_ds  = DeepfakeDataset(DATA_DIR, "test",  transform=eval_transform)

    # ── Model ───────────────────────────────────────────────────────────────────
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # expected: ImageNet head (1000) → 2 classes
    )

    # ── FIX 4: warmup_steps instead of deprecated warmup_ratio ─────────────────
    steps_per_epoch  = len(train_ds) // BATCH_TRAIN
    total_steps      = steps_per_epoch * EPOCHS
    warmup_steps_val = int(0.1 * total_steps)   # 10 % warm-up

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Schedule
        num_train_epochs=EPOCHS,
        warmup_steps=warmup_steps_val,      # ── FIX 4
        weight_decay=0.01,
        learning_rate=LR,

        # Batch sizes
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,

        # Evaluation / saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,

        # Performance
        fp16=True,                      # BF16 also works on Blackwell; fp16 is safer
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,

        # Logging
        logging_steps=50,
        report_to="none",               # MLflow logged manually below

        # Hub
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HF_REPO if PUSH_TO_HUB else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # ── MLflow (optional — skipped gracefully if not installed / no server) ─────
    # ── FIX 7 ──────────────────────────────────────────────────────────────────
    mlflow_available = False
    try:
        import mlflow
        mlflow.set_experiment("deepfake-detection")
        mlflow_run = mlflow.start_run(run_name="vit-celebdf-v2")
        mlflow_available = True
        mlflow.log_params({
            "model":        MODEL_NAME,
            "dataset":      "Celeb-DF v2",
            "epochs":       EPOCHS,
            "batch_size":   BATCH_TRAIN,
            "lr":           LR,
            "fp16":         True,
            "img_size":     IMG_SIZE,
            "augmentation": "flip+rotate+colorjitter",
            "warmup_steps": warmup_steps_val,
        })
    except Exception as e:
        print(f"[INFO] MLflow not available ({e}). Training will proceed without it.")

    # ── Training ────────────────────────────────────────────────────────────────
    print("=" * 56)
    print("  Starting training …")
    print(f"  Epochs          : {EPOCHS}")
    print(f"  Train batch     : {BATCH_TRAIN}")
    print(f"  Steps per epoch : {steps_per_epoch}")
    print(f"  Total steps     : {total_steps}")
    print(f"  Warmup steps    : {warmup_steps_val}")
    print("=" * 56)

    trainer.train()
    

    # ── Test evaluation ─────────────────────────────────────────────────────────
    print("\nEvaluating on test set …")
    test_results = trainer.evaluate(test_ds)
    print(f"\nTest Results: {test_results}")

    test_acc = test_results.get("eval_accuracy", 0)
    test_auc = test_results.get("eval_auc",      0)
    test_f1  = test_results.get("eval_f1_fake",  0)

    print("\n" + "=" * 56)
    print(f"  Test AUC      : {test_auc:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test F1 Fake  : {test_f1:.4f}")
    print("=" * 56)

    # ── Log to MLflow if running ────────────────────────────────────────────────
    if mlflow_available:
        try:
            mlflow.log_metrics({
                "test_accuracy": test_acc,
                "test_auc":      test_auc,
                "test_f1_fake":  test_f1,
            })
            mlflow.log_artifacts(OUTPUT_DIR, artifact_path="model")
            mlflow_run.__exit__(None, None, None)
        except Exception as e:
            print(f"[WARN] MLflow logging failed: {e}")

    # ── Push to Hub ─────────────────────────────────────────────────────────────
    if PUSH_TO_HUB:
        print("\nPushing model to HuggingFace Hub …")
        trainer.push_to_hub(commit_message="Training complete — Celeb-DF v2")
        print(f"Model live at: https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
