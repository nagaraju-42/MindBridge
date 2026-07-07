"""
distilbert/train_distilbert.py
MindBridge — DistilBERT Fine-tuning (GPU Comparison Model).

This script fine-tunes DistilBERT-base-uncased on the Reddit depression
dataset as a comparison to the traditional ML ensemble.

Purpose: Show the accuracy vs. explainability tradeoff in the report.
  - DistilBERT: ~94-96% F1 (likely higher than RF ensemble)
  - RF Ensemble: ~90-92% F1 BUT has SHAP explainability
  - Conclusion: We choose the explainable model (RF) as primary

Requirements:
  - CUDA GPU with >= 8GB VRAM (runs on CPU but takes 3-4 hours)
  - pip install torch transformers datasets accelerate

Runtime: ~90 minutes on a single GPU (T4 or RTX 3060)
         ~4 hours on CPU only

Run:
  python distilbert/train_distilbert.py

Results saved to:
  distilbert/results/distilbert_metrics.json   <- Add to report Table 5.1
"""
import os
import json
import time
import numpy as np

# ── Check GPU availability ────────────────────────────────────────────────────
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DistilBERT] Device: {device}")
if device.type == "cpu":
    print("[DistilBERT] WARNING: Running on CPU. Estimated time: 3-4 hours.")
    print("[DistilBERT] For faster training, use Google Colab (free T4 GPU).")
    print("[DistilBERT] Google Colab: https://colab.research.google.com")
else:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[DistilBERT] GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB")

# ── Load dataset ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_splits

print("\n[DistilBERT] Loading Reddit depression dataset splits...")
X_train, X_test, y_train, y_test = load_splits()
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
from transformers import DistilBertTokenizerFast

MODEL_NAME = "distilbert-base-uncased"
print(f"\n[DistilBERT] Loading tokenizer: {MODEL_NAME}")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# Tokenize
print("[DistilBERT] Tokenizing texts (max_length=512)...")
train_encodings = tokenizer(
    list(X_train), truncation=True, padding=True, max_length=512
)
test_encodings  = tokenizer(
    list(X_test),  truncation=True, padding=True, max_length=512
)

# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class DepressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = DepressionDataset(train_encodings, list(y_train))
test_dataset  = DepressionDataset(test_encodings,  list(y_test))

# ── Model ─────────────────────────────────────────────────────────────────────
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

print(f"\n[DistilBERT] Loading model: {MODEL_NAME} (66M parameters)")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(device)

# ── Training Arguments ─────────────────────────────────────────────────────────
os.makedirs("distilbert/results", exist_ok=True)

training_args = TrainingArguments(
    output_dir="distilbert/results",
    num_train_epochs=3,
    per_device_train_batch_size=16 if device.type == "cuda" else 8,
    per_device_eval_batch_size=16  if device.type == "cuda" else 8,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="distilbert/results/logs",
    logging_steps=50,
    fp16=device.type == "cuda",     # Mixed precision if GPU
    dataloader_num_workers=0,       # Windows-compatible (no fork)
    report_to="none",               # No WandB/TensorBoard
)

# ── Metrics Function ───────────────────────────────────────────────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    proba  = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()

    return {
        "accuracy":  round(accuracy_score(labels, preds) * 100, 2),
        "f1":        round(f1_score(labels, preds) * 100, 2),
        "precision": round(precision_score(labels, preds, zero_division=0) * 100, 2),
        "recall":    round(recall_score(labels, preds, zero_division=0) * 100, 2),
        "auc_roc":   round(roc_auc_score(labels, proba) * 100, 2),
    }

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ── Train ──────────────────────────────────────────────────────────────────────
print(f"\n[DistilBERT] Starting fine-tuning ({training_args.num_train_epochs} epochs)...")
print("[DistilBERT] Estimated time: 90 min (GPU) | 4 hours (CPU)")
t_start = time.time()

trainer.train()

elapsed = time.time() - t_start
print(f"\n[DistilBERT] Training complete in {elapsed/60:.0f} minutes")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n[DistilBERT] Evaluating on held-out test set...")
results = trainer.evaluate()
print("\n[DistilBERT] Evaluation Results:")
for k, v in results.items():
    print(f"  {k}: {v}")

# ── Save Results ───────────────────────────────────────────────────────────────
metrics_out = {
    "model":          "DistilBERT-base-uncased (fine-tuned)",
    "dataset":        "Reddit Depression Cleaned (7,731 posts, 80/20 split)",
    "epochs":         training_args.num_train_epochs,
    "batch_size":     training_args.per_device_train_batch_size,
    "learning_rate":  training_args.learning_rate,
    "device":         str(device),
    "training_minutes": round(elapsed / 60, 1),
    **{k.replace("eval_", ""): v for k, v in results.items()},
}

metrics_path = "distilbert/results/distilbert_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"\n[DistilBERT] Metrics saved to: {metrics_path}")
print("\n" + "=" * 60)
print("ADD THIS ROW TO YOUR REPORT TABLE 5.1 (Model Comparison):")
print("=" * 60)
print(f"  Model:     DistilBERT-base-uncased (fine-tuned, 3 epochs)")
print(f"  Accuracy:  {results.get('eval_accuracy', 'N/A')}%")
print(f"  F1-Score:  {results.get('eval_f1', 'N/A')}%")
print(f"  Precision: {results.get('eval_precision', 'N/A')}%")
print(f"  Recall:    {results.get('eval_recall', 'N/A')}%")
print(f"  AUC-ROC:   {results.get('eval_auc_roc', 'N/A')}%")
print("=" * 60)
print()
print("NOTE: DistilBERT likely achieves higher F1 than the RF ensemble.")
print("Your report conclusion: RF ensemble is preferred because it enables")
print("SHAP explainability, which DistilBERT does not support via TreeExplainer.")
print("This demonstrates the accuracy-vs-explainability tradeoff.")
print()
print("distilbert/train_distilbert.py: COMPLETE")
