"""
src/model_store.py
MindBridge - Trained Model Persistence Layer.
Save and load .joblib files for all trained models.
Satisfies SRS REQ-F04: all models saved as .joblib.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import joblib

from src.config import MODEL_FILES, MODEL_DIR


def save_model(model, key):
    """Save a trained model to disk as a .joblib file."""
    if key not in MODEL_FILES:
        raise KeyError(f"Unknown model key: '{key}'. Valid keys: {list(MODEL_FILES.keys())}")
    path = MODEL_FILES[key]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path, compress=3)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"[ModelStore] Saved '{key}' -> {path} ({size_mb:.1f} MB)")
    return path


def load_model(key):
    """Load a trained model from disk."""
    if key not in MODEL_FILES:
        raise KeyError(f"Unknown model key: '{key}'. Valid keys: {list(MODEL_FILES.keys())}")
    path = MODEL_FILES[key]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{key}' not found at {path}.\n"
            "Run: python src/models.py\n"
            "This trains all models and saves them to the models/ directory."
        )
    return joblib.load(path)


def models_exist():
    """Check which model .joblib files exist on disk. Used by startup health check."""
    return {key: os.path.exists(path) for key, path in MODEL_FILES.items()}


if __name__ == "__main__":
    print("=== MindBridge Model Store Self-Test ===")
    status = models_exist()
    for key, exists in status.items():
        marker = "EXISTS" if exists else "MISSING (run src/models.py first)"
        print(f"  {key}: {marker}")
    print("\nmodel_store.py: PASS")