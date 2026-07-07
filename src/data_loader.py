"""
src/data_loader.py
MindBridge - Dataset Loading and Train/Test Splitting.
Satisfies SRS REQ-M03: 80/20 split, random_state=42, stratified.

Dataset: Reddit Depression Cleaned Dataset
Source:  https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
Size:    7,731 labeled Reddit posts (approximately balanced)

Kaggle Download Flow:
  1. Go to https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
  2. Click Download -> depression-reddit-cleaned.zip
  3. Extract the CSV
  4. Rename/copy to: data/reddit_depression.csv
  5. Run: python src/data_loader.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.preprocessing import batch_clean
from src.config import DATA_PATH, SPLITS_PATH, TEST_SIZE, RANDOM_SEED

_TEXT_COLS  = ["clean_text", "text", "post", "content", "message", "sentence"]
_LABEL_COLS = ["is_depression", "label", "depression", "class", "target"]


def _detect_columns(df):
    """Auto-detect text and label column names."""
    text_col  = next((c for c in _TEXT_COLS  if c in df.columns), None)
    label_col = next((c for c in _LABEL_COLS if c in df.columns), None)
    if not text_col:
        raise ValueError(f"Cannot find text column. Expected one of {_TEXT_COLS}. Got: {df.columns.tolist()}")
    if not label_col:
        raise ValueError(f"Cannot find label column. Expected one of {_LABEL_COLS}. Got: {df.columns.tolist()}")
    return text_col, label_col


def _find_csv():
    """Find the dataset CSV, trying multiple common filenames."""
    candidates = [
        DATA_PATH,
        "data/depression_dataset_reddit_cleaned.csv",
        "data/reddit_clean.csv",
        "data/depression.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Reddit depression CSV not found. Tried: {candidates}\n\n"
        "DOWNLOAD STEPS:\n"
        "1. Go to: https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned\n"
        "2. Click Download -> extract zip\n"
        "3. Copy the CSV to: data/reddit_depression.csv"
    )


def load_and_split(save=True, verbose=True):
    """
    Load the Reddit depression dataset, clean text, and create 80/20 split.
    Saves the split as splits.joblib for reuse (avoids 25s re-cleaning).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    csv_path = _find_csv()
    if verbose:
        print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"Raw dataset shape: {df.shape}")
    text_col, label_col = _detect_columns(df)
    if verbose:
        print(f"Text column: '{text_col}' | Label column: '{label_col}'")

    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    after = len(df)
    if verbose and before != after:
        print(f"Dropped {before - after} rows with missing/empty values")

    label_counts = df[label_col].value_counts()
    if verbose:
        print(f"\nClass distribution:")
        print(f"  Non-depression (0): {label_counts.get(0, 0)}")
        print(f"  Depression     (1): {label_counts.get(1, 0)}")
        balance = label_counts.min() / label_counts.max() * 100
        print(f"  Balance ratio: {balance:.1f}%")

    texts  = df[text_col].astype(str).values
    labels = df[label_col].astype(int).values

    if verbose:
        print(f"\nCleaning {len(texts)} texts with NLTK pipeline...")
    cleaned = batch_clean(texts)

    valid_mask = np.array([len(c.strip()) > 0 for c in cleaned])
    cleaned    = [c for c, v in zip(cleaned, valid_mask) if v]
    labels     = labels[valid_mask]

    if verbose:
        print(f"After cleaning: {len(cleaned)} valid texts")

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    if verbose:
        print(f"\nSplit complete:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(cleaned)*100:.0f}%)")
        print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(cleaned)*100:.0f}%)")

    if save:
        os.makedirs(os.path.dirname(SPLITS_PATH), exist_ok=True)
        joblib.dump((X_train, X_test, y_train, y_test), SPLITS_PATH, compress=3)
        if verbose:
            print(f"\nSplits saved to: {SPLITS_PATH}")

    return X_train, X_test, y_train, y_test


def load_splits():
    """Load pre-computed train-test splits from disk."""
    if not os.path.exists(SPLITS_PATH):
        raise FileNotFoundError(
            f"Splits not found at {SPLITS_PATH}.\nRun: python src/data_loader.py"
        )
    return joblib.load(SPLITS_PATH)


def generate_eda_charts(verbose=True):
    """Generate EDA charts for the project report Chapter 3."""
    csv_path = _find_csv()
    df = pd.read_csv(csv_path)
    text_col, label_col = _detect_columns(df)
    df = df.dropna(subset=[text_col, label_col])
    df["text_length"] = df[text_col].astype(str).str.len()
    df["word_count"]  = df[text_col].astype(str).str.split().str.len()

    os.makedirs("notebooks/eda_charts", exist_ok=True)
    C_GN = "#1E5631"
    C_RD = "#C00000"

    # Chart 1: Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[label_col].value_counts().sort_index()
    bars = ax.bar(["Non-Depression (0)", "Depression (1)"],
                  counts.values, color=[C_GN, C_RD], edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(val), ha="center", fontweight="bold")
    ax.set_title("Reddit Depression Dataset - Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Posts")
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    plt.savefig("notebooks/eda_charts/01_class_distribution.png", dpi=150)
    plt.close()

    # Chart 2: Text length KDE by class
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, name in [(0, C_GN, "Non-Depression"), (1, C_RD, "Depression")]:
        subset = df[df[label_col] == label]["text_length"]
        subset.plot(kind="kde", ax=ax, color=color, label=name, linewidth=2)
    ax.set_title("Text Length Distribution by Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Text Length (characters)")
    ax.set_xlim(0, 3000)
    ax.legend()
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    plt.savefig("notebooks/eda_charts/02_text_length_by_class.png", dpi=150)
    plt.close()

    # Chart 3: Word count KDE by class
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, name in [(0, C_GN, "Non-Depression"), (1, C_RD, "Depression")]:
        subset = df[df[label_col] == label]["word_count"].clip(upper=300)
        subset.plot(kind="kde", ax=ax, color=color, label=name, linewidth=2)
    ax.set_title("Word Count Distribution by Class", fontsize=13, fontweight="bold")
    ax.set_xlabel("Word Count")
    ax.legend()
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()
    plt.savefig("notebooks/eda_charts/03_word_count_by_class.png", dpi=150)
    plt.close()

    if verbose:
        avg_words = df["word_count"].mean()
        print(f"EDA charts saved to notebooks/eda_charts/")
        print(f"Dataset stats: {len(df)} rows, avg {avg_words:.0f} words/post")


if __name__ == "__main__":
    print("=== MindBridge Data Loader ===")
    print()
    X_train, X_test, y_train, y_test = load_and_split(save=True, verbose=True)
    print()
    print("Generating EDA charts...")
    generate_eda_charts()
    print()
    print("Verifying saved splits reload correctly...")
    X_tr2, X_te2, y_tr2, y_te2 = load_splits()
    assert len(X_tr2) == len(X_train), "Split size mismatch after reload"
    print(f"Splits verified: {len(X_tr2)} train, {len(X_te2)} test")
    print()
    print("data_loader.py: ALL PASS")