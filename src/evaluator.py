"""
src/evaluator.py
MindBridge — Model Evaluation Pipeline.
Satisfies SRS REQ-E01 to REQ-E05.

Produces:
  - Model comparison table (Accuracy, Precision, Recall, F1, AUC-ROC)
  - ROC curves for all models on a single plot
  - Confusion matrix for the best model
  - Per-class classification reports
  - Cross-dataset validation hook (CLPsych) — REQ-E05

Run: python src/evaluator.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

from src.model_store import load_model
from src.data_loader import load_splits
from src.config import MODEL_FILES

RESULTS_DIR  = "notebooks/evaluation_results"
MODELS_TO_EVAL = ["lr", "nb", "svm", "rf", "voting"]
MODEL_LABELS = {
    "lr":     "Logistic Regression",
    "nb":     "Naive Bayes",
    "svm":    "SVM (Linear+Calibrated)",
    "rf":     "Random Forest",
    "voting": "Voting Classifier",
}
COLORS = {
    "lr":     "#2E75B6",
    "nb":     "#1E5631",
    "svm":    "#7F3F00",
    "rf":     "#C00000",
    "voting": "#4A0072",
}


def evaluate_model(model, tfidf, X_test: list, y_test, model_key: str) -> dict:
    """
    Evaluate a single model on the held-out test set.

    Returns:
        Dict of metrics plus internal arrays for plotting.
    """
    name = MODEL_LABELS.get(model_key, model_key)
    print(f"  Evaluating {name}...")

    X_vec = tfidf.transform(X_test)
    if model_key in ("rf", "voting"):
        X_vec = X_vec.toarray()

    y_pred  = model.predict(X_vec)
    y_proba = model.predict_proba(X_vec)[:, 1]

    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "AUC-ROC":   round(roc_auc_score(y_test, y_proba) * 100, 2),
        "_y_pred":  y_pred,
        "_y_proba": y_proba,
    }


def plot_roc_curves(results: dict, y_test):
    """Plot ROC curves for all models on one figure. REQ-E04."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Classifier (AUC=0.50)")

    for key, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["_y_proba"])
        ax.plot(fpr, tpr,
                color=COLORS.get(key, "grey"), linewidth=2,
                label=f"{res['Model']} (AUC={res['AUC-ROC']:.2f}%)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All MindBridge Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves saved: {path}")


def plot_confusion_matrices(results: dict, y_test):
    """Plot confusion matrix for the best model by F1. REQ-E03."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_key = max(results, key=lambda k: results[k]["F1-Score"])
    best = results[best_key]

    cm = confusion_matrix(y_test, best["_y_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Non-Depression", "Depression"],
        yticklabels=["Non-Depression", "Depression"],
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        f"Confusion Matrix — {best['Model']}\n(Best by F1-Score)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = f"{RESULTS_DIR}/confusion_matrix_best.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {path}")

    print(f"\n  Classification Report — {best['Model']}:")
    print(classification_report(
        y_test, best["_y_pred"],
        target_names=["Non-Depression", "Depression"]
    ))


def print_results_table(results: dict) -> pd.DataFrame:
    """Print formatted results table and save to CSV."""
    rows = []
    for key in MODELS_TO_EVAL:
        if key in results:
            r = results[key]
            rows.append([
                r["Model"], r["Accuracy"], r["Precision"],
                r["Recall"], r["F1-Score"], r["AUC-ROC"]
            ])

    df = pd.DataFrame(
        rows,
        columns=["Model", "Accuracy%", "Precision%", "Recall%", "F1%", "AUC-ROC%"]
    )
    df = df.sort_values("F1%", ascending=False).reset_index(drop=True)

    print()
    print("=" * 80)
    print("MINDBRIDGE MODEL COMPARISON RESULTS")
    print("Dataset: Reddit Depression Cleaned | Test set: 20% held-out | Metric: macro F1")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    best = df.iloc[0]
    print(f"\nBest model: {best['Model']} with F1={best['F1%']}%")
    target = 92.0
    if float(best["F1%"]) >= target:
        print(f"TARGET MET: F1 >= {target}% (SRS REQ-E01 target achieved)")
    else:
        gap = target - float(best["F1%"])
        print(f"Target: {target}% F1. Gap: {gap:.2f}% — consider GridSearchCV tuning")
        print("Checklist if F1 < 88%:")
        print("  1. Is negation preservation working? (check preprocessing.py)")
        print("  2. Did TF-IDF fit only on X_train? (data leakage check)")
        print("  3. Is class_weight='balanced' set?")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = f"{RESULTS_DIR}/model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    return df


def run_evaluation() -> dict:
    """Run the complete evaluation pipeline on all trained models."""
    print("=" * 60)
    print("MindBridge Model Evaluation")
    print("=" * 60)

    _, X_test, _, y_test = load_splits()
    tfidf = load_model("tfidf")

    results = {}
    print("\nEvaluating models on held-out test set...")
    for key in MODELS_TO_EVAL:
        try:
            model = load_model(key)
            results[key] = evaluate_model(model, tfidf, X_test, y_test, key)
        except FileNotFoundError:
            print(f"  SKIP {key}: model not found — run python src/models.py first")

    print("\nGenerating visualisations...")
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    print_results_table(results)
    return results


def run_cross_dataset_eval(clpsych_csv_path: str):
    """
    REQ-E05: Cross-dataset validation on CLPsych data.
    CLPsych 2015: https://clpsych.org/shared-task-2015/ (free academic registration)

    Args:
        clpsych_csv_path: Path to CLPsych CSV with text and label columns.
    """
    import pandas as pd
    from src.preprocessing import batch_clean

    print(f"\nCross-dataset validation on CLPsych: {clpsych_csv_path}")
    df = pd.read_csv(clpsych_csv_path)

    text_col  = next((c for c in ["text", "post", "clean_text"] if c in df.columns), None)
    label_col = next((c for c in ["label", "is_depression"] if c in df.columns), None)

    if not text_col or not label_col:
        print(f"  Cannot detect columns. Available: {df.columns.tolist()}")
        return

    df = df.dropna(subset=[text_col, label_col])
    texts  = batch_clean(df[text_col].astype(str).tolist())
    labels = df[label_col].astype(int).values

    tfidf = load_model("tfidf")
    rf    = load_model("rf")

    X_vec  = tfidf.transform(texts).toarray()
    y_pred = rf.predict(X_vec)
    f1     = f1_score(labels, y_pred) * 100

    print(f"  CLPsych F1-Score (Random Forest): {f1:.2f}%")
    print(f"  CLPsych Accuracy: {accuracy_score(labels, y_pred)*100:.2f}%")

    if f1 >= 75:
        print("  Generalisation: ACCEPTABLE (F1 >= 75% on unseen dataset)")
    else:
        print("  Generalisation: BELOW TARGET — discuss in limitations chapter")


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_evaluation()
    print()
    print("evaluator.py: COMPLETE")
    print(f"Charts saved to: {RESULTS_DIR}/")
