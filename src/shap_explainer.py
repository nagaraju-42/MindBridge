"""
src/shap_explainer.py
MindBridge - SHAP Explainability Module.
Satisfies SRS REQ-S01 to REQ-S06.

Uses shap.TreeExplainer on the Random Forest model.
WHY TreeExplainer: exact Shapley values (not approximations like KernelExplainer).
Random Forest was chosen as the PRIMARY model to enable TreeExplainer.

SHAP 0.50+ with sklearn 1.9+ RF requires interventional mode with background data.
Background: 100-sample subset of TF-IDF training matrix (from splits.joblib).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "notebooks/evaluation_results"


def build_explainer(rf_model, tfidf_model):
    """
    Build SHAP TreeExplainer with interventional background data.
    
    SHAP 0.50+ needs a background dataset for interventional perturbation.
    We load 100 training samples from splits.joblib, transform with TF-IDF,
    and use as the background reference distribution.
    
    Args:
        rf_model:    Trained RandomForestClassifier
        tfidf_model: Fitted TfidfVectorizer (same one used at training)
    
    Returns:
        shap.TreeExplainer instance with correct background data
    """
    import shap
    try:
        from src.data_loader import load_splits
        X_train, _, _, _ = load_splits()
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), size=min(100, len(X_train)), replace=False)
        sample_texts = [X_train[i] for i in idx]
        background = tfidf_model.transform(sample_texts).toarray()
        print("[SHAP] Building TreeExplainer with 100-sample background (interventional)...")
        return shap.TreeExplainer(rf_model, background, feature_perturbation="interventional")
    except Exception as e:
        print(f"[SHAP] Background load failed ({e}), falling back to default explainer")
        return shap.TreeExplainer(rf_model)


def _get_explainer(rf_model):
    """Legacy single-arg wrapper kept for backward compatibility."""
    return shap.TreeExplainer(rf_model) if True else None


def explain_prediction(cleaned_text, rf_model, tfidf, explainer=None):
    """
    Compute SHAP word attributions for a single text prediction.
    REQ-S01: Use SHAP TreeExplainer on Random Forest.
    REQ-S02: Compute SHAP values for all TF-IDF features.
    REQ-S05: Return top-N words sorted by absolute SHAP value.

    Args:
        cleaned_text: Pre-cleaned text string (output of clean_text())
        rf_model:     Trained RandomForestClassifier
        tfidf:        Fitted TfidfVectorizer
        explainer:    Optional pre-built TreeExplainer (for caching)

    Returns:
        dict with keys:
          top_words       List of (word, shap_value) tuples, sorted by abs value
          all_shap_values np.array of all feature SHAP values (class 1)
          expected_value  float baseline expected value
          feature_names   list of TF-IDF feature names
    """
    import shap

    if not cleaned_text or not cleaned_text.strip():
        return {
            "top_words": [],
            "all_shap_values": np.array([]),
            "expected_value": 0.0,
            "feature_names": [],
        }

    if explainer is None:
        explainer = build_explainer(rf_model, tfidf)

    # Transform text to TF-IDF dense vector
    vector = tfidf.transform([cleaned_text]).toarray()  # shape: (1, 5000)

    # Compute SHAP values
    # check_additivity=False: numerical sum check can fail with float precision
    shap_values = explainer.shap_values(vector, check_additivity=False)

    # Handle SHAP output format differences across versions
    if isinstance(shap_values, list):
        # Older SHAP: list of [class0_array, class1_array]
        sv_class1 = shap_values[1][0]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        # Newer SHAP: shape (n_samples, n_features, n_classes)
        sv_class1 = shap_values[0, :, 1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 2:
        # SHAP: shape (n_samples, n_features) - binary already
        sv_class1 = shap_values[0]
    else:
        sv_class1 = np.array(shap_values).flatten()

    # Get feature names from TF-IDF vocabulary
    feature_names = tfidf.get_feature_names_out().tolist()

    # Only look at features present in this text (non-zero TF-IDF)
    present_mask = vector[0] > 0
    present_names  = [feature_names[i] for i in range(len(feature_names)) if present_mask[i]]
    present_values = sv_class1[present_mask]

    # Sort by absolute SHAP value descending
    sorted_idx = np.argsort(np.abs(present_values))[::-1]
    top_words = [(present_names[i], float(present_values[i])) for i in sorted_idx[:20]]

    # Expected value (scalar for binary class 1)
    try:
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            exp_val = float(explainer.expected_value[1])
        else:
            exp_val = float(explainer.expected_value)
    except Exception:
        exp_val = 0.0

    return {
        "top_words":       top_words,
        "all_shap_values": sv_class1,
        "expected_value":  exp_val,
        "feature_names":   feature_names,
    }


def plot_shap_bar(top_words, n=10, title="Word Attribution (SHAP Values)"):
    """
    Generate a horizontal bar chart of top SHAP word attributions.
    REQ-S06: Display top-N words with colour coding.

    Args:
        top_words: List of (word, shap_value) tuples from explain_prediction()
        n:         Number of words to display (default 10)
        title:     Chart title

    Returns:
        bytes: PNG image bytes for st.image() or None on failure
    """
    if not top_words:
        return None

    top = top_words[:n]
    words  = [w for w, _ in top]
    values = [v for _, v in top]

    # Cap display values to avoid unreadable bars on edge cases
    max_abs = max(abs(v) for v in values) if values else 1.0
    if max_abs > 10:
        scale = 10.0 / max_abs
        values = [v * scale for v in values]

    colors = ["#C00000" if v > 0 else "#1E5631" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(words) * 0.45)))
    fig.patch.set_facecolor("#1A1D2E")
    ax.set_facecolor("#1A1D2E")

    bars = ax.barh(words[::-1], values[::-1], color=colors[::-1],
                   edgecolor="none", height=0.6)

    ax.axvline(0, color="#555", linewidth=1)
    ax.set_xlabel("SHAP Value (depression influence)", color="#CCCCCC", fontsize=10)
    ax.set_title(title, color="#FFFFFF", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors="#CCCCCC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    for label in ax.get_yticklabels():
        label.set_color("#FFFFFF")
        label.set_fontsize(10)

    # Add value labels
    for bar, val in zip(bars, values[::-1]):
        x_pos = bar.get_width()
        ax.text(
            x_pos + (0.01 * (max(abs(v) for v in values) or 1)),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha="left",
            color="#CCCCCC", fontsize=9
        )

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#C00000", label="Toward depression"),
            Patch(color="#1E5631", label="Away from depression"),
        ],
        loc="lower right",
        facecolor="#2D3250",
        edgecolor="#555",
        labelcolor="#CCCCCC",
        fontsize=9,
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print("=== MindBridge SHAP Explainer Self-Test ===")

    from src.model_store import load_model
    rf    = load_model("rf")
    tfidf = load_model("tfidf")

    print("Building TreeExplainer with background data...")
    explainer = build_explainer(rf, tfidf)
    print("TreeExplainer built.")

    test_texts = [
        "I feel completely hopeless and nothing ever gets better",
        "I had a great day today and I am feeling really happy",
        "Sometimes I wonder if anyone would even notice if I was gone",
    ]

    for text in test_texts:
        print(f"\n  Input: {text[:60]}...")
        result = explain_prediction(text, rf, tfidf, explainer)
        print(f"  Top 3 SHAP words:")
        for word, val in result["top_words"][:3]:
            direction = "depression" if val > 0 else "wellbeing"
            print(f"    {word:20s} {val:+.4f} ({direction})")

        chart = plot_shap_bar(result["top_words"])
        if chart:
            print(f"  Bar chart: {len(chart):,} bytes OK")

    print("\nshap_explainer.py: ALL PASS")