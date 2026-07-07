"""
src/models.py
MindBridge — ML Training Pipeline.
Satisfies SRS REQ-F01 to F05, REQ-M01 to M06.

Training order (fastest to slowest — confirms pipeline incrementally):
  1. TF-IDF vectoriser   (~1 second)
  2. Logistic Regression (~2 seconds)
  3. Naive Bayes         (~1 second)
  4. SVM with calibration (~15 seconds)
  5. Random Forest        (~75 seconds) — PRIMARY model for SHAP
  6. Voting Classifier    (~30 seconds)
  7. GridSearchCV on RF   (~8-10 minutes) — optional

Run: python src/models.py --quick    (skip GridSearch for first test)
Run: python src/models.py            (full training with GridSearch)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import time
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

from src.config import (
    TFIDF_NGRAM_RANGE, TFIDF_MAX_FEATURES, TFIDF_SUBLINEAR_TF, TFIDF_MIN_DF,
    RF_N_ESTIMATORS, RF_RANDOM_STATE, RF_CLASS_WEIGHT,
    LR_C, LR_MAX_ITER,
    CV_FOLDS, RANDOM_SEED,
    RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD,
)
from src.model_store import save_model, load_model
from src.data_loader import load_splits


# ── TF-IDF Vectoriser ──────────────────────────────────────────────────────────
def fit_tfidf(X_train: list) -> TfidfVectorizer:
    """
    Fit TF-IDF vectoriser on training data ONLY.
    REQ-F05: Same fitted vectoriser is used at inference — NEVER re-fit on test data.

    Args:
        X_train: List of cleaned text strings from data_loader.

    Returns:
        Fitted TfidfVectorizer saved to models/tfidf_vectorizer.joblib.
    """
    print("[TF-IDF] Fitting vectoriser...")
    t = time.time()
    tfidf = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,    # (1,2) = unigrams + bigrams  REQ-F02
        max_features=TFIDF_MAX_FEATURES,  # 5000 most informative terms REQ-F03
        sublinear_tf=TFIDF_SUBLINEAR_TF,  # log(1+tf) dampening
        min_df=TFIDF_MIN_DF,              # ignore terms in <2 docs
        strip_accents="unicode",
        analyzer="word",
    )
    tfidf.fit(X_train)
    elapsed = time.time() - t
    print(f"[TF-IDF] Fitted in {elapsed:.1f}s | Vocabulary: {len(tfidf.vocabulary_)} terms")
    save_model(tfidf, "tfidf")
    return tfidf


# ── Individual Model Training Functions ───────────────────────────────────────
def train_logistic_regression(X_train_vec, y_train) -> LogisticRegression:
    """Train Logistic Regression. REQ-M01 baseline model 1."""
    print("[LR] Training Logistic Regression...")
    t = time.time()
    lr = LogisticRegression(
        C=LR_C,
        max_iter=LR_MAX_ITER,
        class_weight="balanced",   # REQ-M04
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    lr.fit(X_train_vec, y_train)
    print(f"[LR] Done in {time.time()-t:.1f}s")
    save_model(lr, "lr")
    return lr


def train_naive_bayes(X_train_vec, y_train) -> MultinomialNB:
    """Train Multinomial Naive Bayes. REQ-M01 baseline model 2."""
    print("[NB] Training Multinomial Naive Bayes...")
    t = time.time()
    # MultinomialNB requires non-negative features.
    # TF-IDF values are always >= 0, so this is compatible.
    nb = MultinomialNB(alpha=0.1)  # Laplace smoothing
    nb.fit(X_train_vec, y_train)
    print(f"[NB] Done in {time.time()-t:.1f}s")
    save_model(nb, "nb")
    return nb


def train_svm(X_train_vec, y_train) -> CalibratedClassifierCV:
    """
    Train SVM with probability calibration. REQ-M01 baseline model 3.

    WHY CalibratedClassifierCV:
    LinearSVC is the best-performing SVM for high-dimensional text data
    but does NOT support predict_proba(). The VotingClassifier needs
    probability scores for soft voting. CalibratedClassifierCV wraps
    LinearSVC with Platt scaling, giving both speed and probability output.
    """
    print("[SVM] Training LinearSVC with CalibratedClassifierCV...")
    t = time.time()
    svc = LinearSVC(class_weight="balanced", max_iter=3000, C=1.0)
    svm = CalibratedClassifierCV(svc, cv=3)  # cv=3 speeds up calibration
    svm.fit(X_train_vec, y_train)
    print(f"[SVM] Done in {time.time()-t:.1f}s")
    save_model(svm, "svm")
    return svm


def train_random_forest(X_train_vec, y_train) -> RandomForestClassifier:
    """
    Train Random Forest. REQ-M01 — PRIMARY model for SHAP.
    WHY primary: TreeExplainer gives EXACT Shapley values on RF.
    Note: RF needs dense array (toarray()) — sparse TF-IDF matrix won't work.
    """
    print(f"[RF] Training Random Forest ({RF_N_ESTIMATORS} estimators)...")
    print("[RF] This takes 60-90 seconds on CPU — do not interrupt.")
    t = time.time()
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight=RF_CLASS_WEIGHT,   # "balanced" REQ-M04
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,                      # Use all CPU cores
    )
    rf.fit(X_train_vec.toarray(), y_train)  # RF needs dense
    elapsed = time.time() - t
    print(f"[RF] Done in {elapsed:.1f}s")
    save_model(rf, "rf")
    return rf


def train_voting_classifier(lr, nb, svm, rf, X_train_vec, y_train) -> VotingClassifier:
    """
    Train soft Voting Classifier combining all 4 models. REQ-M02.
    Soft voting averages predicted probabilities — typically most robust.
    """
    print("[Voting] Training Voting Classifier (soft voting)...")
    t = time.time()
    voting = VotingClassifier(
        estimators=[("lr", lr), ("nb", nb), ("svm", svm), ("rf", rf)],
        voting="soft",
    )
    voting.fit(X_train_vec.toarray(), y_train)
    print(f"[Voting] Done in {time.time()-t:.1f}s")
    save_model(voting, "voting")
    return voting


def run_gridsearch_rf(X_train_vec, y_train) -> dict:
    """
    Run GridSearchCV on Random Forest. REQ-M05: 5-fold CV.
    Takes 5-10 minutes. Saves best estimator as models/rf_tuned.joblib.
    """
    print("[GridSearch] Starting RF hyperparameter optimisation (5-10 min)...")
    t = time.time()
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":    [None, 20, 30],
        "max_features": ["sqrt", "log2"],
    }
    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    gs = GridSearchCV(
        rf_base, param_grid,
        cv=CV_FOLDS,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train_vec.toarray(), y_train)
    elapsed = time.time() - t
    print(f"[GridSearch] Done in {elapsed:.0f}s")
    print(f"[GridSearch] Best params: {gs.best_params_}")
    print(f"[GridSearch] Best CV F1: {gs.best_score_*100:.2f}%")
    joblib.dump(gs.best_estimator_, "models/rf_tuned.joblib", compress=3)
    print("[GridSearch] Best RF saved as models/rf_tuned.joblib")
    return {"best_params": gs.best_params_, "best_cv_f1": gs.best_score_}


# ── Main Orchestrator ─────────────────────────────────────────────────────────
def train_all(run_gridsearch: bool = True) -> dict:
    """
    Run the complete training pipeline.

    Args:
        run_gridsearch: If True, run GridSearchCV after base training.
                        Set False for quick smoke test (--quick flag).

    Returns:
        Dict with all trained model objects.
    """
    print("=" * 60)
    print("MindBridge ML Training Pipeline")
    print("=" * 60)
    t_total = time.time()

    print("\n[Step 1/7] Loading data splits...")
    X_train, X_test, y_train, y_test = load_splits()
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[Step 2/7] Fitting TF-IDF vectoriser...")
    tfidf = fit_tfidf(X_train)
    X_train_vec = tfidf.transform(X_train)
    print(f"Feature matrix: {X_train_vec.shape}")

    print("\n[Step 3/7] Training Logistic Regression...")
    lr = train_logistic_regression(X_train_vec, y_train)

    print("\n[Step 4/7] Training Naive Bayes...")
    nb = train_naive_bayes(X_train_vec, y_train)

    print("\n[Step 5/7] Training SVM...")
    svm = train_svm(X_train_vec, y_train)

    print("\n[Step 6/7] Training Random Forest...")
    rf = train_random_forest(X_train_vec, y_train)

    print("\n[Step 7/7] Training Voting Classifier...")
    voting = train_voting_classifier(lr, nb, svm, rf, X_train_vec, y_train)

    total = time.time() - t_total
    print(f"\nAll models trained in {total:.0f}s ({total/60:.1f} minutes)")
    print("Models saved to models/ directory.")

    if run_gridsearch:
        print("\n[GridSearch] Starting hyperparameter optimisation...")
        gs_results = run_gridsearch_rf(X_train_vec, y_train)
    else:
        gs_results = None
        print("[GridSearch] Skipped (--quick mode)")

    return {
        "tfidf": tfidf, "lr": lr, "nb": nb,
        "svm": svm, "rf": rf, "voting": voting,
        "gs_results": gs_results,
    }


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_single(text: str, model=None, tfidf=None) -> dict:
    """
    Predict depression risk for a single text input.
    Used by all Streamlit pages for real-time inference.

    Args:
        text:  Raw or pre-cleaned text string.
        model: Trained classifier (loads RF from disk if None).
        tfidf: Fitted TF-IDF vectoriser (loads from disk if None).

    Returns:
        Dict with keys:
          prediction   (int):   0 = no indicators, 1 = indicators detected
          confidence   (float): 0.0 to 1.0
          risk_level   (str):   'low', 'medium', or 'high'
          label        (str):   Human-readable result label
          cleaned_text (str):   Post-preprocessing text
    """
    from src.preprocessing import clean_text

    if model is None:
        model = load_model("rf")
    if tfidf is None:
        tfidf = load_model("tfidf")

    cleaned = clean_text(text)
    if not cleaned:
        return {
            "prediction": -1,
            "confidence": 0.0,
            "risk_level": "unknown",
            "label": "Insufficient text after processing",
            "cleaned_text": "",
        }

    vector = tfidf.transform([cleaned])

    # RF and Voting need dense arrays
    model_name = type(model).__name__
    if model_name in ("RandomForestClassifier", "VotingClassifier"):
        vector = vector.toarray()

    pred  = int(model.predict(vector)[0])
    proba = float(model.predict_proba(vector)[0][1])  # probability of class 1

    if proba >= RISK_HIGH_THRESHOLD:
        risk = "high"
    elif proba >= RISK_MEDIUM_THRESHOLD:
        risk = "medium"
    else:
        risk = "low"

    return {
        "prediction":   pred,
        "confidence":   round(proba, 4),
        "risk_level":   risk,
        "label":        "Depression Indicators Detected" if pred == 1
                        else "No Depression Indicators",
        "cleaned_text": cleaned,
    }


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    if quick:
        print("Quick mode: skipping GridSearch")

    results = train_all(run_gridsearch=not quick)

    print("\nTesting predict_single() with trained models...")
    TEST_CASES = [
        ("I feel completely hopeless and nothing ever gets better for me. "
         "I am exhausted and empty.", 1),
        ("Great day at cricket practice! Feeling really good and energetic. "
         "Had a wonderful time with friends.", 0),
    ]
    all_pass = True
    for text, expected in TEST_CASES:
        r = predict_single(text)
        status = "PASS" if r["prediction"] == expected else "WARN"
        print(f"  [{status}] \"{text[:50]}...\"")
        print(f"         -> {r['label']} | Confidence: {r['confidence']*100:.1f}% | Risk: {r['risk_level']}")
        if r["prediction"] != expected:
            all_pass = False

    print()
    print("models.py: TRAINING COMPLETE" + (" (all predictions match expected)" if all_pass else " (check predictions)"))

