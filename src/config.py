"""
src/config.py
MindBridge — Single Source of Truth for all configuration.
Satisfies SRS NFR-M02: all constants defined in one place.

Loads secrets from Streamlit secrets (production) with
environment variable fallback for standalone testing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

# ─── Supabase ──────────────────────────────────────────────────────────────────
try:
    import streamlit as st
    SUPABASE_URL  = st.secrets.get("SUPABASE_URL",  os.getenv("SUPABASE_URL",  "NOT_SET"))
    SUPABASE_KEY  = st.secrets.get("SUPABASE_KEY",  os.getenv("SUPABASE_KEY",  "NOT_SET"))
    SUPABASE_SERVICE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_SERVICE_KEY", "NOT_SET"))
    RESEND_API_KEY = st.secrets.get("RESEND_API_KEY", os.getenv("RESEND_API_KEY", "NOT_SET"))
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "NOT_SET"))
    EMAIL_FROM    = st.secrets.get("EMAIL_FROM",    "MindBridge <noreply@example.com>")
    COUNSELOR_PASSWORD = st.secrets.get("COUNSELOR_PASSWORD", "admin")
    # Counselor WhatsApp numbers
    COUNSELOR_CSE_WA  = st.secrets.get("COUNSELOR_CSE_WA",  "919999999901")
    COUNSELOR_ECE_WA  = st.secrets.get("COUNSELOR_ECE_WA",  "919999999902")
    COUNSELOR_MECH_WA = st.secrets.get("COUNSELOR_MECH_WA", "919999999903")
    # Calendly (optional)
    CALENDLY_CSE  = st.secrets.get("CALENDLY_CSE",  "")
    CALENDLY_ECE  = st.secrets.get("CALENDLY_ECE",  "")
    CALENDLY_MECH = st.secrets.get("CALENDLY_MECH", "")
except Exception:
    # Standalone (no Streamlit context) — fall back to env vars
    SUPABASE_URL  = os.getenv("SUPABASE_URL",  "NOT_SET")
    SUPABASE_KEY  = os.getenv("SUPABASE_KEY",  "NOT_SET")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "NOT_SET")
    RESEND_API_KEY = os.getenv("RESEND_API_KEY", "NOT_SET")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "NOT_SET")
    EMAIL_FROM    = os.getenv("EMAIL_FROM",    "MindBridge <noreply@example.com>")
    COUNSELOR_PASSWORD = os.getenv("COUNSELOR_PASSWORD", "admin")
    COUNSELOR_CSE_WA  = os.getenv("COUNSELOR_CSE_WA",  "919999999901")
    COUNSELOR_ECE_WA  = os.getenv("COUNSELOR_ECE_WA",  "919999999902")
    COUNSELOR_MECH_WA = os.getenv("COUNSELOR_MECH_WA", "919999999903")
    CALENDLY_CSE  = os.getenv("CALENDLY_CSE",  "")
    CALENDLY_ECE  = os.getenv("CALENDLY_ECE",  "")
    CALENDLY_MECH = os.getenv("CALENDLY_MECH", "")

# ─── Input Validation Constants (SRS REQ-D01, D08-D12) ────────────────────────
MIN_INPUT_CHARS    = 20       # REQ-D11: reject if fewer than 20 chars
MAX_INPUT_CHARS    = 15_000   # REQ-D01: truncate (not reject) if over 15K
MAX_NUMERIC_RATIO  = 0.30     # REQ-D09: reject if >30% digits (payment data)
MIN_VOCAB_TOKENS   = 5        # REQ-D08: reject if <5 in-vocabulary tokens
LOW_CONF_THRESHOLD = 0.50     # REQ-D10: low-confidence notice below 50%

# ─── TF-IDF Parameters (SRS REQ-F01 to F04) ───────────────────────────────────
TFIDF_NGRAM_RANGE  = (1, 2)   # REQ-F02: unigrams and bigrams
TFIDF_MAX_FEATURES = 5_000    # REQ-F03: top 5000 terms
TFIDF_SUBLINEAR_TF = True     # log(1+tf) dampening
TFIDF_MIN_DF       = 2        # ignore terms appearing in <2 docs

# ─── Random Forest Parameters (SRS REQ-M01, M04) ──────────────────────────────
RF_N_ESTIMATORS    = 200
RF_RANDOM_STATE    = 42
RF_CLASS_WEIGHT    = "balanced"  # REQ-M04

# ─── Logistic Regression Parameters ───────────────────────────────────────────
LR_C               = 1.0
LR_MAX_ITER        = 1_000

# ─── Training Parameters (SRS REQ-M03, M05) ───────────────────────────────────
TEST_SIZE          = 0.20    # REQ-M03: 80/20 split
RANDOM_SEED        = 42      # REQ-M03: reproducibility
CV_FOLDS           = 5       # REQ-M05: 5-fold GridSearchCV

# ─── Risk Thresholds ──────────────────────────────────────────────────────────
RISK_HIGH_THRESHOLD   = 0.75  # confidence >= 0.75 → High Risk
RISK_MEDIUM_THRESHOLD = 0.55  # confidence >= 0.55 → Medium Risk
                               # confidence < 0.55  → Low Risk

# ─── File Paths ────────────────────────────────────────────────────────────────
DATA_PATH          = "data/reddit_depression.csv"
SPLITS_PATH        = "data/train_test_splits/splits.joblib"
MODEL_DIR          = "models/"

MODEL_FILES = {
    "tfidf":  "models/tfidf_vectorizer.joblib",
    "lr":     "models/logistic_regression.joblib",
    "nb":     "models/naive_bayes.joblib",
    "svm":    "models/svm_calibrated.joblib",
    "rf":     "models/random_forest.joblib",
    "voting": "models/voting_classifier.joblib",
}

# ─── Supabase Table Names ──────────────────────────────────────────────────────
TABLE_SUBMISSIONS  = "submissions"
TABLE_CAMPAIGNS    = "campaigns"
TABLE_COUNSELORS   = "counselors"

# ─── Department Groups → Counselor Routing ────────────────────────────────────
# Each dept cluster maps to one counselor's WhatsApp + optional Calendly
DEPARTMENT_GROUPS = {
    "CSE / IT / AI / DS": {
        "name":     "Counselor (CSE Cluster)",
        "wa":       COUNSELOR_CSE_WA,
        "calendly": CALENDLY_CSE,
    },
    "ECE / EEE": {
        "name":     "Counselor (ECE Cluster)",
        "wa":       COUNSELOR_ECE_WA,
        "calendly": CALENDLY_ECE,
    },
    "Mech / Civil / Other": {
        "name":     "Counselor (Engineering Cluster)",
        "wa":       COUNSELOR_MECH_WA,
        "calendly": CALENDLY_MECH,
    },
}

# ─── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== MindBridge Config Self-Test ===")
    print(f"SUPABASE_URL: {'SET' if SUPABASE_URL != 'NOT_SET' else 'NOT SET'}")
    print(f"SUPABASE_KEY: {'SET' if SUPABASE_KEY != 'NOT_SET' else 'NOT SET'}")
    print(f"RESEND_API_KEY: {'SET' if RESEND_API_KEY != 'NOT_SET' else 'PLACEHOLDER'}")
    print(f"MIN_INPUT_CHARS: {MIN_INPUT_CHARS}")
    print(f"MAX_INPUT_CHARS: {MAX_INPUT_CHARS}")
    print(f"TFIDF_MAX_FEATURES: {TFIDF_MAX_FEATURES}")
    print(f"RF_N_ESTIMATORS: {RF_N_ESTIMATORS}")
    print(f"RISK_HIGH_THRESHOLD: {RISK_HIGH_THRESHOLD}")
    print(f"Department groups: {list(DEPARTMENT_GROUPS.keys())}")
    print("config.py: PASS")
