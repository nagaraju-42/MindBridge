# MindBridge
## Explainable AI-Powered Student Mental Wellbeing System
**Anurag University — Department of CSE | 2026-2027**

---

## Overview

MindBridge is a privacy-first explainable depression screening system for Anurag University students. It uses ensemble machine learning (TF-IDF + LR/NB/SVM/RF/Voting) with SHAP-based word-level explanations. Students can analyse text or complete a 10-question wellbeing form. Results are completely anonymous.

**Key Innovations vs anchor paper (Ansari et al., IEEE 2023):**
- SHAP TreeExplainer — per-word attribution (anchor paper has none)
- Student Wellbeing Form — PHQ-9 mapped questionnaire → same ML pipeline
- WhatsApp Click-to-Chat bridge — student-initiated, fully privacy-preserving

---

## Setup Guide

### 1. Clone and enter repo
```bash
git clone <your-repo-url>
cd MindBridge
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (one-time)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5. Fill secrets
Edit `.streamlit/secrets.toml` — replace placeholder values:
- Supabase credentials (already filled — your project)
- Resend API key (from resend.com)
- Real counselor WhatsApp numbers (91XXXXXXXXXX format)
- Dashboard password

### 6. Run Supabase SQL Schema
Go to your Supabase Dashboard → SQL Editor → paste and run the entire contents of `supabase_schema.sql`.

### 7. Download the Dataset

**Reddit Depression Cleaned Dataset:**
1. Visit: https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
2. Click **Download** → extract the zip
3. Rename/copy the CSV to: `data/reddit_depression.csv`

### 8. Preprocess and train models
```bash
# Step 8a: Clean text and create 80/20 splits (~25-30 seconds)
python src/data_loader.py

# Step 8b: Train all 5 models + GridSearchCV (~8-12 minutes total)
python src/models.py

# Quick test (skip GridSearch, ~2 minutes):
python src/models.py --quick
```

### 9. Run the app
```bash
streamlit run app.py
```

The app opens at http://localhost:8501

---

## File Structure

```
MindBridge/
├── .streamlit/
│   ├── config.toml          # Dark theme config
│   └── secrets.toml         # Credentials (gitignored)
├── src/                     # Backend Python modules
│   ├── config.py            # All constants (single source of truth)
│   ├── preprocessing.py     # NLTK NLP pipeline (negation-preserving)
│   ├── validator.py         # 5-check input gateway
│   ├── model_store.py       # Save/load .joblib files
│   ├── data_loader.py       # Dataset loading and 80/20 splits
│   ├── models.py            # TF-IDF + 5 classifiers + GridSearchCV
│   ├── evaluator.py         # Metrics, ROC curves, confusion matrix
│   ├── shap_explainer.py    # SHAP TreeExplainer (EXACT Shapley values)
│   ├── database.py          # Supabase CRUD (no PII ever)
│   └── email_service.py     # Resend email campaigns
├── pages/
│   ├── 1_Text_Analyser.py   # Mode 1: free text analysis
│   ├── 2_Student_Form.py    # Mode 2: 10-Q wellbeing form
│   ├── 3_Dashboard.py       # Counselor dashboard (password-gated)
│   └── 4_Email_Campaign.py  # Email campaign manager
├── distilbert/
│   └── train_distilbert.py  # GPU DistilBERT fine-tuning (comparison)
├── models/                  # Trained .joblib files (gitignored)
├── data/                    # CSV and splits (gitignored)
├── notebooks/
│   ├── eda_charts/          # EDA PNGs for report
│   └── evaluation_results/  # ROC curves, confusion matrix PNGs
├── app.py                   # Streamlit entry point + homepage
├── requirements.txt
├── supabase_schema.sql      # Run in Supabase SQL Editor
└── README.md
```

---

## Module Self-Tests

Each module has a built-in self-test. Run in order:

```bash
python src/config.py           # Check constants + Supabase URL
python src/preprocessing.py    # Negation preservation, URL removal tests
python src/validator.py        # Short text, payment data, truncation tests
python src/model_store.py      # Check model files exist
python src/database.py         # Supabase INSERT + SELECT roundtrip
python src/shap_explainer.py   # SHAP explanation on test text
python src/evaluator.py        # Full model comparison table
```

---

## Kaggle Dataset Flow

```
1. Go to: https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
2. Sign in to Kaggle (free account)
3. Click Download → downloads depression-reddit-cleaned.zip
4. Extract → you get a CSV file
5. Copy it to: MindBridge/data/reddit_depression.csv
6. Run: python src/data_loader.py
```

Kaggle CLI alternative (if you have API configured):
```bash
kaggle datasets download -d infamouscoder/depression-reddit-cleaned -p data/ --unzip
```

---

## DistilBERT Training (Optional — for report Chapter 5)

```bash
# Requires GPU (CUDA). ~90 minutes on T4/RTX 3060.
python distilbert/train_distilbert.py

# Results saved to: distilbert/results/distilbert_metrics.json
# Add these numbers to your model comparison table.
```

**Google Colab (free GPU):**
1. Upload this repo to Google Drive
2. Open Colab → mount Drive → run `train_distilbert.py`

---

## Deployment (Streamlit Community Cloud)

1. Push all code to GitHub (models/, data/ are gitignored — ✅ safe)
2. Go to https://share.streamlit.io → New App → connect your repo
3. Set **Main file path**: `app.py`
4. Go to **Advanced settings → Secrets** → paste the entire contents of `secrets.toml`
5. Click **Deploy** — app starts in ~3 minutes

---

## Reference

> L. Ansari et al., *Ensemble Hybrid Learning Methods for Automated Depression Detection*, IEEE TCSS, 2023.

MindBridge extends this paper with SHAP explainability, structured form input, and WhatsApp counselor connect.

---

## Academic Disclaimer

⚠️ MindBridge is a **screening tool only** — not a clinical diagnosis. Results are based on pattern matching in text. For clinical support, always consult a qualified mental health professional.

---

*Anurag University | Department of CSE | 2026-2027*
