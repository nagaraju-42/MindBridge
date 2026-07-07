"""
app.py
MindBridge — Streamlit Entry Point & Homepage.
This file is the shell. All ML logic lives in pages/.

Multi-page app structure:
  app.py                      <- Homepage + sidebar (you are here)
  pages/1_Text_Analyser.py    <- Mode 1: paste any text
  pages/2_Student_Form.py     <- Mode 2: 10-question wellbeing form
  pages/3_Dashboard.py        <- Counselor dashboard
  pages/4_Email_Campaign.py   <- Email campaign manager

SRS Reference: PRD Section 4.1 — Student Persona
First-time student shall complete wellbeing form without instructions.
Homepage must be self-explanatory in under 10 seconds of reading.
"""
import streamlit as st
import os

# ── Page Config — MUST be the very first Streamlit call ───────────────────────
st.set_page_config(
    page_title="MindBridge — Anurag University",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "MindBridge — Explainable Mental Health Screening System, Anurag University 2026-27",
    }
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0;'>
        <span style='font-size:40px;'>🧠</span><br>
        <span style='font-size:22px; font-weight:700; color:#a855f7;'>MindBridge</span><br>
        <span style='font-size:12px; color:#888;'>Anurag University</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**Navigate**")
    st.page_link("app.py",                         label="🏠 Home")
    st.page_link("pages/1_Text_Analyser.py",        label="🔍 Text Analyser")
    st.page_link("pages/2_Student_Form.py",         label="📋 Wellbeing Check")
    st.page_link("pages/3_Dashboard.py",            label="📊 Counselor Dashboard")
    st.page_link("pages/4_Email_Campaign.py",       label="📧 Email Campaigns")

    st.divider()
    st.info("🔒 All submissions are completely anonymous.\nNo name or roll number is ever stored.")
    st.divider()
    st.caption("v1.0 · Anurag University CSE 2026-27")


# ── Hero Section ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 32px;
    border: 1px solid #6C5CE7;
    position: relative;
    overflow: hidden;
'>
    <div style='position:relative; z-index:1;'>
        <h1 style='
            color: white;
            font-size: 2.8rem;
            font-weight: 800;
            margin: 0 0 8px;
            background: linear-gradient(135deg, #a855f7, #6C5CE7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        '>🧠 MindBridge</h1>
        <p style='color: #c4b5fd; font-size: 1.1rem; margin: 0 0 8px;'>
            Anurag University — Student Mental Wellbeing Platform
        </p>
        <p style='color: #888; font-size: 0.9rem; margin: 0;'>
            Powered by Explainable AI (SHAP) · Grounded in IEEE Research (Ansari et al., 2023) · 100% Anonymous
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Mode Selection Cards ───────────────────────────────────────────────────────
st.markdown("### Choose How You Want to Check In")
st.markdown("")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid #6C5CE7;
        border-radius: 12px;
        padding: 28px;
        min-height: 220px;
    '>
        <div style='font-size:2.5rem; margin-bottom:12px;'>🔍</div>
        <h3 style='color:#a855f7; margin:0 0 12px; font-size:1.2rem;'>Mode 1: Text Analyser</h3>
        <p style='color:#ccc; font-size:0.9rem; line-height:1.6; margin:0 0 16px;'>
            Paste any social media post, journal entry, or personal thoughts.
            The AI analyses it for depression indicators and explains <em>exactly
            which words</em> influenced the result via SHAP.
        </p>
        <div style='color:#888; font-size:0.82rem;'>
            ⏱️ Under 10 seconds &nbsp;|&nbsp; 📝 Any text up to 15,000 chars
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    if st.button("🔍 Open Text Analyser", type="primary", use_container_width=True, key="btn_text"):
        st.switch_page("pages/1_Text_Analyser.py")

with col2:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #1a2e1a, #14532d);
        border: 1px solid #22c55e;
        border-radius: 12px;
        padding: 28px;
        min-height: 220px;
    '>
        <div style='font-size:2.5rem; margin-bottom:12px;'>📋</div>
        <h3 style='color:#4ade80; margin:0 0 12px; font-size:1.2rem;'>Mode 2: Wellbeing Check</h3>
        <p style='color:#ccc; font-size:0.9rem; line-height:1.6; margin:0 0 16px;'>
            10 simple questions about how you've been feeling this week.
            Takes less than 3 minutes. Designed with PHQ-9 clinical criteria
            mapped to student-friendly language.
        </p>
        <div style='color:#888; font-size:0.82rem;'>
            ⏱️ 2–3 minutes &nbsp;|&nbsp; ❓ 10 questions + 1 open sentence
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    if st.button("📋 Take Wellbeing Check", type="primary", use_container_width=True, key="btn_form"):
        st.switch_page("pages/2_Student_Form.py")

st.divider()

# ── How It Works ──────────────────────────────────────────────────────────────
with st.expander("⚙️ How MindBridge Works — The Science"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **1. Text Input**
        Your text (or form responses) is processed by an NLTK NLP pipeline:
        URL removal, stopword filtering with **negation preservation**, and WordNet lemmatization.
        """)
    with col_b:
        st.markdown("""
        **2. ML Classification**
        TF-IDF vectorization feeds an ensemble of 5 classifiers:
        Logistic Regression, Naive Bayes, SVM, Random Forest, and a soft Voting Classifier.
        Trained on 7,731 Reddit posts.
        """)
    with col_c:
        st.markdown("""
        **3. SHAP Explainability**
        SHAP TreeExplainer computes exact Shapley values — showing *which words*
        drove the prediction and by how much. This is our core contribution beyond
        the anchor IEEE paper (Ansari et al., 2023).
        """)

with st.expander("🔒 Privacy Policy — What We Store (and What We Don't)"):
    st.markdown("""
    | What | Stored? |
    |---|---|
    | Your name | ❌ Never |
    | Your roll number | ❌ Never |
    | Your email address | ❌ Never |
    | Your phone number | ❌ Never |
    | Your form responses | ✅ Yes — but identified only by a random UUID, not your name |
    | Your IP address | ❌ Never |

    **You are identified only by a randomly generated UUID (like a serial number).**
    Even Anurag University cannot trace a submission back to you.
    If you contact a counselor via WhatsApp, *you* initiate it — the system never contacts you.
    """)

with st.expander("📚 Academic Reference — Anchor Paper"):
    st.markdown("""
    MindBridge is grounded in:

    > *Ensemble Hybrid Learning Methods for Automated Depression Detection*
    > L. Ansari et al. — IEEE Transactions on Computational Social Systems, 2023

    **MindBridge extends this paper with three contributions:**
    1. **SHAP explainability** — the anchor paper produces predictions only; we add word-level attribution
    2. **Student wellbeing form** — structured questionnaire input feeding the same ML pipeline
    3. **WhatsApp opt-in bridge** — privacy-preserving counselor connect pathway
    """)

# ── Startup Health Check ───────────────────────────────────────────────────────
@st.cache_resource
def _startup_check():
    """Run silently on first load. Shows errors only if configuration is broken."""
    errors = []
    try:
        from src.model_store import models_exist
        status  = models_exist()
        missing = [k for k, v in status.items() if not v]
        if missing:
            errors.append(
                f"Model files missing: {missing}. "
                "Run: `python src/data_loader.py` then `python src/models.py`"
            )
    except Exception as e:
        errors.append(f"Model store error: {e}")

    try:
        from src.database import test_connection
        if not test_connection():
            errors.append("Supabase connection failed. Check SUPABASE_URL and SUPABASE_KEY in .streamlit/secrets.toml")
    except Exception as e:
        errors.append(f"Database import error: {e}")

    return errors

startup_errors = _startup_check()
if startup_errors:
    st.error("**⚠️ System Configuration Issue**")
    for err in startup_errors:
        st.warning(err)
    st.info("The app may still be usable. Contact the administrator if issues persist.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "MindBridge v1.0 | Anurag University Department of CSE | 2026-2027 | "
    "⚠️ This tool is for **screening purposes only** and does not constitute a clinical diagnosis. "
    "For clinical support, always consult a qualified mental health professional."
)
