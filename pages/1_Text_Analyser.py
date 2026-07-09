"""
pages/1_Text_Analyser.py
MindBridge — Text Analyser (Mode 1).
Satisfies SRS REQ-U01 to REQ-U05, REQ-D08 to REQ-D12.

Student flow:
  1. Paste any text
  2. Click Analyse
  3. See: risk level + confidence + SHAP bar chart + top words
  4. (Medium/High risk) WhatsApp connect button
"""
import streamlit as st
import urllib.parse

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Analyser — MindBridge",
    page_icon="🔍",
    layout="wide",
)

# ─── Cached Model Loading ──────────────────────────────────────────────────────
# @st.cache_resource: loads models ONCE, reuses for ALL users.
# Without this, models reload on every button click (15-25 second delay).
@st.cache_resource(show_spinner="🧠 Loading AI models... (30 seconds on first load, instant after)")
def get_models():
    from src.model_store import load_model
    rf    = load_model("rf")
    tfidf = load_model("tfidf")
    return rf, tfidf

@st.cache_resource(show_spinner=False)
def get_explainer(_rf, _tfidf):
    """Cache SHAP TreeExplainer — building takes ~3 seconds."""
    from src.shap_explainer import build_explainer
    return build_explainer(_rf, _tfidf)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0;'>
        <span style='font-size:36px;'>🧠</span><br>
        <span style='font-size:20px; font-weight:700; color:#a855f7;'>MindBridge</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",                   label="🏠 Home")
    st.page_link("pages/1_Text_Analyser.py", label="🔍 Text Analyser")
    st.page_link("pages/2_Student_Form.py",  label="📋 Wellbeing Check")
    st.divider()
    st.markdown("### ⚙️ Advanced Settings")
    enable_llm_safety_net = st.toggle(
        "Enable LLM Safety Net",
        value=False,
        help="When enabled, borderline cases will be sent to a third-party LLM (Llama 3) to check for sarcasm, passive ideation, or hidden context."
    )
    st.divider()
    st.info("🔒 All submissions are anonymous. No name is ever stored.")

# ─── Page Header ──────────────────────────────────────────────────────────────
st.title("🔍 Text Analyser")
st.markdown(
    "Paste any social media post, journal entry, or personal thoughts. "
    "The AI analyses it for depression indicators and explains exactly which words influenced the result."
)
st.divider()

# ─── Load models (cached) ──────────────────────────────────────────────────────
try:
    rf, tfidf = get_models()
    explainer = get_explainer(rf, tfidf)
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.info("Run `python src/models.py` to train and save models first.")
    st.stop()

# ─── Text Input Section ───────────────────────────────────────────────────────
st.markdown("#### Enter Text for Analysis")
st.markdown(
    "You can paste a social media post, a journal entry, or write freely "
    "about how you have been feeling. At least 2–3 sentences works best."
)

raw_text = st.text_area(
    label="Your text:",
    height=180,
    max_chars=15_000,
    placeholder=(
        "Paste your text here or write freely about how you have been feeling...\n\n"
        "Example: I have been feeling really tired lately and nothing seems to excite me "
        "the way it used to. Even small tasks feel completely overwhelming."
    ),
    help="Between 20 and 15,000 characters. Your text is never stored with your name.",
)

char_count = len(raw_text) if raw_text else 0
col_cc, col_btn = st.columns([3, 1])
with col_cc:
    color = "#22c55e" if char_count >= 20 else "#ef4444"
    st.markdown(
        f"<span style='color:{color}; font-size:13px;'>Characters: {char_count} / 15,000</span>",
        unsafe_allow_html=True
    )
with col_btn:
    analyse_clicked = st.button(
        "🧠 Analyse Text",
        type="primary",
        use_container_width=True,
        disabled=(char_count < 1),
    )

# ─── Analysis Pipeline ────────────────────────────────────────────────────────
if analyse_clicked and raw_text:

    # Step 1: Input Validation
    with st.spinner("Validating input..."):
        from src.validator import validate_input, check_confidence
        validation = validate_input(raw_text, tfidf)

    if not validation.valid:
        # REQ-D12: Rejected inputs NOT logged
        st.warning(f"⚠️ {validation.reason}")
        st.info(
            "For best results, write 2–3 sentences in English about how you have "
            "been feeling recently — in your own words, not a transaction or code."
        )
        st.stop()

    text_to_analyse = validation.text or raw_text
    if validation.truncated:
        st.info("✂️ Your text was trimmed to 15,000 characters for analysis.")

    # Step 2: Prediction
    with st.spinner("🧠 Running AI analysis..."):
        from src.models import predict_single
        result = predict_single(text_to_analyse, rf, tfidf)

    # Step 3: LLM Safety Net for Borderline Cases
    llm_reasoning = None
    if enable_llm_safety_net and result["risk_level"] in ["low", "medium"] and 0.35 <= result["confidence"] <= 0.60:
        with st.spinner("🤖 Double-checking with LLM API to prevent false negatives..."):
            from src.llm_evaluator import evaluate_borderline_text
            llm_result = evaluate_borderline_text(text_to_analyse, result["confidence"])
            if llm_result["is_high_risk"]:
                result["risk_level"] = "high"
                result["label"] = "Depressed (LLM Override)"
                result["confidence"] = 0.99  # Override confidence
                llm_reasoning = llm_result["reasoning"]
            else:
                st.toast("🤖 LLM Safety Net verified this text as Low Risk.")

    # Step 3.5: Confidence check (REQ-D10)
    if not check_confidence(result["confidence"]):
        # REQ-D12: Low-confidence NOT logged
        conf_pct = result["confidence"] * 100
        st.warning(f"⚠️ **Borderline Confidence ({conf_pct:.1f}%):** The AI could not determine a clear pattern from your text.")
        st.caption(
            "This can happen when text is very short, or when it contains a complex "
            "mix of positive and negative emotional signals that the model cannot "
            "clearly classify. Analytics are shown below, but may be less reliable."
        )

    # Step 4: SHAP Explanation
    with st.spinner("✨ Generating word explanation..."):
        from src.shap_explainer import explain_prediction, plot_shap_bar
        cleaned    = result["cleaned_text"]
        shap_out   = explain_prediction(cleaned, rf, tfidf, explainer)
        chart_bytes = plot_shap_bar(shap_out["top_words"])

    # Step 5: Save to Supabase (only valid + sufficient confidence — REQ-D12)
    try:
        from src.database import save_submission
        saved_row = save_submission({
            "department":    "Text Analyser Mode",
            "mode":          "text_analyser",
            "generated_text": cleaned[:500],
            "prediction":    result["prediction"],
            "confidence":    result["confidence"],
            "risk_level":    result["risk_level"],
            "shap_top_words": str(shap_out["top_words"][:5]),
        })
        student_token = saved_row.get("id", "unavailable")
    except Exception:
        student_token = "unavailable"

    # Step 6: Display Results
    st.divider()
    st.markdown("## 📊 Analysis Result")

    # Risk level colour-coded banner (REQ-U05)
    risk = result["risk_level"]
    RISK_CONFIG = {
        "low":    ("success", "🟢", "Low Risk",    "Your text does not show strong depression indicators."),
        "medium": ("warning", "🟡", "Medium Risk", "Some indicators detected. Consider speaking with someone you trust."),
        "high":   ("error",   "🔴", "High Risk",  "Strong indicators detected. Please consider reaching out for support."),
    }
    alert_fn, icon, risk_label, msg = RISK_CONFIG.get(risk, RISK_CONFIG["low"])
    getattr(st, alert_fn)(f"{icon} **{risk_label}** — {result['label']}")
    st.markdown(msg)

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction",   result["label"].split()[0])
    m2.metric("AI Confidence", f"{result['confidence']*100:.1f}%")
    m3.metric("Risk Level",    risk_label)
    
    if llm_reasoning:
        st.error(f"**LLM Safety Net Triggered:** {llm_reasoning}")

    # SHAP Explanation (REQ-U03, REQ-S06)
    st.markdown("#### ✨ What Drove This Prediction?")
    st.markdown(
        "The chart shows which words influenced the AI decision. "
        "**Red bars** pushed toward depression. **Blue bars** pushed toward non-depression. "
        "Longer bar = stronger influence."
    )
    if chart_bytes:
        st.image(chart_bytes, caption="SHAP Word Attribution — MindBridge Explainability Layer")

    # Top words text (REQ-S06: top 5 displayed)
    top5 = shap_out["top_words"][:5]
    if top5:
        with st.expander("📝 View top 5 influencing words (text format)"):
            for word, val in top5:
                direction = "toward depression" if val > 0 else "away from depression"
                st.markdown(f"- **{word}**: SHAP {val:+.3f} → pushes *{direction}*")

    # Anonymous token
    if student_token and student_token != "unavailable":
        st.info(f"🔑 Your anonymous reference ID: **{str(student_token)}**")
        st.caption("Save this ID if you want a counselor to view your detailed report.")

    # Disclaimer (REQ-U04: mandatory on every result)
    st.caption(
        "⚠️ **Important:** This is a **screening tool only** — not a medical diagnosis. "
        "If you are experiencing distress, please speak with a qualified mental health professional."
    )

    # Step 7: WhatsApp Connect (Medium/High risk)
    if risk in ("medium", "high"):
        st.divider()
        from src.config import DEPARTMENT_GROUPS

        if risk == "high":
            st.markdown("### 💙 You Don't Have to Face This Alone")
            st.markdown(
                "Strong indicators were detected. A counselor at Anurag University "
                "is available to speak with you **privately and anonymously**. "
                "Your name is never shared — only your anonymous ID, and only if *you* choose."
            )
        else:
            st.markdown("### 💬 A Counselor Is Available")
            st.markdown(
                "If you'd like to speak with someone privately, "
                "a counselor is available to help."
            )

        dept = st.selectbox(
            "Select your department to reach the right counselor:",
            options=list(DEPARTMENT_GROUPS.keys()),
            key="dept_select_ta",
        )
        counselor_info = DEPARTMENT_GROUPS[dept]

        wa_message = (
            f"Hi Counselor, I would like to speak with you privately. "
            f"My MindBridge anonymous ID is: {student_token}. "
            f"Risk level shown to me: {risk_label}. "
            f"I am comfortable with you viewing my report."
        )
        wa_url = f"https://wa.me/{counselor_info['wa']}?text={urllib.parse.quote(wa_message)}"

        st.link_button(
            "💬 Message Counselor on WhatsApp",
            wa_url,
            type="primary",
            use_container_width=True,
        )
        st.caption(
            "Tapping this opens WhatsApp with a pre-written message containing only your anonymous ID. "
            "Your name is NOT included. You decide whether to send it."
        )

        if counselor_info.get("calendly"):
            st.divider()
            st.markdown("**Or schedule a meeting directly:**")
            st.link_button(
                "📅 Book a Private Meeting (Calendly)",
                counselor_info["calendly"],
                use_container_width=True,
            )
    else:
        st.divider()
        st.markdown("### 🌟 Keep up the great work!")
        st.markdown(
            "It looks like you're doing well right now. Remember that mental health is an ongoing journey. "
            "If you ever feel overwhelmed in the future, the **MindBridge** resources are always here for you."
        )
        st.balloons()

    # Student resources
    with st.expander("ℹ️ Wellbeing Resources"):
        st.markdown("""
        - **iCall** (TISS): 9152987821 | Free counseling helpline
        - **Vandrevala Foundation**: 1860-2662-345 | 24/7 mental health helpline
        - **NIMHANS**: 080-46110007
        - **iCall Online Chat**: https://icallhelpline.org
        - **Anurag University Psychology Dept**: Visit Student Services office
        """)

# ─── EDA WordClouds (Real-time Generation) ──────────────────────────────────
st.divider()
st.markdown("## ☁️ Dataset Lexicon (Word Clouds)")
with st.expander("View Global Dataset Word Clouds", expanded=True):
    st.markdown("These word clouds are generated dynamically from the underlying `reddit_depression.csv` dataset, showing the most frequent words in depressed vs. non-depressed texts.")
    
    from src.wordcloud_generator import generate_wordclouds
    import os
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'reddit_depression.csv')
    sad_img, happy_img = generate_wordclouds(csv_path)
    
    if sad_img and happy_img:
        col_sad, col_happy = st.columns(2)
        with col_sad:
            st.markdown("### 🔴 Depression Markers")
            st.image(sad_img, use_container_width=True)
        with col_happy:
            st.markdown("### 🟢 Non-Depression Markers")
            st.image(happy_img, use_container_width=True)
    else:
        st.error("Could not generate Word Clouds. Please check the dataset.")
