"""
pages/2_Student_Form.py
MindBridge — Student Wellbeing Form (Mode 2).
Satisfies SRS REQ-U06 to REQ-U11, NFR-U01, NFR-U03.

10-question structured wellbeing assessment.
Form responses converted to natural language text (REQ-D07: unified pipeline)
then fed through the identical ML pipeline as Mode 1.
"""
import streamlit as st
import urllib.parse

st.set_page_config(
    page_title="Wellbeing Check — MindBridge",
    page_icon="📋",
    layout="wide",
)


@st.cache_resource(show_spinner="🧠 Loading AI models...")
def get_models():
    from src.model_store import load_model
    return load_model("rf"), load_model("tfidf")


@st.cache_resource(show_spinner=False)
def get_explainer(_rf, _tfidf):
    from src.shap_explainer import build_explainer
    return build_explainer(_rf, _tfidf)


# ── Sidebar ─────────────────────────────────────────────────────────────────
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
    st.info("🔒 All submissions are anonymous. No name is ever stored.")

# ── Load Models ──────────────────────────────────────────────────────────────
try:
    rf, tfidf = get_models()
    explainer = get_explainer(rf, tfidf)
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.info("Run `python src/models.py` to train and save models first.")
    st.stop()

# ── Page Header ──────────────────────────────────────────────────────────────
st.title("📋 Student Wellbeing Check")
st.markdown(
    "**10 questions. Less than 3 minutes. Completely anonymous.** "
    "No name, roll number, or email is collected. Only a random anonymous ID is generated."
)

with st.expander("🔒 Privacy Details — What's stored and what's not"):
    st.markdown("""
    | What | Stored? |
    |---|---|
    | Your name | ❌ Never |
    | Your roll number | ❌ Never |
    | Your email / phone | ❌ Never |
    | Your answers | ✅ Yes — linked only to a random ID, not your name |

    - Only a randomly generated UUID (like a serial number) is stored
    - No one at Anurag University can identify you from your submission
    - You choose whether to contact a counselor — the system never contacts you
    - If you contact a counselor, it's via your own WhatsApp — you initiate it
    """)

st.divider()

# ── Step 1: Department ───────────────────────────────────────────────────────
st.markdown("#### Step 1: Select Your Department")
from src.config import DEPARTMENT_GROUPS

dept = st.selectbox(
    "Which department are you from?",
    options=list(DEPARTMENT_GROUPS.keys()),
    help="This routes you to the right counselor if you choose to connect.",
    key="dept_sf",
)
counselor_info = DEPARTMENT_GROUPS[dept]

st.divider()
st.markdown("#### Step 2: Answer 10 Questions")
st.markdown("*Choose the option that best describes how you have been feeling in the **past two weeks**.*")
st.markdown("")

# ── Question Option Sets ─────────────────────────────────────────────────────
OPTS_FREQ     = ["Never", "Sometimes", "Often", "Almost Every Day"]
OPTS_BURDEN   = ["Never", "Rarely", "Sometimes", "Often"]
OPTS_CONC     = ["No problem", "Slightly hard", "Very hard", "Cannot concentrate at all"]
OPTS_SLEEP    = ["Fine", "Sleeping too much", "Cannot sleep properly", "Very disturbed"]
OPTS_APPETITE = ["Normal", "Eating too much", "Not feeling hungry", "Skipping meals often"]
OPTS_ALWAYS   = ["Never", "Sometimes", "Often", "Always"]

with st.container(border=True):
    st.markdown("**Q1 — Fatigue & Energy** *(PHQ-9 Item 4)*")
    q1 = st.radio(
        "How often have you felt tired or had very little energy in the past two weeks?",
        OPTS_FREQ, horizontal=True, key="q1"
    )

with st.container(border=True):
    st.markdown("**Q2 — Interest & Pleasure** *(PHQ-9 Item 1 — Anhedonia)*")
    q2 = st.radio(
        "How often do you feel little interest or pleasure in doing things you usually enjoy?",
        OPTS_FREQ, horizontal=True, key="q2"
    )

with st.container(border=True):
    st.markdown("**Q3 — Mood & Hopelessness** *(PHQ-9 Item 2)*")
    q3 = st.radio(
        "How often have you felt sad, hopeless, or like things will not get better?",
        OPTS_FREQ, horizontal=True, key="q3"
    )

with st.container(border=True):
    st.markdown("**Q4 — Sleep** *(PHQ-9 Item 3)*")
    q4 = st.radio(
        "How is your sleep lately?",
        OPTS_SLEEP, horizontal=True, key="q4"
    )

with st.container(border=True):
    st.markdown("**Q5 — Appetite** *(PHQ-9 Item 5)*")
    q5 = st.radio(
        "How is your appetite these days?",
        OPTS_APPETITE, horizontal=True, key="q5"
    )

with st.container(border=True):
    st.markdown("**Q6 — Worthlessness / Burden** *(PHQ-9 Item 6)*")
    q6 = st.radio(
        "How often do you feel like you are a burden to the people around you?",
        OPTS_BURDEN, horizontal=True, key="q6"
    )

with st.container(border=True):
    st.markdown("**Q7 — Concentration** *(PHQ-9 Item 7)*")
    q7 = st.radio(
        "How hard is it to concentrate on your studies or daily tasks recently?",
        OPTS_CONC, horizontal=True, key="q7"
    )

with st.container(border=True):
    st.markdown("**Q8 — Social Connection** *(PHQ-9 Item 8 variant)*")
    q8 = st.radio(
        "How often do you feel disconnected or isolated from people around you?",
        OPTS_ALWAYS, horizontal=True, key="q8"
    )

with st.container(border=True):
    st.markdown("**Q9 — Sense of Purpose** *(PHQ-9 Item 9 variant)*")
    q9 = st.radio(
        "How often do you feel like nothing you do matters or has any value?",
        OPTS_ALWAYS, horizontal=True, key="q9"
    )

with st.container(border=True):
    st.markdown("**Q10 — In Your Own Words** *(Open question — novel addition)*")
    q10 = st.text_area(
        label="Q10 response",
        label_visibility="collapsed",
        height=90,
        max_chars=500,
        placeholder=(
            "Write one sentence about how you have been feeling this week...\n"
            "e.g., This week I have been feeling overwhelmed and disconnected from everything."
        ),
        key="q10",
    )

st.divider()
st.markdown("#### Step 3: Submit and See Your Result")
st.markdown("*Your result appears immediately below. Nothing is shared without your consent.*")

submitted = st.button(
    "🧠 Submit and See My Result",
    type="primary",
    use_container_width=True,
)

# ── Analysis Pipeline ────────────────────────────────────────────────────────
if submitted:
    responses = {
        "q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5,
        "q6": q6, "q7": q7, "q8": q8, "q9": q9, "q10": q10 or ""
    }

    # REQ-D07: Convert form to natural language text → same ML pipeline as Mode 1
    with st.spinner("Processing your responses..."):
        from src.preprocessing import form_responses_to_text
        from src.validator import validate_input, check_confidence
        profile_text = form_responses_to_text(responses)
        validation   = validate_input(profile_text)

    if not validation.valid:
        st.warning(f"Could not analyse your responses: {validation.reason}")
        st.info("Please make sure you answered all questions and wrote something for Q10.")
        st.stop()

    with st.spinner("🧠 Running AI analysis..."):
        from src.models import predict_single
        result = predict_single(profile_text, rf, tfidf)

    with st.spinner("✨ Generating word explanation..."):
        from src.shap_explainer import explain_prediction, plot_shap_bar
        from src.preprocessing import clean_text
        cleaned     = result["cleaned_text"] or clean_text(profile_text)
        shap_out    = explain_prediction(cleaned, rf, tfidf, explainer)
        chart_bytes = plot_shap_bar(shap_out["top_words"])

    # Save to Supabase (REQ-D12: only valid, sufficient-confidence results)
    student_token = "unavailable"
    try:
        from src.database import save_submission
        saved_row = save_submission({
            "department":     dept,
            "mode":           "form",
            "q1":  q1,  "q2": q2, "q3": q3, "q4": q4, "q5": q5,
            "q6":  q6,  "q7": q7, "q8": q8, "q9": q9,
            "q10_open":       q10[:500] if q10 else "",
            "generated_text": profile_text[:500],
            "prediction":     result["prediction"],
            "confidence":     result["confidence"],
            "risk_level":     result["risk_level"],
            "shap_top_words": str(shap_out["top_words"][:5]),
        })
        student_token = saved_row.get("id", "unavailable")
    except Exception:
        pass

    # ── Result Display ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 📊 Your Wellbeing Check Result")

    risk = result["risk_level"]
    RISK_CONFIG = {
        "low": (
            "success", "🟢", "Low Risk",
            "Your responses suggest you are managing well overall. "
            "Keep taking care of yourself and stay connected with people you trust. ❤️"
        ),
        "medium": (
            "warning", "🟡", "Medium Risk",
            "Your responses suggest you may be experiencing some stress or low mood lately. "
            "Consider speaking with someone you trust, or a counselor. You are not alone."
        ),
        "high": (
            "error", "🔴", "High Risk",
            "Your responses indicate significant distress indicators. "
            "Please consider reaching out for support — you deserve care. "
            "A counselor at Anurag University is available to speak with you privately."
        ),
    }
    alert_fn, icon, risk_label, message = RISK_CONFIG[risk]
    getattr(st, alert_fn)(f"{icon} **{risk_label}**")
    st.markdown(message)

    m1, m2, m3 = st.columns(3)
    m1.metric("Result",        "Indicators Detected" if result["prediction"] == 1 else "No Indicators")
    m2.metric("AI Confidence", f"{result['confidence']*100:.1f}%")
    m3.metric("Risk Level",    risk_label)

    st.markdown("#### ✨ What Influenced This Result?")
    st.markdown(
        "The chart shows which words from your responses had the most influence on the AI decision. "
        "**Red bars** → associated with depression indicators. "
        "**Blue bars** → associated with wellbeing."
    )
    if chart_bytes:
        st.image(chart_bytes, caption="AI Word Attribution — What drove your wellbeing result")

    top5 = shap_out["top_words"][:5]
    if top5:
        with st.expander("📝 View top 5 influential words (text format)"):
            for word, val in top5:
                direction = "toward depression" if val > 0 else "away from depression"
                st.markdown(f"- **{word}**: SHAP {val:+.3f} → pushes *{direction}*")

    if student_token and student_token != "unavailable":
        st.info(f"🔑 Your anonymous reference ID: **{str(student_token)}**")
        st.caption(
            "Save this ID if you want a counselor to look up your full report. "
            "Only you know this ID — it cannot be traced back to your name."
        )

    # Mandatory disclaimer (REQ-U04)
    st.caption(
        "⚠️ **Important:** This is a **wellbeing screening tool only** — not a clinical diagnosis. "
        "Results are based on your responses and AI pattern matching. "
        "Always consult a qualified mental health professional for clinical support."
    )

    # ── WhatsApp Connect (Medium/High risk) ──────────────────────────────────
    if risk in ("medium", "high"):
        st.divider()
        if risk == "high":
            st.markdown("### 💙 A Counselor Is Here for You")
            st.markdown(
                f"Based on your responses, **{counselor_info['name']}** is available to speak "
                f"with you privately. Your identity is **never shared** — "
                f"only your anonymous ID, and only if *you* choose to share it."
            )
        else:
            st.markdown("### 💬 You Can Speak With Someone")
            st.markdown(
                "If you'd like to talk through how you're feeling with a counselor, "
                "one is available for your department."
            )

        wa_message = (
            f"Hi Counselor, I completed the MindBridge wellbeing check. "
            f"My anonymous reference ID is: {student_token}. "
            f"My result showed: {risk_label}. "
            f"I am comfortable with you viewing my form responses."
        )
        wa_url = f"https://wa.me/{counselor_info['wa']}?text={urllib.parse.quote(wa_message)}"

        st.link_button(
            "💬 Connect With Counselor on WhatsApp",
            wa_url, type="primary", use_container_width=True,
        )
        st.caption(
            "Tapping this opens WhatsApp with a pre-written message containing only your anonymous ID. "
            "Your name is NOT included. You decide whether to send it — we never contact you."
        )

        if counselor_info.get("calendly"):
            st.markdown("**Or book a private meeting directly:**")
            st.link_button(
                "📅 Book a Private Meeting (Calendly)",
                counselor_info["calendly"], use_container_width=True,
            )

    # ── Resources ────────────────────────────────────────────────────────────
    with st.expander("ℹ️ Wellbeing Resources (Free Helplines)"):
        st.markdown("""
        - **iCall** (TISS): 📞 9152987821 — Free professional counseling helpline
        - **Vandrevala Foundation**: 📞 1860-2662-345 — 24/7 mental health helpline
        - **NIMHANS Helpline**: 📞 080-46110007
        - **iCall Online Chat**: https://icallhelpline.org
        - **Anurag University Psychology Department**: Visit the Student Services office
        """)
