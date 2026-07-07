"""
pages/3_Dashboard.py
MindBridge — Counselor Dashboard.
Satisfies SRS REQ-U12 to REQ-U15, NFR-S03, NFR-S04.

Password-gated dashboard for counselors to:
  - View overview metrics and risk distribution
  - See daily trend charts
  - Filter submissions by risk, department, date range
  - Look up individual reports by student anonymous token
  - Export filtered data as CSV
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Counselor Dashboard — MindBridge",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0;'>
        <span style='font-size:36px;'>🧠</span><br>
        <span style='font-size:20px; font-weight:700; color:#a855f7;'>MindBridge</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.page_link("app.py",                    label="🏠 Home")
    st.page_link("pages/1_Text_Analyser.py",  label="🔍 Text Analyser")
    st.page_link("pages/2_Student_Form.py",   label="📋 Wellbeing Check")
    st.page_link("pages/3_Dashboard.py",      label="📊 Dashboard")
    st.page_link("pages/4_Email_Campaign.py", label="📧 Email Campaigns")

st.title("📊 Counselor Dashboard")
st.markdown("*Monitor student wellbeing data, look up individual reports, and track trends.*")

# ── Authentication ────────────────────────────────────────────────────────────
if "dashboard_auth" not in st.session_state:
    st.session_state.dashboard_auth = False

if not st.session_state.dashboard_auth:
    st.divider()
    st.markdown("### 🔒 Counselor Login")
    st.markdown("This dashboard is restricted to Anurag University counselors and administrators.")

    with st.container(border=True):
        entered_pw = st.text_input("Password:", type="password", key="dashboard_pw",
                                   placeholder="Enter counselor password")
        login_btn = st.button("🔐 Login", type="primary", use_container_width=True)

    if login_btn:
        from src.config import COUNSELOR_PASSWORD
        if entered_pw == COUNSELOR_PASSWORD:
            st.session_state.dashboard_auth = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Contact the MindBridge administrator.")

    st.divider()
    st.caption("Forgot the password? Check `.streamlit/secrets.toml` → `COUNSELOR_PASSWORD`")
    st.stop()

# ── Logout ────────────────────────────────────────────────────────────────────
col_title, col_logout = st.columns([5, 1])
with col_logout:
    if st.button("🚪 Logout", key="logout"):
        st.session_state.dashboard_auth = False
        st.rerun()

st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
st.markdown("#### 🔧 Filters")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
with col_f1:
    days_back = st.selectbox(
        "Time Range",
        [7, 14, 30, 60, 90], index=2,
        format_func=lambda x: f"Last {x} days"
    )
with col_f2:
    risk_filter = st.selectbox("Risk Level", ["All", "high", "medium", "low"])
with col_f3:
    dept_filter = st.selectbox(
        "Department",
        ["All", "CSE / IT / AI / DS", "ECE / EEE", "Mech / Civil / Other"]
    )
with col_f4:
    mode_filter = st.selectbox("Mode", ["All", "form", "text_analyser"])

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading submissions from Supabase..."):
    try:
        from src.database import get_all_submissions
        rows = get_all_submissions(
            limit=1000,
            risk_filter=None  if risk_filter == "All" else risk_filter,
            department_filter=None if dept_filter == "All" else dept_filter,
            days_back=days_back,
        )
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        st.info("Check your SUPABASE_URL and SUPABASE_SERVICE_KEY in `.streamlit/secrets.toml`")
        st.stop()

df = pd.DataFrame(rows) if rows else pd.DataFrame()

if mode_filter != "All" and not df.empty:
    df = df[df["mode"] == mode_filter]

# ── Overview Metrics ──────────────────────────────────────────────────────────
st.markdown("#### 📊 Overview")
m1, m2, m3, m4, m5 = st.columns(5)
total  = len(df)
high   = len(df[df["risk_level"] == "high"])   if not df.empty else 0
medium = len(df[df["risk_level"] == "medium"]) if not df.empty else 0
low    = len(df[df["risk_level"] == "low"])    if not df.empty else 0
wa_cnt = len(df[df["whatsapp_connected"] == True]) if not df.empty and "whatsapp_connected" in df.columns else 0

m1.metric("📊 Total", total)
m2.metric("🔴 High Risk",   high)
m3.metric("🟡 Medium Risk", medium)
m4.metric("🟢 Low Risk",    low)
m5.metric("💬 WA Connected", wa_cnt)

if df.empty:
    st.info("📢 No submissions found for the selected filters. Try expanding the time range or removing filters.")
    st.stop()

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
col_chart1, col_chart2 = st.columns(2, gap="large")
COLOR_MAP = {"high": "#C00000", "medium": "#FFA500", "low": "#1E8449"}

with col_chart1:
    st.markdown("**Risk Level Distribution**")
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    fig = px.pie(
        risk_counts, names="Risk Level", values="Count",
        color="Risk Level", color_discrete_map=COLOR_MAP,
        hole=0.42,
    )
    fig.update_layout(
        paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
        font_color="#E0E0E0", margin=dict(t=10, b=10),
        legend=dict(bgcolor="#1A1D2E"),
        height=320,
    )
    fig.update_traces(textfont_color="white")
    st.plotly_chart(fig, use_container_width=True)

with col_chart2:
    st.markdown("**Daily Submissions by Risk Level**")
    if "created_at" in df.columns:
        df["date"] = pd.to_datetime(df["created_at"]).dt.date
        daily = df.groupby(["date", "risk_level"]).size().reset_index(name="count")
        if not daily.empty:
            fig2 = px.line(
                daily, x="date", y="count", color="risk_level",
                color_discrete_map=COLOR_MAP,
                labels={"date": "Date", "count": "Submissions", "risk_level": "Risk"},
                markers=True,
            )
            fig2.update_layout(
                paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
                font_color="#E0E0E0", margin=dict(t=10, b=10),
                legend=dict(bgcolor="#1A1D2E"),
                height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough data for trend chart.")

# ── Dept Breakdown ─────────────────────────────────────────────────────────────
if "department" in df.columns:
    st.markdown("**Department × Risk Breakdown**")
    dept_risk = df.groupby(["department", "risk_level"]).size().reset_index(name="count")
    if not dept_risk.empty:
        fig3 = px.bar(
            dept_risk, x="department", y="count", color="risk_level",
            color_discrete_map=COLOR_MAP, barmode="group",
            labels={"department": "Department", "count": "Submissions", "risk_level": "Risk"},
        )
        fig3.update_layout(
            paper_bgcolor="#1A1D2E", plot_bgcolor="#1A1D2E",
            font_color="#E0E0E0", margin=dict(t=10, b=10),
            legend=dict(bgcolor="#1A1D2E"), height=280,
        )
        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Token Lookup ──────────────────────────────────────────────────────────────
st.markdown("#### 🔍 Token Lookup — Student Shares Their Anonymous ID")
st.markdown(
    "When a student chooses to contact a counselor, they share their anonymous reference ID. "
    "Paste it here to view their full report."
)

col_tok, col_tok_btn = st.columns([4, 1])
with col_tok:
    token_input = st.text_input(
        "Anonymous student token:",
        placeholder="e.g. 7f3d9a2e-1234-5678-abcd-ef0123456789",
        label_visibility="collapsed",
        key="token_lookup"
    )
with col_tok_btn:
    fetch_btn = st.button("🔍 Fetch Report", type="secondary", use_container_width=True)

if fetch_btn and token_input:
    try:
        from src.database import get_submission_by_id
        report = get_submission_by_id(token_input.strip())
        if report:
            st.success("✅ Report found")
            with st.container(border=True):
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown(f"**Submitted:** `{report.get('created_at', 'N/A')}`")
                    st.markdown(f"**Department:** {report.get('department', 'N/A')}")
                    st.markdown(f"**Mode:** `{report.get('mode', 'N/A')}`")
                    risk_val = report.get('risk_level', 'N/A')
                    risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk_val, "⚪")
                    st.markdown(f"**Risk Level:** {risk_emoji} `{risk_val.upper()}`")
                    conf_val = report.get('confidence', 0)
                    st.markdown(f"**AI Confidence:** `{float(conf_val)*100:.1f}%`")
                    st.markdown(f"**WhatsApp Connected:** {'✅ Yes' if report.get('whatsapp_connected') else '❌ No'}")

                with col_r2:
                    st.markdown("**SHAP Top Words (Influential Features):**")
                    st.code(report.get('shap_top_words', 'N/A'), language=None)

                # Form Q&A
                q_labels = [
                    "Q1: Energy/Tiredness (PHQ-9 Q4)",
                    "Q2: Interest/Pleasure (PHQ-9 Q1)",
                    "Q3: Mood/Hopelessness (PHQ-9 Q2)",
                    "Q4: Sleep (PHQ-9 Q3)",
                    "Q5: Appetite (PHQ-9 Q5)",
                    "Q6: Burden feelings (PHQ-9 Q6)",
                    "Q7: Concentration (PHQ-9 Q7)",
                    "Q8: Social connection (PHQ-9 Q8)",
                    "Q9: Sense of purpose (PHQ-9 Q9)",
                ]
                answered = [(q_labels[i-1], report.get(f"q{i}")) for i in range(1, 10) if report.get(f"q{i}")]
                if answered:
                    st.markdown("**Form Responses:**")
                    for label, val in answered:
                        st.markdown(f"- {label}: **{val}**")
                    if report.get("q10_open"):
                        st.markdown(f'- Q10 (Open): *"{report["q10_open"]}"*')
                else:
                    if report.get("generated_text"):
                        st.markdown("**Analysed Text (first 200 chars):**")
                        st.markdown(f'*"{report["generated_text"][:200]}..."*')
        else:
            st.error("❌ No report found for that token. Check the ID and try again.")
    except Exception as e:
        st.error(f"Lookup failed: {e}")

st.divider()

# ── Submissions Table ─────────────────────────────────────────────────────────
st.markdown("#### 📝 Submission Records")

display_cols = ["id", "created_at", "department", "mode",
                "risk_level", "confidence", "whatsapp_connected"]
avail_cols   = [c for c in display_cols if c in df.columns]
df_display   = df[avail_cols].copy()

if "confidence" in df_display.columns:
    df_display["confidence"] = df_display["confidence"].apply(
        lambda x: f"{float(x)*100:.1f}%" if x else "N/A"
    )
if "created_at" in df_display.columns:
    df_display["created_at"] = pd.to_datetime(
        df_display["created_at"]
    ).dt.strftime("%Y-%m-%d %H:%M")
if "id" in df_display.columns:
    df_display["id"] = df_display["id"].astype(str).str[:8].str.upper()

# Rename for display
df_display.columns = [
    c.replace("_", " ").title() if c != "id" else "ID (first 8)"
    for c in df_display.columns
]

st.dataframe(df_display, use_container_width=True, hide_index=True, height=350)

# Export button
csv_bytes = df.to_csv(index=False).encode()
st.download_button(
    "⬇️ Export All Filtered Records (CSV)",
    csv_bytes,
    file_name=f"mindbridge_submissions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)

# ── Model Performance Panel ───────────────────────────────────────────────────
st.divider()
st.markdown("#### 🤖 Model Performance (Last Training Run)")

import os
eval_csv = "notebooks/evaluation_results/model_comparison.csv"
if os.path.exists(eval_csv):
    eval_df = pd.read_csv(eval_csv)
    st.dataframe(eval_df, use_container_width=True, hide_index=True)
    best = eval_df.sort_values("F1%", ascending=False).iloc[0]
    st.success(
        f"Best model: **{best['Model']}** with F1 = **{best['F1%']}%** on held-out test set"
    )
else:
    st.info(
        "Model evaluation results not found. "
        "Run `python src/evaluator.py` to generate the comparison table."
    )

st.divider()
st.caption(
    "MindBridge Counselor Dashboard · Anurag University 2026-27 · "
    "All data is anonymized — no PII is stored."
)
