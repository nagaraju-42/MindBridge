"""
pages/4_Email_Campaign.py
MindBridge — Email Campaign Manager.
PRD Section 5.4 — v2 Scale/Outreach feature.

Counselors can:
  - Compose a wellbeing check invitation email
  - Upload a CSV of student email addresses
  - Preview the rendered HTML template
  - Send to all uploaded recipients via Resend API (free: 3,000/month)
  - View campaign history from Supabase
"""
import streamlit as st
import pandas as pd
import io

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Campaigns — MindBridge",
    page_icon="📧",
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
    st.page_link("pages/3_Dashboard.py",      label="📊 Dashboard")
    st.page_link("pages/4_Email_Campaign.py", label="📧 Email Campaigns")
    st.divider()
    st.info("Resend API free tier: **3,000 emails/month**, 100/day")

st.title("📧 Email Campaign Manager")
st.markdown(
    "Send anonymous wellbeing check invitation emails to students. "
    "Uses [Resend](https://resend.com) — free tier: 3,000 emails/month."
)

# ── Auth check (reuse dashboard session) ─────────────────────────────────────
if not st.session_state.get("dashboard_auth", False):
    st.warning("🔒 Please log in via the **Counselor Dashboard** first.")
    st.markdown("")
    st.page_link("pages/3_Dashboard.py", label="→ Go to Dashboard Login")
    st.stop()

st.divider()

# ── Compose Campaign ──────────────────────────────────────────────────────────
st.markdown("#### ✏️ Compose Campaign")
col_compose, col_preview = st.columns([1, 1], gap="large")

with col_compose:
    subject = st.text_input(
        "Email Subject:",
        value="MindBridge: Take Your Wellbeing Check — 3 Minutes, Completely Anonymous",
        key="campaign_subject",
    )

    from src.config import DEPARTMENT_GROUPS
    dept_choice = st.selectbox(
        "Target Department (for counselor name in email):",
        ["All Departments"] + list(DEPARTMENT_GROUPS.keys()),
        key="campaign_dept",
    )

    counselor_name = st.text_input(
        "Counselor Name (shown in email footer):",
        value="Your Counselor",
        key="counselor_name_input",
    )

    form_url = st.text_input(
        "MindBridge App URL:",
        value="https://your-app.streamlit.app",
        help="Paste your Streamlit Community Cloud deployment URL here.",
        key="form_url_input",
    )

    # Get Calendly for chosen dept
    if dept_choice != "All Departments":
        calendly = DEPARTMENT_GROUPS[dept_choice].get("calendly", "")
    else:
        calendly = ""

    st.markdown(f"**Estimated recipients:** Upload CSV below to see count")

with col_preview:
    st.markdown("**📧 Email Template Preview**")
    from src.email_service import render_campaign_template

    preview_html = render_campaign_template(
        form_url=form_url,
        counselor_name=counselor_name,
        calendly_url=calendly or None,
        college_name="Anurag University",
    )

    with st.container(border=True):
        st.markdown("""
        <div style='font-size:13px; color:#aaa;'>
        Preview: Email includes MindBridge header, CTA button, privacy box, and footer.
        Download HTML to open in a browser for full preview.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        st.download_button(
            "⬇️ Download HTML Preview",
            preview_html.encode("utf-8"),
            "mindbridge_email_preview.html",
            "text/html",
            key="preview_download",
        )
        st.markdown("")
        # Show a text snippet of the email
        st.caption("**Subject:** " + subject)
        st.caption("**From:** MindBridge <mindbridge@yourdomain.com>")
        st.caption("**Template:** Wellbeing check invitation with CTA button, privacy guarantee, and counselor footer")

st.divider()

# ── Upload Email List ─────────────────────────────────────────────────────────
st.markdown("#### 📄 Upload Student Email List")

with st.expander("📋 Required CSV Format"):
    st.markdown("""
    Your CSV must have a column named **`email`**.
    The `name` column is optional.

    ```csv
    name,email
    Student 1,student1@anurag.edu.in
    Student 2,student2@anurag.edu.in
    ```

    - Emails without `@` are automatically skipped
    - Maximum 100 emails per batch (Resend free daily limit)
    - For larger campaigns, split into batches of 100
    """)

uploaded_file = st.file_uploader(
    "Upload student email CSV:",
    type=["csv"],
    key="email_csv",
    help="CSV file with an 'email' column.",
)

if uploaded_file:
    try:
        email_df = pd.read_csv(uploaded_file)

        if "email" not in email_df.columns:
            st.error("❌ Your CSV must have a column named `email`. Please check and re-upload.")
            st.stop()

        all_emails = email_df["email"].dropna().astype(str).str.strip().tolist()
        valid_emails = [e for e in all_emails if "@" in e and "." in e.split("@")[-1]]
        invalid_count = len(all_emails) - len(valid_emails)

        st.success(f"✅ **{len(valid_emails)}** valid email addresses loaded from CSV")
        if invalid_count > 0:
            st.warning(f"⚠️ {invalid_count} rows skipped (invalid or missing email format)")

        if len(valid_emails) > 100:
            st.warning(
                f"⚠️ Resend free tier limit is 100 emails/day. "
                f"You have {len(valid_emails)} emails. "
                f"Only the first 100 will be sent. Split into batches for more."
            )
            send_emails = valid_emails[:100]
        else:
            send_emails = valid_emails

        with st.expander(f"Preview first 10 of {len(valid_emails)} emails"):
            for e in valid_emails[:10]:
                st.markdown(f"- `{e}`")

        st.divider()
        st.markdown("#### 🚀 Send Campaign")

        col_send, col_info = st.columns([2, 3])
        with col_info:
            st.info(
                f"📧 **Ready to send:**\n"
                f"- Recipients: **{len(send_emails)}** addresses\n"
                f"- Subject: {subject[:60]}...\n"
                f"- Target dept: {dept_choice}\n"
                f"- From: {counselor_name}"
            )

        with col_send:
            send_confirmed = st.checkbox(
                f"I confirm I want to send to **{len(send_emails)}** recipients",
                key="send_confirm",
            )
            send_btn = st.button(
                f"🚀 Send to {len(send_emails)} Students",
                type="primary",
                disabled=not send_confirmed,
                use_container_width=True,
                key="send_campaign",
            )

        if send_btn:
            from src.email_service import send_bulk_campaign
            from src.database import save_campaign

            progress_bar = st.progress(0, text="Sending emails...")
            with st.spinner(f"Sending {len(send_emails)} emails via Resend..."):
                results = send_bulk_campaign(
                    emails=send_emails,
                    subject=subject,
                    body_html=preview_html,
                    delay_seconds=0.1,
                )
            progress_bar.progress(100, text="Done!")

            if results["sent"] > 0:
                st.success(
                    f"✅ Campaign sent!\n"
                    f"- Sent: **{results['sent']}** emails\n"
                    f"- Failed: {results['failed']}"
                )
                # Log to Supabase
                try:
                    save_campaign({
                        "subject":    subject,
                        "body":       preview_html[:1000],
                        "department": dept_choice,
                        "sent_count": results["sent"],
                        "status":     "sent",
                        "sender_note": f"Counselor: {counselor_name}",
                    })
                    st.info("💾 Campaign logged to Supabase database.")
                except Exception:
                    pass
            else:
                st.error(
                    f"❌ All {results['failed']} sends failed.\n"
                    "Check RESEND_API_KEY in `.streamlit/secrets.toml`.\n"
                    "Sign up free at: https://resend.com"
                )

    except Exception as e:
        st.error(f"CSV parsing error: {e}")
        st.info("Make sure your CSV is well-formed and has an 'email' column.")

st.divider()

# ── Campaign History ──────────────────────────────────────────────────────────
st.markdown("#### 📜 Campaign History")
try:
    from src.database import get_all_campaigns
    campaigns = get_all_campaigns(limit=25)
    if campaigns:
        camp_df = pd.DataFrame(campaigns)
        display_cols = [c for c in
                        ["created_at", "subject", "department", "sent_count", "status", "sender_note"]
                        if c in camp_df.columns]
        if "created_at" in camp_df.columns:
            camp_df["created_at"] = pd.to_datetime(
                camp_df["created_at"]
            ).dt.strftime("%Y-%m-%d %H:%M")
        if "subject" in camp_df.columns:
            camp_df["subject"] = camp_df["subject"].str[:60] + "..."
        st.dataframe(camp_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("📭 No campaigns have been sent yet. Upload a CSV above and send your first campaign!")
except Exception as e:
    st.warning(f"Could not load campaign history: {e}")

st.divider()
st.caption(
    "MindBridge Email Campaigns · Powered by Resend API · Free tier: 3,000 emails/month · "
    "Anurag University 2026-27"
)
