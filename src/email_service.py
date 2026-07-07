"""
src/email_service.py
MindBridge — Email Campaign Module.
Sends awareness and outreach emails via Resend API.
Free tier: 3,000 emails/month, 100/day.

Sign up at: https://resend.com
Create API key → paste into secrets.toml as RESEND_API_KEY.
"""
import time
from typing import Optional

from src.config import RESEND_API_KEY, EMAIL_FROM


def send_campaign_email(to_email: str, subject: str, body_html: str) -> bool:
    """
    Send a single HTML email via Resend API.

    Args:
        to_email:   Recipient email address.
        subject:    Email subject line.
        body_html:  HTML body content.

    Returns:
        True on success, False on error.
    """
    if RESEND_API_KEY in ("NOT_SET", "re_PLACEHOLDER_replace_with_your_resend_api_key", ""):
        print(f"[Email] SKIP: RESEND_API_KEY not configured. Would send to: {to_email}")
        return True  # Return True so app doesn't break during dev

    try:
        import resend
        resend.api_key = RESEND_API_KEY
        response = resend.Emails.send({
            "from":    EMAIL_FROM,
            "to":      [to_email],
            "subject": subject,
            "html":    body_html,
        })
        return bool(response.get("id"))
    except ImportError:
        print("[Email] resend library not installed. Run: pip install resend")
        return False
    except Exception as e:
        print(f"[Email] send_campaign_email failed for {to_email}: {e}")
        return False


def send_bulk_campaign(
    emails: list,
    subject: str,
    body_html: str,
    delay_seconds: float = 0.1
) -> dict:
    """
    Send an HTML email to a list of recipients.
    Uses a small delay between sends to respect Resend rate limits.

    Args:
        emails:         List of email address strings.
        subject:        Email subject line.
        body_html:      HTML body content.
        delay_seconds:  Delay between sends (default 0.1s = 10/sec).

    Returns:
        Dict: {'sent': N, 'failed': N, 'total': N}
    """
    sent = 0
    failed = 0

    for email in emails:
        email = email.strip()
        if not email or "@" not in email:
            failed += 1
            continue

        ok = send_campaign_email(email, subject, body_html)
        if ok:
            sent += 1
        else:
            failed += 1

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    print(f"[Email] Campaign complete: {sent} sent, {failed} failed out of {len(emails)} total")
    return {"sent": sent, "failed": failed, "total": len(emails)}


def render_campaign_template(
    form_url: str,
    counselor_name: str = "your counselor",
    calendly_url: Optional[str] = None,
    college_name: str = "Anurag University",
) -> str:
    """
    Render the standard wellbeing check invitation email template.

    Args:
        form_url:       URL of the MindBridge student form.
        counselor_name: Counselor's name (for personalization).
        calendly_url:   Optional Calendly booking link.
        college_name:   College name for footer.

    Returns:
        HTML string ready to send as email body.
    """
    calendly_section = ""
    if calendly_url:
        calendly_section = f"""
        <p style="text-align:center; margin-top:16px;">
            <a href="{calendly_url}"
               style="background:#6C5CE7; color:white; padding:10px 24px;
                      border-radius:6px; text-decoration:none; font-size:14px;">
                📅 Book a Private Meeting
            </a>
        </p>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MindBridge Wellbeing Check</title>
</head>
<body style="margin:0; padding:0; background:#f4f4f8; font-family:Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f8; padding:24px 0;">
    <tr>
      <td align="center">
        <table width="580" cellpadding="0" cellspacing="0"
               style="background:#ffffff; border-radius:12px;
                      box-shadow:0 2px 12px rgba(0,0,0,0.08); overflow:hidden;">

          <!-- Header -->
          <tr>
            <td style="background:linear-gradient(135deg,#6C5CE7,#a855f7);
                       padding:32px; text-align:center;">
              <h1 style="color:white; margin:0; font-size:28px; font-weight:700;">
                🧠 MindBridge
              </h1>
              <p style="color:rgba(255,255,255,0.85); margin:8px 0 0; font-size:14px;">
                {college_name} — Student Mental Wellbeing Platform
              </p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:32px;">
              <h2 style="color:#1a1a2e; font-size:20px; margin:0 0 16px;">
                How are you doing this semester?
              </h2>
              <p style="color:#444; line-height:1.7; margin:0 0 16px;">
                {college_name} cares about your mental wellbeing — not just
                your academic performance. We've launched a completely
                <strong>anonymous, 3-minute wellbeing check</strong> to help
                you understand how you're feeling and connect with support
                if you need it.
              </p>
              <p style="color:#444; line-height:1.7; margin:0 0 24px;">
                It's <strong>10 simple questions</strong>, powered by AI, and
                your identity is <em>never stored</em>. If you'd like to
                speak with a counselor, you choose — we never contact you.
              </p>

              <!-- CTA Button -->
              <p style="text-align:center; margin:24px 0;">
                <a href="{form_url}"
                   style="background:linear-gradient(135deg,#6C5CE7,#a855f7);
                          color:white; padding:14px 36px; border-radius:8px;
                          text-decoration:none; font-size:16px; font-weight:bold;
                          display:inline-block;">
                  ✅ Take the Wellbeing Check (3 min)
                </a>
              </p>

              {calendly_section}

              <!-- Privacy box -->
              <div style="background:#f8f4ff; border-left:4px solid #6C5CE7;
                          padding:16px; border-radius:4px; margin-top:24px;">
                <p style="margin:0; color:#333; font-size:13px; line-height:1.6;">
                  🔒 <strong>Privacy Guarantee:</strong> No name, roll number,
                  email, or phone is ever collected. Only a random anonymous ID
                  is generated. Even your counselor cannot identify you unless
                  you choose to reach out.
                </p>
              </div>

              <p style="color:#666; font-size:13px; margin:24px 0 0; line-height:1.6;">
                This initiative is coordinated by <strong>{counselor_name}</strong>
                at the {college_name} Psychology Department.
                If you have any questions, you can reply to this email.
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background:#f4f4f8; padding:20px; text-align:center;
                       border-top:1px solid #e8e8e8;">
              <p style="color:#999; font-size:12px; margin:0; line-height:1.6;">
                {college_name} | Mental Wellbeing Initiative<br>
                <em>This is a screening tool only — not a medical diagnosis.
                For clinical support, consult a qualified mental health professional.</em>
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
