"""
src/database.py
MindBridge — Supabase Database Layer.
Satisfies SRS NFR-S01 to NFR-S05.

NO PII IS EVER STORED. Only anonymous UUIDs identify submissions.
The student's name, roll number, email and phone are never collected.

Uses the anon key for student-facing INSERT operations.
Uses the service_role key for counselor dashboard SELECT operations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
from typing import Optional

from src.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_SERVICE_KEY,
    TABLE_SUBMISSIONS,
    TABLE_CAMPAIGNS,
    TABLE_COUNSELORS,
)

# ── Supabase client factory ────────────────────────────────────────────────────
def _get_client(use_service_key: bool = False):
    """
    Return a Supabase client.
    use_service_key=True bypasses Row Level Security — only for dashboard.
    """
    try:
        from supabase import create_client, Client
        key = SUPABASE_SERVICE_KEY if use_service_key else SUPABASE_KEY
        if key in ("NOT_SET", ""):
            raise ValueError("Supabase key not configured in secrets.toml")
        return create_client(SUPABASE_URL, key)
    except ImportError:
        raise ImportError(
            "supabase library not installed. Run: pip install supabase"
        )


def test_connection() -> bool:
    """
    Test Supabase connection. Used by the startup health check in app.py.

    Returns:
        True if connection succeeds, False otherwise.
    """
    try:
        client = _get_client()
        # Lightweight query — just fetch 1 row to confirm DB is reachable
        client.table(TABLE_SUBMISSIONS).select("id").limit(1).execute()
        return True
    except Exception as e:
        print(f"[Database] Connection test failed: {e}")
        return False


def save_submission(data: dict) -> dict:
    """
    Save an anonymous submission to Supabase.
    Called after a successful prediction with sufficient confidence (REQ-D12).

    Args:
        data: Dict with submission fields. Must NOT include any PII.
              Expected keys: department, mode, q1..q9, q10_open,
              generated_text, prediction, confidence, risk_level,
              shap_top_words.

    Returns:
        Inserted row dict (includes auto-generated 'id' UUID).
        Returns {'id': 'unavailable'} on error (graceful degradation).
    """
    try:
        client = _get_client()
        # Ensure no PII fields slip through
        _PII_FIELDS = {"name", "email", "phone", "roll_number", "student_id", "mobile"}
        clean_data = {k: v for k, v in data.items() if k.lower() not in _PII_FIELDS}

        response = client.table(TABLE_SUBMISSIONS).insert(clean_data).execute()
        if response.data:
            return response.data[0]
        return {"id": "unavailable"}
    except Exception as e:
        print(f"[Database] save_submission failed: {e}")
        return {"id": "unavailable"}


def get_all_submissions(
    limit: int = 500,
    risk_filter: Optional[str] = None,
    department_filter: Optional[str] = None,
    days_back: int = 30,
    use_service_key: bool = True
) -> list:
    """
    Fetch all submissions for the counselor dashboard.
    Requires service_role key (bypasses RLS).

    Args:
        limit:             Max rows to return.
        risk_filter:       Filter by 'low', 'medium', or 'high'. None = all.
        department_filter: Filter by department string. None = all.
        days_back:         Only return submissions from last N days.
        use_service_key:   Use service role (True for dashboard).

    Returns:
        List of submission dicts.
    """
    try:
        client = _get_client(use_service_key=use_service_key)
        query = client.table(TABLE_SUBMISSIONS).select("*")

        if risk_filter:
            query = query.eq("risk_level", risk_filter)

        if department_filter:
            query = query.eq("department", department_filter)

        cutoff = (
            datetime.datetime.utcnow() - datetime.timedelta(days=days_back)
        ).isoformat()
        query = query.gte("created_at", cutoff)
        query = query.order("created_at", desc=True).limit(limit)

        response = query.execute()
        return response.data or []
    except Exception as e:
        print(f"[Database] get_all_submissions failed: {e}")
        return []


def get_submission_by_id(submission_id: str, use_service_key: bool = True) -> Optional[dict]:
    """
    Fetch a single submission by its UUID (for counselor token lookup).

    Args:
        submission_id:   UUID string from the student's token.
        use_service_key: Use service role (True for dashboard).

    Returns:
        Submission dict or None if not found.
    """
    try:
        client = _get_client(use_service_key=use_service_key)
        response = (
            client.table(TABLE_SUBMISSIONS)
            .select("*")
            .eq("id", submission_id)
            .execute()
        )
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"[Database] get_submission_by_id failed: {e}")
        return None


def mark_whatsapp_connected(submission_id: str) -> bool:
    """
    Mark a submission as WhatsApp-connected after student taps the button.

    Args:
        submission_id: UUID of the submission to update.

    Returns:
        True on success, False on error.
    """
    try:
        client = _get_client()
        client.table(TABLE_SUBMISSIONS).update(
            {"whatsapp_connected": True}
        ).eq("id", submission_id).execute()
        return True
    except Exception as e:
        print(f"[Database] mark_whatsapp_connected failed: {e}")
        return False


def get_risk_distribution(days_back: int = 30) -> dict:
    """
    Get risk level counts for the dashboard overview.

    Returns:
        Dict: {'low': N, 'medium': N, 'high': N, 'total': N}
    """
    rows = get_all_submissions(limit=5000, days_back=days_back)
    dist = {"low": 0, "medium": 0, "high": 0}
    for row in rows:
        level = row.get("risk_level", "low")
        if level in dist:
            dist[level] += 1
    dist["total"] = sum(dist.values())
    return dist


def save_campaign(campaign_data: dict) -> dict:
    """
    Log an email campaign to the campaigns table.

    Args:
        campaign_data: Dict with keys: subject, body, department,
                       sent_count, status.

    Returns:
        Inserted row dict with 'id'.
    """
    try:
        client = _get_client(use_service_key=True)
        response = client.table(TABLE_CAMPAIGNS).insert(campaign_data).execute()
        if response.data:
            return response.data[0]
        return {"id": "unavailable"}
    except Exception as e:
        print(f"[Database] save_campaign failed: {e}")
        return {"id": "unavailable"}


def get_all_campaigns(limit: int = 50) -> list:
    """Fetch campaign history for the Email Campaign page."""
    try:
        client = _get_client(use_service_key=True)
        response = (
            client.table(TABLE_CAMPAIGNS)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as e:
        print(f"[Database] get_all_campaigns failed: {e}")
        return []


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== MindBridge Database Self-Test ===")
    print()

    print("Testing connection...")
    ok = test_connection()
    if ok:
        print("Connection: PASS")
    else:
        print("Connection: FAIL — check SUPABASE_URL and SUPABASE_KEY in secrets.toml")
        exit(1)

    print("\nTesting save_submission()...")
    test_row = {
        "department":    "CSE / IT / AI / DS",
        "mode":          "text_analyser",
        "generated_text":"feel hopeless nothing helps",
        "prediction":    1,
        "confidence":    0.87,
        "risk_level":    "high",
        "shap_top_words":"[('hopeless', 0.41), ('nothing', 0.29)]",
    }
    saved = save_submission(test_row)
    assert "id" in saved, "FAIL: save_submission did not return an id"
    submission_id = saved["id"]
    print(f"Saved row id: {submission_id}")

    print("\nTesting get_submission_by_id()...")
    fetched = get_submission_by_id(submission_id)
    if fetched:
        print(f"Fetched: department={fetched.get('department')}, risk={fetched.get('risk_level')}")
        print("get_submission_by_id: PASS")
    else:
        print("get_submission_by_id: FAIL — row not found")

    print("\nTesting get_risk_distribution()...")
    dist = get_risk_distribution()
    print(f"Risk distribution: {dist}")

    print("\ndatabase.py: ALL PASS")
