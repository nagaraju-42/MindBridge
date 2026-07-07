import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
src/validator.py
MindBridge — Input Validation Gateway
Satisfies SRS REQ-D01, REQ-D08 through REQ-D12.

All 5 checks run BEFORE any ML inference.
REQ-D12: Rejected or low-confidence inputs are NEVER saved to Supabase.
"""
from dataclasses import dataclass, field
from typing import Optional

from src.config import (
    MIN_INPUT_CHARS,
    MAX_INPUT_CHARS,
    MAX_NUMERIC_RATIO,
    MIN_VOCAB_TOKENS,
    LOW_CONF_THRESHOLD,
)


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid:     bool             # True if input passed all checks
    reason:    str = ""        # Human-readable rejection reason (empty if valid)
    truncated: bool = False    # True if text was truncated to MAX_INPUT_CHARS
    text:      Optional[str] = None  # Processed text (may be truncated)


def validate_input(raw_text: str, vectorizer=None) -> ValidationResult:
    """
    Run all input validation checks sequentially.

    Checks (in order):
    1. REQ-D11: Minimum length (>= 20 characters)
    2. REQ-D09: Numeric ratio (<= 30% of all characters)
    3. REQ-D01: Maximum length (> 15,000 → truncate, not reject)
    4. REQ-D08: Vocabulary coverage (>= 5 in-vocabulary tokens)
               Requires a fitted TfidfVectorizer. Skipped if None.

    Args:
        raw_text:   Raw input string from user.
        vectorizer: Fitted TfidfVectorizer (optional). If provided,
                    enables vocabulary check (REQ-D08).

    Returns:
        ValidationResult with valid=True if all checks pass.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ValidationResult(
            valid=False,
            reason="Please enter some text before clicking Analyse."
        )

    text = raw_text.strip()

    # ── Check 1: REQ-D11 — Minimum length ─────────────────────────────────────
    if len(text) < MIN_INPUT_CHARS:
        return ValidationResult(
            valid=False,
            reason=(
                f"Your text is too short ({len(text)} characters). "
                f"Please write at least {MIN_INPUT_CHARS} characters — "
                "ideally 2-3 sentences about how you have been feeling."
            )
        )

    # ── Check 2: REQ-D09 — Numeric character ratio ─────────────────────────────
    # Catches payment transaction data, phone numbers, OTPs
    digit_count = sum(c.isdigit() for c in text)
    numeric_ratio = digit_count / len(text)
    if numeric_ratio > MAX_NUMERIC_RATIO:
        return ValidationResult(
            valid=False,
            reason=(
                f"Your text appears to contain mainly numbers ({numeric_ratio*100:.0f}% digits). "
                "MindBridge analyses emotional language, not numeric data. "
                "Please paste a personal text — a journal entry, a post, or "
                "write a few sentences about how you are feeling."
            )
        )

    # ── Check 3: REQ-D01 — Maximum length (truncate, not reject) ───────────────
    truncated = False
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        truncated = True

    # ── Check 4: REQ-D08 — Vocabulary coverage ─────────────────────────────────
    # Only runs if a fitted TF-IDF vectoriser is provided
    if vectorizer is not None:
        try:
            from src.preprocessing import clean_text
            cleaned = clean_text(text)
            tokens = cleaned.split()
            vocab = vectorizer.vocabulary_
            in_vocab = [t for t in tokens if t in vocab]
            if len(in_vocab) < MIN_VOCAB_TOKENS:
                return ValidationResult(
                    valid=False,
                    reason=(
                        f"Only {len(in_vocab)} recognisable word(s) were found in your text. "
                        "The AI needs at least 5 meaningful words to analyse. "
                        "Try writing in English — ideally about how you have been feeling this week."
                    )
                )
        except Exception:
            pass  # If vocab check fails for any reason, allow through

    return ValidationResult(valid=True, truncated=truncated, text=text)


def check_confidence(confidence: float) -> bool:
    """
    REQ-D10: Check if prediction confidence meets the minimum threshold.

    Low-confidence predictions (< 60%) are shown a notice but NOT
    logged to Supabase (REQ-D12).

    Args:
        confidence: Float between 0.0 and 1.0 from model.predict_proba().

    Returns:
        True if confidence >= LOW_CONF_THRESHOLD, False otherwise.
    """
    return confidence >= LOW_CONF_THRESHOLD


# ─── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== MindBridge Validator Self-Test ===")

    # Test 1: Too short
    r = validate_input("ok")
    assert not r.valid, "FAIL: too-short text should be rejected"
    print(f"Test 1 PASS — short text rejected: {r.reason[:60]}...")

    # Test 2: Payment transaction (numeric ratio)
    r = validate_input("Txn 123456789 Rs.1499 SUCCESS OTP 8827")
    assert not r.valid, "FAIL: payment data should be rejected"
    print(f"Test 2 PASS — payment data rejected: {r.reason[:60]}...")

    # Test 3: Valid depression text
    r = validate_input("I feel completely hopeless and nothing ever gets better for me. I am exhausted all the time.")
    assert r.valid, "FAIL: valid depression text should pass"
    print(f"Test 3 PASS — valid text accepted")

    # Test 4: Max length truncation
    long_text = "feeling bad " * 2000  # ~24,000 chars
    r = validate_input(long_text)
    assert r.valid, "FAIL: long text should be truncated not rejected"
    assert r.truncated, "FAIL: truncated flag should be True"
    assert len(r.text) == 15000, f"FAIL: truncated text length {len(r.text)} != 15000"
    print(f"Test 4 PASS — long text truncated to {len(r.text)} chars")

    # Test 5: check_confidence
    assert check_confidence(0.80),  "FAIL: 80% confidence should PASS"
    assert not check_confidence(0.55), "FAIL: 55% confidence should FAIL"
    print("Test 5 PASS — confidence check working")

    print("\nvalidator.py: ALL PASS")
