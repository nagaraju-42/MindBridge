"""
src/preprocessing.py
MindBridge — NLTK Text Cleaning Pipeline
Satisfies SRS REQ-D02 to REQ-D07.

Key methodological decision: NEGATION WORDS ARE PRESERVED.
Standard NLTK stopword removal deletes 'not', 'never', 'nothing',
making 'I am happy' and 'I am not happy' identical after cleaning.
For depression detection, negation is a critical signal — these
words are KEPT even though NLTK marks them as stopwords.

This is documented as a methodological contribution vs the anchor
paper (Ansari et al., 2023) which does not address negation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─── One-time NLTK downloads ───────────────────────────────────────────────────
# Run once: these are cached to ~/nltk_data after first download
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else
                       f"corpora/{pkg}" if pkg != "punkt_tab" else
                       f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ─── Negation words to PRESERVE despite being NLTK stopwords ──────────────────
# These are critical depression markers:
# "I am NOT happy" must differ from "I am happy"
# "NOTHING helps" must be distinguishable from "everything helps"
NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor", "cannot", "cant", "wont", "dont", "didnt",
    "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
    "shouldnt", "wouldnt", "couldnt", "mightnt", "mustnt",
    "without", "hardly", "barely", "scarcely"
}

# ─── Build custom stopword set ─────────────────────────────────────────────────
_STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS

# ─── Lemmatizer (shared instance) ─────────────────────────────────────────────
_LEMMATIZER = WordNetLemmatizer()

# ─── Compiled regex patterns (compile once for speed) ──────────────────────────
_URL_RE     = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
_HTML_RE    = re.compile(r"<[^>]+>")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_PUNCT_RE   = re.compile(r"[^a-z\s]")   # keep only lowercase alpha + spaces
_SPACE_RE   = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean a single text string through the MindBridge NLP pipeline.

    Pipeline steps (order matters):
    1. None/empty guard                          (REQ-D06)
    2. Lowercase                                 (REQ-D02)
    3. Remove URLs                               (REQ-D03)
    4. Remove HTML tags                          (REQ-D03)
    5. Remove @mentions and #hashtags            (REQ-D03)
    6. Remove punctuation/non-alpha chars        (REQ-D03)
    7. Tokenize
    8. Remove stopwords (keep NEGATION_WORDS)    (REQ-D04)
    9. Lemmatize                                 (REQ-D05)
    10. Remove tokens shorter than 2 characters
    11. Rejoin to string

    Args:
        text: Raw input text string.

    Returns:
        Cleaned text string, or empty string if input was empty/None.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()                          # Step 2
    text = _URL_RE.sub(" ", text)               # Step 3
    text = _HTML_RE.sub(" ", text)              # Step 4
    text = _MENTION_RE.sub(" ", text)           # Step 5
    text = _HASHTAG_RE.sub(" ", text)           # Step 5
    text = _PUNCT_RE.sub(" ", text)             # Step 6
    text = _SPACE_RE.sub(" ", text).strip()     # Normalize whitespace

    tokens = word_tokenize(text)                 # Step 7

    cleaned_tokens = []
    for token in tokens:
        if token in _STOP_WORDS:                 # Step 8: remove stopword
            continue
        if len(token) < 2:                       # Step 10: skip single chars
            continue
        lemma = _LEMMATIZER.lemmatize(token)     # Step 9: lemmatize
        cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


def batch_clean(texts) -> list:
    """
    Clean a list (or array) of text strings.
    Used in data_loader.py during dataset preprocessing (~25s for 7,731 posts).

    Args:
        texts: List or numpy array of raw text strings.

    Returns:
        List of cleaned text strings.
    """
    total = len(texts)
    cleaned = []
    for i, text in enumerate(texts):
        if i % 1000 == 0 and i > 0:
            print(f"  Cleaning: {i}/{total} texts processed...")
        cleaned.append(clean_text(str(text)))
    return cleaned


def form_responses_to_text(responses: dict) -> str:
    """
    Convert 10-question form responses to a natural language paragraph.
    This implements REQ-D07: unified pipeline — the same ML pipeline
    handles both free text (Mode 1) and form responses (Mode 2).

    The conversion maps each radio selection to a sentence that
    naturally expresses the response. TF-IDF will then pick up
    depression markers like 'hopeless', 'nothing', 'tired' from
    the generated sentences.

    Args:
        responses: Dict with keys 'q1' through 'q10'.

    Returns:
        A paragraph of 8-12 sentences representing the student's
        wellbeing state, ready to be passed through clean_text().
    """
    q1  = responses.get("q1", "Never")
    q2  = responses.get("q2", "Never")
    q3  = responses.get("q3", "Never")
    q4  = responses.get("q4", "Fine")
    q5  = responses.get("q5", "Normal")
    q6  = responses.get("q6", "Never")
    q7  = responses.get("q7", "No problem")
    q8  = responses.get("q8", "Never")
    q9  = responses.get("q9", "Never")
    q10 = responses.get("q10", "").strip()

    # Q1: Tiredness / energy (PHQ-9 Q4)
    q1_map = {
        "Never":           "I feel energetic and am not tired.",
        "Sometimes":       "I sometimes feel tired and low on energy.",
        "Often":           "I often feel tired and have very little energy.",
        "Almost Every Day":"I feel exhausted and drained of energy almost every day. Nothing seems to restore my energy.",
    }

    # Q2: Anhedonia (PHQ-9 Q1)
    q2_map = {
        "Never":           "I enjoy my usual activities and hobbies.",
        "Sometimes":       "I sometimes feel less interested in things I used to enjoy.",
        "Often":           "I often feel little interest or pleasure in doing things I used to enjoy.",
        "Almost Every Day":"I have almost no interest or pleasure in anything anymore. Nothing feels enjoyable.",
    }

    # Q3: Depressed mood (PHQ-9 Q2)
    q3_map = {
        "Never":           "I feel okay and positive most of the time.",
        "Sometimes":       "I sometimes feel sad or down.",
        "Often":           "I often feel sad, hopeless, or like things will not get better.",
        "Almost Every Day":"I feel completely hopeless and sad almost every day. It feels like nothing will ever get better.",
    }

    # Q4: Sleep (PHQ-9 Q3)
    q4_map = {
        "Fine":                 "My sleep is fine.",
        "Sleeping too much":    "I have been sleeping too much and still feel tired.",
        "Cannot sleep properly":"I cannot sleep properly. I lie awake and cannot rest.",
        "Very disturbed":       "My sleep is very disturbed and broken. I feel terrible after sleeping.",
    }

    # Q5: Appetite (PHQ-9 Q5)
    q5_map = {
        "Normal":            "My appetite is normal.",
        "Eating too much":   "I have been eating too much even when not hungry.",
        "Not feeling hungry":"I am not feeling hungry and have to force myself to eat.",
        "Skipping meals often":"I am skipping meals often and have lost appetite completely.",
    }

    # Q6: Worthlessness / burden (PHQ-9 Q6)
    q6_map = {
        "Never":     "I do not feel like a burden to others.",
        "Rarely":    "I rarely feel like a burden.",
        "Sometimes": "I sometimes feel like I am a burden to the people around me.",
        "Often":     "I often feel worthless and like a burden to everyone around me.",
    }

    # Q7: Concentration (PHQ-9 Q7)
    q7_map = {
        "No problem":         "I can concentrate on my studies without any problem.",
        "Slightly hard":      "It is slightly hard to concentrate on my studies lately.",
        "Very hard":          "It is very hard to concentrate on anything. My mind keeps wandering.",
        "Cannot concentrate at all": "I cannot concentrate at all. Even simple tasks feel impossible and overwhelming.",
    }

    # Q8: Social withdrawal (PHQ-9 Q8 variant)
    q8_map = {
        "Never":     "I feel connected to the people around me.",
        "Sometimes": "I sometimes feel disconnected and distant from people.",
        "Often":     "I often feel disconnected and isolated from everyone around me.",
        "Always":    "I always feel completely disconnected and alone. Nobody understands me.",
    }

    # Q9: Hopelessness (PHQ-9 Q9 variant)
    q9_map = {
        "Never":     "What I do feels meaningful.",
        "Sometimes": "I sometimes feel that nothing I do makes any difference.",
        "Often":     "I often feel like nothing I do matters or has any value.",
        "Always":    "I always feel like nothing I do matters at all. Everything feels pointless and empty.",
    }

    sentences = [
        q1_map.get(q1, f"Regarding energy: {q1}."),
        q2_map.get(q2, f"Regarding interest: {q2}."),
        q3_map.get(q3, f"Regarding mood: {q3}."),
        q4_map.get(q4, f"Sleep situation: {q4}."),
        q5_map.get(q5, f"Appetite: {q5}."),
        q6_map.get(q6, f"Feeling of burden: {q6}."),
        q7_map.get(q7, f"Concentration: {q7}."),
        q8_map.get(q8, f"Social connection: {q8}."),
        q9_map.get(q9, f"Sense of purpose: {q9}."),
    ]

    if q10:
        sentences.append(f"In my own words this week: {q10}")

    return " ".join(sentences)


# ─── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== MindBridge Preprocessing Self-Test ===")

    # Test 1: Basic cleaning
    result = clean_text("I feel hopeless and nothing helps me!!!")
    assert "hopeless" in result, "FAIL: 'hopeless' should be preserved"
    assert "nothing" in result, "FAIL: 'nothing' (negation) should be preserved"
    print(f"Test 1 PASS — clean_text: {repr(result)}")

    # Test 2: URL removal
    result = clean_text("Visit https://example.com for help")
    assert "http" not in result, "FAIL: URL not removed"
    print(f"Test 2 PASS — URL removed: {repr(result)}")

    # Test 3: Negation preservation
    result = clean_text("I am not okay and cannot sleep")
    assert "not" in result or "cannot" in result, "FAIL: negation words removed"
    print(f"Test 3 PASS — negation preserved: {repr(result)}")

    # Test 4: HTML removal
    result = clean_text("<p>I feel <b>terrible</b></p>")
    assert "<" not in result, "FAIL: HTML not removed"
    print(f"Test 4 PASS — HTML removed: {repr(result)}")

    # Test 5: form_responses_to_text
    worst_case = {
        "q1": "Almost Every Day",
        "q2": "Almost Every Day",
        "q3": "Almost Every Day",
        "q4": "Very disturbed",
        "q5": "Skipping meals often",
        "q6": "Often",
        "q7": "Cannot concentrate at all",
        "q8": "Always",
        "q9": "Always",
        "q10": "I feel very alone and nothing helps me."
    }
    text = form_responses_to_text(worst_case)
    assert "hopeless" in text, "FAIL: depression markers missing from form conversion"
    assert "nothing" in text, "FAIL: 'nothing' missing from form conversion"
    cleaned = clean_text(text)
    assert len(cleaned.split()) >= 10, "FAIL: cleaned form text too short"
    print(f"Test 5 PASS — form_responses_to_text: {len(text)} chars, {len(cleaned.split())} tokens after cleaning")

    print("\npreprocessing.py: ALL PASS")
