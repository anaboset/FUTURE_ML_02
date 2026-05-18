"""
preprocess.py
-------------
Clinical text preprocessing pipeline for pharmacy support tickets.

Designed specifically for healthcare/pharmacy text:
  - Preserves medically significant terms (drug names, dosages, frequencies)
  - Removes noise without destroying clinical signal
  - Normalizes pharmacy-specific abbreviations
  - Flags potential PHI (Protected Health Information) patterns

Pipeline order matters — steps are applied in sequence.
"""

import re
import string
import unicodedata
from typing import Optional
from helpers.patterns import PHARMACY_ABBREVIATIONS, PHI_PATTERNS


def detect_phi(text: str) -> dict:
    """
    Detect potential PHI patterns in ticket text.
    Returns a dict of {PHI_type: [found_matches]}.
    
    In a production system, this would trigger a HIPAA compliance workflow.
    For this project, it demonstrates awareness of healthcare data obligations.
    """
    found = {}
    for phi_type, pattern in PHI_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found[phi_type] = matches
    return found


def redact_phi(text: str) -> str:
    """
    Replace detected PHI with safe placeholder tokens.
    Preserves text structure while removing identifiable information.
    """
    for phi_type, pattern in PHI_PATTERNS.items():
        placeholder = f"[{phi_type}]"
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    return text


# ─────────────────────────────────────────────
# CORE PREPROCESSING STEPS
# ─────────────────────────────────────────────

def to_lowercase(text: str) -> str:
    return text.lower()


def normalize_unicode(text: str) -> str:
    """Normalize accented characters (patients use various keyboard layouts)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")


def expand_pharmacy_abbreviations(text: str) -> str:
    """
    Expand pharmacy-specific abbreviations before stopword removal.
    Must run BEFORE lowercasing to preserve pattern matching accuracy,
    OR after — here we apply it after lowercase since patterns are lowercase.
    """
    for pattern, expansion in PHARMACY_ABBREVIATIONS.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", "", text)


def remove_email_addresses(text: str) -> str:
    return re.sub(r"\S+@\S+", "[EMAIL]", text)


def normalize_dosages(text: str) -> str:
    """
    Preserve dosage information as a single token instead of splitting it.
    e.g. '500mg' -> '500mg', '10 mg' -> '10mg'
    This prevents the vectorizer from treating '500' and 'mg' as separate features.
    """
    # Merge number + unit (e.g. "10 mg" -> "10mg")
    text = re.sub(r"(\d+)\s*(mg|mcg|ml|g|iu|units?|tabs?|caps?)", r"\1\2", text, flags=re.IGNORECASE)
    return text


def remove_punctuation(text: str, keep_hyphens: bool = True) -> str:
    """
    Remove punctuation, optionally keeping hyphens (important for drug names
    like 'co-amoxiclav', 'non-steroidal').
    """
    if keep_hyphens:
        # Remove all punctuation except hyphens
        punct = string.punctuation.replace("-", "")
        text = text.translate(str.maketrans("", "", punct))
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_numbers_standalone(text: str) -> str:
    """
    Remove standalone numbers (e.g. ticket IDs, zip codes) but NOT dosages.
    Call AFTER normalize_dosages to preserve clinical values.
    """
    # Remove numbers that are not attached to a unit
    text = re.sub(r"\b\d+\b(?!\s*(?:mg|mcg|ml|g|iu|units?|tabs?|caps?))", " ", text)
    return text


# ─────────────────────────────────────────────
# STOPWORD HANDLING
# We use a MODIFIED stopword list for healthcare text.
# Standard NLP stopwords remove too many clinically relevant terms.
# ─────────────────────────────────────────────

# Words to always KEEP even if in standard stopword list
CLINICAL_KEEP_WORDS = {
    "not", "no", "never", "without",     # Negation is clinically critical
    "more", "less", "worse", "better",   # Symptom progression
    "before", "after", "during",         # Temporal context
    "still", "again", "always", "every", # Frequency
    "only", "just", "severe", "mild",    # Intensity
    "cannot", "can't", "unable",         # Inability to take medication
}

def get_stopwords() -> set:
    """
    Return a clinically-appropriate stopword set.
    Attempts to use NLTK; falls back to a built-in minimal list.
    """
    try:
        from nltk.corpus import stopwords
        import nltk
        try:
            sw = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            sw = set(stopwords.words("english"))
        # Remove clinically important words that NLTK would strip
        sw -= CLINICAL_KEEP_WORDS
        return sw
    except ImportError:
        # Fallback minimal stopword list
        return {
            "i", "me", "my", "myself", "we", "our", "you", "your",
            "he", "she", "it", "they", "them", "their", "what", "which",
            "who", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should",
            "a", "an", "the", "and", "but", "or", "for", "of", "at",
            "by", "from", "with", "about", "as", "into", "through",
            "to", "in", "on", "up", "out", "so", "than", "too", "very",
        } - CLINICAL_KEEP_WORDS


def remove_stopwords(tokens: list, stopwords: Optional[set] = None) -> list:
    if stopwords is None:
        stopwords = get_stopwords()
    return [t for t in tokens if t not in stopwords]


# ─────────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────────

def tokenize(text: str) -> list:
    """
    Tokenize text, preserving hyphenated drug names as single tokens.
    e.g. 'co-amoxiclav' stays as one token.
    """
    # Split on whitespace, keep hyphenated terms whole
    tokens = re.findall(r"\b[\w][\w-]*[\w]\b|\b\w\b", text)
    return [t for t in tokens if len(t) > 1]  # drop single chars


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

_STOPWORDS = None  # Lazy-loaded once

def preprocess(
    text: str,
    redact_phi_data: bool = True,
    expand_abbreviations: bool = True,
    remove_stops: bool = True,
    return_tokens: bool = False,
) -> str:
    """
    Full preprocessing pipeline for a pharmacy support ticket.

    Parameters
    ----------
    text : str
        Raw ticket text (subject + body).
    redact_phi_data : bool
        Replace detected PHI with placeholder tokens (recommended: True).
    expand_abbreviations : bool
        Expand pharmacy abbreviations before vectorization.
    remove_stops : bool
        Remove stopwords (keeping clinical negation terms).
    return_tokens : bool
        If True, return list of tokens; if False, return joined string.

    Returns
    -------
    str or list
        Cleaned text string or token list.
    """
    global _STOPWORDS

    if not isinstance(text, str) or not text.strip():
        return [] if return_tokens else ""

    # Step 1: Redact PHI before any other processing
    if redact_phi_data:
        text = redact_phi(text)

    # Step 2: Normalize encoding
    text = normalize_unicode(text)

    # Step 3: Lowercase
    text = to_lowercase(text)

    # Step 4: Remove URLs
    text = remove_urls(text)

    # Step 5: Expand pharmacy abbreviations
    if expand_abbreviations:
        text = expand_pharmacy_abbreviations(text)

    # Step 6: Normalize dosages BEFORE removing numbers
    text = normalize_dosages(text)

    # Step 7: Remove punctuation (keep hyphens for drug names)
    text = remove_punctuation(text, keep_hyphens=True)

    # Step 8: Remove standalone numbers
    text = remove_numbers_standalone(text)

    # Step 9: Tokenize
    tokens = tokenize(text)

    # Step 10: Remove stopwords
    if remove_stops:
        if _STOPWORDS is None:
            _STOPWORDS = get_stopwords()
        tokens = remove_stopwords(tokens, _STOPWORDS)

    # Step 11: Remove very short tokens (likely noise after cleaning)
    tokens = [t for t in tokens if len(t) >= 2]

    if return_tokens:
        return tokens

    return " ".join(tokens)


def preprocess_batch(
    texts: list,
    redact_phi_data: bool = True,
    expand_abbreviations: bool = True,
    remove_stops: bool = True,
    verbose: bool = True,
) -> list:
    """
    Apply preprocessing pipeline to a list of ticket texts.

    Parameters
    ----------
    texts : list of str
    verbose : bool
        Print progress every 500 records.

    Returns
    -------
    list of str
        Cleaned text strings.
    """
    cleaned = []
    for i, text in enumerate(texts):
        cleaned.append(preprocess(
            text,
            redact_phi_data=redact_phi_data,
            expand_abbreviations=expand_abbreviations,
            remove_stops=remove_stops,
        ))
        if verbose and (i + 1) % 500 == 0:
            print(f"  Preprocessed {i + 1}/{len(texts)} tickets...")

    if verbose:
        print(f"  Done. Preprocessed {len(texts)} tickets total.")
    return cleaned


# ─────────────────────────────────────────────
# QUICK DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sample_tickets = [
        "Hi, I was taking metformin 500mg BID but I started getting severe nausea and vomiting. "
        "Should I stop taking it? My Dr said to take it PC.",

        "My insurance denied my prior authorization for Humira again. "
        "I've been on this medication for 2 years and now they want step therapy. "
        "This is urgent — I only have 3 doses left.",

        "I accidentally took double my warfarin dose this morning (10mg instead of 5mg). "
        "What should I do? I'm scared.",

        "Can you refill my lisinopril 10mg? I have 4 tabs left and my appt is next week.",
    ]

    print("=" * 60)
    print("PHARMACY TICKET PREPROCESSING DEMO")
    print("=" * 60)

    for i, ticket in enumerate(sample_tickets, 1):
        phi = detect_phi(ticket)
        cleaned = preprocess(ticket)

        print(f"\n[Ticket {i}]")
        print(f"  Original : {ticket[:80]}...")
        print(f"  PHI found: {phi if phi else 'None'}")
        print(f"  Cleaned  : {cleaned[:80]}...")
