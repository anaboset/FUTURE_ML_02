# ─────────────────────────────────────────────
# PHARMACY ABBREVIATION NORMALIZER
# Expands shorthand that patients and staff use in tickets.
# This improves model feature quality significantly.
# ─────────────────────────────────────────────

PHARMACY_ABBREVIATIONS = {
    # Dosage frequencies
    r"\bqd\b":    "once daily",
    r"\bbid\b":   "twice daily",
    r"\btid\b":   "three times daily",
    r"\bqid\b":   "four times daily",
    r"\bprn\b":   "as needed",
    r"\bac\b":    "before meals",
    r"\bpc\b":    "after meals",
    r"\bhs\b":    "at bedtime",
    r"\bstat\b":  "immediately",
    r"\bq\d+h\b": "every hours",   # e.g. q8h -> every hours (freq preserved)

    # Routes
    r"\bpo\b":    "by mouth",
    r"\biv\b":    "intravenous",
    r"\bim\b":    "intramuscular",
    r"\bsc\b":    "subcutaneous",
    r"\bsl\b":    "sublingual",
    r"\binh\b":   "inhaled",
    r"\btop\b":   "topical",

    # Prescription shorthand
    r"\brx\b":    "prescription",
    r"\bdr\b":    "doctor",
    r"\bmd\b":    "doctor",
    r"\bdo\b":    "doctor",
    r"\bnp\b":    "nurse practitioner",
    r"\bpa\b":    "prior authorization",
    r"\botc\b":   "over the counter",
    r"\bgeneric\b": "generic",

    # Common clinical shorthand
    r"\badr\b":   "adverse drug reaction",
    r"\bddi\b":   "drug drug interaction",
    r"\bpdmp\b":  "prescription drug monitoring program",
    r"\bbp\b":    "blood pressure",
    r"\bhr\b":    "heart rate",
    r"\bdm\b":    "diabetes",
    r"\bhtm\b":   "hypertension",
    r"\bchf\b":   "heart failure",
    r"\bcopd\b":  "lung disease",
}

# ─────────────────────────────────────────────
# PHI DETECTION PATTERNS (HIPAA)
# We detect but do NOT store or log these in production.
# In this system, they are replaced with placeholder tokens.
# ─────────────────────────────────────────────

PHI_PATTERNS = {
    "SSN":          r"\b\d{3}-\d{2}-\d{4}\b",
    "PHONE":        r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "EMAIL":        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "DATE_OF_BIRTH":r"\b(?:dob|date of birth|born on|born)\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    "MRN":          r"\b(?:mrn|patient id|patient number|chart)\s*:?\s*\d{4,10}\b",
}
