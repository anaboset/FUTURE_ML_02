# ─────────────────────────────────────────────
# PRIORITY KEYWORD LISTS  (clinical judgment)
# ─────────────────────────────────────────────

CRITICAL_KEYWORDS = [
    "allergic reaction", "anaphylaxis", "anaphylactic",
    "overdose", "took too much", "accidental ingestion",
    "chest pain", "difficulty breathing", "can't breathe",
    "unconscious", "not responding", "seizure", "severe bleeding",
    "suicidal", "want to end", "self harm",
    "swallowed wrong medication", "child swallowed",
    "accidentally swallowed", "accidental overdose", "swallowed my",
    "my child", "my toddler", "my baby", "my kid",
]

HIGH_KEYWORDS = [
    "drug interaction", "interact", "taking both",
    "wrong medication", "wrong drug", "wrong dose", "dispensing error",
    "side effect", "adverse", "reaction", "rash", "hives",
    "severe nausea", "vomiting blood", "blood in stool",
    "last pill", "ran out", "no more medication", "running out",
    "missed insulin", "out of insulin", "diabetic",
    "blood pressure medication", "heart medication", "blood thinner",
    "anticoagulant", "warfarin", "heparin",
    "controlled", "narcotic", "opioid", "schedule ii",
]

MEDIUM_KEYWORDS = [
    "prior authorization", "pa denied", "not covered", "formulary",
    "insurance denied", "step therapy",
    "refill", "running low", "need refill",
    "prescription expired", "no refills left",
    "transfer", "transfer prescription",
    "copay too high", "can't afford",
]

PRIORITY_LEVELS = ["CRITICAL", "High", "Medium", "Low"]

PRIORITY_SLA = {
    "CRITICAL": "Immediate — call emergency services / on-call pharmacist",
    "High":     "Within 2 hours",
    "Medium":   "Within 24 hours",
    "Low":      "Within 48-72 hours",
}


ADR_SIGNAL_KEYWORDS = [
    "rash", "hives", "swelling", "itching",
    "nausea", "vomiting", "diarrhea",
    "dizziness", "fainting", "lightheaded",
    "heart pounding", "palpitations",
    "vision changes", "blurred vision",
    "unusual bleeding", "bruising",
    "yellowing", "jaundice",
    "difficulty breathing", "wheezing",
    "muscle pain", "weakness",
    "confusion", "memory loss",
]