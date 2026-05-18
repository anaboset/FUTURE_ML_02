"""
predict.py
----------
Inference module — wraps the trained models into a clean prediction API.

This is what gets called by the Streamlit app and any downstream service.
Designed to be stateless and importable anywhere.
"""

import pickle
import os
from dataclasses import dataclass, field
from typing import Optional

from src.preprocess import preprocess
from src.taxonomy   import assign_priority, flag_for_adr_reporting, get_routing, PRIORITY_SLA


# ─────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class TriageResult:
    """
    Full triage output for a single support ticket.
    Structured as a dataclass for clean access and easy serialization.
    """
    # Input
    raw_text:          str

    # Predictions
    category:          str
    category_confidence: float           # 0.0 – 1.0

    priority:          str               # CRITICAL / High / Medium / Low
    priority_confidence: float

    # Clinical flags
    adr_flag:          bool              # Potential adverse drug reaction
    adr_note:          str = ""

    # Routing
    routing_team:      str = ""
    routing_action:    str = ""
    routing_escalate:  str = ""
    routing_icon:      str = ""

    # SLA
    sla:               str = ""

    # Model metadata
    category_model:    str = ""
    priority_model:    str = ""

    def to_dict(self) -> dict:
        return self.__dict__

    def summary(self) -> str:
        lines = [
            f"📋  Category   : {self.routing_icon}  {self.category}  "
            f"(confidence: {self.category_confidence:.0%})",
            f"🚨  Priority   : {self.priority}  "
            f"(confidence: {self.priority_confidence:.0%})",
            f"⏱️   SLA        : {self.sla}",
            f"👥  Route to   : {self.routing_team}",
            f"📝  Action     : {self.routing_action}",
            f"🔬  ADR Flag   : {'⚠️  YES — pharmacovigilance review required' if self.adr_flag else 'No'}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# MODEL LOADER  (singleton pattern — load once)
# ─────────────────────────────────────────────

_LOADED_MODELS = {}

def _load_model(task: str, models_dir: str = "../models") -> dict:
    """Load and cache a model from disk."""
    if task not in _LOADED_MODELS:
        path = os.path.join(models_dir, f"{task}_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run `python train.py` first to train and save models."
            )
        with open(path, "rb") as f:
            _LOADED_MODELS[task] = pickle.load(f)
    return _LOADED_MODELS[task]


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_ticket(
    raw_text:   str,
    models_dir: str = "../models",
) -> TriageResult:
    """
    Run full triage on a single support ticket.

    Parameters
    ----------
    raw_text   : str  —  raw ticket text (subject + body)
    models_dir : str  —  directory containing saved .pkl model files

    Returns
    -------
    TriageResult dataclass with all predictions and routing info.
    """
    # 1. Preprocess
    cleaned_text = preprocess(raw_text, redact_phi_data=True)

    # 2. Load models
    cat_bundle  = _load_model("category", models_dir)
    prio_bundle = _load_model("priority", models_dir)

    cat_pipeline  = cat_bundle["model"]
    prio_pipeline = prio_bundle["model"]

    # 3. Predict category
    category     = cat_pipeline.predict([cleaned_text])[0]
    cat_proba    = cat_pipeline.predict_proba([cleaned_text])[0]
    cat_conf     = float(cat_proba.max())

    # 4. Predict priority (ML model)
    ml_priority  = prio_pipeline.predict([cleaned_text])[0]
    prio_proba   = prio_pipeline.predict_proba([cleaned_text])[0]
    prio_conf    = float(prio_proba.max())

    # 5. Rule-based priority override (clinical safety net)
    # The rule-based system always has veto power over the ML model
    # for CRITICAL escalation — patient safety is non-negotiable.
    rule_priority = assign_priority(raw_text, category)
    if rule_priority == "CRITICAL":
        priority = "CRITICAL"
        prio_conf = 1.0   # Rules are deterministic
    else:
        priority = ml_priority

    # 6. Clinical flags
    adr_flag = flag_for_adr_reporting(raw_text)
    adr_note = (
        "⚠️  Possible Adverse Drug Reaction detected. "
        "Clinical review and FDA MedWatch reporting evaluation required."
        if adr_flag else ""
    )

    # 7. Routing
    routing = get_routing(category)

    return TriageResult(
        raw_text              = raw_text,
        category              = category,
        category_confidence   = cat_conf,
        priority              = priority,
        priority_confidence   = prio_conf,
        adr_flag              = adr_flag,
        adr_note              = adr_note,
        routing_team          = routing["team"],
        routing_action        = routing["action"],
        routing_escalate      = routing["escalate_to"],
        routing_icon          = routing["icon"],
        sla                   = PRIORITY_SLA.get(priority, ""),
        category_model        = cat_bundle["model_name"],
        priority_model        = prio_bundle["model_name"],
    )


def predict_batch(
    texts:      list,
    models_dir: str = "../models",
    verbose:    bool = True,
) -> list:
    """
    Run triage on a list of ticket texts.
    Returns a list of TriageResult objects.
    """
    results = []
    for i, text in enumerate(texts):
        results.append(predict_ticket(text, models_dir))
        if verbose and (i + 1) % 100 == 0:
            print(f"  Triaged {i + 1}/{len(texts)} tickets...")
    return results


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_tickets = [
        # Should be: Prescription Issue / High
        "I received atorvastatin 40mg but my prescription clearly says 20mg. "
        "I already took this morning's dose.",

        # Should be: Drug Interaction Query / High
        "I started taking warfarin last week and I also take aspirin daily. "
        "My GP didn't mention any interaction — is this safe?",

        # Should be: CRITICAL
        "I think my child accidentally swallowed some of my metformin tablets. "
        "About 5 tablets of 500mg. She's 3 years old. What do I do?",

        # Should be: Prior Authorization / Medium
        "My insurance denied my prior authorization for Humira again. "
        "I've been on biologics for 3 years. Third denial this year.",

        # Should be: Side Effect / ADR + ADR flag
        "Since starting lisinopril two weeks ago I have a constant dry cough "
        "and last night I had some facial swelling. Is this normal?",

        # Should be: General Inquiry / Low
        "What time does the pharmacy close on Sundays?",
    ]

    print("=" * 60)
    print("PHARMACY TRIAGE SYSTEM — INFERENCE DEMO")
    print("=" * 60)

    for i, ticket in enumerate(test_tickets, 1):
        print(f"\n{'─'*60}")
        print(f"Ticket {i}: {ticket[:70]}...")
        try:
            result = predict_ticket(ticket)
            print(result.summary())
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")
