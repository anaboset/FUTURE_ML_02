"""
taxonomy.py
-----------
Clinical category taxonomy, priority logic, and routing rules
for pharmacy/healthcare support ticket triage.

Domain knowledge authored by a pharmacy professional.
This module is the clinical brain of the system — it encodes
the decision rules that a pharmacist would apply manually.
"""
from helpers.categories import CATEGORIES, CATEGORY_DESCRIPTIONS
from helpers.keywords import *
from helpers.routing import ROUTING_MAP


def assign_priority(text: str, category: str) -> str:
    """
    Assign a clinical priority level to a support ticket.

    Priority ladder (first match wins):
      1. CRITICAL safety keyword  ->  CRITICAL
      2. High-risk category       ->  High
      3. High-urgency keyword     ->  High
      4. Medium-urgency category  ->  Medium
      5. Medium-urgency keyword   ->  Medium
      6. Default                  ->  Low
    """
    text_lower = text.lower()

    if any(kw in text_lower for kw in CRITICAL_KEYWORDS):
        return "CRITICAL"

    if category in ("Drug Interaction Query",
                    "Side Effect / Adverse Drug Reaction",
                    "Controlled Substance",
                    "Prescription Issue"):
        return "High"

    if any(kw in text_lower for kw in HIGH_KEYWORDS):
        return "High"

    if category in ("Prior Authorization / Insurance", "Refill Request"):
        return "Medium"

    if any(kw in text_lower for kw in MEDIUM_KEYWORDS):
        return "Medium"

    return "Low"



def flag_for_adr_reporting(text: str) -> bool:
    """
    Flag a ticket as a potential ADR requiring pharmacovigilance
    review and possible FDA MedWatch reporting.
    """
    return any(signal in text.lower() for signal in ADR_SIGNAL_KEYWORDS)


def get_routing(category: str) -> dict:
    """Return the routing recommendation dict for a given category."""
    return ROUTING_MAP.get(category, ROUTING_MAP["General Inquiry"])
