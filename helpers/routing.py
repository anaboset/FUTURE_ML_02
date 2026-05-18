ROUTING_MAP = {
    "Prescription Issue": {
        "team":        "Dispensing Pharmacist",
        "icon":        "🟠",
        "action":      "Verify dispensing record and original Rx; contact prescriber if needed",
        "escalate_to": "Pharmacy Manager",
    },
    "Drug Interaction Query": {
        "team":        "Clinical Pharmacist",
        "icon":        "🔴",
        "action":      "Clinical review required; use Lexicomp / Micromedex",
        "escalate_to": "Chief Pharmacist",
    },
    "Prior Authorization / Insurance": {
        "team":        "PA / Insurance Specialist",
        "icon":        "🟡",
        "action":      "Review formulary status, initiate PA if applicable",
        "escalate_to": "Pharmacy Director",
    },
    "Refill Request": {
        "team":        "Pharmacy Technician",
        "icon":        "🟡",
        "action":      "Check refill eligibility; contact prescriber if no refills remain",
        "escalate_to": "Pharmacist on duty",
    },
    "Side Effect / Adverse Drug Reaction": {
        "team":        "Clinical Pharmacist + ADR Team",
        "icon":        "🔴",
        "action":      "Clinical assessment; document ADR; evaluate MedWatch reporting obligation",
        "escalate_to": "Chief Pharmacist + Pharmacovigilance Officer",
    },
    "Medication Counseling": {
        "team":        "Staff Pharmacist",
        "icon":        "🟢",
        "action":      "Schedule counseling; provide patient education materials",
        "escalate_to": "Clinical Pharmacist (if complex)",
    },
    "Billing & Copay": {
        "team":        "Billing Department",
        "icon":        "🟢",
        "action":      "Review billing record; check copay assistance eligibility",
        "escalate_to": "Pharmacy Manager",
    },
    "Controlled Substance": {
        "team":        "Compliance Team + Pharmacist",
        "icon":        "🟠",
        "action":      "Verify Rx validity; check PDMP; document all actions",
        "escalate_to": "Pharmacy Director + DEA Compliance Officer",
    },
    "General Inquiry": {
        "team":        "Pharmacy Technician",
        "icon":        "🟢",
        "action":      "Provide standard information; escalate if clinical question emerges",
        "escalate_to": "Staff Pharmacist",
    },
}