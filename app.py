import sys
import os
import time
from pathlib import Path
import pickle

import streamlit as st
import pandas as pd

sys.path.insert(0, "src")

from preprocess import preprocess
from taxonomy import (
    assign_priority,
    flag_for_adr_reporting,
    get_routing,
    PRIORITY_SLA,
    CATEGORIES,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PharmaTriage",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS (FROM app2.py UI)
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace;
    }

    .stTextArea textarea {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.92rem;
    }

    .priority-critical {
        background: #1a0a0a;
        border: 2px solid #ff2222;
        border-radius: 10px;
        padding: 18px;
        margin: 10px 0;
        color: #ff6666;
        font-family: 'IBM Plex Mono', monospace;
    }

    .priority-high {
        background: #1a0f0a;
        border: 2px solid #ff6600;
        border-radius: 10px;
        padding: 18px;
        margin: 10px 0;
        color: #ff9944;
        font-family: 'IBM Plex Mono', monospace;
    }

    .priority-medium {
        background: #1a1a0a;
        border: 2px solid #ddcc00;
        border-radius: 10px;
        padding: 18px;
        margin: 10px 0;
        color: #eecc44;
        font-family: 'IBM Plex Mono', monospace;
    }

    .priority-low {
        background: #0a1a0a;
        border: 2px solid #22aa44;
        border-radius: 10px;
        padding: 18px;
        margin: 10px 0;
        color: #44cc66;
        font-family: 'IBM Plex Mono', monospace;
    }

    .routing-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.92rem;
    }

    .adr-warning {
        background: #1a1200;
        border: 2px solid #ffaa00;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        color: #ffcc44;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    .subtitle {
        color: #9ca3af;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: #238636;
        color: white;
        border: none;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        width: 100%;
    }

    .stButton > button:hover {
        background: #2ea043;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# MODEL LOADING (ORIGINAL app(copy).py FUNCTIONALITY)
# ─────────────────────────────────────────────────────────────

@st.cache_resource

def load_models():
    """Load trained models once and cache them."""
    models = {}

    for task in ["category", "priority"]:
        path = f"models/{task}_model.pkl"

        if os.path.exists(path):
            with open(path, "rb") as f:
                models[task] = pickle.load(f)

    return models


MODELS = load_models()
MODELS_AVAILABLE = len(MODELS) == 2

# ─────────────────────────────────────────────────────────────
# TRIAGE PIPELINE (ORIGINAL FUNCTIONALITY)
# ─────────────────────────────────────────────────────────────


def run_triage(raw_text: str) -> dict:
    """Run full triage pipeline on raw ticket text."""

    cleaned = preprocess(raw_text, redact_phi_data=True)

    if MODELS_AVAILABLE:
        cat_pipeline = MODELS["category"]["model"]
        prio_pipeline = MODELS["priority"]["model"]

        category = cat_pipeline.predict([cleaned])[0]
        cat_proba = cat_pipeline.predict_proba([cleaned])[0]
        cat_conf = float(cat_proba.max())

        ml_priority = prio_pipeline.predict([cleaned])[0]
        prio_proba = prio_pipeline.predict_proba([cleaned])[0]
        prio_conf = float(prio_proba.max())

        cat_model_name = MODELS["category"]["model_name"]
        prio_model_name = MODELS["priority"]["model_name"]

    else:
        category = "General Inquiry"
        cat_conf = 0.0
        ml_priority = "Low"
        prio_conf = 0.0
        cat_model_name = "Not trained"
        prio_model_name = "Not trained"

    COUNSELING_SIGNALS = [
        "how do i take",
        "how should i take",
        "when should i take",
        "should i take with food",
        "avoid any foods",
        "avoid foods",
        "can i take with",
        "safe to take with",
        "what foods",
        "how to use",
        "how to store",
        "can i split",
        "can i cut",
        "missed dose",
        "forgot to take",
        "what if i miss",
        "how long does it take",
        "when will it start working",
        "can i drink alcohol",
        "safe during pregnancy",
        "safe while breastfeeding",
        "didn't explain",
        "pharmacist didn't",
        "doctor didn't explain",
        "how do i use",
        "instructions",
        "how many times",
    ]

    raw_lower = raw_text.lower()

    if any(sig in raw_lower for sig in COUNSELING_SIGNALS):
        category = "Medication Counseling"
        cat_conf = max(cat_conf, 0.72)

    rule_priority = assign_priority(raw_text, category)

    if rule_priority == "CRITICAL":
        priority = "CRITICAL"
    elif rule_priority == "High" and ml_priority in ("Medium", "Low"):
        priority = "High"
    else:
        priority = ml_priority

    confidence = cat_conf
    needs_human_review = confidence < 0.65

    adr_flag = flag_for_adr_reporting(raw_text)
    routing = get_routing(category)
    sla = PRIORITY_SLA.get(priority, "")

    priority_icons = {
        "CRITICAL": "🚨",
        "High": "🔴",
        "Medium": "🟡",
        "Low": "🟢",
    }

    return {
        "category": category,
        "confidence": confidence,
        "priority": priority,
        "priority_icon": priority_icons.get(priority, "🟢"),
        "priority_confidence": prio_conf,
        "adr_flag": adr_flag,
        "needs_human_review": needs_human_review,
        "routing": routing,
        "sla": sla,
        "cleaned_text": cleaned,
        "cat_model": cat_model_name,
        "prio_model": prio_model_name,
    }


# ─────────────────────────────────────────────────────────────
# SAMPLE TICKETS
# ─────────────────────────────────────────────────────────────

SAMPLE_TICKETS = {
    "🚨 Critical — Allergic Reaction": "I took my first dose of amoxicillin an hour ago and my throat is starting to swell. I'm having difficulty breathing. I also have hives all over my chest.",
    "🔴 High — Dispensing Error": "You gave me the completely WRONG medication! My prescription was for metoprolol 25mg for my heart but you gave me metformin. I've been taking it for 2 days. This is a serious mistake!!",
    "🔴 High — Drug Interaction": "My cardiologist just put me on warfarin and I'm already taking aspirin daily. My GP never told me if this is safe. I've been reading about bleeding risks online and I'm very worried.",
    "🟡 Medium — PA Denial": "My insurance denied my prior authorization for Humira again. This is the second time. My rheumatologist says I need this medication specifically and there are no alternatives for me.",
    "🟡 Medium — Refill": "Hi, can you please refill my lisinopril 10mg? I have about 4 tablets left and my next appointment with my doctor is in 2 weeks.",
    "🟢 Low — General Inquiry": "Hi, what time does the pharmacy close on Saturdays? I need to pick up a prescription.",
    "🟢 Low — Counseling": "how should i take gabapentin? should it be with food or on an empty stomach?",
}

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">💊 PharmaTriage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Automatic classification, prioritization, clinical routing, and pharmacovigilance analysis for pharmacy support tickets.</div>',
    unsafe_allow_html=True,
)

st.divider()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ System Settings")

    confidence_threshold = st.slider(
        "Human Review Threshold",
        min_value=0.40,
        max_value=0.90,
        value=0.65,
        step=0.05,
    )

    st.divider()

    st.markdown("## 📡 Model Status")

    if MODELS_AVAILABLE:
        st.success("✅ ML models loaded")
        st.caption(f"Category model: {MODELS['category']['model_name']}")
        st.caption(f"Priority model: {MODELS['priority']['model_name']}")
    else:
        st.warning("⚠️ Models not trained yet")

    st.divider()

    st.markdown("## 🏷️ Categories")
    for cat in CATEGORIES:
        st.caption(f"• {cat}")

    st.divider()

    st.markdown("## 🚦 Priority Guide")
    st.markdown(
        """
🚨 **CRITICAL** — Immediate intervention  
🔴 **High** — 2 hour SLA  
🟡 **Medium** — 24 hour SLA  
🟢 **Low** — 48–72 hour SLA
"""
    )

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────


tab1, tab2, tab3 = st.tabs([
    "🔍 Single Ticket",
    "📁 Batch Processing",
    "📊 About System",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — SINGLE TICKET
# ─────────────────────────────────────────────────────────────

with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 📨 Ticket Input")

        sample_choice = st.selectbox(
            "Load example ticket:",
            ["— type your own —"] + list(SAMPLE_TICKETS.keys()),
        )

        default_text = ""

        if sample_choice != "— type your own —":
            default_text = SAMPLE_TICKETS[sample_choice]

        ticket_text = st.text_area(
            "Ticket text",
            value=default_text,
            height=220,
            placeholder="Paste or type a pharmacy support ticket here...",
            label_visibility="collapsed",
        )

        word_count = len(ticket_text.split()) if ticket_text.strip() else 0
        st.caption(f"{word_count} words")

        analyze_btn = st.button(
            "⚡ Analyze Ticket",
            type="primary",
        )

    with col_output:
        st.markdown("### 🎯 Triage Result")

        if analyze_btn and ticket_text.strip():
            with st.spinner("Running clinical triage analysis..."):
                time.sleep(0.25)
                result = run_triage(ticket_text)

            priority = result["priority"]
            css_class = f"priority-{priority.lower()}"
            icon = result["priority_icon"]

            st.markdown(
                f'<div class="{css_class}">'
                f'<strong>{icon} {priority} PRIORITY</strong><br>'
                f'<small>SLA: {result["sla"]}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(f"### 📋 {result['category']}")

            conf = result["confidence"]
            st.progress(conf, text=f"Category Confidence: {conf:.0%}")

            if result["needs_human_review"] or conf < confidence_threshold:
                st.warning(
                    f"👤 Human review recommended — confidence below {confidence_threshold:.0%}."
                )

            routing = result["routing"]

            st.markdown(
                f'<div class="routing-box">'
                f'<strong>🏥 Route To:</strong> {routing["icon"]} {routing["team"]}<br><br>'
                f'<strong>Action:</strong> {routing["action"]}<br><br>'
                f'<strong>Escalate:</strong> {routing["escalate_to"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

            if result["adr_flag"]:
                st.markdown(
                    '<div class="adr-warning">'
                    '⚠️ <strong>Potential ADR detected.</strong><br>'
                    'Clinical review recommended. Evaluate for FDA MedWatch reporting.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.success("✅ No ADR signals detected")

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f'<div class="metric-card"><h4>Priority</h4><h2>{priority}</h2></div>',
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f'<div class="metric-card"><h4>Confidence</h4><h2>{conf:.0%}</h2></div>',
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f'<div class="metric-card"><h4>SLA</h4><h2>{result["sla"]}</h2></div>',
                    unsafe_allow_html=True,
                )

            with st.expander("🔎 Analysis Details"):
                st.markdown("**Preprocessed text sent to models:**")
                st.code(result["cleaned_text"])
                st.caption(
                    f"Category model: {result['cat_model']} | Priority model: {result['prio_model']}"
                )

        elif analyze_btn:
            st.warning("Please enter ticket text before analyzing.")

        else:
            st.info("Enter a ticket and click Analyze Ticket.")

# ─────────────────────────────────────────────────────────────
# TAB 2 — BATCH PROCESSING
# ─────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### 📁 Batch Ticket Processing")
    st.markdown("Upload a CSV with a `text` column to process tickets in bulk.")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)

        if "text" not in df_raw.columns:
            st.error("CSV must contain a 'text' column.")
            st.info(f"Columns found: {list(df_raw.columns)}")

        else:
            st.success(f"Loaded {len(df_raw)} tickets.")
            st.dataframe(df_raw.head(5), use_container_width=True)

            if st.button("🚀 Run Batch Triage"):
                results = []
                progress = st.progress(0)
                status = st.empty()

                for i, row in df_raw.iterrows():
                    raw_text = str(row["text"])
                    ticket_id = (
                        str(row["ticket_id"])
                        if "ticket_id" in df_raw.columns
                        else f"TKT-{i+1:04d}"
                    )

                    r = run_triage(raw_text)

                    results.append(
                        {
                            "ticket_id": ticket_id,
                            "text_preview": raw_text[:80]
                            + ("..." if len(raw_text) > 80 else ""),
                            "triage_category": str(r["category"]),
                            "triage_priority": str(r["priority"]),
                            "adr_flag": "⚠️ Yes" if r["adr_flag"] else "No",
                            "route_to": str(r["routing"]["team"]),
                            "sla": str(r["sla"]),
                            "cat_confidence": f"{r['confidence']:.0%}",
                        }
                    )

                    progress.progress((i + 1) / len(df_raw))
                    status.caption(
                        f"Triaging ticket {i + 1} of {len(df_raw)}..."
                    )

                progress.empty()
                status.empty()

                results_df = pd.DataFrame(results)

                st.success("✅ Batch analysis completed")

                priority_series = results_df["triage_priority"]
                adr_series = results_df["adr_flag"]

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "🚨 CRITICAL",
                    int((priority_series == "CRITICAL").sum()),
                )

                c2.metric(
                    "🔴 High",
                    int((priority_series == "High").sum()),
                )

                c3.metric(
                    "⚠️ ADR Flags",
                    int((adr_series == "⚠️ Yes").sum()),
                )

                c4.metric("📊 Total", len(results_df))

                st.dataframe(results_df, use_container_width=True)

                st.markdown("### 📈 Priority Breakdown")

                priority_counts = priority_series.value_counts().reindex(
                    ["CRITICAL", "High", "Medium", "Low"],
                    fill_value=0,
                )

                st.bar_chart(priority_counts)

                csv_out = results_df.to_csv(index=False)

                st.download_button(
                    "⬇️ Download Results CSV",
                    csv_out,
                    file_name="triaged_tickets.csv",
                    mime="text/csv",
                )

# ─────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────

with tab3:
    st.markdown("## 📊 About PharmaTriage AI")

    st.markdown(
        """
This version keeps the full functionality from your original `app(copy).py` while using the cleaner and more modern UI design from `app2.py`.

### Included Functionalities

- ML-based category classification
- Rule-based clinical priority escalation
- ADR / pharmacovigilance detection
- Human review recommendation system
- Clinical routing suggestions
- Batch CSV triage processing
- Confidence scoring
- Model transparency / preprocessing inspection

### Clinical Safety Features

- CRITICAL tickets cannot be downgraded by the ML model
- Counseling keyword override prevents common misclassification
- ADR signals trigger pharmacovigilance warnings
- Human review is recommended for low-confidence predictions

### Tech Stack

- Streamlit
- scikit-learn
- NLP preprocessing pipeline
- Hybrid ML + rules clinical decision support
"""
    )
