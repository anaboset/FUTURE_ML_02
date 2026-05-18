"""
data_generator.py
-----------------
Synthetic pharmacy support ticket generator.

Generates realistic, domain-expert-labeled tickets across all 9 categories.
Used to:
  1. Bootstrap training data before a real dataset is obtained
  2. Validate preprocessing and model pipelines
  3. Demonstrate clinically grounded annotation methodology

NOTE: In a real deployment, this is replaced with actual ticket data.
Synthetic data generation is a legitimate ML technique for rare-class
bootstrapping and pipeline testing.
"""

import random
import pandas as pd
from src.taxonomy import CATEGORIES, assign_priority, flag_for_adr_reporting
import yaml

with open("data/templates/templates.yaml", "r", encoding="utf-8") as f:
    TEMPLATES = yaml.safe_load(f)

with open("data/templates/fillers.yaml", "r", encoding="utf-8") as f:
    FILLERS = yaml.safe_load(f)

DRUGS = FILLERS["drugs"]
CONTROLLED_SUBSTANCES = FILLERS["controlled_substances"]
DOSES = FILLERS["doses"]
ALT_DOSES = FILLERS["alt_doses"]
PHARMACIES = FILLERS["pharmacies"]
DURATIONS = FILLERS["durations"]
SYMPTOMS = FILLERS["symptoms"]
COUNTS = FILLERS["counts"]
LOW_COSTS = FILLERS["low_costs"]
HIGH_COSTS = FILLERS["high_costs"]

random.seed(42)


def fill_template(template: str) -> str:
    """Fill a ticket template with random realistic values."""
    drugs = random.sample(DRUGS, 3)
    return (
        template
        .replace("{drug}", drugs[0])
        .replace("{drug2}", drugs[1])
        .replace("{drug3}", drugs[2])
        .replace("{wrong_drug}", random.choice(DRUGS))
        .replace("{cs_drug}", random.choice(CONTROLLED_SUBSTANCES))
        .replace("{dose}", random.choice(DOSES))
        .replace("{alt_dose}", random.choice(ALT_DOSES))
        .replace("{pharmacy}", random.choice(PHARMACIES))
        .replace("{duration}", random.choice(DURATIONS))
        .replace("{symptom}", random.choice(SYMPTOMS))
        .replace("{count}", random.choice(COUNTS))
        .replace("{low_cost}", random.choice(LOW_COSTS))
        .replace("{high_cost}", random.choice(HIGH_COSTS))
    )


def generate_dataset(n_per_category: int = 80, noise_level: float = 0.1) -> pd.DataFrame:
    """
    Generate a synthetic labeled pharmacy ticket dataset.

    Parameters
    ----------
    n_per_category : int
        Number of tickets per category (total = n * 9 categories).
    noise_level : float
        Fraction of tickets with added realistic noise (typos, informal language).

    Returns
    -------
    pd.DataFrame
        Columns: ticket_id, text, category, priority, adr_flag
    """
    records = []
    ticket_id = 1

    for category in CATEGORIES:
        templates = TEMPLATES[category]
        for _ in range(n_per_category):
            styles = TEMPLATES[category]
            style = random.choice(list(styles.keys()))
            template = random.choice(styles[style])
            text = fill_template(template)

            # Add noise: informal language, slight typos
            if random.random() < noise_level:
                text = add_noise(text)

            priority = assign_priority(text, category)
            adr_flag = flag_for_adr_reporting(text)

            records.append({
                "ticket_id": f"TKT-{ticket_id:05d}",
                "text":      text,
                "category":  category,
                "priority":  priority,
                "adr_flag":  adr_flag,
            })
            ticket_id += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    return df


def add_noise(text: str) -> str:
    """Add realistic patient-language noise to a ticket."""
    noise_variants = [
        lambda t: t.replace(".", "").replace(",", ""),          # Missing punctuation
        lambda t: t.lower(),                                     # All lowercase
        lambda t: t.replace(" not ", " nt "),                   # Informal shorthand
        lambda t: "Hello, " + t,                                # Added greeting
        lambda t: t + " Please help asap.",                     # Urgency suffix
        lambda t: t.replace("medication", "meds"),              # Informal word
        lambda t: t.replace("prescription", "script"),          # Street slang
        lambda t: "Hi there! " + t,                             # Casual opener
    ]
    transform = random.choice(noise_variants)
    return transform(text)


# ─────────────────────────────────────────────
# MAIN — Generate and save dataset
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("Generating synthetic pharmacy ticket dataset...")
    df = generate_dataset(n_per_category=300)

    # Save
    os.makedirs("./data/raw", exist_ok=True)
    output_path = "./data/raw/pharmacy_tickets_synthetic.csv"
    
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to: {output_path}")
    print(f"Total tickets   : {len(df)}")
    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())
    print(f"\nPriority distribution:")
    print(df["priority"].value_counts().to_string())
    print(f"\nADR flags: {df['adr_flag'].sum()} tickets flagged for pharmacovigilance review")
    print(f"\nSample ticket:")
    sample = df.sample(1).iloc[0]
    print(f"  Category : {sample['category']}")
    print(f"  Priority : {sample['priority']}")
    print(f"  ADR Flag : {sample['adr_flag']}")
    print(f"  Text     : {sample['text']}")