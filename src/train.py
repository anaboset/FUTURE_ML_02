"""
train.py
--------
Model training pipeline for pharmacy support ticket classification.

Trains and compares multiple classifiers for:
  1. Category classification (9 classes)
  2. Priority classification (4 classes: CRITICAL, High, Medium, Low)

Models compared:
  - Naive Bayes (baseline)
  - Logistic Regression (strong text baseline)
  - Linear SVC (often best for high-dim text)
  - Random Forest (non-linear ensemble)

All models use TF-IDF features with pharmacy-tuned parameters.
Results are logged to reports/ for comparison.
"""

import os
import pickle
import json
import time
import numpy as np
import pandas as pd
from sklearn.pipeline        import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes     import MultinomialNB
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import LabelEncoder
from sklearn.calibration     import CalibratedClassifierCV

from src.preprocess import preprocess_batch
from src.evaluate   import evaluate_model, print_report


# ─────────────────────────────────────────────
# TF-IDF CONFIGURATION
# Tuned for pharmacy/clinical text characteristics.
# ─────────────────────────────────────────────

TFIDF_PARAMS = {
    "ngram_range":  (1, 3),    # Unigrams + bigrams + trigrams
                               # "drug interaction" and "prior authorization" 
                               # are meaningful as multi-word phrases
    "max_features": 15000,     # Vocabulary cap — pharmacy vocab is specialized
    "min_df":       2,         # Ignore terms appearing in only 1 document
    "max_df":       0.90,      # Ignore terms in 90%+ of documents (too common)
    "sublinear_tf": True,      # Apply log normalization to term frequencies
    "analyzer":     "word",
    "strip_accents": "ascii",
}


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

def build_models() -> dict:
    """
    Return a dict of {model_name: sklearn Pipeline}.
    Each pipeline: TF-IDF vectorizer -> classifier.
    """
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),

        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf",   LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )),
        ]),

        "Linear SVC": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            # CalibratedClassifierCV wraps LinearSVC to enable predict_proba()
            # so we can show confidence scores in the app
            ("clf",   CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, random_state=42),
                cv=3,
            )),
        ]),

        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf",   RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_all_models(
    df: pd.DataFrame,
    text_col:     str = "text_cleaned",
    category_col: str = "category",
    priority_col: str = "priority",
    test_size:    float = 0.20,
    models_dir:   str = "./models",
    reports_dir:  str = "./reports",
) -> dict:
    """
    Train category and priority classifiers; save best model artifacts.

    Parameters
    ----------
    df           : DataFrame with cleaned text + labels
    text_col     : Column containing preprocessed text
    category_col : Target column for category classification
    priority_col : Target column for priority classification
    test_size    : Fraction held out for final evaluation
    models_dir   : Where to save .pkl model files
    reports_dir  : Where to save evaluation JSON

    Returns
    -------
    dict with keys 'category_results', 'priority_results',
    'best_category_model', 'best_priority_model'
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    X = df[text_col].astype(str).to_numpy()

    results = {}

    for task, label_col in [("category", category_col), ("priority", priority_col)]:
        print(f"\n{'='*60}")
        print(f" Training: {task.upper()} classifier")
        print(f"{'='*60}")

        y = df[label_col].astype(str).to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=42,
        )

        print(f" Train: {len(X_train)} | Test: {len(X_test)}")

        models = build_models()
        task_results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, pipeline in models.items():
            print(f"\n  [{name}]")
            t0 = time.time()

            # 5-fold cross-validation on training set
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv, scoring="f1_macro", n_jobs=-1,
            )
            cv_mean = cv_scores.mean()
            cv_std  = cv_scores.std()

            # Final fit on full training set
            pipeline.fit(X_train, y_train)
            elapsed = time.time() - t0

            # Evaluate on held-out test set
            metrics = evaluate_model(pipeline, X_test, y_test)
            metrics["cv_f1_mean"] = round(cv_mean, 4)
            metrics["cv_f1_std"]  = round(cv_std,  4)
            metrics["train_time_s"] = round(elapsed, 2)

            print(f"    CV F1 (5-fold): {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"    Test Accuracy : {metrics['accuracy']:.4f}")
            print(f"    Test F1 Macro : {metrics['f1_macro']:.4f}")
            print(f"    Train time    : {elapsed:.2f}s")

            task_results[name] = {
                "pipeline": pipeline,
                "metrics":  metrics,
            }

        # Select best model by test F1 macro
        best_name = max(
            task_results,
            key=lambda n: task_results[n]["metrics"]["f1_macro"],
        )
        best_pipeline = task_results[best_name]["pipeline"]

        print(f"\n  ✅  Best {task} model: {best_name}")
        print_report(task_results[best_name]["metrics"], label=f"Best {task} model ({best_name})")

        # Save best model
        model_path = os.path.join(models_dir, f"{task}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"model": best_pipeline, "model_name": best_name}, f)
        print(f"  💾  Saved to {model_path}")

        # Save comparison report
        report = {
            name: {k: v for k, v in res["metrics"].items() if k != "confusion_matrix"}
            for name, res in task_results.items()
        }
        report_path = os.path.join(reports_dir, f"{task}_model_comparison.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        results[f"{task}_results"]      = task_results
        results[f"best_{task}_model"]   = best_name
        results[f"best_{task}_pipeline"]= best_pipeline

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    DATA_PATH = "./data/processed/pharmacy_tickets_processed.csv"

    if not os.path.exists(DATA_PATH):
        print(f"Processed data not found at {DATA_PATH}")
        print("Generating synthetic data and preprocessing...")

        from src.data_generator import generate_dataset
        from src.preprocess     import preprocess_batch

        df_raw = generate_dataset(n_per_category=300)
        df_raw["text_cleaned"] = preprocess_batch(df_raw["text"].tolist())

        os.makedirs("../data/processed", exist_ok=True)
        df_raw.to_csv(DATA_PATH, index=False)
        df = df_raw
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} tickets from {DATA_PATH}")

    # Ensure cleaned text exists
    if "text_cleaned" not in df.columns:
        print("Cleaning text...")
        df["text_cleaned"] = preprocess_batch(df["text"].tolist())

    results = train_all_models(df)

    print("\n\n" + "="*60)
    print(" TRAINING COMPLETE")
    print("="*60)
    print(f" Best category model : {results['best_category_model']}")
    print(f" Best priority model : {results['best_priority_model']}")
    print(" Models saved to     : ../models/")
    print(" Reports saved to    : ../reports/")
