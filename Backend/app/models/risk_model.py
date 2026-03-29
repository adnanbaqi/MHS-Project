"""
Risk Model
XGBoost-based binary classifier for mental health risk assessment.
Trained on synthetic data by default; replace with real dataset for production.
"""

import os
import pickle
import numpy as np
from xgboost import XGBClassifier


MODEL_PATH = "app/models/saved_model.pkl"
N_FEATURES = 12


def generate_synthetic_training_data(n_samples: int = 2000):
    """
    Generate synthetic training data based on clinical heuristics.
    
    High risk = high GAD/PHQ scores + negative text features
    Low risk  = low GAD/PHQ scores + neutral/positive text features
    
    Replace this with real labeled data (Reddit Mental Health dataset, DAIC-WOZ, etc.)
    """
    np.random.seed(42)

    X = np.zeros((n_samples, N_FEATURES), dtype=np.float32)
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Randomly assign risk label with 40% positive rate
        is_high_risk = np.random.random() < 0.4

        if is_high_risk:
            # High-risk profile: high clinical scores + negative text
            gad_norm = np.random.uniform(0.5, 1.0)
            phq_norm = np.random.uniform(0.5, 1.0)
            gad_sev = np.random.choice([0.5, 0.75, 1.0])
            phq_sev = np.random.choice([0.5, 0.75, 1.0])
            combined_sev = (gad_sev + phq_sev) / 2.0
            sentiment = np.random.uniform(0.55, 1.0)
            neg_kw_ratio = np.random.uniform(0.05, 0.4)
            word_count = np.random.uniform(0.2, 1.0)
            avg_word_len = np.random.uniform(0.3, 0.8)
            exclamation = np.random.uniform(0, 0.3)
            question = np.random.uniform(0, 0.2)
            combined = (combined_sev + sentiment) / 2.0
        else:
            # Low-risk profile: low clinical scores + neutral/positive text
            gad_norm = np.random.uniform(0.0, 0.45)
            phq_norm = np.random.uniform(0.0, 0.45)
            gad_sev = np.random.choice([0.0, 0.25, 0.5])
            phq_sev = np.random.choice([0.0, 0.25, 0.5])
            combined_sev = (gad_sev + phq_sev) / 2.0
            sentiment = np.random.uniform(0.0, 0.45)
            neg_kw_ratio = np.random.uniform(0.0, 0.05)
            word_count = np.random.uniform(0.0, 0.5)
            avg_word_len = np.random.uniform(0.2, 0.6)
            exclamation = np.random.uniform(0, 0.1)
            question = np.random.uniform(0, 0.15)
            combined = (combined_sev + sentiment) / 2.0

        X[i] = [
            gad_norm, phq_norm, gad_sev, phq_sev, combined_sev,
            sentiment, neg_kw_ratio, word_count, avg_word_len,
            exclamation, question, combined
        ]
        y[i] = int(is_high_risk)

    return X, y


def train_model() -> XGBClassifier:
    """Train XGBoost model on synthetic data."""
    print("[Model] Generating training data...")
    X, y = generate_synthetic_training_data(n_samples=2000)

    print(f"[Model] Training on {len(X)} samples, {N_FEATURES} features...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X, y)
    print("[Model] Training complete.")
    return model


def save_model(model: XGBClassifier, path: str = MODEL_PATH):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[Model] Saved to {path}")


def load_model(path: str = MODEL_PATH) -> XGBClassifier:
    """Load model from disk, or train fresh if not found."""
    if os.path.exists(path):
        print(f"[Model] Loading from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"[Model] No saved model found at {path}. Training new model...")
        model = train_model()
        save_model(model, path)
        return model


# Load or train model at import time
model = load_model()
