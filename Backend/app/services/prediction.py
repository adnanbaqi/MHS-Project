"""
Prediction Service
Handles loading the production XGBoost model and TF-IDF vectorizer,
executing predictions, and generating actionable recommendations.
"""

import os
import pickle
import numpy as np
from fastapi import HTTPException

MODEL_PATH = "app/models/saved_model.pkl"
OPERATING_THRESHOLD = 0.208  # The exact threshold optimized during training

_calibrated_model = None
_tfidf_vectorizer = None

RECOMMENDATIONS = {
    "low": (
        "Your responses suggest low risk. Continue maintaining healthy habits. "
        "Consider mindfulness exercises and regular physical activity. "
        "Check in with yourself weekly."
    ),
    "moderate": (
        "Your responses suggest moderate mental health concerns. "
        "We recommend speaking with a counselor or therapist. "
        "Practice stress-reduction techniques like deep breathing or journaling."
    ),
    "high": (
        "Your responses suggest high risk. Please seek professional help immediately. "
        "Contact a mental health crisis line: National Suicide Prevention Lifeline: 988 (US). "
        "Talk to a mental health professional as soon as possible. "
        "You are not alone — help is available."
    ),
}

def load_artifacts():
    """Loads the model and TF-IDF vectorizer into memory on startup."""
    global _calibrated_model, _tfidf_vectorizer
    
    if not os.path.exists(MODEL_PATH):
        print(f"[Warning] Production model not found at {MODEL_PATH}. Run training script first.")
        return
        
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)
        
    _calibrated_model = artifacts["model"]
    _tfidf_vectorizer = artifacts["tfidf"]

# Initialize on import
load_artifacts()

def get_tfidf_vectorizer():
    return _tfidf_vectorizer

def predict_risk(feature_vector: np.ndarray) -> dict:
    """
    Generate risk prediction from the 161-dimensional feature vector.
    """
    if _calibrated_model is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded on server.")

    # Get probability of high-risk class (Class 1)
    risk_score = float(_calibrated_model.predict_proba(feature_vector)[0, 1])
    
    # Apply our custom recall-optimized threshold
    if risk_score < 0.15:
        risk_level = "low"
    elif risk_score < OPERATING_THRESHOLD:
        risk_level = "moderate"
    else:
        risk_level = "high"

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "confidence": round(abs(risk_score - OPERATING_THRESHOLD) * 100, 1), # Distance from decision boundary
        "recommendation": RECOMMENDATIONS[risk_level],
    }