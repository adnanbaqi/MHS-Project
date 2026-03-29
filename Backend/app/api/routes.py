"""
API Routes
All HTTP endpoints for the Mental Health Risk Assessment System.
"""

from fastapi import APIRouter, HTTPException
from app.schemas.request import AssessmentInput, AssessmentOutput
from app.services.clinical import compute_scores
from app.services.text_processing import process_text
from app.services.feature_engineering import build_feature_vector
from app.services.prediction import predict_risk, OPERATING_THRESHOLD
from app.utils.helpers import get_timestamp
from app.schemas.request import QuickScreenInput


router = APIRouter()

@router.post("/analyze", response_model=AssessmentOutput, summary="Full mental health risk assessment")
def analyze(data: AssessmentInput):
    clinical_data = compute_scores(data.gad7, data.phq9)
    text_data = process_text(data.text)
    feature_vector = build_feature_vector(clinical_data, text_data, data.text)
    prediction = predict_risk(feature_vector)

    return AssessmentOutput(
        user_id=data.user_id or "anonymous",
        clinical={
            "gad_score": clinical_data["gad_score"],
            "phq_score": clinical_data["phq_score"],
            "gad_severity": clinical_data["gad_severity"],
            "phq_severity": clinical_data["phq_severity"],
        },
        text_analysis={
            "word_count": text_data["word_count"],
            "cleaned_text": text_data["cleaned_text"],
            "sentiment_label": text_data["sentiment_label"],
            "negative_keywords_found": text_data["negative_keywords_found"],
        },
        prediction={
            "risk_score": prediction["risk_score"],
            "risk_level": prediction["risk_level"],
            "confidence": prediction["confidence"],
            "recommendation": prediction["recommendation"],
        },
        timestamp=get_timestamp(),
    )

# @router.post("/analyze", response_model=AssessmentOutput, summary="Full mental health risk assessment")
# def analyze(data: AssessmentInput):
#     """
#     Perform a complete mental health risk assessment.
    
#     Inputs:
#     - gad7: 7 integers (0-3) for Generalized Anxiety Disorder scale
#     - phq9: 9 integers (0-3) for Patient Health Questionnaire
#     - text: Free-form text (journal entry, chat message, etc.)
#     - user_id: Optional user identifier
    
#     Returns:
#     - Clinical score analysis
#     - Text NLP analysis
#     - Risk prediction with recommendation
#     """
#     try:
#         # Step 1: Process clinical scores
#         clinical_data = compute_scores(data.gad7, data.phq9)

#         # Step 2: Process text (extracts dense NLP features)
#         text_data = process_text(data.text)

#         # Step 3: Build feature vector (merges clinical, dense NLP, and sparse TF-IDF)
#         # We pass data.text so the TF-IDF vectorizer can process the raw string
#         feature_vector = build_feature_vector(clinical_data, text_data, data.text)

#         # Step 4: Predict risk
#         prediction = predict_risk(feature_vector)

#         return AssessmentOutput(
#             user_id=data.user_id or "anonymous",
#             clinical={
#                 "gad_score": clinical_data["gad_score"],
#                 "phq_score": clinical_data["phq_score"],
#                 "gad_severity": clinical_data["gad_severity"],
#                 "phq_severity": clinical_data["phq_severity"],
#             },
#             text_analysis={
#                 "word_count": text_data["word_count"],
#                 "cleaned_text": text_data["cleaned_text"],
#                 "sentiment_label": text_data["sentiment_label"],
#                 "negative_keywords_found": text_data["negative_keywords_found"],
#             },
#             prediction={
#                 "risk_score": prediction["risk_score"],
#                 "risk_level": prediction["risk_level"],
#                 "confidence": prediction["confidence"],
#                 "recommendation": prediction["recommendation"],
#             },
#             timestamp=get_timestamp(),
#         )

#     except ValueError as e:
#         raise HTTPException(status_code=422, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/features", summary="Get feature names used by the model")
def get_features():
    """Returns the list of features used in the ML model (for transparency)."""
    return {
        "feature_count": 161,
        "description": "11 Dense Features + 150 TF-IDF N-Gram Features",
        "dense_features": [
            "gad_score_normalized",
            "phq_score_normalized",
            "phq_severity_numeric",
            "sentiment_score",
            "negative_keyword_ratio",
            "word_count_normalized",
            "avg_word_length_norm",
            "exclamation_norm",
            "question_norm",
            "clinical_text_combined",
            "negation_ratio"
        ],
        "sparse_features": "150 top TF-IDF unigrams and bigrams extracted from training corpus."
    }


@router.get("/risk-levels", summary="Get risk level thresholds")
def get_risk_levels():
    """Returns the risk level thresholds and their meanings based on training telemetry."""
    return {
        "operating_threshold": OPERATING_THRESHOLD,
        "levels": {
            "low": {
                "range": "0.000 - 0.150",
                "description": "Minimal mental health concern",
                "action": "Self-monitoring recommended",
            },
            "moderate": {
                "range": f"0.150 - {OPERATING_THRESHOLD}",
                "description": "Elevated mental health concern",
                "action": "Professional consultation recommended",
            },
            "high": {
                "range": f"{OPERATING_THRESHOLD} - 1.000",
                "description": "High mental health risk",
                "action": "Immediate professional help required",
            },
        }
    }



@router.post("/quick-screen", summary="Quick risk screen from clinical scores only")
def quick_screen(data: QuickScreenInput): # <-- Changed this line
    """
    Quick screen using only GAD-7 and PHQ-9 scores (no text required).
    """
    if len(data.gad7) != 7 or len(data.phq9) != 9:
        raise HTTPException(status_code=422, detail="GAD-7 must have exactly 7 items and PHQ-9 must have exactly 9 items")
    
    if any(x < 0 or x > 3 for x in data.gad7) or any(x < 0 or x > 3 for x in data.phq9):
        raise HTTPException(status_code=422, detail="All questionnaire items must be between 0 and 3")

    # Update variable references
    clinical_data = compute_scores(data.gad7, data.phq9)

    text_data_zero_state = {
        "sentiment_score": 0.0,
        "negative_keyword_ratio": 0.0,
        "word_count_normalized": 0.0,
        "avg_word_length_norm": 0.0,
        "exclamation_norm": 0.0,
        "question_norm": 0.0,
        "negation_ratio": 0.0
    }

    feature_vector = build_feature_vector(clinical_data, text_data_zero_state, raw_text="")
    prediction = predict_risk(feature_vector)

    return {
        "clinical": clinical_data,
        "prediction": prediction,
        "timestamp": get_timestamp(),
    }