"""
API Routes
All HTTP endpoints for the Mental Health Risk Assessment System.
"""

import traceback  # <--- NEW IMPORT
from fastapi import APIRouter, HTTPException
from app.schemas.request import AssessmentInput, AssessmentOutput, QuickScreenInput
from app.services.clinical import compute_scores
from app.services.text_processing import process_text
from app.services.prediction import predict_risk, OPERATING_THRESHOLD
from app.utils.helpers import get_timestamp

router = APIRouter()

@router.post("/analyze", response_model=AssessmentOutput, summary="Full mental health risk assessment")
def analyze(data: AssessmentInput):
    """
    Perform a complete mental health risk assessment using both clinical scores 
    and free-form text via a Deep Learning (LSTM) + ML Ensemble pipeline.
    """
    try:
        # Step 1: Process clinical scores (GAD-7 and PHQ-9)
        clinical_data = compute_scores(data.gad7, data.phq9)
        
        # Step 2: Process text
        text_data = process_text(data.text)
        
        # Step 3: Normalize the clinical scores
        gad_norm = clinical_data["gad_score"] / 21.0
        phq_norm = clinical_data["phq_score"] / 27.0
        structured_features = [gad_norm, phq_norm]
        
        # Step 4: Predict Risk using the Hybrid Ensemble
        prediction = predict_risk(data.text, structured_features)

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
                "confidence": prediction.get("model_breakdown", {}).get("lstm_prob", 0.0) * 100, 
                "recommendation": prediction["recommendation"],
            },
            timestamp=get_timestamp(),
        )

    except ValueError as e:
        # Capture the full trace for ValueErrors
        error_trace = traceback.format_exc()
        print(error_trace) # Print to terminal just in case
        raise HTTPException(status_code=422, detail=f"Validation Error:\n{error_trace}")
        
    except Exception as e:
        # Capture the FULL crash report and throw it to the frontend!
        error_trace = traceback.format_exc()
        print(error_trace) # Print to terminal just in case
        raise HTTPException(status_code=500, detail=f"CRITICAL BACKEND CRASH:\n{error_trace}")


@router.get("/features", summary="Get feature architecture used by the model")
def get_features():
    return {
        "architecture": "Deep Learning to Machine Learning Pipeline (BERT -> LSTM -> LR/RF/SVM)",
        "description": "Text is processed via BERT embeddings into a bidirectional LSTM. The LSTM probability is fused with normalized clinical features and passed into an ensemble of classifiers.",
        "text_features": "128-token BERT embeddings",
        "structured_features": [
            "gad_score_normalized (0.0 - 1.0)",
            "phq_score_normalized (0.0 - 1.0)"
        ]
    }


@router.get("/risk-levels", summary="Get risk level thresholds")
def get_risk_levels():
    return {
        "operating_threshold": OPERATING_THRESHOLD,
        "levels": {
            "low": {
                "range": "0.000 - 0.250",
                "description": "Minimal mental health concern",
                "action": "Self-monitoring recommended",
            },
            "moderate": {
                "range": f"0.250 - {OPERATING_THRESHOLD}",
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
def quick_screen(data: QuickScreenInput): 
    try:
        if len(data.gad7) != 7 or len(data.phq9) != 9:
            raise ValueError("GAD-7 must have exactly 7 items and PHQ-9 must have exactly 9 items")
        
        if any(x < 0 or x > 3 for x in data.gad7) or any(x < 0 or x > 3 for x in data.phq9):
            raise ValueError("All questionnaire items must be between 0 and 3")

        clinical_data = compute_scores(data.gad7, data.phq9)
        gad_norm = clinical_data["gad_score"] / 21.0
        phq_norm = clinical_data["phq_score"] / 27.0
        structured_features = [gad_norm, phq_norm]

        prediction = predict_risk("", structured_features)

        return {
            "clinical": clinical_data,
            "prediction": prediction,
            "timestamp": get_timestamp(),
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"CRITICAL BACKEND CRASH:\n{error_trace}")