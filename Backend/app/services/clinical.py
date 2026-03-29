"""
Clinical Score Processing Service
Handles GAD-7 (Generalized Anxiety Disorder) and PHQ-9 (Patient Health Questionnaire)
scoring and severity classification.
"""


GAD7_SEVERITY_THRESHOLDS = {
    "minimal": (0, 4),
    "mild": (5, 9),
    "moderate": (10, 14),
    "severe": (15, 21),
}

PHQ9_SEVERITY_THRESHOLDS = {
    "minimal": (0, 4),
    "mild": (5, 9),
    "moderate": (10, 14),
    "moderately_severe": (15, 19),
    "severe": (20, 27),
}


def classify_gad7(score: int) -> str:
    """Classify GAD-7 score into severity level."""
    if score <= 4:
        return "minimal"
    elif score <= 9:
        return "mild"
    elif score <= 14:
        return "moderate"
    else:
        return "severe"


def classify_phq9(score: int) -> str:
    """Classify PHQ-9 score into severity level."""
    if score <= 4:
        return "minimal"
    elif score <= 9:
        return "mild"
    elif score <= 14:
        return "moderate"
    elif score <= 19:
        return "moderately_severe"
    else:
        return "severe"


def compute_scores(gad7: list, phq9: list) -> dict:
    """
    Compute and classify GAD-7 and PHQ-9 scores.
    
    Args:
        gad7: List of 7 integers (0-3) for GAD-7 questionnaire
        phq9: List of 9 integers (0-3) for PHQ-9 questionnaire
    
    Returns:
        Dictionary with scores and severity labels
    """
    gad_score = sum(gad7)
    phq_score = sum(phq9)

    gad_severity = classify_gad7(gad_score)
    phq_severity = classify_phq9(phq_score)

    return {
        "gad_score": gad_score,
        "phq_score": phq_score,
        "gad_severity": gad_severity,
        "phq_severity": phq_severity,
    }


def severity_to_numeric(severity: str) -> float:
    """Convert severity label to numeric value for ML features."""
    mapping = {
        "minimal": 0.0,
        "mild": 0.25,
        "moderate": 0.5,
        "moderately_severe": 0.75,
        "severe": 1.0,
    }
    return mapping.get(severity, 0.0)


def normalize_score(score: int, max_score: int) -> float:
    """Normalize a score to 0-1 range."""
    return score / max_score if max_score > 0 else 0.0
