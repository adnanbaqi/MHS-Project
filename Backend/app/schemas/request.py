from pydantic import BaseModel, validator
from typing import List, Optional


class QuickScreenInput(BaseModel):
    gad7: List[int]   # 7 answers, each 0-3
    phq9: List[int]   # 9 answers, each 0-3

    @validator("gad7")
    def validate_gad7(cls, v):
        if len(v) != 7:
            raise ValueError("GAD-7 must have exactly 7 items")
        if any(x < 0 or x > 3 for x in v):
            raise ValueError("GAD-7 items must be between 0 and 3")
        return v

    @validator("phq9")
    def validate_phq9(cls, v):
        if len(v) != 9:
            raise ValueError("PHQ-9 must have exactly 9 items")
        if any(x < 0 or x > 3 for x in v):
            raise ValueError("PHQ-9 items must be between 0 and 3")
        return v


class AssessmentInput(QuickScreenInput):
    # Inherits gad7 and phq9 (and their validators) from QuickScreenInput
    text: str          # free-form user text
    user_id: Optional[str] = "anonymous"

    @validator("text")
    def validate_text(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Text must be at least 3 characters")
        return v


class ClinicalResult(BaseModel):
    gad_score: int
    phq_score: int
    gad_severity: str
    phq_severity: str


class TextResult(BaseModel):
    word_count: int
    cleaned_text: str
    sentiment_label: str
    negative_keywords_found: List[str]


class PredictionResult(BaseModel):
    risk_score: float
    risk_level: str
    confidence: float
    recommendation: str


class AssessmentOutput(BaseModel):
    user_id: str
    clinical: ClinicalResult
    text_analysis: TextResult
    prediction: PredictionResult
    timestamp: str