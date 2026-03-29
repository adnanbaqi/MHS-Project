export interface ClinicalResult {
  gad_score: number;
  phq_score: number;
  gad_severity: string;
  phq_severity: string;
}

export interface TextResult {
  word_count: number;
  cleaned_text: string;
  sentiment_label: string;
  negative_keywords_found: string[];
}

export interface PredictionResult {
  risk_score: number;
  risk_level: string;
  confidence: number;
  recommendation: string;
}

export interface AssessmentOutput {
  user_id: string;
  clinical: ClinicalResult;
  text_analysis: TextResult;
  prediction: PredictionResult;
  timestamp: string;
}
