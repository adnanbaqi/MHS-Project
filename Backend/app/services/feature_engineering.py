"""
Feature Engineering Service
Combines clinical scores and text features into the 161-dimensional 
feature vector expected by the production XGBoost model.
"""

import numpy as np
from app.services.clinical import normalize_score, severity_to_numeric
from app.services.prediction import get_tfidf_vectorizer

def build_feature_vector(clinical_data: dict, text_data: dict, raw_text: str) -> np.ndarray:
    
    # 1. Map API data to the exact 11 active dense features from training
    gad_score_norm = normalize_score(clinical_data["gad_score"], max_score=21)
    phq_score_norm = normalize_score(clinical_data["phq_score"], max_score=27)
    phq_sev_num = severity_to_numeric(clinical_data["phq_severity"])
    gad_sev_num = severity_to_numeric(clinical_data["gad_severity"])
    combined_severity = (gad_sev_num + phq_sev_num) / 2.0
    clinical_text_combined = (combined_severity + text_data["sentiment_score"]) / 2.0

    dense_features = [
        gad_score_norm,
        phq_score_norm,
        phq_sev_num,
        text_data["sentiment_score"],
        text_data["negative_keyword_ratio"],
        text_data["word_count_normalized"],
        text_data["avg_word_length_norm"],
        text_data["exclamation_norm"],
        text_data["question_norm"],
        clinical_text_combined,
        text_data["negation_ratio"]
    ]
    
    X_dense = np.array(dense_features, dtype=np.float32).reshape(1, -1)
    
    # 2. Extract TF-IDF features
    tfidf = get_tfidf_vectorizer()
    if tfidf is None:
        X_tfidf = np.zeros((1, 150), dtype=np.float32)
    else:
        X_tfidf = tfidf.transform([raw_text]).toarray()
        
    # 3. Concatenate horizontally to form shape (1, 161)
    return np.hstack([X_dense, X_tfidf])