"""
Text Processing Service
Extracts NLP features. EXACTLY mirrors the training script to prevent
training-serving skew.
"""
import numpy as np
import re
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

DEPRESSION_KW = {"hopeless", "worthless", "empty", "numb", "depressed", "suicide", "self-harm"}
ANXIETY_KW = {"anxious", "panic", "worried", "scared", "phobia", "ptsd", "overthinking"}
NEGATIVE_WORDS = {"bad", "terrible", "pain", "struggle", "broken", "death"} | DEPRESSION_KW | ANXIETY_KW
NEGATION_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "without", "lack"}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s\-\!\?]", "", text) 
    return re.sub(r"\s+", " ", text).strip()

def _kw_density_negation_aware(text: str, keywords: set) -> tuple:
    try:
        words = word_tokenize(text.lower())
    except:
        words = re.findall(r"\b\w+\b", text.lower())
    if not words: return 0.0, 0.0

    pos_hits, negated_hits, neg_window = 0, 0, 3
    for idx, w in enumerate(words):
        if w in keywords:
            if any(words[j] in NEGATION_WORDS for j in range(max(0, idx-neg_window), idx)):
                negated_hits += 1
            else: pos_hits += 1

    total_hits = pos_hits + negated_hits
    if total_hits == 0: return 0.0, 0.0
    return min(pos_hits / len(words), 1.0), negated_hits / total_hits

def process_text(text: str) -> dict:
    """Extracts features exactly as the training script did."""
    cleaned = clean_text(text)
    words = re.findall(r"\b\w+\b", cleaned.lower())
    n_words = max(len(words), 1)

    gad_dens, gad_neg = _kw_density_negation_aware(cleaned, ANXIETY_KW)
    phq_dens, phq_neg = _kw_density_negation_aware(cleaned, DEPRESSION_KW)
    neg_dens, _ = _kw_density_negation_aware(cleaned, NEGATIVE_WORDS)
    
    sentiment = _vader.polarity_scores(cleaned)['neg']

    # For the API response visual
    negative_keywords_found = [w for w in words if w in NEGATIVE_WORDS]

    return {
        "cleaned_text": cleaned,
        "word_count": len(words),
        "word_count_normalized": min(np.log1p(n_words)/np.log1p(500), 1.0),
        "avg_word_length_norm": min(np.mean([len(w) for w in words])/10.0, 1.0) if words else 0.0,
        "exclamation_norm": min(text.count("!")/n_words, 1.0),
        "question_norm": min(text.count("?")/n_words, 1.0),
        "sentiment_score": sentiment,
        "negative_keyword_ratio": neg_dens,
        "negation_ratio": (gad_neg + phq_neg)/2.0,
        "negative_keywords_found": list(set(negative_keywords_found)),
        "sentiment_label": "negative" if sentiment > 0.3 else "neutral"
    }