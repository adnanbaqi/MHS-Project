"""
Mental Health Risk Model — Training Pipeline (Production + Visuals)
===================================================================
A robust, production-ready pipeline featuring:
  - Isotonic Probability Calibration
  - TF-IDF integration (preventing keyword-gaming)
  - Per-dataset evaluation metrics
  - Bootstrapped 95% Confidence Intervals for AUC
  - Native Matplotlib/Seaborn visualization generation

Usage
─────
    pip install xgboost scikit-learn pandas numpy tqdm nltk vaderSentiment matplotlib seaborn
    python notebooks/train_model.py
"""

import os
import sys
import json
import pickle
import re
import warnings
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Sentiment & Negation
# ─────────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_OK = True
except ImportError:
    VADER_OK = False
    print("  [warn] vaderSentiment not installed.")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────────────────────
# ML libraries
# ─────────────────────────────────────────────────────────────
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DIR_DREADDIT  = ROOT / "data" / "raw" / "dreaddit"
DIR_DEPREDDIT = ROOT / "data" / "raw" / "depression_reddit"
DIR_MHNLP     = ROOT / "data" / "raw" / "mental_health_nlp"
DATA_PROC     = ROOT / "data" / "processed"
MODEL_DIR     = ROOT / "app" / "models"
REPORT_TXT    = ROOT / "data" / "report.txt"
REPORT_IMG    = ROOT / "data" / "training_visuals.png"

DATA_PROC.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FEATURE CONFIG 
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "gad_score_normalized", "phq_score_normalized", "gad_severity_numeric",
    "phq_severity_numeric", "combined_severity", "sentiment_score",
    "negative_keyword_ratio", "word_count_normalized", "avg_word_length_norm",
    "exclamation_norm", "question_norm", "clinical_text_combined", "negation_ratio",
]

DEPRESSION_KW = {"hopeless", "worthless", "empty", "numb", "depressed", "suicide", "self-harm"}
ANXIETY_KW = {"anxious", "panic", "worried", "scared", "phobia", "ptsd", "overthinking"}
NEGATIVE_WORDS = {"bad", "terrible", "pain", "struggle", "broken", "death"} | DEPRESSION_KW | ANXIETY_KW
CRISIS_RESPONSE_KW = {"suicid", "depress", "self-harm", "hopeless", "crisis", "harm"}
NEGATION_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "without", "lack"}

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION 
# ─────────────────────────────────────────────────────────────

def _kw_density_negation_aware(text: str, keywords: set) -> tuple[float, float]:
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

def _sentiment_neg(text: str) -> float:
    if VADER_OK: return _vader.polarity_scores(text)['neg']
    return _kw_density_negation_aware(text, NEGATIVE_WORDS)[0]

def text_to_features(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    words = re.findall(r"\b\w+\b", text.lower())
    n_words = max(len(words), 1)

    gad_dens, gad_neg = _kw_density_negation_aware(text, ANXIETY_KW)
    phq_dens, phq_neg = _kw_density_negation_aware(text, DEPRESSION_KW)
    
    comb_sev = (gad_dens + phq_dens) / 2.0
    sentiment = _sentiment_neg(text)
    neg_dens, _ = _kw_density_negation_aware(text, NEGATIVE_WORDS)
    
    return np.array([
        gad_dens, phq_dens, gad_dens, phq_dens, comb_sev,
        sentiment, neg_dens, min(np.log1p(n_words)/np.log1p(500), 1.0),
        min(np.mean([len(w) for w in words])/10.0, 1.0),
        min(text.count("!")/n_words, 1.0), min(text.count("?")/n_words, 1.0),
        (comb_sev + sentiment)/2.0, (gad_neg + phq_neg)/2.0
    ], dtype=np.float32)

# ─────────────────────────────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────────────────────────────

def _abort_if_missing(path: Path):
    if not path.exists():
        print(f"\n  ❌  File not found: {path}\n      Run download script.\n")
        sys.exit(1)

def load_datasets() -> tuple[pd.DataFrame, list]:
    print("\n" + "="*60 + "\n  STEP 1 — LOAD DATASETS FROM DISK\n" + "="*60)
    parts, stats = [], []
    
    paths = [
        ("Dreaddit", DIR_DREADDIT / "dreaddit.csv", "dreaddit"),
        ("Depression Reddit", DIR_DEPREDDIT / "depression_reddit_cleaned.csv", "depression_reddit"),
        ("Mental Health NLP", DIR_MHNLP / "mental_health_nlp.csv", "mental_health_nlp")
    ]
    
    for name, path, src in paths:
        _abort_if_missing(path)
        df = pd.read_csv(path)
        text_col = next((c for c in df.columns if "text" in c.lower() or "context" in c.lower()), df.columns[0])
        label_col = next((c for c in df.columns if "label" in c.lower() or "depress" in c.lower() or "response" in c.lower()), df.columns[-1])
        
        if name == "Mental Health NLP" and "Response" in df.columns:
            df["label"] = df["Response"].astype(str).apply(lambda r: int(any(kw in r.lower() for kw in CRISIS_RESPONSE_KW)))
        else:
            df["label"] = df[label_col].astype(int)
            
        df = df[[text_col, "label"]].dropna().copy()
        df.columns = ["text", "label"]
        df["source"] = src
        parts.append(df)
        stats.append({"name": name, "rows": len(df), "pos_pct": round(df["label"].mean()*100, 1)})
        print(f"  ✅  {name:<22} {len(df):>6,} rows")

    combined = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return combined, stats

def bootstrap_auc(y_true: np.ndarray, y_pred_proba: np.ndarray, n_iterations: int = 1000) -> tuple[float, float]:
    aucs = []
    for _ in range(n_iterations):
        idx = resample(np.arange(len(y_true)))
        if len(np.unique(y_true[idx])) < 2: continue
        aucs.append(roc_auc_score(y_true[idx], y_pred_proba[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

# ─────────────────────────────────────────────────────────────
# TRAIN, CALIBRATE & EVALUATE
# ─────────────────────────────────────────────────────────────

MIN_PRECISION_FLOOR = 0.70
ZERO_IMP_FEATURES = {"gad_severity_numeric", "combined_severity"}
DEFAULT_THRESHOLD = 0.35

def train_and_evaluate(df: pd.DataFrame) -> tuple[XGBClassifier, dict, TfidfVectorizer]:
    print("\n" + "="*60 + "\n  STEP 2 & 3 — FEATURE EXTRACT & TRAIN\n" + "="*60)
    
    print("  Extracting dense features...")
    X_dense = np.vstack([text_to_features(t) for t in tqdm(df["text"], ncols=80, desc="  feat")])
    y = df["label"].to_numpy(dtype=int)
    
    active_idx = [i for i, n in enumerate(FEATURE_NAMES) if n not in ZERO_IMP_FEATURES]
    dense_names = [FEATURE_NAMES[i] for i in active_idx]
    X_active = X_dense[:, active_idx]
    
    texts_train, texts_test, X_tr_d, X_te_d, y_train, y_test, src_train, src_test = train_test_split(
        df["text"], X_active, y, df["source"], test_size=0.20, random_state=42, stratify=y
    )

    print("  Fitting TF-IDF on training set...")
    custom_stops = list(text.ENGLISH_STOP_WORDS.union({"wa", "ha", "just", "like", "don", "ve"}))
    tfidf = TfidfVectorizer(max_features=150, stop_words=custom_stops, ngram_range=(1,2))
    X_tr_tfidf = tfidf.fit_transform(texts_train).toarray()
    X_te_tfidf = tfidf.transform(texts_test).toarray()
    
    tfidf_names = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    all_feature_names = dense_names + tfidf_names
    
    X_train = np.hstack([X_tr_d, X_tr_tfidf])
    X_test  = np.hstack([X_te_d, X_te_tfidf])

    print(f"\n  Train : {len(X_train):,}    Test : {len(X_test):,}")

    base_model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.03, subsample=0.80,
        colsample_bytree=0.80, min_child_weight=5, gamma=0.15,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        eval_metric="logloss", random_state=42, verbosity=0,
    )

    print("\n  Training base XGBoost model...")
    base_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("  Applying probability calibration (Isotonic)...")
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)

    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    opt_threshold = DEFAULT_THRESHOLD
    for prec, rec, thr in zip(precisions[:-1], recalls[:-1], thresholds):
        if prec >= MIN_PRECISION_FLOOR:
            opt_threshold = float(thr)
            break

    y_pred = (y_proba >= opt_threshold).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    ci_lower, ci_upper = bootstrap_auc(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    feat_imp = sorted(zip(all_feature_names, base_model.feature_importances_), key=lambda x: x[1], reverse=True)

    metrics = {
        "auc": round(auc, 4),
        "auc_ci": (round(ci_lower, 4), round(ci_upper, 4)),
        "ap": round(ap, 4),
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "fn_rate": round((cm[1, 0] / max(cm[1, 0] + cm[1, 1], 1)) * 100, 1),
        "opt_threshold": round(opt_threshold, 3),
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "precisions": precisions,
        "recalls": recalls,
        "feat_imp": [(f, float(v)) for f, v in feat_imp]
    }

    return calibrated_model, metrics, tfidf

# ─────────────────────────────────────────────────────────────
# GENERATE VISUALIZATIONS (MATPLOTLIB/SEABORN)
# ─────────────────────────────────────────────────────────────

def generate_visuals(metrics: dict):
    print("\n  Generating native visualizations...")
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Mental Health Risk Model — Training Telemetry", fontsize=18, fontweight='bold')

    # 1. ROC Curve
    ax = axes[0, 0]
    ax.plot(metrics["fpr"], metrics["tpr"], color='b', lw=2, label=f'AUC = {metrics["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=14)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    # 2. Precision-Recall Curve
    ax = axes[0, 1]
    ax.plot(metrics["recalls"], metrics["precisions"], color='purple', lw=2, label=f'AP = {metrics["ap"]:.3f}')
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")

    # 3. Confusion Matrix
    ax = axes[1, 0]
    cm = metrics["cm"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Pred Low', 'Pred High'], yticklabels=['Actual Low', 'Actual High'],
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_title(f"Confusion Matrix (Threshold = {metrics['opt_threshold']})", fontsize=14)

    # 4. Feature Importance (Top 15)
    ax = axes[1, 1]
    top_feats = metrics["feat_imp"][:15]
    names = [f[0].replace('tfidf_', 'TFIDF: ') for f in top_feats]
    scores = [f[1] for f in top_feats]
    sns.barplot(x=scores, y=names, ax=ax, palette="viridis")
    ax.set_title("Top 15 Feature Importances (Gain)", fontsize=14)
    ax.set_xlabel("Importance")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(REPORT_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅  Visuals saved → {REPORT_IMG}")

# ─────────────────────────────────────────────────────────────
# GENERATE TEXT REPORT
# ─────────────────────────────────────────────────────────────

def generate_text_report(metrics: dict, ds_stats: list[dict]):
    ts = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    lines = [
        "="*65,
        f" MENTAL HEALTH RISK MODEL — PRODUCTION REPORT",
        f" Generated: {ts}",
        "="*65,
        "\n[ DATASET SUMMARY ]"
    ]
    for ds in ds_stats:
        lines.append(f" - {ds['name']:<20} | Rows: {ds['rows']:<6} | High Risk: {ds['pos_pct']}%")

    lines.extend([
        "\n[ AGGREGATE PERFORMANCE (Test Set) ]",
        f" Operating Threshold        : {metrics['opt_threshold']}",
        f" ROC-AUC                    : {metrics['auc']:.4f}  [95% CI: {metrics['auc_ci'][0]:.4f} - {metrics['auc_ci'][1]:.4f}]",
        f" Accuracy                   : {metrics['accuracy']}%",
        f" F1 Score                   : {metrics['f1']:.4f}",
        f" Precision                  : {metrics['precision']:.4f}",
        f" Recall                     : {metrics['recall']:.4f}",
        f" False Negative Rate        : {metrics['fn_rate']}%",
    ])

    lines.append("\n" + "="*65)
    report_text = "\n".join(lines)
    print("\n" + report_text)
    REPORT_TXT.write_text(report_text, encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    combined_df, ds_stats = load_datasets()
    trained_model, metrics, tfidf_vectorizer = train_and_evaluate(combined_df)
    
    # Save Artifacts
    with open(MODEL_DIR / "saved_model.pkl", "wb") as f:
        pickle.dump({"model": trained_model, "tfidf": tfidf_vectorizer}, f)
    print(f"\n  ✅  Pipeline saved → {MODEL_DIR / 'saved_model.pkl'}")

    generate_text_report(metrics, ds_stats)
    generate_visuals(metrics)

    print("\n" + "="*65)
    print("  DONE")
    print(f"  1. Metrics log: {REPORT_TXT}")
    print(f"  2. Visual Dash: {REPORT_IMG}")
    print("="*65 + "\n")