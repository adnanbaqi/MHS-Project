"""
Prediction Service
Handles loading the PyTorch LSTM model, Scikit-Learn ensemble, and tokenizer.
Executes multi-model predictions and generates actionable recommendations.
"""

import os
import joblib
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
from fastapi import HTTPException

# --- CONFIGURATION ---
ML_MODEL_PATH = "app/models/ml_ensemble.pkl"
DL_MODEL_PATH = "app/models/text_lstm_model.pth"

# Ensure we run inference on CPU for web servers unless explicitly configured for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
OPERATING_THRESHOLD = 0.50  # You may need to tune this based on your ensemble's ROC curve

# --- GLOBALS ---
_lr = None
_rf = None
_svm = None
_lstm_model = None
_tokenizer = None

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
        "Contact a mental health crisis line: "
        "Govt HELPLINE KIRAN 1800-599-0019. "
        "Talk to a mental health professional as soon as possible. "
        "You are not alone — help is available."
    ),
}

# --- PYTORCH ARCHITECTURE ---
# This must perfectly match the architecture used in your training script
class TextLSTMModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=64):
        super(TextLSTMModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size, 
            hidden_size=hidden_dim, 
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = bert_outputs.last_hidden_state
        lstm_out, (hn, cn) = self.lstm(sequence_output) 
        
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        out = self.fc(hidden)
        return self.sigmoid(out)

# --- STARTUP LOADERS ---
def load_artifacts():
    """Loads all models and the tokenizer into memory on startup."""
    global _lr, _rf, _svm, _lstm_model, _tokenizer
    
    if not os.path.exists(ML_MODEL_PATH) or not os.path.exists(DL_MODEL_PATH):
        print("[Warning] Production models not found. Ensure training script has been run and models are in app/models/")
        return
        
    print("[*] Loading Machine Learning Ensemble...")
    ml_artifacts = joblib.load(ML_MODEL_PATH)
    _lr = ml_artifacts['lr']
    _rf = ml_artifacts['rf']
    _svm = ml_artifacts['svm']
    
    print("[*] Loading BERT Tokenizer & Deep Learning Model...")
    _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    _lstm_model = TextLSTMModel().to(DEVICE)
    _lstm_model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=DEVICE))
    _lstm_model.eval() # Set to evaluation mode
    print("[*] Prediction Service Ready.")

# Initialize on import
load_artifacts()

# --- INFERENCE PIPELINE ---
def predict_risk(text: str, structured_data: list) -> dict:
    """
    Generate risk prediction by piping text through BERT/LSTM, 
    fusing it with structured data, and evaluating via ML ensemble.
    """
    if _lstm_model is None or _lr is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded on server.")

    # 1. Tokenize Text
    # If it's a quick screen, text might be empty, so we provide a fallback
    safe_text = text if text and text.strip() else "No text provided."
    
    encoding = _tokenizer(
        safe_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    # 2. Get LSTM Probability
    with torch.no_grad():
        lstm_prob = _lstm_model(input_ids, attention_mask).item()

    # 3. Fuse Features (Structured Data + LSTM Prob)
    struct_array = np.array(structured_data, dtype=np.float32)
    fusion_vector = np.hstack((struct_array, [lstm_prob])).reshape(1, -1)

    # 4. Get ML Ensemble Probabilities
    prob_lr = _lr.predict_proba(fusion_vector)[0, 1]
    prob_rf = _rf.predict_proba(fusion_vector)[0, 1]
    prob_svm = _svm.predict_proba(fusion_vector)[0, 1]

    # 5. Calculate Final Hybrid Score
    final_risk_score = (prob_lr + prob_rf + prob_svm + lstm_prob) / 4.0

    # 6. DYNAMIC CLINICAL OVERRIDE LOGIC
    # structured_data[0] is GAD-7 normalized (score / 21.0)
    # structured_data[1] is PHQ-9 normalized (score / 27.0)
    gad_severe = structured_data[0] >= (15.0 / 21.0)  # GAD-7 >= 15 is Severe
    phq_severe = structured_data[1] >= (15.0 / 27.0)  # PHQ-9 >= 15 is Mod. Severe/Severe
    
    is_clinical_override = False

    if gad_severe or phq_severe:
        risk_level = "high"
        
        # Calculate how high the clinical scores are to scale the override dynamically
        max_clinical_norm = max(structured_data[0], structured_data[1])
        
        # Create a dynamic override score between 0.85 and 0.99 based on severity
        # This prevents the UI from showing a static 85% every time it triggers
        dynamic_override_score = min(0.85 + (max_clinical_norm * 0.14), 0.99)
        
        # Use the ML score if it's somehow higher, otherwise use our dynamic safety net
        final_risk_score = max(final_risk_score, dynamic_override_score)
        is_clinical_override = True
    else:
        # Standard Ensemble Logic
        if final_risk_score < 0.25:
            risk_level = "low"
        elif final_risk_score < OPERATING_THRESHOLD:
            risk_level = "moderate"
        else:
            risk_level = "high"

    return {
        "risk_score": round(final_risk_score, 4),
        "risk_level": risk_level,
        "model_breakdown": {
            "lstm_prob": round(lstm_prob, 4),
            "lr_prob": round(prob_lr, 4),
            "rf_prob": round(prob_rf, 4),
            "svm_prob": round(prob_svm, 4),
            "clinical_override_triggered": is_clinical_override
        },
        "recommendation": RECOMMENDATIONS[risk_level],
    }