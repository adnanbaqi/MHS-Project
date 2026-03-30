"""
Mental Health Risk Prediction - Multi-Dataset Pipeline
Architecture: PyTorch on GPU (BERT/LSTM) + Scikit-Learn on CPU (LR/RF/SVM)
Outputs: Trained Weights (.pth, .pkl) + Full Comparative Visual Dashboards (.png)
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve, 
    confusion_matrix
)

warnings.filterwarnings('ignore')

# ==========================================
# 1. CUDA & HARDWARE OPTIMIZATION SETUP
# ==========================================
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. Please install the CUDA-enabled version of PyTorch for Windows.")

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True 

print(f"[*] Initializing Training Pipeline on: {torch.cuda.get_device_name(0)}")

MAX_LEN = 128
BATCH_SIZE = 64 
EPOCHS = 5
LSTM_HIDDEN_DIM = 64
LEARNING_RATE = 0.001

# ==========================================
# 2. DATASET & DATALOADER
# ==========================================
class MentalHealthDataset(Dataset):
    def __init__(self, texts, structured_data, labels, tokenizer, max_len):
        self.texts = texts
        self.structured_data = structured_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        struct_feats = np.array(self.structured_data[item], dtype=np.float32) 
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'structured_features': torch.tensor(struct_feats, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

# ==========================================
# 3. DEEP LEARNING MODEL (BERT -> LSTM)
# ==========================================
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

# ==========================================
# 4. MULTI-DATASET PREPARATION
# ==========================================
# ==========================================
# 4. MULTI-DATASET PREPARATION ENGINE
# ==========================================
def load_and_prepare_data(base_path_str):
    print("\n[*] Loading and unifying datasets...")
    base_dir = Path(base_path_str)
    
    dataframes = []
    
    # ---------------------------------------------------------
    # DATASET 1: Depression Reddit (Text, Label)
    # ---------------------------------------------------------
    path_dep = base_dir / "depression_reddit" / "depression_reddit_cleaned.csv"
    if path_dep.exists():
        print("    -> Parsing Depression Reddit...")
        try:
            df1 = pd.read_csv(path_dep)
            # Handle cases with standard headers or no headers
            if 'clean_text' in df1.columns and 'is_depression' in df1.columns:
                df1 = df1[['clean_text', 'is_depression']].rename(columns={'clean_text': 'text', 'is_depression': 'label'})
            else:
                # If the CSV has no headers (just text,1)
                df1 = pd.read_csv(path_dep, names=['text', 'label'], header=None)
            
            # Ensure label is strictly 0 or 1
            df1['label'] = pd.to_numeric(df1['label'], errors='coerce').fillna(0).astype(int)
            dataframes.append(df1[['text', 'label']].dropna())
        except Exception as e:
            print(f"       [!] Error parsing Depression Reddit: {e}")

    # ---------------------------------------------------------
    # DATASET 2: Dreaddit (100+ Columns -> Isolate 'text' & 'label')
    # ---------------------------------------------------------
    path_dread = base_dir / "dreaddit" / "dreaddit.csv"
    if path_dread.exists():
        print("    -> Parsing Dreaddit...")
        try:
            df2 = pd.read_csv(path_dread)
            df2 = df2[['text', 'label']].copy()
            df2['label'] = pd.to_numeric(df2['label'], errors='coerce').fillna(0).astype(int)
            dataframes.append(df2.dropna())
        except Exception as e:
            print(f"       [!] Error parsing Dreaddit: {e}")

    # ---------------------------------------------------------
    # DATASET 3: Mental Health NLP (Context, Response -> Derive Label)
    # ---------------------------------------------------------
    path_nlp = base_dir / "mental_health_nlp" / "mental_health_nlp.csv"
    if path_nlp.exists():
        print("    -> Parsing Mental Health NLP...")
        try:
            df3 = pd.read_csv(path_nlp)
            if 'Context' in df3.columns:
                df3 = df3[['Context']].rename(columns={'Context': 'text'})
                
                # Derive a binary label based on clinical crisis keywords in the user's text
                crisis_kw = ["suicid", "depress", "die", "worthless", "self-harm", "hopeless", "kill", "numb", "empty"]
                df3['label'] = df3['text'].astype(str).apply(
                    lambda x: 1 if any(kw in x.lower() for kw in crisis_kw) else 0
                )
                dataframes.append(df3[['text', 'label']].dropna())
        except Exception as e:
            print(f"       [!] Error parsing Mental Health NLP: {e}")

    # ---------------------------------------------------------
    # UNIFICATION & FUSION
    # ---------------------------------------------------------
    if not dataframes:
        raise ValueError("No datasets were successfully loaded. Check your file paths.")

    # Combine all 3 datasets into one massive dataframe
    unified_df = pd.concat(dataframes, ignore_index=True)
    
    # Shuffle the dataset so the neural network learns generalizing features, not just dataset order
    unified_df = unified_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[*] Successfully unified {len(dataframes)} datasets into {len(unified_df)} total rows.")

    # Generate Dummy Structured Data (Until you have actual lifestyle data arrays)
    unified_df['dummy_feature_1'] = 0.0
    unified_df['dummy_feature_2'] = 0.0
    struct_cols = ['dummy_feature_1', 'dummy_feature_2']

    X_texts = unified_df['text'].values
    X_struct = unified_df[struct_cols].values.astype(np.float32) 
    y = unified_df['label'].values.astype(np.float32)

    X_train_t, X_test_t, X_train_s, X_test_s, y_train, y_test = train_test_split(
        X_texts, X_struct, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train_t, X_test_t, X_train_s, X_test_s, y_train, y_test

# ==========================================
# 5. TRAINING ROUTINES
# ==========================================
def train_lstm_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, leave=False, desc="Training Batches")

    for batch in loop:
        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
        attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
        labels = batch['label'].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

def get_lstm_predictions(model, data_loader):
    model.eval()
    predictions = []
    loop = tqdm(data_loader, leave=False, desc="Extracting Probs")

    with torch.no_grad():
        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids, attention_mask).squeeze()

            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            predictions.extend(outputs.cpu().numpy())
    return np.array(predictions, dtype=np.float32)

# ==========================================
# 6. COMPREHENSIVE VISUALIZATION DASHBOARDS
# ==========================================
def generate_comprehensive_dashboards(y_true, models_dict, epoch_losses, save_dir="saved_models"):
    print("\n[*] Generating Comprehensive Visual Dashboards...")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # --- DASHBOARD 1: Performance Comparisons (ROC, PR, Accuracy) ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(22, 6))
    fig1.suptitle("Mental Health Risk Models - Performance Comparison", fontsize=18, fontweight='bold')

    for name, data in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, data['prob'])
        auc_score = roc_auc_score(y_true, data['prob'])
        axes1[0].plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.3f})')
    axes1[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes1[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=14)
    axes1[0].set_xlabel("False Positive Rate")
    axes1[0].set_ylabel("True Positive Rate")
    axes1[0].legend(loc="lower right")

    for name, data in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, data['prob'])
        axes1[1].plot(recall, precision, lw=2, label=name)
    axes1[1].set_title("Precision-Recall Curves", fontsize=14)
    axes1[1].set_xlabel("Recall")
    axes1[1].set_ylabel("Precision")
    axes1[1].legend(loc="lower left")

    accuracies = [accuracy_score(y_true, m['pred']) * 100 for m in models_dict.values()]
    sns.barplot(x=list(models_dict.keys()), y=accuracies, ax=axes1[2], palette="viridis")
    axes1[2].set_ylim(0, 105)
    axes1[2].set_title("Model Accuracy Comparison (%)", fontsize=14)
    axes1[2].set_ylabel("Accuracy (%)")
    for i, acc in enumerate(accuracies):
        axes1[2].text(i, acc + 1.5, f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(os.path.join(save_dir, "model_comparison_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # --- DASHBOARD 2: Confusion Matrices & Loss ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle("Model Telemetry & Confusion Matrices", fontsize=18, fontweight='bold')
    axes2 = axes2.flatten()

    axes2[0].plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='crimson', lw=2)
    axes2[0].set_title("LSTM Training Loss over Epochs", fontsize=14)
    axes2[0].set_xlabel("Epoch")
    axes2[0].set_ylabel("Binary Cross Entropy Loss")
    axes2[0].set_xticks(range(1, len(epoch_losses) + 1))

    for i, (name, data) in enumerate(models_dict.items()):
        ax = axes2[i+1]
        cm = confusion_matrix(y_true, data['pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Pred Low Risk', 'Pred High Risk'], 
                    yticklabels=['Actual Low Risk', 'Actual High Risk'],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{name} Confusion Matrix", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(os.path.join(save_dir, "confusion_matrices_and_loss.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("[*] Visual Dashboards saved to: " + save_dir)

# ==========================================
# 7. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    base_raw_data_dir = "C:/Users/Adnan Baqi/Downloads/MHS Project/data/raw"
    X_train_t, X_test_t, X_train_s, X_test_s, y_train, y_test = load_and_prepare_data(base_raw_data_dir)

    train_dataset = MentalHealthDataset(X_train_t, X_train_s, y_train, tokenizer, MAX_LEN)
    test_dataset = MentalHealthDataset(X_test_t, X_test_s, y_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 1. Train LSTM Branch
    print("\n[*] Initializing and Training PyTorch LSTM on CUDA...")
    lstm_model = TextLSTMModel().to(DEVICE)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    epoch_losses = []
    for epoch in range(EPOCHS):
        avg_loss = train_lstm_model(lstm_model, train_loader, optimizer, criterion)
        epoch_losses.append(avg_loss)
        print(f"    Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 2. Extract Deep Learning Probabilities
    print("\n[*] Extracting LSTM probabilities for fusion...")
    train_lstm_probs = get_lstm_predictions(lstm_model, train_loader)
    test_lstm_probs = get_lstm_predictions(lstm_model, test_loader)

    X_train_fusion = np.hstack((X_train_s, train_lstm_probs.reshape(-1, 1))).astype(np.float32)
    X_test_fusion = np.hstack((X_test_s, test_lstm_probs.reshape(-1, 1))).astype(np.float32)

    # 3. Train Traditional ML Classifiers
    print("\n[*] Training Scikit-Learn Classifiers (LR, RF, SVM) on CPU...")
    lr = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(probability=True, random_state=42) 

    lr.fit(X_train_fusion, y_train)
    rf.fit(X_train_fusion, y_train)
    svm.fit(X_train_fusion, y_train)

    # 4. Generate Predictions & Probabilities for ALL models
    prob_lstm = test_lstm_probs
    prob_lr = lr.predict_proba(X_test_fusion)[:, 1]
    prob_rf = rf.predict_proba(X_test_fusion)[:, 1]
    prob_svm = svm.predict_proba(X_test_fusion)[:, 1]
    prob_ensemble = (prob_lr + prob_rf + prob_svm + prob_lstm) / 4.0

    # Package them into a dictionary for easy evaluation and plotting
    models_dict = {
        "LSTM (Deep Learning)": {"prob": prob_lstm, "pred": (prob_lstm > 0.5).astype(int)},
        "Logistic Regression":  {"prob": prob_lr, "pred": (prob_lr > 0.5).astype(int)},
        "Random Forest":        {"prob": prob_rf, "pred": (prob_rf > 0.5).astype(int)},
        "Support Vector Mach":  {"prob": prob_svm, "pred": (prob_svm > 0.5).astype(int)},
        "Final Hybrid Ensemble":{"prob": prob_ensemble, "pred": (prob_ensemble > 0.5).astype(int)}
    }

    # 5. Print Text Reports for ALL models
    print("\n" + "="*60)
    print("FULL MODEL COMPARISON REPORT")
    print("="*60)
    
    for name, data in models_dict.items():
        acc = accuracy_score(y_test, data['pred'])
        auc = roc_auc_score(y_test, data['prob'])
        print(f"\n[{name.upper()}]")
        print(f"Accuracy : {acc:.4f} | ROC-AUC : {auc:.4f}")
        print("-" * 40)
        print(classification_report(y_test, data['pred']))

    # 6. Generate Visuals and Save Artifacts
    generate_comprehensive_dashboards(y_test, models_dict, epoch_losses, save_dir='saved_models')
    
    print("\n[*] Saving model artifacts to disk...")
    torch.save(lstm_model.state_dict(), 'saved_models/text_lstm_model.pth')
    joblib.dump({'lr': lr, 'rf': rf, 'svm': svm}, 'saved_models/ml_ensemble.pkl')
    
    print("[*] Models saved successfully! Ready for inference.\n")