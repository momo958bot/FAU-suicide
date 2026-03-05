import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
import sys
from sklearn.utils.class_weight import compute_class_weight
import warnings
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OMP_NUM_THREADS"] = "8"
warnings.filterwarnings('ignore')

# Write print output bidirectionally to terminal and txt file
class Logger(object):
    def __init__(self, filename="deberta_training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure real-time writing to disk
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    # Added this section: tell transformers library we support terminal output features
    def isatty(self):
        return getattr(self.terminal, 'isatty', lambda: False)()

sys.stdout = Logger("deberta_training_log.txt")
sys.stderr = sys.stdout

# ==========================================
# 1. Basic Configuration
# ==========================================
DATA_PATH = "autodl-tmp/Anonymizing_suicide_datasets.csv"
MODEL_NAME = "microsoft/deberta-base"
MAX_LEN = 512
BATCH_SIZE = 8  # BATCH SIZE
EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Currently using device: {DEVICE}")

# ==========================================
# 2. Data Processing and Time Feature Extraction
# ==========================================
def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by=['users', 'time']).reset_index(drop=True)
    df['text'] = df['text'].fillna('').astype(str)
    
    user_data = []
    
    # Use sliding window to build dataset (restore to ~14k data volume)
    for user, group in df.groupby('users'):
        group = group.reset_index(drop=True)
        for i in range(len(group)):
            # Extract current post, and trace back up to 4 historical posts (forming a stable 5-element window)
            window_start = max(0, i - 4)
            window_df = group.iloc[window_start:i+1]
            
            # Target label is the sentiment of the latest post
            label = window_df['sentiment'].iloc[-1]
            
            # [Critical Fix] Reverse concatenation: put the latest post first to prevent truncation when exceeding 512 tokens!
            texts = window_df['text'].tolist()
            texts.reverse() 
            combined_text = " [SEP] ".join(texts)
            
            # Time features are still based on the latest post
            latest_time = window_df['time'].iloc[-1]
            hour = latest_time.hour
            dayofweek = latest_time.dayofweek
            is_night = 1 if (hour >= 22 or hour <= 4) else 0
            is_weekend = 1 if dayofweek >= 5 else 0
            
            user_data.append({
                'users': user,  # Reserved for GroupKFold
                'text': combined_text,
                'hour': hour,
                'week': dayofweek,
                'date': latest_time.day,
                'month': latest_time.month,
                'is_night': is_night,
                'is_weekend': is_weekend,
                'label': label
            })
            
    df_users = pd.DataFrame(user_data)
    
    # Standardize time features
    scaler = StandardScaler()
    time_cols = ['hour', 'week', 'date', 'month', 'is_night', 'is_weekend']
    df_users[time_cols] = scaler.fit_transform(df_users[time_cols])
    
    # Label encoding
    le = LabelEncoder()
    df_users['label_encoded'] = le.fit_transform(df_users['label'])
    
    print(f"Sliding window construction completed, total training samples: {len(df_users)}")
    return df_users, le.classes_, time_cols

# ==========================================
# 3. Custom Dataset
# ==========================================
class SuicideRiskDataset(Dataset):
    def __init__(self, texts, time_features, labels, tokenizer, max_len):
        self.texts = texts
        self.time_features = time_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        time_feat = self.time_features[item]
        label = self.labels[item]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'time_features': torch.tensor(time_feat, dtype=torch.float),
            'targets': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 4. DeBERTa Model Architecture Integrating Time Features 
# ==========================================
class DebertaWithTimeFeatures(nn.Module):
    def __init__(self, num_classes, time_dim):
        super(DebertaWithTimeFeatures, self).__init__()
        self.deberta = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.deberta.config.hidden_size
        
        # Feature projection layer: map time features to semantic space 
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Classifier: fuse text and time features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask, time_features):
        # Extract text representation
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # DeBERTa does not have pooler_output, we take the hidden state of the [CLS] token
        text_representation = outputs.last_hidden_state[:, 0, :]
        
        # Map time representation 
        time_representation = self.time_projection(time_features)
        
        # Concatenate and fuse 
        fused_representation = torch.cat((text_representation, time_representation), dim=1)
        
        logits = self.classifier(fused_representation)
        return logits

# ==========================================
# 5. Training and Evaluation Functions
# ==========================================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        time_features = batch['time_features'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, time_features)
        loss = criterion(outputs, targets)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_model(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            time_features = batch['time_features'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids, attention_mask, time_features)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
            
    return true_labels, predictions

# ==========================================
# 6. Main Execution Logic (5-Fold Cross Validation)
# ==========================================
def main():
    print("Loading and processing data...")
    df, class_names, time_cols = load_and_preprocess_data(DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    X_texts = df['text'].values
    X_times = df[time_cols].values
    y = df['label_encoded'].values
    groups = df['users'].values  # Extract user groups
    
    # Must switch back to GroupKFold to strictly prevent sliding window data from the same user from appearing in both training and test sets simultaneously!
    gkf = GroupKFold(n_splits=5)
    
    fold_metrics = []
    
    print("\nStarting 5-Fold GroupKFold cross-validation fine-tuning of DeBERTa...")
    # Note: changed to gkf.split(X_texts, y, groups) here
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_texts, y, groups), 1):
        print(f"\n========== Fold {fold} ==========")
        # ... Keep subsequent DataLoader and training logic unchanged ...
        
        train_texts, val_texts = X_texts[train_idx], X_texts[val_idx]
        train_times, val_times = X_times[train_idx], X_times[val_idx]
        train_labels, val_labels = y[train_idx], y[val_idx]
        
        train_dataset = SuicideRiskDataset(train_texts, train_times, train_labels, tokenizer, MAX_LEN)
        val_dataset = SuicideRiskDataset(val_texts, val_times, val_labels, tokenizer, MAX_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = DebertaWithTimeFeatures(num_classes=len(class_names), time_dim=len(time_cols))
        model = model.to(DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        
        # Calculate class weights for current fold training set, reverse sample imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        # Convert weights to tensor and put on GPU
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        
        # Pass loss function with weights to model
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        
        best_mac_f1 = 0
        best_preds, best_trues = [], []
        
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
            trues, preds = eval_model(model, val_loader, DEVICE)
            
            val_acc = accuracy_score(trues, preds)
            val_mac_f1 = f1_score(trues, preds, average='macro')
            
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Mac-F1: {val_mac_f1:.4f}")
            
            if val_mac_f1 > best_mac_f1:
                best_mac_f1 = val_mac_f1
                best_preds, best_trues = preds, trues
                
        # Calculate optimal class F1 for this fold
        class_f1s = f1_score(best_trues, best_preds, average=None)
        f1_dict = {class_names[i]: class_f1s[i] for i in range(len(class_names))}
        
        # Strictly map according to paper metrics 
        metrics = {
            'Acc': accuracy_score(best_trues, best_preds),
            'Mac-F1': best_mac_f1,
            'IN-F1': f1_dict.get('Indicator', 0),
            'ID-F1': f1_dict.get('Ideation', 0),
            'BR-F1': f1_dict.get('Behavior', 0),
            'AT-F1': f1_dict.get('Attempt', 0)
        }
        fold_metrics.append(metrics)
        print(f"Fold {fold} Best Result: Acc: {metrics['Acc']:.4f}, Mac-F1: {metrics['Mac-F1']:.4f}, IN-F1: {metrics['IN-F1']:.4f}, ID-F1: {metrics['ID-F1']:.4f}, BR-F1: {metrics['BR-F1']:.4f}, AT-F1: {metrics['AT-F1']:.4f}")

    # ==========================================
    # Summarize and output final metrics
    # ==========================================
    print("\n" + "="*50)
    print("5-Fold Cross-Validation Average Results:")
    avg_metrics = {k: np.mean([f[k] for f in fold_metrics]) * 100 for k in fold_metrics[0].keys()}
    
    print(f"Acc. (%): {avg_metrics['Acc']:.1f}")
    print(f"Mac-F1 (%): {avg_metrics['Mac-F1']:.1f}")
    print(f"IN-F1 (%): {avg_metrics['IN-F1']:.1f}")
    print(f"ID-F1 (%): {avg_metrics['ID-F1']:.1f}")
    print(f"BR-F1 (%): {avg_metrics['BR-F1']:.1f}")
    print(f"AT-F1 (%): {avg_metrics['AT-F1']:.1f}")
    print("="*50)

if __name__ == "__main__":
    main()