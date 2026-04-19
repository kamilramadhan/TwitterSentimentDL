"""
=============================================================================
Mini Project: Klasifikasi Emosi Tweet Indonesia
=============================================================================
Dataset : Indonesian Twitter Emotion Dataset (Saputri et al., 2018)
Models  : RNN, LSTM, GRU
Embedding: Word2Vec Bahasa Indonesia (Wikipedia)
Reference: IEEE 8629262 - F1-score baseline = 69.73%
=============================================================================
"""

import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'word2vec_id')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
EMBEDDING_DIM = 100
MAX_SEQ_LEN = 50
BATCH_SIZE = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
PATIENCE = 5  # early stopping

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)

# =============================================================================
# 1. Data Loading & Exploration
# =============================================================================
print("\n[1] Loading Dataset...")

df = pd.read_csv(os.path.join(DATA_DIR, 'Twitter_Emotion_Dataset.csv'))
print(f"    Total samples: {len(df)}")
print(f"    Columns: {list(df.columns)}")
print(f"\n    Label distribution:")
label_counts = df['label'].value_counts()
for label, count in label_counts.items():
    print(f"      {label:>8s}: {count:4d} ({count/len(df)*100:.1f}%)")

# Load abbreviation dictionary
abbrev_df = pd.read_csv(
    os.path.join(DATA_DIR, 'kamus_singkatan.csv'),
    sep=';', header=None, names=['abbrev', 'full']
)
abbrev_dict = dict(zip(abbrev_df['abbrev'].str.strip(), abbrev_df['full'].str.strip()))
print(f"\n    Abbreviation dictionary: {len(abbrev_dict)} entries")

# =============================================================================
# 2. Text Preprocessing
# =============================================================================
print("\n[2] Preprocessing Text...")

def preprocess_text(text, abbrev_dict):
    """Clean and normalize Indonesian tweet text."""
    text = str(text).lower()
    # Remove [USERNAME], [URL], [SENSITIVE-NO] tokens
    text = re.sub(r'\[username\]', '', text)
    text = re.sub(r'\[url\]', '', text)
    text = re.sub(r'\[sensitive-no\]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize abbreviations
    words = text.split()
    words = [abbrev_dict.get(w, w) for w in words]
    # Remove extra whitespace
    text = ' '.join(words).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

df['clean_tweet'] = df['tweet'].apply(lambda x: preprocess_text(x, abbrev_dict))
df['tokens'] = df['clean_tweet'].apply(lambda x: x.split())

# Show example
print("    Example preprocessing:")
print(f"      Original: {df['tweet'].iloc[0][:80]}...")
print(f"      Cleaned : {df['clean_tweet'].iloc[0][:80]}...")

# Token length statistics
lengths = df['tokens'].apply(len)
print(f"\n    Token length stats:")
print(f"      Mean: {lengths.mean():.1f}, Median: {lengths.median():.0f}, "
      f"Max: {lengths.max()}, 95th pct: {lengths.quantile(0.95):.0f}")

# =============================================================================
# 3. Label Encoding
# =============================================================================
print("\n[3] Encoding Labels...")

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
label_names = list(le.classes_)
num_classes = len(label_names)
print(f"    Classes: {label_names}")
print(f"    Num classes: {num_classes}")

# =============================================================================
# 4. Load Word2Vec Embedding
# =============================================================================
print("\n[4] Loading Word2Vec Model...")

w2v_path = os.path.join(MODEL_DIR, f'idwiki_word2vec_{EMBEDDING_DIM}.model')
w2v_model = Word2Vec.load(w2v_path)
w2v_vocab = set(w2v_model.wv.key_to_index.keys())
print(f"    Word2Vec vocabulary size: {len(w2v_vocab)}")
print(f"    Embedding dimension: {w2v_model.wv.vector_size}")

# =============================================================================
# 5. Build Vocabulary & Embedding Matrix
# =============================================================================
print("\n[5] Building Vocabulary & Embedding Matrix...")

# Build vocabulary from dataset
word_freq = {}
for tokens in df['tokens']:
    for token in tokens:
        word_freq[token] = word_freq.get(token, 0) + 1

# Create vocab: PAD=0, UNK=1, then words sorted by frequency
word2idx = {'<PAD>': 0, '<UNK>': 1}
idx = 2
for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
    word2idx[word] = idx
    idx += 1

vocab_size = len(word2idx)
print(f"    Dataset vocabulary size: {vocab_size}")

# Build embedding matrix
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
found = 0
for word, i in word2idx.items():
    if word in w2v_vocab:
        embedding_matrix[i] = w2v_model.wv[word]
        found += 1
    elif i > 1:  # not PAD or UNK
        embedding_matrix[i] = np.random.normal(0, 0.1, EMBEDDING_DIM)

print(f"    Words found in Word2Vec: {found}/{vocab_size} ({found/vocab_size*100:.1f}%)")

# Free memory
del w2v_model

# =============================================================================
# 6. Dataset & DataLoader
# =============================================================================
print("\n[6] Preparing DataLoaders...")

def encode_tokens(tokens, word2idx, max_len):
    """Convert tokens to padded index sequence."""
    encoded = [word2idx.get(t, word2idx['<UNK>']) for t in tokens[:max_len]]
    # Pad
    if len(encoded) < max_len:
        encoded += [word2idx['<PAD>']] * (max_len - len(encoded))
    return encoded

df['encoded'] = df['tokens'].apply(lambda x: encode_tokens(x, word2idx, MAX_SEQ_LEN))

class EmotionDataset(Dataset):
    def __init__(self, encodeds, labels):
        self.encodeds = torch.LongTensor(encodeds)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encodeds[idx], self.labels[idx]

# Split: 80% train, 10% val, 10% test (stratified)
X = np.array(df['encoded'].tolist())
y = np.array(df['label_encoded'].tolist())

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

train_dataset = EmotionDataset(X_train, y_train)
val_dataset = EmotionDataset(X_val, y_val)
test_dataset = EmotionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============================================================================
# 7. Model Definitions: RNN, LSTM, GRU
# =============================================================================
print("\n[7] Defining Models...")

class EmotionClassifier(nn.Module):
    """
    Unified emotion classifier supporting RNN, LSTM, and GRU.
    Uses pre-trained Word2Vec embeddings.
    """
    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim,
                 num_layers, num_classes, embedding_matrix, dropout=0.3):
        super(EmotionClassifier, self).__init__()

        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer (pre-trained Word2Vec)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.FloatTensor(embedding_matrix), requires_grad=True
        )

        # RNN variant
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        if self.rnn_type == 'LSTM':
            output, (hidden, _) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)

        # Use last layer's hidden state
        hidden_last = hidden[-1]              # (batch, hidden_dim)
        hidden_last = self.dropout(hidden_last)
        logits = self.fc(hidden_last)         # (batch, num_classes)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# 8. Training & Evaluation Functions
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def train_model(model, model_name, train_loader, val_loader, criterion,
                optimizer, scheduler, device, num_epochs, patience):
    """Train with early stopping, return history and best model."""
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training {model_name} | Parameters: {count_parameters(model):,}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best val F1: {best_val_f1:.4f})")
                break

    # Restore best model
    model.load_state_dict(best_model_state)
    return model, history

# =============================================================================
# 9. Train All Models
# =============================================================================
print("\n[8] Training Models...")

model_types = ['RNN', 'LSTM', 'GRU']
results = {}
histories = {}
all_test_preds = {}

embedding_tensor = embedding_matrix.copy()

for rnn_type in model_types:
    set_seed(SEED)

    model = EmotionClassifier(
        rnn_type=rnn_type,
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        embedding_matrix=embedding_tensor,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    model, history = train_model(
        model, rnn_type, train_loader, val_loader,
        criterion, optimizer, scheduler, DEVICE, NUM_EPOCHS, PATIENCE
    )

    # Evaluate on test set
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    test_prec = precision_score(test_labels, test_preds, average='macro')
    test_rec = recall_score(test_labels, test_preds, average='macro')

    results[rnn_type] = {
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1_score': test_f1,
        'params': count_parameters(model),
    }
    histories[rnn_type] = history
    all_test_preds[rnn_type] = (test_preds, test_labels)

    print(f"\n  {rnn_type} Test Results:")
    print(f"    Accuracy:  {test_acc:.4f}")
    print(f"    Precision: {test_prec:.4f}")
    print(f"    Recall:    {test_rec:.4f}")
    print(f"    F1-Score:  {test_f1:.4f}")
    print(f"\n  Classification Report ({rnn_type}):")
    print(classification_report(test_labels, test_preds, target_names=label_names, digits=4))

# =============================================================================
# 10. Comparison with Paper (IEEE 8629262)
# =============================================================================
print("\n" + "=" * 60)
print("  COMPARISON: RNN Variants vs Paper (Saputri et al., 2018)")
print("=" * 60)

paper_f1 = 69.73  # from IEEE 8629262

print(f"\n  {'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Params':>12}")
print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

for model_name, r in results.items():
    print(f"  {model_name:<12} {r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% "
          f"{r['recall']*100:>9.2f}% {r['f1_score']*100:>9.2f}% {r['params']:>12,}")

print(f"  {'Paper':<12} {'N/A':>10} {'N/A':>10} {'N/A':>10} {paper_f1:>9.2f}% {'N/A':>12}")
print()

best_model_name = max(results, key=lambda k: results[k]['f1_score'])
best_f1 = results[best_model_name]['f1_score'] * 100
diff = best_f1 - paper_f1
print(f"  Best Model: {best_model_name} (F1: {best_f1:.2f}%)")
if diff > 0:
    print(f"  >> Outperforms paper by {diff:.2f}%")
else:
    print(f"  >> Below paper by {abs(diff):.2f}%")

# =============================================================================
# 11. Visualizations
# =============================================================================
print("\n[9] Generating Visualizations...")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#2196F3', '#FF5722', '#4CAF50']  # RNN=blue, LSTM=orange, GRU=green

# --- Plot 1: Training Loss ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (name, hist) in enumerate(histories.items()):
    axes[0].plot(hist['train_loss'], label=f'{name} (Train)', color=colors[i], linewidth=2)
    axes[0].plot(hist['val_loss'], label=f'{name} (Val)', color=colors[i], linewidth=2, linestyle='--')
axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

for i, (name, hist) in enumerate(histories.items()):
    axes[1].plot(hist['train_acc'], label=f'{name} (Train)', color=colors[i], linewidth=2)
    axes[1].plot(hist['val_acc'], label=f'{name} (Val)', color=colors[i], linewidth=2, linestyle='--')
axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
print("    Saved: training_curves.png")

# --- Plot 2: Model Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 6))

model_names = list(results.keys()) + ['Paper\n(Saputri 2018)']
f1_scores = [results[m]['f1_score'] * 100 for m in results] + [paper_f1]
bar_colors = colors + ['#9E9E9E']

bars = ax.bar(model_names, f1_scores, color=bar_colors, edgecolor='white', linewidth=2, width=0.6)

# Add value labels on bars
for bar, score in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{score:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('F1-Score (%)', fontsize=13)
ax.set_title('Perbandingan F1-Score: RNN Variants vs Paper Baseline',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(f1_scores) + 10)
ax.axhline(y=paper_f1, color='#9E9E9E', linestyle='--', alpha=0.7, label=f'Paper Baseline ({paper_f1}%)')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
print("    Saved: model_comparison.png")

# --- Plot 3: Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, (preds, labels)) in enumerate(all_test_preds.items()):
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=label_names, yticklabels=label_names)
    axes[i].set_title(f'{name}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.suptitle('Confusion Matrix - Test Set', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
print("    Saved: confusion_matrices.png")

# --- Plot 4: Per-class F1-Score ---
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(label_names))
width = 0.25

for i, (name, (preds, labels)) in enumerate(all_test_preds.items()):
    report = classification_report(labels, preds, target_names=label_names, output_dict=True)
    class_f1 = [report[c]['f1-score'] * 100 for c in label_names]
    bars = ax.bar(x + i * width, class_f1, width, label=name, color=colors[i], edgecolor='white')

ax.set_xlabel('Emotion Class', fontsize=13)
ax.set_ylabel('F1-Score (%)', fontsize=13)
ax.set_title('Per-Class F1-Score by Model', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(label_names, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'per_class_f1.png'), dpi=150, bbox_inches='tight')
print("    Saved: per_class_f1.png")

# --- Plot 5: Data Distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
palette = sns.color_palette("Set2", num_classes)
ax.bar(label_counts.index, label_counts.values, color=palette, edgecolor='white', linewidth=2)
for i, (label, count) in enumerate(label_counts.items()):
    ax.text(i, count + 20, str(count), ha='center', fontsize=12, fontweight='bold')
ax.set_title('Distribusi Label Emosi dalam Dataset', fontsize=14, fontweight='bold')
ax.set_xlabel('Emosi', fontsize=13)
ax.set_ylabel('Jumlah Tweet', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'data_distribution.png'), dpi=150, bbox_inches='tight')
print("    Saved: data_distribution.png")

# =============================================================================
# 12. Save Summary Report
# =============================================================================
print("\n[10] Saving Summary Report...")

summary_path = os.path.join(RESULTS_DIR, 'summary_report.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("  LAPORAN HASIL KLASIFIKASI EMOSI TWEET INDONESIA\n")
    f.write("  Mini Project - Sequence Model\n")
    f.write("=" * 70 + "\n\n")

    f.write("DATASET:\n")
    f.write(f"  - Indonesian Twitter Emotion Dataset (Saputri et al., 2018)\n")
    f.write(f"  - Total samples: {len(df)}\n")
    f.write(f"  - Classes: {label_names}\n")
    f.write(f"  - Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)}\n\n")

    f.write("WORD EMBEDDING:\n")
    f.write(f"  - Word2Vec Bahasa Indonesia (Wikipedia)\n")
    f.write(f"  - Dimension: {EMBEDDING_DIM}\n")
    f.write(f"  - Vocabulary coverage: {found}/{vocab_size} ({found/vocab_size*100:.1f}%)\n\n")

    f.write("HYPERPARAMETERS:\n")
    f.write(f"  - Max sequence length: {MAX_SEQ_LEN}\n")
    f.write(f"  - Hidden dim: {HIDDEN_DIM}\n")
    f.write(f"  - Num layers: {NUM_LAYERS}\n")
    f.write(f"  - Dropout: {DROPOUT}\n")
    f.write(f"  - Learning rate: {LEARNING_RATE}\n")
    f.write(f"  - Batch size: {BATCH_SIZE}\n")
    f.write(f"  - Max epochs: {NUM_EPOCHS}\n")
    f.write(f"  - Early stopping patience: {PATIENCE}\n\n")

    f.write("HASIL EVALUASI (Test Set):\n")
    f.write(f"  {'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
    f.write(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n")
    for model_name, r in results.items():
        f.write(f"  {model_name:<12} {r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% "
                f"{r['recall']*100:>9.2f}% {r['f1_score']*100:>9.2f}%\n")
    f.write(f"  {'Paper':<12} {'N/A':>10} {'N/A':>10} {'N/A':>10} {paper_f1:>9.2f}%\n\n")

    f.write("PERBANDINGAN DENGAN PAPER (IEEE 8629262):\n")
    f.write(f"  Paper baseline F1-score: {paper_f1}%\n")
    f.write(f"  Best model: {best_model_name} (F1: {best_f1:.2f}%)\n")
    if diff > 0:
        f.write(f"  >> Model terbaik mengungguli paper sebesar {diff:.2f}%\n")
    else:
        f.write(f"  >> Model terbaik di bawah paper sebesar {abs(diff):.2f}%\n")
    f.write("\n")

    f.write("DETAIL CLASSIFICATION REPORT PER MODEL:\n\n")
    for name, (preds, labels) in all_test_preds.items():
        f.write(f"--- {name} ---\n")
        f.write(classification_report(labels, preds, target_names=label_names, digits=4))
        f.write("\n")

    f.write("=" * 70 + "\n")
    f.write("  Reference Paper:\n")
    f.write("  Mei Silviana Saputri, Rahmad Mahendra, and Mirna Adriani,\n")
    f.write("  'Emotion Classification on Indonesian Twitter Dataset',\n")
    f.write("  Proc. IALP 2018. (IEEE 8629262)\n")
    f.write("=" * 70 + "\n")

print(f"    Saved: {summary_path}")

print("\n" + "=" * 60)
print("  ALL DONE! Check the 'results/' folder for outputs.")
print("=" * 60)
