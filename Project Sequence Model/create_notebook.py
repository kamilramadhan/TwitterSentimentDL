import json

cells = []

def md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source})

def code(source):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': source, 'outputs': [], 'execution_count': None})

# ==================== CELLS ====================

md([
    "# 📊 Klasifikasi Emosi Tweet Indonesia\n",
    "## Mini Project - Sequence Model\n",
    "\n",
    "**Dataset:** Indonesian Twitter Emotion Dataset (Saputri et al., 2018)  \n",
    "**Models:** RNN, LSTM, GRU  \n",
    "**Embedding:** Word2Vec Bahasa Indonesia (Wikipedia)  \n",
    "**Reference Paper:** IEEE 8629262 — F1-score baseline = 69.73%\n"
])

md(["## 1. Import Libraries"])
code([
    "import os\n",
    "import re\n",
    "import random\n",
    "import warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "from gensim.models import Word2Vec\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix, f1_score,\n",
    "    accuracy_score, precision_score, recall_score\n",
    ")\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "%matplotlib inline\n",
    "\n",
    "print(f'PyTorch: {torch.__version__}')\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else\n",
    "                       'mps' if torch.backends.mps.is_available() else 'cpu')\n",
    "print(f'Device: {DEVICE}')"
])

md(["## 2. Konfigurasi & Hyperparameter"])
code([
    "# Paths\n",
    "DATA_DIR = 'data'\n",
    "MODEL_DIR = os.path.join('models', 'word2vec_id')\n",
    "RESULTS_DIR = 'results'\n",
    "os.makedirs(RESULTS_DIR, exist_ok=True)\n",
    "\n",
    "# Hyperparameters\n",
    "SEED = 42\n",
    "EMBEDDING_DIM = 100\n",
    "MAX_SEQ_LEN = 50\n",
    "BATCH_SIZE = 64\n",
    "HIDDEN_DIM = 128\n",
    "NUM_LAYERS = 2\n",
    "DROPOUT = 0.3\n",
    "LEARNING_RATE = 1e-3\n",
    "NUM_EPOCHS = 30\n",
    "PATIENCE = 5  # early stopping\n",
    "\n",
    "def set_seed(seed):\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    if torch.cuda.is_available():\n",
    "        torch.cuda.manual_seed_all(seed)\n",
    "\n",
    "set_seed(SEED)\n",
    "print('Configuration loaded.')"
])

md(["## 3. Load Dataset & Eksplorasi Data"])
code([
    "df = pd.read_csv(os.path.join(DATA_DIR, 'Twitter_Emotion_Dataset.csv'))\n",
    "print(f'Total samples: {len(df)}')\n",
    "print(f'Columns: {list(df.columns)}')\n",
    "print(f'\\nSample data:')\n",
    "df.head()"
])

code([
    "# Label distribution\n",
    "label_counts = df['label'].value_counts()\n",
    "print('Label distribution:')\n",
    "print(label_counts)\n",
    "print(f'\\nPercentage:')\n",
    "print((label_counts / len(df) * 100).round(1))"
])

code([
    "# Visualize distribution\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "palette = sns.color_palette('Set2', len(label_counts))\n",
    "bars = ax.bar(label_counts.index, label_counts.values, color=palette, edgecolor='white', linewidth=2)\n",
    "for bar, count in zip(bars, label_counts.values):\n",
    "    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,\n",
    "            str(count), ha='center', fontsize=12, fontweight='bold')\n",
    "ax.set_title('Distribusi Label Emosi dalam Dataset', fontsize=14, fontweight='bold')\n",
    "ax.set_xlabel('Emosi', fontsize=13)\n",
    "ax.set_ylabel('Jumlah Tweet', fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(RESULTS_DIR, 'data_distribution.png'), dpi=150, bbox_inches='tight')\n",
    "plt.show()"
])

md(["## 4. Text Preprocessing"])
code([
    "# Load abbreviation dictionary\n",
    "abbrev_df = pd.read_csv(\n",
    "    os.path.join(DATA_DIR, 'kamus_singkatan.csv'),\n",
    "    sep=';', header=None, names=['abbrev', 'full']\n",
    ")\n",
    "abbrev_dict = dict(zip(abbrev_df['abbrev'].str.strip(), abbrev_df['full'].str.strip()))\n",
    "print(f'Abbreviation dictionary: {len(abbrev_dict)} entries')\n",
    "print('\\nContoh:')\n",
    "list(abbrev_dict.items())[:10]"
])

code([
    "def preprocess_text(text, abbrev_dict):\n",
    "    \"\"\"Clean and normalize Indonesian tweet text.\"\"\"\n",
    "    text = str(text).lower()\n",
    "    text = re.sub(r'\\[username\\]', '', text)\n",
    "    text = re.sub(r'\\[url\\]', '', text)\n",
    "    text = re.sub(r'\\[sensitive-no\\]', '', text)\n",
    "    text = re.sub(r'http\\S+|www\\.\\S+', '', text)\n",
    "    text = re.sub(r'\\d+', '', text)\n",
    "    text = re.sub(r'[^\\w\\s]', ' ', text)\n",
    "    words = text.split()\n",
    "    words = [abbrev_dict.get(w, w) for w in words]\n",
    "    text = ' '.join(words).strip()\n",
    "    text = re.sub(r'\\s+', ' ', text)\n",
    "    return text\n",
    "\n",
    "df['clean_tweet'] = df['tweet'].apply(lambda x: preprocess_text(x, abbrev_dict))\n",
    "df['tokens'] = df['clean_tweet'].apply(lambda x: x.split())\n",
    "\n",
    "print('Contoh preprocessing:')\n",
    "for i in range(3):\n",
    "    print(f'\\n--- Sample {i+1} ---')\n",
    "    print(f'  Original : {df[\"tweet\"].iloc[i][:100]}...')\n",
    "    print(f'  Cleaned  : {df[\"clean_tweet\"].iloc[i][:100]}...')"
])

code([
    "# Token length statistics\n",
    "lengths = df['tokens'].apply(len)\n",
    "print(f'Token length statistics:')\n",
    "print(f'  Mean  : {lengths.mean():.1f}')\n",
    "print(f'  Median: {lengths.median():.0f}')\n",
    "print(f'  Max   : {lengths.max()}')\n",
    "print(f'  95th  : {lengths.quantile(0.95):.0f}')\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 4))\n",
    "ax.hist(lengths, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)\n",
    "ax.axvline(x=MAX_SEQ_LEN, color='red', linestyle='--', label=f'MAX_SEQ_LEN={MAX_SEQ_LEN}')\n",
    "ax.set_title('Distribusi Panjang Token', fontsize=14, fontweight='bold')\n",
    "ax.set_xlabel('Jumlah Token')\n",
    "ax.set_ylabel('Frekuensi')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
])

md(["## 5. Label Encoding"])
code([
    "le = LabelEncoder()\n",
    "df['label_encoded'] = le.fit_transform(df['label'])\n",
    "label_names = list(le.classes_)\n",
    "num_classes = len(label_names)\n",
    "print(f'Classes: {label_names}')\n",
    "print(f'Num classes: {num_classes}')\n",
    "print(f'\\nMapping:')\n",
    "for i, name in enumerate(label_names):\n",
    "    print(f'  {i} -> {name}')"
])

md(["## 6. Load Word2Vec Embedding"])
code([
    "w2v_path = os.path.join(MODEL_DIR, f'idwiki_word2vec_{EMBEDDING_DIM}.model')\n",
    "w2v_model = Word2Vec.load(w2v_path)\n",
    "w2v_vocab = set(w2v_model.wv.key_to_index.keys())\n",
    "print(f'Word2Vec vocabulary size: {len(w2v_vocab):,}')\n",
    "print(f'Embedding dimension: {w2v_model.wv.vector_size}')\n",
    "\n",
    "print(f'\\nContoh kata mirip dengan \"senang\":')\n",
    "try:\n",
    "    for word, sim in w2v_model.wv.most_similar('senang', topn=5):\n",
    "        print(f'  {word}: {sim:.4f}')\n",
    "except KeyError:\n",
    "    print('  (kata tidak ditemukan di vocabulary)')"
])

md(["## 7. Build Vocabulary & Embedding Matrix"])
code([
    "# Build vocabulary from dataset\n",
    "word_freq = {}\n",
    "for tokens in df['tokens']:\n",
    "    for token in tokens:\n",
    "        word_freq[token] = word_freq.get(token, 0) + 1\n",
    "\n",
    "word2idx = {'<PAD>': 0, '<UNK>': 1}\n",
    "idx = 2\n",
    "for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):\n",
    "    word2idx[word] = idx\n",
    "    idx += 1\n",
    "\n",
    "vocab_size = len(word2idx)\n",
    "print(f'Dataset vocabulary size: {vocab_size:,}')\n",
    "\n",
    "# Build embedding matrix\n",
    "embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))\n",
    "found = 0\n",
    "for word, i in word2idx.items():\n",
    "    if word in w2v_vocab:\n",
    "        embedding_matrix[i] = w2v_model.wv[word]\n",
    "        found += 1\n",
    "    elif i > 1:\n",
    "        embedding_matrix[i] = np.random.normal(0, 0.1, EMBEDDING_DIM)\n",
    "\n",
    "print(f'Words found in Word2Vec: {found}/{vocab_size} ({found/vocab_size*100:.1f}%)')\n",
    "\n",
    "del w2v_model\n",
    "print('Embedding matrix ready.')"
])

md(["## 8. Dataset & DataLoader"])
code([
    "def encode_tokens(tokens, word2idx, max_len):\n",
    "    \"\"\"Convert tokens to padded index sequence.\"\"\"\n",
    "    encoded = [word2idx.get(t, word2idx['<UNK>']) for t in tokens[:max_len]]\n",
    "    if len(encoded) < max_len:\n",
    "        encoded += [word2idx['<PAD>']] * (max_len - len(encoded))\n",
    "    return encoded\n",
    "\n",
    "df['encoded'] = df['tokens'].apply(lambda x: encode_tokens(x, word2idx, MAX_SEQ_LEN))\n",
    "\n",
    "class EmotionDataset(Dataset):\n",
    "    def __init__(self, encodeds, labels):\n",
    "        self.encodeds = torch.LongTensor(encodeds)\n",
    "        self.labels = torch.LongTensor(labels)\n",
    "    def __len__(self):\n",
    "        return len(self.labels)\n",
    "    def __getitem__(self, idx):\n",
    "        return self.encodeds[idx], self.labels[idx]\n",
    "\n",
    "# Split: 80% train, 10% val, 10% test (stratified)\n",
    "X = np.array(df['encoded'].tolist())\n",
    "y = np.array(df['label_encoded'].tolist())\n",
    "\n",
    "X_train, X_temp, y_train, y_temp = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=SEED, stratify=y\n",
    ")\n",
    "X_val, X_test, y_val, y_test = train_test_split(\n",
    "    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp\n",
    ")\n",
    "\n",
    "print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')\n",
    "\n",
    "train_dataset = EmotionDataset(X_train, y_train)\n",
    "val_dataset = EmotionDataset(X_val, y_val)\n",
    "test_dataset = EmotionDataset(X_test, y_test)\n",
    "\n",
    "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n",
    "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)\n",
    "test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)\n",
    "print('DataLoaders ready.')"
])

md([
    "## 9. Definisi Model: RNN, LSTM, GRU\n",
    "\n",
    "Tiga varian model Recurrent Neural Network:\n",
    "1. **Simple RNN** — Vanilla recurrent network\n",
    "2. **LSTM** — Long Short-Term Memory, mengatasi vanishing gradient\n",
    "3. **GRU** — Gated Recurrent Unit, lebih sederhana dari LSTM\n",
    "\n",
    "Semua model menggunakan:\n",
    "- Pre-trained Word2Vec embedding layer\n",
    "- 2-layer RNN dengan dropout\n",
    "- Dense output layer (5 classes)"
])

code([
    "class EmotionClassifier(nn.Module):\n",
    "    \"\"\"\n",
    "    Unified emotion classifier supporting RNN, LSTM, and GRU.\n",
    "    Uses pre-trained Word2Vec embeddings.\n",
    "    \"\"\"\n",
    "    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim,\n",
    "                 num_layers, num_classes, embedding_matrix, dropout=0.3):\n",
    "        super(EmotionClassifier, self).__init__()\n",
    "        self.rnn_type = rnn_type\n",
    "        self.hidden_dim = hidden_dim\n",
    "        self.num_layers = num_layers\n",
    "\n",
    "        # Embedding layer (pre-trained Word2Vec)\n",
    "        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)\n",
    "        self.embedding.weight = nn.Parameter(\n",
    "            torch.FloatTensor(embedding_matrix), requires_grad=True\n",
    "        )\n",
    "\n",
    "        # RNN variant\n",
    "        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]\n",
    "        self.rnn = rnn_cls(\n",
    "            input_size=embedding_dim,\n",
    "            hidden_size=hidden_dim,\n",
    "            num_layers=num_layers,\n",
    "            batch_first=True,\n",
    "            dropout=dropout if num_layers > 1 else 0,\n",
    "            bidirectional=False\n",
    "        )\n",
    "\n",
    "        self.dropout = nn.Dropout(dropout)\n",
    "        self.fc = nn.Linear(hidden_dim, num_classes)\n",
    "\n",
    "    def forward(self, x):\n",
    "        embedded = self.embedding(x)\n",
    "        embedded = self.dropout(embedded)\n",
    "\n",
    "        if self.rnn_type == 'LSTM':\n",
    "            output, (hidden, _) = self.rnn(embedded)\n",
    "        else:\n",
    "            output, hidden = self.rnn(embedded)\n",
    "\n",
    "        hidden_last = hidden[-1]\n",
    "        hidden_last = self.dropout(hidden_last)\n",
    "        logits = self.fc(hidden_last)\n",
    "        return logits\n",
    "\n",
    "def count_parameters(model):\n",
    "    return sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
    "\n",
    "print('Model class defined.')"
])

md(["## 10. Training & Evaluation Functions"])
code([
    "def train_epoch(model, loader, criterion, optimizer, device):\n",
    "    model.train()\n",
    "    total_loss, correct, total = 0, 0, 0\n",
    "    for X_batch, y_batch in loader:\n",
    "        X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
    "        optimizer.zero_grad()\n",
    "        logits = model(X_batch)\n",
    "        loss = criterion(logits, y_batch)\n",
    "        loss.backward()\n",
    "        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n",
    "        optimizer.step()\n",
    "        total_loss += loss.item() * X_batch.size(0)\n",
    "        preds = logits.argmax(dim=1)\n",
    "        correct += (preds == y_batch).sum().item()\n",
    "        total += X_batch.size(0)\n",
    "    return total_loss / total, correct / total\n",
    "\n",
    "\n",
    "def evaluate(model, loader, criterion, device):\n",
    "    model.eval()\n",
    "    total_loss = 0\n",
    "    all_preds, all_labels = [], []\n",
    "    with torch.no_grad():\n",
    "        for X_batch, y_batch in loader:\n",
    "            X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
    "            logits = model(X_batch)\n",
    "            loss = criterion(logits, y_batch)\n",
    "            total_loss += loss.item() * X_batch.size(0)\n",
    "            preds = logits.argmax(dim=1)\n",
    "            all_preds.extend(preds.cpu().numpy())\n",
    "            all_labels.extend(y_batch.cpu().numpy())\n",
    "    return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_labels)\n",
    "\n",
    "\n",
    "def train_model(model, model_name, train_loader, val_loader, criterion,\n",
    "                optimizer, scheduler, device, num_epochs, patience):\n",
    "    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}\n",
    "    best_val_f1 = 0\n",
    "    best_model_state = None\n",
    "    patience_counter = 0\n",
    "\n",
    "    print(f'\\n{\"=\"*60}')\n",
    "    print(f'  Training {model_name} | Parameters: {count_parameters(model):,}')\n",
    "    print(f'{\"=\"*60}')\n",
    "\n",
    "    for epoch in range(1, num_epochs + 1):\n",
    "        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)\n",
    "        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)\n",
    "        val_acc = accuracy_score(val_labels, val_preds)\n",
    "        val_f1 = f1_score(val_labels, val_preds, average='macro')\n",
    "        scheduler.step(val_loss)\n",
    "\n",
    "        history['train_loss'].append(train_loss)\n",
    "        history['val_loss'].append(val_loss)\n",
    "        history['train_acc'].append(train_acc)\n",
    "        history['val_acc'].append(val_acc)\n",
    "\n",
    "        if epoch % 5 == 0 or epoch == 1:\n",
    "            print(f'  Epoch {epoch:02d}/{num_epochs} | '\n",
    "                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '\n",
    "                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}')\n",
    "\n",
    "        if val_f1 > best_val_f1:\n",
    "            best_val_f1 = val_f1\n",
    "            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}\n",
    "            patience_counter = 0\n",
    "        else:\n",
    "            patience_counter += 1\n",
    "            if patience_counter >= patience:\n",
    "                print(f'  Early stopping at epoch {epoch} (best val F1: {best_val_f1:.4f})')\n",
    "                break\n",
    "\n",
    "    model.load_state_dict(best_model_state)\n",
    "    return model, history\n",
    "\n",
    "print('Training functions defined.')"
])

md([
    "## 11. Training Semua Model\n",
    "\n",
    "Melatih 3 model: **RNN**, **LSTM**, **GRU**"
])

code([
    "model_types = ['RNN', 'LSTM', 'GRU']\n",
    "results = {}\n",
    "histories = {}\n",
    "all_test_preds = {}\n",
    "embedding_tensor = embedding_matrix.copy()\n",
    "\n",
    "for rnn_type in model_types:\n",
    "    set_seed(SEED)\n",
    "\n",
    "    model = EmotionClassifier(\n",
    "        rnn_type=rnn_type,\n",
    "        vocab_size=vocab_size,\n",
    "        embedding_dim=EMBEDDING_DIM,\n",
    "        hidden_dim=HIDDEN_DIM,\n",
    "        num_layers=NUM_LAYERS,\n",
    "        num_classes=num_classes,\n",
    "        embedding_matrix=embedding_tensor,\n",
    "        dropout=DROPOUT\n",
    "    ).to(DEVICE)\n",
    "\n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)\n",
    "    scheduler = optim.lr_scheduler.ReduceLROnPlateau(\n",
    "        optimizer, mode='min', factor=0.5, patience=3\n",
    "    )\n",
    "\n",
    "    model, history = train_model(\n",
    "        model, rnn_type, train_loader, val_loader,\n",
    "        criterion, optimizer, scheduler, DEVICE, NUM_EPOCHS, PATIENCE\n",
    "    )\n",
    "\n",
    "    # Evaluate on test set\n",
    "    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)\n",
    "    test_acc = accuracy_score(test_labels, test_preds)\n",
    "    test_f1 = f1_score(test_labels, test_preds, average='macro')\n",
    "    test_prec = precision_score(test_labels, test_preds, average='macro')\n",
    "    test_rec = recall_score(test_labels, test_preds, average='macro')\n",
    "\n",
    "    results[rnn_type] = {\n",
    "        'accuracy': test_acc, 'precision': test_prec,\n",
    "        'recall': test_rec, 'f1_score': test_f1,\n",
    "        'params': count_parameters(model),\n",
    "    }\n",
    "    histories[rnn_type] = history\n",
    "    all_test_preds[rnn_type] = (test_preds, test_labels)\n",
    "\n",
    "    print(f'\\n  {rnn_type} Test Results:')\n",
    "    print(f'    Accuracy:  {test_acc:.4f}')\n",
    "    print(f'    Precision: {test_prec:.4f}')\n",
    "    print(f'    Recall:    {test_rec:.4f}')\n",
    "    print(f'    F1-Score:  {test_f1:.4f}')\n",
    "\n",
    "print('\\n✅ All models trained!')"
])

md(["## 12. Classification Report per Model"])
code([
    "for name, (preds, labels) in all_test_preds.items():\n",
    "    print(f'\\n{\"=\"*50}')\n",
    "    print(f'  Classification Report: {name}')\n",
    "    print(f'{\"=\"*50}')\n",
    "    print(classification_report(labels, preds, target_names=label_names, digits=4))"
])

md([
    "## 13. Perbandingan dengan Paper (IEEE 8629262)\n",
    "\n",
    "Paper reference: Mei Silviana Saputri, Rahmad Mahendra, and Mirna Adriani,  \n",
    "\"Emotion Classification on Indonesian Twitter Dataset\",  \n",
    "Proc. IALP 2018. **F1-score = 69.73%**"
])

code([
    "paper_f1 = 69.73\n",
    "\n",
    "print(f'{\"=\"*70}')\n",
    "print(f'  PERBANDINGAN: RNN Variants vs Paper (Saputri et al., 2018)')\n",
    "print(f'{\"=\"*70}')\n",
    "print()\n",
    "print(f'  {\"Model\":<12} {\"Accuracy\":>10} {\"Precision\":>10} {\"Recall\":>10} {\"F1-Score\":>10} {\"Params\":>12}')\n",
    "print(f'  {\"-\"*12} {\"-\"*10} {\"-\"*10} {\"-\"*10} {\"-\"*10} {\"-\"*12}')\n",
    "\n",
    "for model_name, r in results.items():\n",
    "    print(f'  {model_name:<12} {r[\"accuracy\"]*100:>9.2f}% {r[\"precision\"]*100:>9.2f}% '\n",
    "          f'{r[\"recall\"]*100:>9.2f}% {r[\"f1_score\"]*100:>9.2f}% {r[\"params\"]:>12,}')\n",
    "\n",
    "print(f'  {\"Paper\":<12} {\"N/A\":>10} {\"N/A\":>10} {\"N/A\":>10} {paper_f1:>9.2f}% {\"N/A\":>12}')\n",
    "print()\n",
    "\n",
    "best_model_name = max(results, key=lambda k: results[k]['f1_score'])\n",
    "best_f1 = results[best_model_name]['f1_score'] * 100\n",
    "diff = best_f1 - paper_f1\n",
    "print(f'  Best Model: {best_model_name} (F1: {best_f1:.2f}%)')\n",
    "if diff > 0:\n",
    "    print(f'  >> Mengungguli paper sebesar {diff:.2f}%')\n",
    "else:\n",
    "    print(f'  >> Di bawah paper sebesar {abs(diff):.2f}%')"
])

code([
    "# Tabel perbandingan\n",
    "comp_data = []\n",
    "for model_name, r in results.items():\n",
    "    comp_data.append({\n",
    "        'Model': model_name,\n",
    "        'Accuracy (%)': round(r['accuracy'] * 100, 2),\n",
    "        'Precision (%)': round(r['precision'] * 100, 2),\n",
    "        'Recall (%)': round(r['recall'] * 100, 2),\n",
    "        'F1-Score (%)': round(r['f1_score'] * 100, 2),\n",
    "        'Parameters': f'{r[\"params\"]:,}'\n",
    "    })\n",
    "comp_data.append({\n",
    "    'Model': 'Paper (Saputri 2018)',\n",
    "    'Accuracy (%)': '-', 'Precision (%)': '-',\n",
    "    'Recall (%)': '-', 'F1-Score (%)': paper_f1,\n",
    "    'Parameters': '-'\n",
    "})\n",
    "comp_df = pd.DataFrame(comp_data)\n",
    "comp_df"
])

md(["## 14. Visualisasi Hasil"])
md(["### 14.1 Training & Validation Curves"])
code([
    "colors = ['#2196F3', '#FF5722', '#4CAF50']\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "for i, (name, hist) in enumerate(histories.items()):\n",
    "    axes[0].plot(hist['train_loss'], label=f'{name} (Train)', color=colors[i], linewidth=2)\n",
    "    axes[0].plot(hist['val_loss'], label=f'{name} (Val)', color=colors[i], linewidth=2, linestyle='--')\n",
    "axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xlabel('Epoch')\n",
    "axes[0].set_ylabel('Loss')\n",
    "axes[0].legend()\n",
    "\n",
    "for i, (name, hist) in enumerate(histories.items()):\n",
    "    axes[1].plot(hist['train_acc'], label=f'{name} (Train)', color=colors[i], linewidth=2)\n",
    "    axes[1].plot(hist['val_acc'], label=f'{name} (Val)', color=colors[i], linewidth=2, linestyle='--')\n",
    "axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')\n",
    "axes[1].set_xlabel('Epoch')\n",
    "axes[1].set_ylabel('Accuracy')\n",
    "axes[1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')\n",
    "plt.show()"
])

md(["### 14.2 Perbandingan F1-Score"])
code([
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "\n",
    "model_names_plot = list(results.keys()) + ['Paper\\n(Saputri 2018)']\n",
    "f1_scores = [results[m]['f1_score'] * 100 for m in results] + [paper_f1]\n",
    "bar_colors = colors + ['#9E9E9E']\n",
    "\n",
    "bars = ax.bar(model_names_plot, f1_scores, color=bar_colors, edgecolor='white', linewidth=2, width=0.6)\n",
    "\n",
    "for bar, score in zip(bars, f1_scores):\n",
    "    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,\n",
    "            f'{score:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')\n",
    "\n",
    "ax.set_ylabel('F1-Score (%)', fontsize=13)\n",
    "ax.set_title('Perbandingan F1-Score: RNN Variants vs Paper Baseline',\n",
    "             fontsize=14, fontweight='bold')\n",
    "ax.set_ylim(0, max(f1_scores) + 10)\n",
    "ax.axhline(y=paper_f1, color='#9E9E9E', linestyle='--', alpha=0.7, label=f'Paper Baseline ({paper_f1}%)')\n",
    "ax.legend(fontsize=11)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')\n",
    "plt.show()"
])

md(["### 14.3 Confusion Matrix"])
code([
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "for i, (name, (preds, labels)) in enumerate(all_test_preds.items()):\n",
    "    cm = confusion_matrix(labels, preds)\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],\n",
    "                xticklabels=label_names, yticklabels=label_names)\n",
    "    axes[i].set_title(f'{name}', fontsize=14, fontweight='bold')\n",
    "    axes[i].set_xlabel('Predicted')\n",
    "    axes[i].set_ylabel('Actual')\n",
    "\n",
    "plt.suptitle('Confusion Matrix - Test Set', fontsize=15, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')\n",
    "plt.show()"
])

md(["### 14.4 Per-Class F1-Score"])
code([
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "\n",
    "x = np.arange(len(label_names))\n",
    "width = 0.25\n",
    "\n",
    "for i, (name, (preds, labels)) in enumerate(all_test_preds.items()):\n",
    "    report = classification_report(labels, preds, target_names=label_names, output_dict=True)\n",
    "    class_f1 = [report[c]['f1-score'] * 100 for c in label_names]\n",
    "    ax.bar(x + i * width, class_f1, width, label=name, color=colors[i], edgecolor='white')\n",
    "\n",
    "ax.set_xlabel('Emotion Class', fontsize=13)\n",
    "ax.set_ylabel('F1-Score (%)', fontsize=13)\n",
    "ax.set_title('Per-Class F1-Score by Model', fontsize=14, fontweight='bold')\n",
    "ax.set_xticks(x + width)\n",
    "ax.set_xticklabels(label_names, fontsize=12)\n",
    "ax.legend(fontsize=12)\n",
    "ax.set_ylim(0, 100)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(RESULTS_DIR, 'per_class_f1.png'), dpi=150, bbox_inches='tight')\n",
    "plt.show()"
])

md([
    "## 15. Kesimpulan\n",
    "\n",
    "### Ringkasan Hasil\n",
    "- Tiga model RNN variants (RNN, LSTM, GRU) telah dilatih untuk klasifikasi emosi tweet Indonesia\n",
    "- Dataset terdiri dari 4.401 tweet dengan 5 kelas emosi: anger, fear, happy, love, sadness\n",
    "- Word2Vec Bahasa Indonesia (100 dimensi) digunakan sebagai pre-trained word embedding\n",
    "- Hasil dibandingkan dengan paper baseline (Saputri et al., 2018) yang mencapai F1-score 69.73%\n",
    "\n",
    "### Reference\n",
    "Mei Silviana Saputri, Rahmad Mahendra, and Mirna Adriani,  \n",
    "\"Emotion Classification on Indonesian Twitter Dataset\",  \n",
    "in Proceeding of International Conference on Asian Language Processing 2018. (IEEE 8629262)"
])

# Build notebook
notebook = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.13.0'
        }
    },
    'cells': cells
}

output_path = '/Users/kamil/Deep Learning/Project Sequence Model/emotion_classification.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f'Notebook saved: {output_path}')
print(f'Total cells: {len(cells)}')
