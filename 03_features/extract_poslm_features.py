#!/usr/bin/env python3
# 03_features/extract_poslm_features.py
"""
POS-LM Features con 3 métodos:
1. kneser-ney: Kneser-Ney smoothing (estado del arte n-grams)
2. backoff: Backoff simple con Laplace smoothing
3. lstm: LSTM Language Model (neural)

Genera columnas separadas por método:
- poslm_kn_bigram_ce_mean, poslm_kn_trigram_ce_mean, ...
- poslm_bo_bigram_ce_mean, poslm_bo_trigram_ce_mean, ...
- poslm_lstm_ce_mean, poslm_lstm_ppl_mean, ...
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# POS Tagging
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
except:
    USE_SPACY = False
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('universal_tagset', quiet=True)

# PyTorch (para LSTM)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("  PyTorch no disponible. LSTM deshabilitado.")
    print("   Instala con: pip install torch")

# ==================== CONFIG ====================
UNIVERSAL_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 
                  'ADP', 'NUM', 'CONJ', 'PRT', '.', 'X']

def safe_skewness(x):
    """Skewness con manejo de valores constantes"""
    try:
        s = pd.Series(x)
        if s.std() == 0:  # Valores constantes
            return 0.0
        return s.skew()
    except:
        return np.nan

def safe_kurtosis(x):
    """Kurtosis con manejo de valores constantes"""
    try:
        s = pd.Series(x)
        if s.std() == 0:  # Valores constantes
            return 0.0
        return s.kurtosis()
    except:
        return np.nan

STATISTICS = {
    'mean': np.mean, 
    'std': np.std, 
    'min': np.min,
    'q1': lambda x: np.percentile(x, 25),
    'median': np.median,
    'q3': lambda x: np.percentile(x, 75),
    'max': np.max,
    'iqr': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
    'p1': lambda x: np.percentile(x, 1),
    'p99': lambda x: np.percentile(x, 99),
    'range_p1_p99': lambda x: np.percentile(x, 99) - np.percentile(x, 1),
    'skewness': safe_skewness, 
    'kurtosis': safe_kurtosis  
}

# ==================== POS EXTRACTION ====================
def extract_pos_sequence_spacy(text):
    doc = nlp(text)
    spacy_to_universal = {
        'NOUN': 'NOUN', 'PROPN': 'NOUN', 'VERB': 'VERB', 'AUX': 'VERB',
        'ADJ': 'ADJ', 'ADV': 'ADV', 'PRON': 'PRON', 'DET': 'DET',
        'ADP': 'ADP', 'NUM': 'NUM', 'CONJ': 'CONJ', 'CCONJ': 'CONJ',
        'SCONJ': 'CONJ', 'PART': 'PRT', 'PUNCT': '.',
        'SYM': 'X', 'INTJ': 'X', 'X': 'X', 'SPACE': 'X'
    }
    return [spacy_to_universal.get(token.pos_, 'X') for token in doc]

def extract_pos_sequence_nltk(text):
    import nltk
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens, tagset='universal')
    return [tag for word, tag in pos_tags]

def extract_pos_sequences(texts):
    sequences = []
    for text in texts:
        if isinstance(text, str) and len(text.strip()) > 0:
            try:
                seq = (extract_pos_sequence_spacy(text) if USE_SPACY 
                       else extract_pos_sequence_nltk(text))
                if len(seq) > 0:
                    sequences.append(seq)
            except:
                continue
    return sequences

# ==================== N-GRAM MODELS ====================
class POSLanguageModelNGram:
    """N-gram LM con Kneser-Ney o Backoff"""
    
    def __init__(self, n=2, method='kneser-ney', alpha=1.0, discount=0.75):
        self.n = n
        self.method = method
        self.alpha = alpha
        self.discount = discount
        
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set(['<s>', '</s>'])
        self.lower_order_model = None
        
        # Para Kneser-Ney
        self.continuation_counts = defaultdict(int)
    
    def train(self, sequences):
        """Entrena modelo"""
        for seq in sequences:
            self.vocab.update(seq)
        
        for seq in sequences:
            padded = ['<s>'] * (self.n - 1) + seq + ['</s>']
            
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i+self.n])
                context = ngram[:-1]
                word = ngram[-1]
                
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
                
                if self.method == 'kneser-ney':
                    self.continuation_counts[word] += 1
    
    def probability_backoff(self, word, context):
        """Backoff simple con Laplace smoothing"""
        context = tuple(context)
        count = self.ngram_counts[context][word]
        context_total = self.context_counts[context]
        
        if context_total > 0:
            # Laplace smoothing
            prob = (count + self.alpha) / (context_total + self.alpha * len(self.vocab))
        else:
            # Backoff a modelo de orden inferior
            if self.lower_order_model:
                prob = 0.4 * self.lower_order_model.probability(word, context[1:])
            else:
                prob = 1.0 / len(self.vocab)
        
        return max(prob, 1e-10)
    
    def probability_kneser_ney(self, word, context):
        """Kneser-Ney smoothing"""
        context = tuple(context)
        count = self.ngram_counts[context][word]
        context_total = self.context_counts[context]
        d = self.discount
        
        if context_total > 0:
            num_unique_continuations = len(self.ngram_counts[context])
            lambda_weight = (d * num_unique_continuations) / context_total
            
            total_continuation_count = sum(self.continuation_counts.values())
            p_continuation = (self.continuation_counts[word] / 
                            total_continuation_count if total_continuation_count > 0 
                            else 1.0 / len(self.vocab))
            
            prob = (max(count - d, 0) / context_total) + (lambda_weight * p_continuation)
        else:
            if self.lower_order_model:
                prob = 0.4 * self.lower_order_model.probability(word, context[1:])
            else:
                prob = 1.0 / len(self.vocab)
        
        return max(prob, 1e-10)
    
    def probability(self, word, context):
        """Wrapper"""
        if self.method == 'kneser-ney':
            return self.probability_kneser_ney(word, context)
        elif self.method == 'backoff':
            return self.probability_backoff(word, context)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def cross_entropy(self, sequence):
        """Cross-entropy de una utterance"""
        if len(sequence) < self.n:
            return np.nan
        
        padded = ['<s>'] * (self.n - 1) + sequence + ['</s>']
        log_probs = []
        
        for i in range(self.n - 1, len(padded)):
            context = padded[i-self.n+1:i]
            word = padded[i]
            
            prob = self.probability(word, context)
            log_probs.append(-np.log2(prob))
        
        return np.mean(log_probs) if log_probs else np.nan
    
    def perplexity(self, sequence):
        ce = self.cross_entropy(sequence)
        return 2 ** ce if not np.isnan(ce) else np.nan

# ==================== LSTM MODEL ====================
if TORCH_AVAILABLE:
    class POSDataset(Dataset):
        def __init__(self, sequences, vocab_to_idx):
            self.sequences = sequences
            self.vocab_to_idx = vocab_to_idx
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            seq = self.sequences[idx]
            # Convertir a índices
            indices = [self.vocab_to_idx.get(tag, self.vocab_to_idx['<unk>']) for tag in seq]
            # Añadir <s> y </s>
            indices = [self.vocab_to_idx['<s>']] + indices + [self.vocab_to_idx['</s>']]
            return torch.LongTensor(indices)
    
    def collate_fn(batch):
        """Agrupa secuencias con padding"""
        lengths = [len(seq) for seq in batch]
        max_len = max(lengths)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        
        for i, seq in enumerate(batch):
            padded[i, :len(seq)] = seq
        
        return padded, torch.LongTensor(lengths)
    
    class POSLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                               batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_dim, vocab_size)
        
        def forward(self, x, lengths):
            embeds = self.embedding(x)
            
            # Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            lstm_out, _ = self.lstm(packed)
            
            # Unpack
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            
            logits = self.fc(unpacked)
            return logits
    
    class POSLanguageModelLSTM:
        """LSTM Language Model"""
        
        def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = POSLSTM(vocab_size, embedding_dim, hidden_dim).to(self.device)
            self.vocab_to_idx = {}
            self.idx_to_vocab = {}
        
        def train_model(self, sequences, epochs=10, batch_size=32, lr=0.001):
            """Entrena LSTM"""
            print(f"    Entrenando LSTM en {self.device} ({epochs} epochs)...")
            
            # Crear vocabulario
            vocab = set(['<pad>', '<unk>', '<s>', '</s>'])
            for seq in sequences:
                vocab.update(seq)
            
            self.vocab_to_idx = {tag: idx for idx, tag in enumerate(sorted(vocab))}
            self.idx_to_vocab = {idx: tag for tag, idx in self.vocab_to_idx.items()}
            
            # Dataset
            dataset = POSDataset(sequences, self.vocab_to_idx)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn)
            
            # Optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_seqs, lengths in loader:
                    batch_seqs = batch_seqs.to(self.device)
                    
                    # Input: todas menos la última
                    # Target: todas menos la primera
                    input_seqs = batch_seqs[:, :-1]
                    target_seqs = batch_seqs[:, 1:]
                    input_lengths = lengths - 1
                    
                    optimizer.zero_grad()
                    logits = self.model(input_seqs, input_lengths)
                    
                    # Reshape para cross-entropy
                    logits_flat = logits.reshape(-1, logits.size(-1))
                    targets_flat = target_seqs.reshape(-1)
                    
                    loss = criterion(logits_flat, targets_flat)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 5 == 0:
                    avg_loss = total_loss / len(loader)
                    print(f" Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        def cross_entropy(self, sequence):
            """Calcula CE de una utterance"""
            if len(sequence) < 1:
                return np.nan
            
            self.model.eval()
            with torch.no_grad():
                # Convertir a índices
                indices = [self.vocab_to_idx.get(tag, self.vocab_to_idx['<unk>']) 
                          for tag in sequence]
                indices = [self.vocab_to_idx['<s>']] + indices + [self.vocab_to_idx['</s>']]
                
                # Input/target
                input_seq = torch.LongTensor(indices[:-1]).unsqueeze(0).to(self.device)
                target_seq = torch.LongTensor(indices[1:]).to(self.device)
                lengths = torch.LongTensor([len(input_seq[0])])
                
                # Forward
                logits = self.model(input_seq, lengths).squeeze(0)
                
                # Cross-entropy
                log_probs = F.log_softmax(logits, dim=-1)
                ce = -log_probs[range(len(target_seq)), target_seq].mean().item() / np.log(2)
                
                return ce
        
        def perplexity(self, sequence):
            ce = self.cross_entropy(sequence)
            return 2 ** ce if not np.isnan(ce) else np.nan

# ==================== TRAINING ====================
def train_ngram_models(control_sequences, method='kneser-ney'):
    """Entrena modelos n-gram con backoff"""
    models = {}
    
    print(f"  Entrenando n-gram models ({method})...")
    
    # Unigram
    unigram = POSLanguageModelNGram(n=1, method=method)
    unigram.train(control_sequences)
    models['unigram'] = unigram
    
    # Bigram
    bigram = POSLanguageModelNGram(n=2, method=method)
    bigram.lower_order_model = unigram
    bigram.train(control_sequences)
    models['bigram'] = bigram
    
    # Trigram
    trigram = POSLanguageModelNGram(n=3, method=method)
    trigram.lower_order_model = bigram
    trigram.train(control_sequences)
    models['trigram'] = trigram
    
    return models

def train_lstm_model(control_sequences, epochs=15):
    """Entrena modelo LSTM"""
    if not TORCH_AVAILABLE:
        print(" PyTorch no disponible. Saltando LSTM.")
        return None
    
    print(f"  Entrenando LSTM model...")
    
    # Determinar vocab_size
    vocab = set(['<pad>', '<unk>', '<s>', '</s>'])
    for seq in control_sequences:
        vocab.update(seq)
    
    lstm = POSLanguageModelLSTM(vocab_size=len(vocab))
    lstm.train_model(control_sequences, epochs=epochs)
    
    return lstm

# ==================== FEATURE EXTRACTION ====================
def calculate_poslm_features_ngram(patient_transcripts, models, prefix):
    """Calcula features de n-gram models"""
    pos_sequences = extract_pos_sequences(patient_transcripts)
    
    if len(pos_sequences) == 0:
        return {}
    
    features = {}
    
    for model_name in ['bigram', 'trigram']:
        if model_name not in models:
            continue
        
        model = models[model_name]
        ce_scores = []
        ppl_scores = []
        
        min_len = 2 if model_name == 'bigram' else 3
        
        for pos_seq in pos_sequences:
            if len(pos_seq) >= min_len:
                ce = model.cross_entropy(pos_seq)
                if not np.isnan(ce):
                    ce_scores.append(ce)
                    ppl_scores.append(2 ** ce)
        
        # 13 estadísticas
        if ce_scores:
            feat_prefix = f'poslm_{prefix}_{model_name}_ce'
            for stat_name, stat_func in STATISTICS.items():
                try:
                    features[f'{feat_prefix}_{stat_name}'] = stat_func(ce_scores)
                except:
                    features[f'{feat_prefix}_{stat_name}'] = np.nan
            
            # Perplexity
            features[f'poslm_{prefix}_{model_name}_ppl_mean'] = np.mean(ppl_scores)
            features[f'poslm_{prefix}_{model_name}_ppl_median'] = np.median(ppl_scores)
    
    return features

def calculate_poslm_features_lstm(patient_transcripts, lstm_model):
    """Calcula features de LSTM"""
    if lstm_model is None:
        return {}
    
    pos_sequences = extract_pos_sequences(patient_transcripts)
    
    if len(pos_sequences) == 0:
        return {}
    
    ce_scores = []
    ppl_scores = []
    
    for pos_seq in pos_sequences:
        if len(pos_seq) >= 1:
            ce = lstm_model.cross_entropy(pos_seq)
            if not np.isnan(ce):
                ce_scores.append(ce)
                ppl_scores.append(2 ** ce)
    
    features = {}
    
    if ce_scores:
        # 13 estadísticas
        prefix = 'poslm_lstm_ce'
        for stat_name, stat_func in STATISTICS.items():
            try:
                features[f'{prefix}_{stat_name}'] = stat_func(ce_scores)
            except:
                features[f'{prefix}_{stat_name}'] = np.nan
        
        # Perplexity
        features['poslm_lstm_ppl_mean'] = np.mean(ppl_scores)
        features['poslm_lstm_ppl_median'] = np.median(ppl_scores)
    
    return features

# ==================== MAIN PIPELINE ====================
def extract_poslm_features_for_dataset(df, text_column='transcript', 
                                       group_column='group', 
                                       patient_column='patient_id',
                                       methods=['kneser-ney', 'backoff', 'lstm'],
                                       lstm_epochs=15,
                                       save_models=True):
    """
    Pipeline completo - genera features para TODOS los métodos solicitados
    
    Args:
        methods: lista con 'kneser-ney', 'backoff', 'lstm'
    """
    print("="*70)
    print("POS-LM FEATURES EXTRACTION")
    print("="*70)
    print(f"Métodos: {', '.join(methods)}")
    
    # Separar Control y PWA
    df_control = df[df[group_column] == 'control'].copy()
    
    print(f"\nDatos:")
    print(f"  Control: {len(df_control)} utterances")
    print(f"  Total: {len(df)} utterances")
    
    if len(df_control) == 0:
        raise ValueError(" No hay datos Control para entrenar POS-LM")
    
    # Extraer POS de Control
    print("\n Extrayendo secuencias POS de Control...")
    control_texts = df_control[text_column].dropna().tolist()
    control_pos_sequences = extract_pos_sequences(control_texts)
    print(f"  ✓ {len(control_pos_sequences)} secuencias POS")
    
    # Entrenar modelos
    print("\n Entrenando modelos...")
    
    models_dict = {}
    
    if 'kneser-ney' in methods:
        print("\n1. Kneser-Ney smoothing:")
        models_dict['kn'] = train_ngram_models(control_pos_sequences, method='kneser-ney')
    
    if 'backoff' in methods:
        print("\n2. Backoff simple:")
        models_dict['bo'] = train_ngram_models(control_pos_sequences, method='backoff')
    
    if 'lstm' in methods:
        print("\n3. LSTM:")
        models_dict['lstm'] = train_lstm_model(control_pos_sequences, epochs=lstm_epochs)
    
    # Guardar modelos
    if save_models:
        import pickle
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        with open(models_dir / "poslm_models_all.pkl", 'wb') as f:
            pickle.dump(models_dict, f)
        print(f"\n Modelos guardados: models/poslm_models_all.pkl")
    
    # Calcular features por paciente
    print("\n Calculando features por paciente...")
    
    all_features = []
    unique_patients = df[patient_column].unique()
    
    for i, patient_id in enumerate(unique_patients, 1):
        if i % 50 == 0 or i == len(unique_patients):
            print(f"  {i}/{len(unique_patients)}")
        
        patient_data = df[df[patient_column] == patient_id]
        patient_texts = patient_data[text_column].dropna().tolist()
        
        if len(patient_texts) == 0:
            continue
        
        features = {patient_column: patient_id}
        
        # Group
        if group_column in patient_data.columns:
            features[group_column] = patient_data[group_column].iloc[0]
        
        # Kneser-Ney
        if 'kn' in models_dict:
            feat_kn = calculate_poslm_features_ngram(patient_texts, models_dict['kn'], 'kn')
            features.update(feat_kn)
        
        # Backoff
        if 'bo' in models_dict:
            feat_bo = calculate_poslm_features_ngram(patient_texts, models_dict['bo'], 'bo')
            features.update(feat_bo)
        
        # LSTM
        if 'lstm' in models_dict and models_dict['lstm'] is not None:
            feat_lstm = calculate_poslm_features_lstm(patient_texts, models_dict['lstm'])
            features.update(feat_lstm)
        
        all_features.append(features)
    
    # DataFrame final
    features_df = pd.DataFrame(all_features)
    
    # Rellenar NaN
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(
        features_df[numeric_cols].median()
    )
    
    print(f"\n Extraídas: {len(features_df)} pacientes")
    poslm_cols = [col for col in features_df.columns if 'poslm' in col]
    print(f"   Columnas POS-LM totales: {len(poslm_cols)}")
    
    # Contar por método
    for method_prefix in ['kn', 'bo', 'lstm']:
        method_cols = [col for col in poslm_cols if f'poslm_{method_prefix}_' in col]
        if method_cols:
            print(f"   - {method_prefix.upper()}: {len(method_cols)} features")
    
    return features_df

# ==================== CLI ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='POS-LM features (3 métodos)')
    parser.add_argument('--input', required=True, help='CSV con transcripciones')
    parser.add_argument('--output', default=None, help='CSV output')
    parser.add_argument('--methods', nargs='+', 
                       choices=['kneser-ney', 'backoff', 'lstm'],
                       default=['kneser-ney', 'backoff'],
                       help='Métodos a usar')
    parser.add_argument('--lstm-epochs', type=int, default=15,
                       help='Epochs para LSTM')
    parser.add_argument('--text-col', default='transcript')
    parser.add_argument('--group-col', default='group')
    parser.add_argument('--patient-col', default='patient_id')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f" No existe: {args.input}")
        sys.exit(1)
    
    df = pd.read_csv(args.input)
    
    poslm_features = extract_poslm_features_for_dataset(
        df,
        text_column=args.text_col,
        group_column=args.group_col,
        patient_column=args.patient_col,
        methods=args.methods,
        lstm_epochs=args.lstm_epochs
    )
    
    output_path = args.output or 'data/poslm_features_ALL.csv'
    poslm_features.to_csv(output_path, index=False)
    
    print(f"\n Guardado: {output_path}")
    print("="*70)