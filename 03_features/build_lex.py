#!/usr/bin/env python3
# build_lex.py
"""
Construye LEX features leyendo directamente de archivos .CHA
"""

import os
import re
import glob
import argparse
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

# Paths
PROJECT_BASE = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025"
WORD_DB_DIR = os.path.join(PROJECT_BASE, "data/word_databases")
OUT_DIR = os.path.join(PROJECT_BASE, "data/lex_features")

os.makedirs(OUT_DIR, exist_ok=True)

# Verificar NLTK data
print("Verificando NLTK data...")
for corpus in ['wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'universal_tagset']:
    try:
        nltk.data.find(f'corpora/{corpus}' if corpus in ['wordnet', 'omw-1.4'] else f'taggers/{corpus}')
    except:
        print(f"  Descargando {corpus}...")
        nltk.download(corpus, quiet=True)

print("="*70)
print("BUILD: LEX FEATURES (desde archivos .CHA)")
print("="*70)

# ======================== CARGAR WORD DATABASE ========================
print("\n[1/4] Cargando word database...")

db_file = os.path.join(WORD_DB_DIR, "word_database_master.csv")
if not os.path.exists(db_file):
    print(f"\nERROR: No existe {db_file}")
    print("\nEjecuta primero:")
    print("  cd /lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025/03_features")
    print("  python3 download_lex_databases.py")
    sys.exit(1)

word_db = pd.read_csv(db_file)
print(f"      Palabras cargadas: {len(word_db)}")

# Diccionarios para lookup
freq_dict = dict(zip(word_db['word'], word_db['frequency']))
img_dict = dict(zip(word_db['word'], word_db['imageability']))
aoa_dict = dict(zip(word_db['word'], word_db['aoa']))
fam_dict = dict(zip(word_db['word'], word_db['familiarity']))
phone_dict = dict(zip(word_db['word'], word_db['n_phones']))

# ======================== PARSER .CHA ========================
def parse_cha_file(filepath):
    """Lee archivo .CHA y extrae utterances del paciente"""
    patient_id = os.path.splitext(os.path.basename(filepath))[0]
    utterances = []
    language = 'en'  # default
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # Detectar idioma
                if line.startswith('@ID:') or line.startswith('@Languages:'):
                    if 'spa' in line.lower() or 'spanish' in line.lower():
                        language = 'es'
                    elif 'cat' in line.lower() or 'catalan' in line.lower():
                        language = 'ca'
                
                # Extraer utterances del paciente (*PAR:)
                if line.startswith('*PAR:'):
                    utt = line[5:].strip()
                    # Limpiar códigos CHAT
                    utt = re.sub(r'[.!?]+$', '', utt)
                    utt = re.sub(r'<[^>]+>', '', utt)  # Quitar tags
                    utt = re.sub(r'\[/?\]', '', utt)   # Quitar códigos disfluencia
                    if utt:
                        utterances.append(utt)
        
        return {
            'patient_id': patient_id,
            'utterances': utterances,
            'language': language
        }
    
    except Exception as e:
        print(f"  Error leyendo {filepath}: {e}")
        return None

# ======================== UTILIDADES ========================
lemmatizer = WordNetLemmatizer()

def is_open_class(pos_tag):
    return pos_tag in ['NOUN', 'VERB', 'ADJ', 'ADV']

def get_word_score(word, score_dict):
    word_lower = word.lower()
    
    if word_lower in score_dict:
        score = score_dict[word_lower]
        if pd.notna(score):
            return float(score)
    
    lemma = lemmatizer.lemmatize(word_lower)
    if lemma in score_dict and lemma != word_lower:
        score = score_dict[lemma]
        if pd.notna(score):
            return float(score)
    
    return np.nan

def stats13_array(arr, prefix):
    s = pd.Series(arr).dropna()
    if len(s) == 0:
        keys = ["q1","q2","q3","iqr12","iqr23","iqr13","p01","p99",
                "range01_99","mean","std","skew","kurt"]
        return {f"{prefix}_{k}": np.nan for k in keys}
    
    q1 = s.quantile(0.25)
    q2 = s.quantile(0.50)
    q3 = s.quantile(0.75)
    p01 = s.quantile(0.01)
    p99 = s.quantile(0.99)
    
    return {
        f"{prefix}_q1": float(q1),
        f"{prefix}_q2": float(q2),
        f"{prefix}_q3": float(q3),
        f"{prefix}_iqr12": float(q2 - q1),
        f"{prefix}_iqr23": float(q3 - q2),
        f"{prefix}_iqr13": float(q3 - q1),
        f"{prefix}_p01": float(p01),
        f"{prefix}_p99": float(p99),
        f"{prefix}_range01_99": float(p99 - p01),
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_std": float(s.std(ddof=0)),
        f"{prefix}_skew": float(s.skew()),
        f"{prefix}_kurt": float(s.kurt())
    }

# ======================== EXTRACCION POR PACIENTE ========================
def extract_LEX_for_patient(cha_data):
    """Extrae 66 LEX features desde datos .CHA"""
    
    utterances = cha_data['utterances']
    
    if not utterances:
        return {
            'lex_ttr': np.nan,
            **stats13_array(np.array([]), 'lex_freq'),
            **stats13_array(np.array([]), 'lex_img'),
            **stats13_array(np.array([]), 'lex_aoa'),
            **stats13_array(np.array([]), 'lex_fam'),
            **stats13_array(np.array([]), 'lex_phones')
        }
    
    # Juntar todo el texto
    all_text = ' '.join(utterances)
    
    # Tokenizar
    words = re.findall(r'\b\w+\b', all_text)
    
    if len(words) == 0:
        return {
            'lex_ttr': np.nan,
            **stats13_array(np.array([]), 'lex_freq'),
            **stats13_array(np.array([]), 'lex_img'),
            **stats13_array(np.array([]), 'lex_aoa'),
            **stats13_array(np.array([]), 'lex_fam'),
            **stats13_array(np.array([]), 'lex_phones')
        }
    
    # POS tagging
    pos_tags = nltk.pos_tag(words, tagset='universal')
    
    # Feature 29: Type-Token Ratio
    open_class_lemmas = [
        lemmatizer.lemmatize(w.lower()) 
        for w, pos in pos_tags 
        if is_open_class(pos)
    ]
    
    if len(open_class_lemmas) > 0:
        ttr = len(set(open_class_lemmas)) / len(open_class_lemmas)
    else:
        ttr = np.nan
    
    # Features 30-34: Word scores
    freq_scores = []
    img_scores = []
    aoa_scores = []
    fam_scores = []
    phone_counts = []
    
    for word in words:
        freq = get_word_score(word, freq_dict)
        if pd.notna(freq):
            freq_scores.append(freq)
        
        img = get_word_score(word, img_dict)
        if pd.notna(img):
            img_scores.append(img)
        
        aoa = get_word_score(word, aoa_dict)
        if pd.notna(aoa):
            aoa_scores.append(aoa)
        
        fam = get_word_score(word, fam_dict)
        if pd.notna(fam):
            fam_scores.append(fam)
        
        phones = get_word_score(word, phone_dict)
        if pd.notna(phones):
            phone_counts.append(phones)
    
    out = {'lex_ttr': float(ttr)}
    out.update(stats13_array(np.array(freq_scores), 'lex_freq'))
    out.update(stats13_array(np.array(img_scores), 'lex_img'))
    out.update(stats13_array(np.array(aoa_scores), 'lex_aoa'))
    out.update(stats13_array(np.array(fam_scores), 'lex_fam'))
    out.update(stats13_array(np.array(phone_counts), 'lex_phones'))
    
    return out

# ======================== MAIN ========================
print("\n[2/4] Buscando archivos .CHA...")

CHA_DIR = "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones"
cha_files = glob.glob(os.path.join(CHA_DIR, "**/*.cha"), recursive=True)

print(f"      Archivos .CHA encontrados: {len(cha_files)}")

if len(cha_files) == 0:
    print(f"\nERROR: No se encontraron archivos .CHA en {CHA_DIR}")
    sys.exit(1)

print("\n[3/4] Parseando archivos .CHA...")

cha_data_dict = {}
for i, cha_path in enumerate(cha_files, 1):
    if i % 100 == 0:
        print(f"        {i}/{len(cha_files)} archivos...")
    
    parsed = parse_cha_file(cha_path)
    if parsed and len(parsed['utterances']) > 0:
        cha_data_dict[parsed['patient_id']] = parsed

print(f"      Pacientes válidos: {len(cha_data_dict)}")

# Filtrar solo inglés
cha_data_en = {k: v for k, v in cha_data_dict.items() if v['language'] == 'en'}
print(f"      Pacientes EN: {len(cha_data_en)}")

print("\n[4/4] Extrayendo LEX features...")

results = []
patients = sorted(cha_data_en.keys())

for i, pid in enumerate(patients, 1):
    if i % 50 == 0:
        print(f"        Progreso: {i}/{len(patients)} pacientes")
    
    cha_data = cha_data_en[pid]
    feats = extract_LEX_for_patient(cha_data)
    
    row = {
        'patient_id': pid,
        'language': cha_data['language']
    }
    row.update(feats)
    results.append(row)

df_lex = pd.DataFrame(results)

# Cargar metadata (QA) si existe
metadata_file = os.path.join(PROJECT_BASE, "data/patient_metadata_WAB.csv")
if os.path.exists(metadata_file):
    df_meta = pd.read_csv(metadata_file)
    df_lex = df_lex.merge(
        df_meta[['patient_id', 'QA']],
        on='patient_id',
        how='left'
    )

print(f"\n      Pacientes procesados: {len(df_lex)}")

# Coverage
print("\n      Coverage por feature group:")
print(f"        TTR:    {df_lex['lex_ttr'].notna().sum()} ({100*df_lex['lex_ttr'].notna().mean():.1f}%)")
print(f"        Freq:   {df_lex['lex_freq_mean'].notna().sum()} ({100*df_lex['lex_freq_mean'].notna().mean():.1f}%)")
print(f"        Img:    {df_lex['lex_img_mean'].notna().sum()} ({100*df_lex['lex_img_mean'].notna().mean():.1f}%)")
print(f"        AoA:    {df_lex['lex_aoa_mean'].notna().sum()} ({100*df_lex['lex_aoa_mean'].notna().mean():.1f}%)")
print(f"        Fam:    {df_lex['lex_fam_mean'].notna().sum()} ({100*df_lex['lex_fam_mean'].notna().mean():.1f}%)")
print(f"        Phones: {df_lex['lex_phones_mean'].notna().sum()} ({100*df_lex['lex_phones_mean'].notna().mean():.1f}%)")

# Guardar
out_file = os.path.join(OUT_DIR, "lex_features_en.csv")
df_lex.to_csv(out_file, index=False)

print(f"\n      Output: {out_file}")
print("\n" + "="*70)
print("COMPLETADO")
print("="*70)
print(f"Features extraídas: {len([c for c in df_lex.columns if c.startswith('lex_')])}")
print(f"  - Feature 29 (TTR): 1 feature")
print(f"  - Features 30-34 (5 grupos × 13 stats): 65 features")
print(f"  - TOTAL: 66 LEX features")
print("="*70)