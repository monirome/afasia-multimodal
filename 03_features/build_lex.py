#!/usr/bin/env python3
# build_lex.py
"""
Construye LEX features (PAPER MODE: solo 6 features)
Para inglés (EN), español (ES) y catalán (CA)
"""

import os
import re
import glob
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

# ======================== CONFIGURACION ========================
LEX_MODE = "paper"  # FIJO: Solo mean (6 features como Fraser et al.)

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
print("BUILD: LEX FEATURES - PAPER MODE (6 features)")
print("Languages: EN, ES, CA")
print("="*70)

# ======================== CARGAR WORD DATABASE ========================
print("\n[1/5] Loading word database...")

db_file = os.path.join(WORD_DB_DIR, "word_database_master.csv")
if not os.path.exists(db_file):
    print(f"\nERROR: File not found {db_file}")
    print("\nRun first:")
    print("  cd /lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025/03_features")
    print("  python3 download_lex_databases.py")
    sys.exit(1)

word_db = pd.read_csv(db_file)
print(f"      Words loaded: {len(word_db)}")

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
        print(f"  Error reading {filepath}: {e}")
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

# ======================== EXTRACCION POR PACIENTE ========================
def extract_LEX_for_patient(cha_data):
    """
    Extrae 6 LEX features (PAPER MODE):
    - Feature 29: TTR
    - Feature 30: freq_mean
    - Feature 31: img_mean
    - Feature 32: aoa_mean
    - Feature 33: fam_mean
    - Feature 34: phones_mean
    """
    
    utterances = cha_data['utterances']
    
    if not utterances:
        return {
            'lex_ttr': np.nan,
            'lex_freq_mean': np.nan,
            'lex_img_mean': np.nan,
            'lex_aoa_mean': np.nan,
            'lex_fam_mean': np.nan,
            'lex_phones_mean': np.nan
        }
    
    # Juntar todo el texto
    all_text = ' '.join(utterances)
    
    # Tokenizar
    words = re.findall(r'\b\w+\b', all_text)
    
    if len(words) == 0:
        return {
            'lex_ttr': np.nan,
            'lex_freq_mean': np.nan,
            'lex_img_mean': np.nan,
            'lex_aoa_mean': np.nan,
            'lex_fam_mean': np.nan,
            'lex_phones_mean': np.nan
        }
    
    # POS tagging
    pos_tags = nltk.pos_tag(words, tagset='universal')
    
    # Feature 29: Type-Token Ratio (Fergadiotis & Wright 2011)
    open_class_lemmas = [
        lemmatizer.lemmatize(w.lower()) 
        for w, pos in pos_tags 
        if is_open_class(pos)
    ]
    
    if len(open_class_lemmas) > 0:
        ttr = len(set(open_class_lemmas)) / len(open_class_lemmas)
    else:
        ttr = np.nan
    
    # Features 30-34: Word scores (mean only, like Fraser et al. 2013)
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
    
    return {
        'lex_ttr': float(ttr) if not np.isnan(ttr) else np.nan,
        'lex_freq_mean': float(np.mean(freq_scores)) if len(freq_scores) > 0 else np.nan,
        'lex_img_mean': float(np.mean(img_scores)) if len(img_scores) > 0 else np.nan,
        'lex_aoa_mean': float(np.mean(aoa_scores)) if len(aoa_scores) > 0 else np.nan,
        'lex_fam_mean': float(np.mean(fam_scores)) if len(fam_scores) > 0 else np.nan,
        'lex_phones_mean': float(np.mean(phone_counts)) if len(phone_counts) > 0 else np.nan
    }

# ======================== MAIN ========================
print("\n[2/5] Finding .CHA files...")

CHA_DIR = "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones"
cha_files = glob.glob(os.path.join(CHA_DIR, "**/*.cha"), recursive=True)

print(f"      .CHA files found: {len(cha_files)}")

if len(cha_files) == 0:
    print(f"\nERROR: No .CHA files found in {CHA_DIR}")
    sys.exit(1)

print("\n[3/5] Parsing .CHA files...")

cha_data_dict = {}
for i, cha_path in enumerate(cha_files, 1):
    if i % 100 == 0:
        print(f"        {i}/{len(cha_files)} files...")
    
    parsed = parse_cha_file(cha_path)
    if parsed and len(parsed['utterances']) > 0:
        cha_data_dict[parsed['patient_id']] = parsed

print(f"      Valid patients: {len(cha_data_dict)}")

# Separar por idioma
cha_data_by_lang = {
    'en': {k: v for k, v in cha_data_dict.items() if v['language'] == 'en'},
    'es': {k: v for k, v in cha_data_dict.items() if v['language'] == 'es'},
    'ca': {k: v for k, v in cha_data_dict.items() if v['language'] == 'ca'}
}

print(f"      EN patients: {len(cha_data_by_lang['en'])}")
print(f"      ES patients: {len(cha_data_by_lang['es'])}")
print(f"      CA patients: {len(cha_data_by_lang['ca'])}")

# ======================== EXTRACCION POR IDIOMA ========================
print("\n[4/5] Extracting LEX features by language...")

# Cargar metadata (QA) si existe
metadata_file = os.path.join(PROJECT_BASE, "data/patient_metadata_WAB.csv")
df_meta = None
if os.path.exists(metadata_file):
    df_meta = pd.read_csv(metadata_file)
    print(f"      Metadata loaded: {len(df_meta)} patients")

for lang in ['en', 'es', 'ca']:
    print(f"\n      Processing {lang.upper()}...")
    
    cha_data_lang = cha_data_by_lang[lang]
    
    if len(cha_data_lang) == 0:
        print(f"        No patients for {lang.upper()}, skipping...")
        continue
    
    results = []
    patients = sorted(cha_data_lang.keys())
    
    for i, pid in enumerate(patients, 1):
        if i % 50 == 0:
            print(f"          Progress: {i}/{len(patients)} patients")
        
        cha_data = cha_data_lang[pid]
        feats = extract_LEX_for_patient(cha_data)
        
        row = {
            'patient_id': pid,
            'language': lang
        }
        row.update(feats)
        results.append(row)
    
    df_lex = pd.DataFrame(results)
    
    # Merge con metadata si existe
    if df_meta is not None:
        df_lex = df_lex.merge(
            df_meta[['patient_id', 'QA']],
            on='patient_id',
            how='left'
        )
    
    # Coverage
    print(f"          Patients processed: {len(df_lex)}")
    print(f"          Coverage:")
    print(f"            TTR:    {df_lex['lex_ttr'].notna().sum()} ({100*df_lex['lex_ttr'].notna().mean():.1f}%)")
    print(f"            Freq:   {df_lex['lex_freq_mean'].notna().sum()} ({100*df_lex['lex_freq_mean'].notna().mean():.1f}%)")
    print(f"            Img:    {df_lex['lex_img_mean'].notna().sum()} ({100*df_lex['lex_img_mean'].notna().mean():.1f}%)")
    print(f"            AoA:    {df_lex['lex_aoa_mean'].notna().sum()} ({100*df_lex['lex_aoa_mean'].notna().mean():.1f}%)")
    print(f"            Fam:    {df_lex['lex_fam_mean'].notna().sum()} ({100*df_lex['lex_fam_mean'].notna().mean():.1f}%)")
    print(f"            Phones: {df_lex['lex_phones_mean'].notna().sum()} ({100*df_lex['lex_phones_mean'].notna().mean():.1f}%)")
    
    # Guardar
    out_file = os.path.join(OUT_DIR, f"lex_features_{lang}.csv")
    df_lex.to_csv(out_file, index=False)
    print(f"          Saved: {out_file}")

# ======================== RESUMEN FINAL ========================
print("\n[5/5] Summary")
print("="*70)
print("COMPLETED")
print("="*70)
print("Mode: PAPER (replicates Fraser et al. 2013)")
print("Features extracted: 6 LEX features")
print("")
print("  Feature 29: lex_ttr          (Type-Token Ratio)")
print("  Feature 30: lex_freq_mean    (Word frequency)")
print("  Feature 31: lex_img_mean     (Word imageability)")
print("  Feature 32: lex_aoa_mean     (Age of acquisition)")
print("  Feature 33: lex_fam_mean     (Word familiarity)")
print("  Feature 34: lex_phones_mean  (Phones per word)")
print("")
print("Output files:")
print(f"  {os.path.join(OUT_DIR, 'lex_features_en.csv')}")
print(f"  {os.path.join(OUT_DIR, 'lex_features_es.csv')}")
print(f"  {os.path.join(OUT_DIR, 'lex_features_ca.csv')}")
print("="*70)