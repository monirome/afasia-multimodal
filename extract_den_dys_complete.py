#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
EXTRACCIÓN FEATURES DEN (1-18) + DYS (19-28) - PAPER VERSION
Replica EXACTA de Le et al. (2018) - Solo las 28 features originales
==============================================================================

FEATURES GENERADAS:
- DEN (18 features): words/min, phones/min, W, OCW, words/utt (mean),
  phones/utt (mean), POS ratios (nouns, verbs, etc.)
- DYS (10 features): fillers/min, fillers/word, fillers/phone,
  pauses/min (total, long, short), pauses/word (total, long, short),
  seconds/pause (mean)

TOTAL: 28 features + metadata (patient_id, language)
==============================================================================
"""

import os
import re
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import spacy
from collections import Counter

# ======================== CONFIGURACIÓN ========================
PAUSE_MIN = 0.150   # 150 ms
PAUSE_LONG = 0.400  # 400 ms

SPACY_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "ca": "ca_core_news_sm",
}

# ==================== PATRONES .CHA ====================
CHA_FILLER_PATTERNS = [
    r'&uh+', r'&um+', r'&er+', r'&mm+', r'&hmm+',
    r'&eh+', r'&em+', r'&ah+', r'&oh+',
]

CHA_FILLER_WORDS = {
    'uh', 'um', 'er', 'mm', 'hmm', 'mhm', 'uh-huh', 'um-hum',
    'eh', 'em', 'este', 'pues', 'doncs',
}

CHA_DISFLUENCY_CODES = [r'\[/\]', r'\[//\]', r'\[/\?\]']

# ==================== POS TAGS ====================
OPEN_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
FUNC_POS = {"ADP", "AUX", "CCONJ", "DET", "INTJ", "NUM", "PART", "PRON", "SCONJ"}

LIGHT_VERBS = {
    "en": {"be", "have", "do", "get", "go", "make", "take"},
    "es": {"ser", "estar", "haber", "tener", "hacer", "ir", "dar", "poner"},
    "ca": {"ser", "estar", "haver", "tenir", "fer", "anar", "donar", "posar"},
}

DEMONSTRATIVES = {
    "en": {"this", "that", "these", "those"},
    "es": {"este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", 
           "aquel", "aquella", "aquellos", "aquellas"},
    "ca": {"aquest", "aquesta", "aquests", "aquestes", 
           "aquell", "aquella", "aquells", "aquelles"},
}

# ======================== PARSER .CHA ========================
def parse_cha_file(filepath):
    """Lee archivo .CHA y extrae utterances del paciente"""
    patient_id = os.path.splitext(os.path.basename(filepath))[0]
    utterances = []
    metadata = {'language': 'en'}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('@ID:'):
                    parts = line.split('|')
                    if len(parts) > 2:
                        lang_code = parts[1].lower()
                        if 'eng' in lang_code:
                            metadata['language'] = 'en'
                        elif 'spa' in lang_code:
                            metadata['language'] = 'es'
                        elif 'cat' in lang_code:
                            metadata['language'] = 'ca'
                
                if line.startswith('*PAR:'):
                    utt = line[5:].strip()
                    utt = re.sub(r'[.!?]+$', '', utt)
                    if utt:
                        utterances.append(utt)
        
        return {
            'patient_id': patient_id,
            'utterances': utterances,
            'language': metadata['language']
        }
    
    except Exception as e:
        print(f"  Error leyendo {filepath}: {e}")
        return None

def count_fillers_cha(utterance):
    """Cuenta fillers en una utterance"""
    count = 0
    
    for pattern in CHA_FILLER_PATTERNS:
        count += len(re.findall(pattern, utterance, re.IGNORECASE))
    
    words = re.findall(r'\b\w+\b', utterance.lower())
    for word in words:
        if word in CHA_FILLER_WORDS:
            count += 1
    
    for pattern in CHA_DISFLUENCY_CODES:
        count += len(re.findall(pattern, utterance))
    
    return count

# ======================== SPACY ========================
def load_nlp(lang):
    """Carga modelo spaCy"""
    model = SPACY_MODELS.get(lang, SPACY_MODELS["en"])
    try:
        return spacy.load(model, disable=["ner", "lemmatizer"])
    except:
        raise RuntimeError(
            f"Error cargando {model}. Instálalo con:\n"
            f"  python -m spacy download {model}"
        )

def estimate_phones(words):
    """Estima fonemas: ~3.5 por palabra"""
    return int(len(words) * 3.5)

def analyze_utterance_spacy(nlp, text, lang):
    """Analiza una utterance con spaCy"""
    doc = nlp(text)
    
    light_vbs = LIGHT_VERBS.get(lang, LIGHT_VERBS["en"])
    demos = DEMONSTRATIVES.get(lang, DEMONSTRATIVES["en"])
    
    words = [t for t in doc if not t.is_space and (t.is_alpha or t.pos_ != "PUNCT")]
    n_words = len(words)
    n_phones = estimate_phones(words)
    
    if n_words == 0:
        return {
            'words': 0, 'phones': 0, 'open_words': 0, 'interj': 0,
            'nouns': 0, 'verbs': 0, 'light_verbs': 0, 'determiners': 0,
            'demonstratives': 0, 'prepositions': 0, 'adjectives': 0,
            'adverbs': 0, 'pronouns': 0, 'function_words': 0
        }
    
    counts = Counter()
    open_count = 0
    func_count = 0
    light_count = 0
    demo_count = 0
    
    for t in words:
        counts[t.pos_] += 1
        
        if t.pos_ in OPEN_POS:
            open_count += 1
        if t.pos_ in FUNC_POS:
            func_count += 1
        
        lemma = t.lemma_.lower()
        
        if t.pos_ == "VERB" and lemma in light_vbs:
            light_count += 1
        if t.pos_ == "DET" and lemma in demos:
            demo_count += 1
    
    return {
        'words': n_words,
        'phones': n_phones,
        'open_words': open_count,
        'interj': counts.get('INTJ', 0),
        'nouns': counts.get('NOUN', 0) + counts.get('PROPN', 0),
        'verbs': counts.get('VERB', 0),
        'light_verbs': light_count,
        'determiners': counts.get('DET', 0),
        'demonstratives': demo_count,
        'prepositions': counts.get('ADP', 0),
        'adjectives': counts.get('ADJ', 0),
        'adverbs': counts.get('ADV', 0),
        'pronouns': counts.get('PRON', 0),
        'function_words': func_count,
    }

# ======================== UTILIDADES ========================
def safe_float_series(s):
    """Convierte Series a numérico"""
    return pd.to_numeric(s, errors="coerce")

def safe_div(num, den):
    """División segura"""
    if den is None or not np.isfinite(den) or den <= 0:
        return np.nan
    return float(num / den)

def minutes(seconds):
    """Segundos a minutos"""
    return max(float(seconds) / 60.0, 1e-6)

# ======================== BUILD DEN (18 features) ========================
def build_DEN_from_cha(cha_data, speech_sec):
    """
    Construye 18 features DEN desde .CHA
    """
    nlp = load_nlp(cha_data['language'])
    
    all_stats = []
    for utt in cha_data['utterances']:
        stats = analyze_utterance_spacy(nlp, utt, cha_data['language'])
        stats['utt_words'] = stats['words']
        stats['utt_phones'] = stats['phones']
        all_stats.append(stats)
    
    if not all_stats:
        return {}
    
    df = pd.DataFrame(all_stats)
    
    total_words = float(df['words'].sum())
    total_phones = float(df['phones'].sum())
    total_interj = float(df['interj'].sum())
    total_open = float(df['open_words'].sum())
    
    speech_min = minutes(speech_sec)
    
    def ratio(n, d):
        return float(n / max(d, 1e-6))
    
    # DEN 1-4
    den = {
        'den_words_per_min': total_words / speech_min,
        'den_phones_per_min': total_phones / speech_min,
        'den_W': ratio(total_words, total_words + total_interj),
        'den_OCW': ratio(total_open, total_words),
    }
    
    # DEN 5-6: SOLO MEDIA (no todas las stats)
    den['den_words_utt_mean'] = float(df['utt_words'].mean()) if len(df) > 0 else np.nan
    den['den_phones_utt_mean'] = float(df['utt_phones'].mean()) if len(df) > 0 else np.nan
    
    # DEN 7-18: POS ratios
    nouns = float(df['nouns'].sum())
    verbs = float(df['verbs'].sum())
    
    den.update({
        'den_nouns': ratio(nouns, total_words),
        'den_verbs': ratio(verbs, total_words),
        'den_nouns_per_verb': ratio(nouns, verbs),
        'den_noun_ratio': ratio(nouns, nouns + verbs),
        'den_light_verbs': ratio(df['light_verbs'].sum(), verbs) if verbs > 0 else np.nan,
        'den_determiners': ratio(df['determiners'].sum(), total_words),
        'den_demonstratives': ratio(df['demonstratives'].sum(), total_words),
        'den_prepositions': ratio(df['prepositions'].sum(), total_words),
        'den_adjectives': ratio(df['adjectives'].sum(), total_words),
        'den_adverbs': ratio(df['adverbs'].sum(), total_words),
        'den_pronoun_ratio': ratio(df['pronouns'].sum(), 
                                   df['pronouns'].sum() + nouns),
        'den_function_words': ratio(df['function_words'].sum(), total_words),
    })
    
    den['_total_words'] = total_words
    den['_total_phones'] = total_phones
    
    return den

# ======================== BUILD DYS (10 features) ========================
def build_DYS_from_cha_and_align(cha_data, df_words, speech_sec):
    """
    Construye 10 features DYS desde .CHA + alignments
    """
    # DYS 19-21: Fillers
    total_fillers = 0
    for utt in cha_data['utterances']:
        total_fillers += count_fillers_cha(utt)
    
    # DYS 22-28: Pauses
    if df_words.empty or len(df_words) < 2:
        pauses = []
    else:
        df_words = df_words.sort_values(['start_sec', 'end_sec']).reset_index(drop=True)
        gaps = (df_words['start_sec'].iloc[1:].values - 
                df_words['end_sec'].iloc[:-1].values)
        pauses = [float(g) for g in gaps if g >= PAUSE_MIN]
    
    n_pauses = len(pauses)
    n_long = sum(1 for p in pauses if p > PAUSE_LONG)
    n_short = n_pauses - n_long
    
    total_words = len(df_words) if not df_words.empty else 1
    total_phones = int(total_words * 3.5)
    speech_min = minutes(speech_sec)
    
    # DYS 19-27
    dys = {
        'dys_fillers_per_min': total_fillers / speech_min,
        'dys_fillers_per_word': safe_div(total_fillers, total_words),
        'dys_fillers_per_phone': safe_div(total_fillers, total_phones),
        'dys_pauses_per_min': n_pauses / speech_min,
        'dys_long_pauses_per_min': n_long / speech_min,
        'dys_short_pauses_per_min': n_short / speech_min,
        'dys_pauses_per_word': safe_div(n_pauses, total_words),
        'dys_long_pauses_per_word': safe_div(n_long, total_words),
        'dys_short_pauses_per_word': safe_div(n_short, total_words),
    }
    
    # DYS 28: SOLO MEDIA (no todas las stats)
    dys['dys_pause_sec_mean'] = float(np.mean(pauses)) if len(pauses) > 0 else np.nan
    
    return dys

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(
        description="Extrae 28 features DEN+DYS (Paper Version)"
    )
    parser.add_argument("--cha_dir", required=True)
    parser.add_argument("--word_align_csv", required=True)
    parser.add_argument("--output", default="features_den_dys_COMPLETO.csv")
    
    args = parser.parse_args()
    
    print("="*70)
    print("EXTRACCIÓN FEATURES DEN+DYS - PAPER VERSION (28 features)")
    print("="*70)
    
    # Cargar .CHA
    print("\nBuscando archivos .CHA...")
    cha_files = glob.glob(os.path.join(args.cha_dir, "**/*.cha"), recursive=True)
    print(f"Archivos .CHA encontrados: {len(cha_files)}")
    
    if len(cha_files) == 0:
        print("No se encontraron archivos .CHA")
        return
    
    cha_data_dict = {}
    for cha_path in cha_files:
        parsed = parse_cha_file(cha_path)
        if parsed and len(parsed['utterances']) > 0:
            cha_data_dict[parsed['patient_id']] = parsed
    
    print(f"Pacientes con .CHA válidos: {len(cha_data_dict)}")
    
    # Cargar alignments
    print("\nCargando word alignments...")
    if not os.path.exists(args.word_align_csv):
        print(f"No existe: {args.word_align_csv}")
        return
    
    df_align = pd.read_csv(args.word_align_csv)
    print(f"Word alignments cargados: {len(df_align)}")
    
    required_cols = {'patient_id', 'word', 'start_sec', 'end_sec'}
    if not required_cols.issubset(df_align.columns):
        print(f"Faltan columnas: {required_cols - set(df_align.columns)}")
        return
    
    df_align['start_sec'] = safe_float_series(df_align['start_sec'])
    df_align['end_sec'] = safe_float_series(df_align['end_sec'])
    
    # Procesar pacientes
    print("\n" + "="*70)
    print("PROCESANDO PACIENTES")
    print("="*70)
    
    patients = sorted(set(cha_data_dict.keys()) & set(df_align['patient_id'].unique()))
    print(f"Pacientes con .CHA + alignments: {len(patients)}")
    
    if len(patients) == 0:
        print("No hay pacientes con ambos datos")
        return
    
    results = []
    
    for i, pid in enumerate(patients, 1):
        if i % 50 == 0 or i == len(patients):
            print(f"  [{i:4d}/{len(patients)}] {pid}")
        
        cha_data = cha_data_dict[pid]
        df_words = df_align[df_align['patient_id'] == pid].copy()
        
        if not df_words.empty:
            speech_sec = float((df_words['end_sec'] - df_words['start_sec']).sum())
        else:
            speech_sec = 60.0
        
        den_feat = build_DEN_from_cha(cha_data, speech_sec)
        dys_feat = build_DYS_from_cha_and_align(cha_data, df_words, speech_sec)
        
        row = {'patient_id': pid, 'language': cha_data['language']}
        row.update(den_feat)
        row.update(dys_feat)
        
        row.pop('_total_words', None)
        row.pop('_total_phones', None)
        
        results.append(row)
    
    # Guardar
    print("\n" + "="*70)
    print("GUARDANDO RESULTADOS")
    print("="*70)
    
    df_out = pd.DataFrame(results).sort_values('patient_id')
    df_out.to_csv(args.output, index=False)
    
    den_cols = [c for c in df_out.columns if c.startswith('den_')]
    dys_cols = [c for c in df_out.columns if c.startswith('dys_')]
    
    print(f"Pacientes procesados: {len(df_out)}")
    print(f"\nFeatures extraídas:")
    print(f"  DEN: {len(den_cols)}")
    print(f"  DYS: {len(dys_cols)}")
    print(f"  TOTAL: {len(den_cols) + len(dys_cols)}")
    
    print(f"\nDistribución por idioma:")
    print(df_out['language'].value_counts())
    
    print(f"\nGuardado en: {args.output}")
    
    # Verificar que son exactamente 28
    if len(den_cols) == 18 and len(dys_cols) == 10:
        print("\n✓ CORRECTO: 18 DEN + 10 DYS = 28 features")
    else:
        print(f"\nADVERTENCIA: Se esperaban 18 DEN + 10 DYS")
    
    print("="*70)


if __name__ == "__main__":
    main()
