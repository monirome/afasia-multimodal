#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEX features SIMPLIFICADO - solo TTR
Funciona para EN/ES/CA sin bases de datos externas
"""

import os
import re
import glob
import sys
import argparse
import pandas as pd
import nltk
from collections import Counter

# Descargar recursos NLTK
for pkg in ['punkt', 'averaged_perceptron_tagger', 'universal_tagset', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        try:
            nltk.data.find(f'taggers/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

try:
    from nltk.stem import WordNetLemmatizer
    LEMM_EN = WordNetLemmatizer()
    HAS_LEMM_EN = True
except:
    HAS_LEMM_EN = False

try:
    from nltk.stem.snowball import SnowballStemmer
    STEM_ES = SnowballStemmer('spanish')
    STEM_CA = SnowballStemmer('catalan')
    HAS_STEM = True
except:
    HAS_STEM = False

OPEN_TAGS = {'NOUN', 'VERB', 'ADJ', 'ADV'}

def detect_language_from_path(path):
    """Detecta idioma por ruta o contenido"""
    path_lower = path.lower()
    if 'english' in path_lower or '/en/' in path_lower:
        return 'en'
    if 'spanish' in path_lower or '/es/' in path_lower:
        return 'es'
    if 'catalan' in path_lower or '/ca/' in path_lower:
        return 'ca'
    
    # Detectar por contenido
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(3000)
        if re.search(r'@Languages:.*(spa|spanish)', content, re.I):
            return 'es'
        if re.search(r'@Languages:.*(cat|catalan)', content, re.I):
            return 'ca'
        if re.search(r'@Languages:.*(eng|english)', content, re.I):
            return 'en'
    except:
        pass
    
    return 'en'  # default

def lemmatize_word(word, lang):
    """Lematiza palabra según idioma"""
    word = word.lower()
    
    if lang == 'en' and HAS_LEMM_EN:
        return LEMM_EN.lemmatize(word)
    elif lang == 'es' and HAS_STEM:
        return STEM_ES.stem(word)
    elif lang == 'ca' and HAS_STEM:
        return STEM_CA.stem(word)
    else:
        return word  # fallback: palabra original

def calculate_ttr(text, lang):
    """Calcula Type-Token Ratio con lematización"""
    tokens = re.findall(r"\b[\w'-]+\b", text)
    if not tokens:
        return float('nan')
    
    # POS tagging (solo para inglés funciona bien con NLTK universal)
    if lang == 'en':
        try:
            pos = nltk.pos_tag(tokens, tagset='universal')
            open_words = [w for w, t in pos if t in OPEN_TAGS]
        except:
            open_words = tokens
    else:
        # Para ES/CA: usar todas las palabras (sin POS filtering)
        open_words = tokens
    
    if not open_words:
        return float('nan')
    
    # Lematizar
    lemmas = [lemmatize_word(w, lang) for w in open_words]
    
    # TTR = tipos únicos / total tokens
    ttr = float(len(set(lemmas)) / max(len(lemmas), 1))
    
    return ttr

def process_cha_file(cha_path):
    """Extrae utterances del paciente y calcula LEX"""
    patient_id = os.path.splitext(os.path.basename(cha_path))[0]
    lang = detect_language_from_path(cha_path)
    
    try:
        with open(cha_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except:
        return None
    
    # Extraer *PAR:
    utterances = []
    for line in lines:
        if line.startswith('*PAR:'):
            utt = line[5:].strip()
            utt = re.sub(r'[.!?]+$', '', utt)  # quitar puntuación final
            if utt:
                utterances.append(utt)
    
    if not utterances:
        return None
    
    text = " ".join(utterances)
    ttr = calculate_ttr(text, lang)
    
    if pd.isna(ttr):
        return None
    
    return {
        'patient_id': patient_id,
        'language': lang,
        'lex_ttr': ttr
    }

def main():
    parser = argparse.ArgumentParser(
        description="LEX features simplificado (solo TTR) - EN/ES/CA"
    )
    parser.add_argument("--cha_dir", required=True, help="Directorio con .CHA")
    parser.add_argument("--output", default="data/lex_features_ALL.csv", help="CSV de salida")
    parser.add_argument("--language", choices=['en', 'es', 'ca', 'all'], default='all',
                       help="Filtrar por idioma (default: all)")
    args = parser.parse_args()
    
    print("="*70)
    print("EXTRACCION LEX FEATURES - SIMPLIFICADO (solo TTR)")
    print("="*70)
    
    # Buscar .CHA
    cha_files = glob.glob(os.path.join(args.cha_dir, "**/*.cha"), recursive=True)
    print(f"\nArchivos .CHA encontrados: {len(cha_files)}")
    
    if len(cha_files) == 0:
        print("ERROR: No .CHA")
        sys.exit(1)
    
    # Procesar
    print("\nProcesando...")
    rows = []
    lang_counts = Counter()
    
    for i, path in enumerate(cha_files, 1):
        result = process_cha_file(path)
        if result:
            # Filtrar por idioma si se especificó
            if args.language != 'all' and result['language'] != args.language:
                continue
            
            rows.append(result)
            lang_counts[result['language']] += 1
        
        if i % 100 == 0:
            print(f"  {i}/{len(cha_files)}")
    
    if not rows:
        print("\nERROR: No se generaron features LEX")
        sys.exit(1)
    
    # Crear DataFrame
    df = pd.DataFrame(rows).sort_values('patient_id')
    
    # Guardar
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    df.to_csv(args.output, index=False)
    
    # Estadísticas
    print("\n" + "="*70)
    print("RESULTADOS")
    print("="*70)
    print(f"Total pacientes: {len(df)}")
    print("\nPor idioma:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang.upper()}: {count}")
    
    print(f"\nEstadísticas TTR:")
    print(f"  Media: {df['lex_ttr'].mean():.3f}")
    print(f"  Std:   {df['lex_ttr'].std():.3f}")
    print(f"  Range: [{df['lex_ttr'].min():.3f}, {df['lex_ttr'].max():.3f}]")
    
    print(f"\nGuardado: {args.output}")
    print("="*70)

if __name__ == "__main__":
    main()