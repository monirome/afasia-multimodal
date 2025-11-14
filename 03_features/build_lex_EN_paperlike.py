#!/usr/bin/env python3
# build_lex_EN_paperlike.py
# -*- coding: utf-8 -*-
"""
LEX features (EN) en estilo del paper:
- lex_ttr (open-class lemmas)
- {Freq/word, Img/word, AoA/word, Fam/word, Phones/word}: 13 estadísticas cada uno
Requiere normas reales: SUBTLEX-US y Brysbaert (AoA, Imageability, Familiarity).
"""

import os
import re
import glob
import sys
import argparse
import numpy as np
import pandas as pd
import nltk
from collections import defaultdict

from nltk.stem import WordNetLemmatizer

# Taggers y lexicones NLTK
for pkg in ['punkt', 'averaged_perceptron_tagger', 'universal_tagset', 'wordnet', 'omw-1.4', 'cmudict']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        try:
            nltk.data.find(f'taggers/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

from nltk.corpus import cmudict
CMU = cmudict.dict()
LEMM = WordNetLemmatizer()

OPEN_TAGS = {'NOUN', 'VERB', 'ADJ', 'ADV'}

def stats_13(x):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "min": np.nan, "p10": np.nan, "q1": np.nan, "median": np.nan, "q3": np.nan,
            "p90": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan,
            "skew": np.nan, "kurt": np.nan, "iqr": np.nan, "mad": np.nan
        }
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    from scipy.stats import skew, kurtosis
    skewv = float(skew(arr, bias=False)) if arr.size > 2 else np.nan
    kurtv = float(kurtosis(arr, bias=False)) if arr.size > 3 else np.nan
    iqr = float(q[3] - q[1])
    mad = float(np.median(np.abs(arr - np.median(arr))))
    return {
        "min": float(np.min(arr)),
        "p10": float(q[0]),
        "q1": float(q[1]),
        "median": float(q[2]),
        "q3": float(q[3]),
        "p90": float(q[4]),
        "max": float(np.max(arr)),
        "mean": mean,
        "std": std,
        "skew": skewv,
        "kurt": kurtv,
        "iqr": iqr,
        "mad": mad
    }

def add_stats(prefix, arr, out_dict):
    s = stats_13(arr)
    for k, v in s.items():
        out_dict[f"{prefix}_{k}"] = v

def load_norms(base_dir):
    base_dir = os.path.abspath(base_dir)
    paths = {
        "subtlex": os.path.join(base_dir, "SUBTLEX-US.csv"),
        "aoa": os.path.join(base_dir, "Brysbaert_AoA.csv"),
        "img": os.path.join(base_dir, "Brysbaert_Imageability.csv"),
        "fam": os.path.join(base_dir, "Brysbaert_Familiarity.csv"),
    }
    norms = {}

    if os.path.exists(paths["subtlex"]):
        df = pd.read_csv(paths["subtlex"])
        cols = {c.lower(): c for c in df.columns}
        word_col = cols.get('word', None) or cols.get('words', None)
        lg_col = cols.get('lg10wf', None) or cols.get('zipfus', None)
        if word_col and lg_col:
            norms['freq'] = dict(zip(df[word_col].astype(str).str.lower(), df[lg_col].astype(float)))
        else:
            norms['freq'] = {}
    else:
        norms['freq'] = {}

    for key, colname in [('aoa','aoa'), ('img','img'), ('fam','fam')]:
        path = paths[key]
        if os.path.exists(path):
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            word_col = cols.get('word', None) or cols.get('words', None)
            val_col = cols.get(colname, None)
            if word_col and val_col:
                norms[key] = dict(zip(df[word_col].astype(str).str.lower(), df[val_col].astype(float)))
            else:
                norms[key] = {}
        else:
            norms[key] = {}

    return norms, paths

def phones_per_word(w):
    w = w.lower()
    if w in CMU:
        return float(len(CMU[w][0]))
    # prueba con lema
    lemma = LEMM.lemmatize(w)
    if lemma in CMU:
        return float(len(CMU[lemma][0]))
    return np.nan

def lex_for_patient(text):
    tokens = re.findall(r"\b[\w'-]+\b", text)
    if not tokens:
        return None

    pos = nltk.pos_tag(tokens, tagset='universal')
    open_words = [w for w, t in pos if t in OPEN_TAGS]
    if not open_words:
        ttr = np.nan
    else:
        lemmas = [LEMM.lemmatize(w.lower()) for w in open_words]
        ttr = float(len(set(lemmas)) / max(len(lemmas), 1))

    return {
        "tokens": tokens,
        "ttr": ttr
    }

def main():
    parser = argparse.ArgumentParser(description="LEX EN paper-like (13 estadísticas)")
    parser.add_argument("--cha_dir", required=True, help="Directorio con .CHA")
    parser.add_argument("--output", default="../data/lex_features_en_paperlike.csv", help="CSV de salida")
    parser.add_argument("--lex_resources", default="../data/lex_resources", help="Carpeta con SUBTLEX y Brysbaert CSVs")
    args = parser.parse_args()

    norms, paths = load_norms(args.lex_resources)
    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"Aviso: no se encontró {name}: {p}")

    cha_files = glob.glob(os.path.join(args.cha_dir, "**/*.cha"), recursive=True)
    cha_files_en = [p for p in cha_files if re.search(r'@ID:.*(eng|english)|@Languages:.*(eng|english)', open(p, 'r', errors='ignore').read(), flags=re.IGNORECASE) or p.lower().endswith("_en.cha")]
    print(f"Archivos EN: {len(cha_files_en)}")

    rows = []
    for i, cha_path in enumerate(cha_files_en, 1):
        patient_id = os.path.splitext(os.path.basename(cha_path))[0]
        with open(cha_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        utts = [re.sub(r'[.!?]+$', '', ln[5:].strip())
                for ln in lines if ln.startswith('*PAR:')]

        if not utts:
            continue

        text = " ".join(utts)
        base = lex_for_patient(text)
        if base is None:
            continue

        tokens = base["tokens"]
        ttr = base["ttr"]

        freq_vals, img_vals, aoa_vals, fam_vals, phn_vals = [], [], [], [], []

        for w in tokens:
            wl = w.lower()
            if norms['freq']:
                v = norms['freq'].get(wl)
                if pd.notna(v):
                    freq_vals.append(float(v))
            if norms['img']:
                v = norms['img'].get(wl)
                if pd.notna(v):
                    img_vals.append(float(v))
            if norms['aoa']:
                v = norms['aoa'].get(wl)
                if pd.notna(v):
                    aoa_vals.append(float(v))
            if norms['fam']:
                v = norms['fam'].get(wl)
                if pd.notna(v):
                    fam_vals.append(float(v))
            ph = phones_per_word(wl)
            if pd.notna(ph):
                phn_vals.append(float(ph))

        row = {"patient_id": patient_id, "language": "en", "lex_ttr": ttr}

        if len(freq_vals) > 0:
            add_stats("lex_freq", freq_vals, row)
        if len(img_vals) > 0:
            add_stats("lex_img", img_vals, row)
        if len(aoa_vals) > 0:
            add_stats("lex_aoa", aoa_vals, row)
        if len(fam_vals) > 0:
            add_stats("lex_fam", fam_vals, row)
        if len(phn_vals) > 0:
            add_stats("lex_phones", phn_vals, row)

        rows.append(row)

        if i % 100 == 0:
            print(f"Procesados {i}/{len(cha_files_en)}")

    if not rows:
        print("No se generaron LEX.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("patient_id")
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Guardado: {args.output}")
    print(f"Pacientes EN: {len(df)}")

if __name__ == "__main__":
    main()
