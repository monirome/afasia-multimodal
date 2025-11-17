#!/usr/bin/env python3
# 03_features/build_den_dys.py
# -*- coding: utf-8 -*-
"""
EXTRACCION FEATURES DEN (1-18) + DYS (19-28) - PAPER VERSION
Alineado con definiciones del paper:
- DEN en formato ratio donde corresponde (W, nouns, verbs, etc.)
- {Words/utt}, {Phones/utt}, {Seconds/pause}: 13 estadísticas
- Filtro de pausas: descartar gaps que no sean silencio (evita contar turnos del entrevistador)
"""

import os
import re
import glob
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import spacy
from collections import Counter

# VAD simple por energía para verificar silencio en gaps
try:
    import librosa
except Exception:
    librosa = None

# ======================== CONFIG ========================
PAUSE_MIN = 0.150
PAUSE_LONG = 0.400

SILENCE_DB = -35.0    # umbral dBFS para considerar silencio
SILENCE_FRAC = 0.8    # fracción mínima de frames en silencio

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

# Light verbs tal como en el paper (EN)
LIGHT_VERBS = {
    "en": {"be", "have", "come", "go", "give", "take", "make", "do", "get", "move", "put"},
    # Para ES/CA mantenemos listas razonables; el paper solo detalla EN
    "es": {"ser", "estar", "haber", "tener", "hacer", "ir", "dar", "poner", "venir", "llevar", "dejar"},
    "ca": {"ser", "estar", "haver", "tenir", "fer", "anar", "donar", "posar", "venir", "dur", "deixar"},
}

DEMONSTRATIVES = {
    "en": {"this", "that", "these", "those"},
    "es": {"este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
           "aquel", "aquella", "aquellos", "aquellas"},
    "ca": {"aquest", "aquesta", "aquests", "aquestes",
           "aquell", "aquella", "aquells", "aquelles"},
}

# ======================== UTILS ========================
def stats_13(x):
    """Devuelve 13 estadísticas sobre una lista/array x."""
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

def safe_float_series(s):
    return pd.to_numeric(s, errors="coerce")

def safe_div(num, den):
    if den is None or not np.isfinite(den) or den <= 0:
        return np.nan
    return float(num / den)

def minutes(seconds):
    return max(float(seconds) / 60.0, 1e-6)

def is_silent_segment(audio_path, start_sec, end_sec, sr=16000,
                      silence_db=SILENCE_DB, silence_frac=SILENCE_FRAC):
    """Devuelve True si el segmento [start_sec, end_sec] es silencioso."""
    if librosa is None or audio_path is None or not os.path.exists(audio_path):
        return False
    dur = max(0.0, float(end_sec) - float(start_sec))
    if dur < PAUSE_MIN:
        return False
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True, offset=float(start_sec), duration=dur)
        if y.size == 0:
            return True
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).flatten()
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
        frac_silent = float(np.mean(rms_db < silence_db))
        return frac_silent >= silence_frac
    except Exception:
        return False

# ======================== PARSER .CHA ========================
def parse_cha_file(filepath):
    patient_id = os.path.splitext(os.path.basename(filepath))[0]
    utterances = []
    metadata = {'language': 'en', 'group': None}

    GROUP_SYNONYMS = {
        'control': 'control', 'typical': 'control', 'nonaphasic': 'control', 'healthy': 'control',
        'pwa': 'pwa', 'aphasia': 'pwa', 'aphasic': 'pwa'
    }

    def norm_group(text):
        if not text:
            return None
        t = str(text).lower()
        for k, v in GROUP_SYNONYMS.items():
            if k in t:
                return v
        return None

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()

                if line.startswith('@ID:'):
                    parts = line.split(':', 1)[-1].strip()
                    parts = [p.strip() for p in parts.split('|')]
                    raw = '|'.join(parts).lower()
                    if 'spa' in raw or 'spanish' in raw:
                        metadata['language'] = 'es'
                    elif 'cat' in raw or 'catalan' in raw or 'ctg' in raw:
                        metadata['language'] = 'ca'
                    elif 'eng' in raw or 'english' in raw:
                        metadata['language'] = 'en'
                    for idx in [5, 6, 7, 8]:
                        if 0 <= idx < len(parts) and metadata['group'] is None:
                            g = norm_group(parts[idx])
                            if g:
                                metadata['group'] = g
                    if metadata['group'] is None:
                        for cand in parts:
                            g = norm_group(cand)
                            if g:
                                metadata['group'] = g
                                break

                if line.startswith('@Participants') and metadata['group'] is None:
                    g = norm_group(line)
                    if g:
                        metadata['group'] = g

                if line.startswith('*PAR:'):
                    utt = line[5:].strip()
                    utt = re.sub(r'[.!?]+$', '', utt)
                    if utt:
                        utterances.append(utt)

        if metadata['group'] is None:
            lower = filepath.lower()
            if re.search(r'(^|/|\\b)pwa($|/|\\b)', lower):
                metadata['group'] = 'pwa'
            elif re.search(r'(^|/|\\b)control($|/|\\b)', lower):
                metadata['group'] = 'control'

        return {
            'patient_id': patient_id,
            'utterances': utterances,
            'language': metadata['language'],
            'group': metadata['group']
        }

    except Exception as e:
        print("  Error leyendo {}: {}".format(filepath, e))
        return None

def count_fillers_cha(utterance):
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
    model = SPACY_MODELS.get(lang, SPACY_MODELS["en"])
    try:
        return spacy.load(model, disable=["ner", "lemmatizer"])
    except Exception:
        raise RuntimeError(
            "Error cargando {}. Instalalo con:\n"
            "  python -m spacy download {}".format(model, model)
        )

def estimate_phones(words):
    return int(len(words) * 3.5)

def analyze_utterance_spacy(nlp, text, lang):
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

# ======================== BUILD DEN (1-18) ========================
def build_DEN_from_cha(cha_data, speech_sec):
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

    den = {
        'den_words_per_min': total_words / speech_min,
        'den_phones_per_min': total_phones / speech_min,
        'den_W': ratio(total_words, total_words + total_interj),                     # Words / (Words + Interjections)
        'den_OCW': ratio(total_open, total_open + (total_words - total_open)),      # Open / (Open + Closed)
    }

    # {Words/utt} y {Phones/utt}: 13 estadísticas
    add_stats("den_words_utt", df['utt_words'].tolist(), den)
    add_stats("den_phones_utt", df['utt_phones'].tolist(), den)

    nouns = float(df['nouns'].sum())
    verbs = float(df['verbs'].sum())

    den.update({
        'den_nouns': ratio(nouns, total_words),                                     # Nouns / Words
        'den_verbs': ratio(verbs, total_words),                                     # Verbs / Words
        'den_nouns_per_verb': ratio(nouns, verbs),
        'den_noun_ratio': ratio(nouns, nouns + verbs),
        'den_light_verbs': ratio(df['light_verbs'].sum(), verbs) if verbs > 0 else np.nan,  # Light / Verbs
        'den_determiners': ratio(df['determiners'].sum(), total_words),             # DET / Words
        'den_demonstratives': ratio(df['demonstratives'].sum(), total_words),       # DEM / Words
        'den_prepositions': ratio(df['prepositions'].sum(), total_words),           # ADP / Words
        'den_adjectives': ratio(df['adjectives'].sum(), total_words),               # ADJ / Words
        'den_adverbs': ratio(df['adverbs'].sum(), total_words),                     # ADV / Words
        'den_pronoun_ratio': ratio(df['pronouns'].sum(), df['pronouns'].sum() + nouns),
        'den_function_words': ratio(df['function_words'].sum(), total_words),       # Closed / Words
    })

    return den

# ======================== BUILD DYS (19-28) ========================
def build_DYS_from_cha_and_align(cha_data, df_words, speech_sec, audio_path=None):
    total_fillers = 0
    for utt in cha_data['utterances']:
        total_fillers += count_fillers_cha(utt)

    pauses = []
    if not df_words.empty and len(df_words) >= 2:
        df_words = df_words.sort_values(['start_sec', 'end_sec']).reset_index(drop=True)
        starts = df_words['start_sec'].values
        ends = df_words['end_sec'].values

        for i in range(len(df_words) - 1):
            gap = float(starts[i+1] - ends[i])
            if gap >= PAUSE_MIN:
                if gap >= 3.0:
                    continue
                # aceptar gap solo si hay silencio acústico
                if is_silent_segment(audio_path, ends[i], starts[i+1]):
                    pauses.append(gap)

    n_pauses = len(pauses)
    n_long = sum(1 for p in pauses if p > PAUSE_LONG)
    n_short = n_pauses - n_long

    total_words = len(df_words) if not df_words.empty else 1
    total_phones = int(total_words * 3.5)
    speech_min = minutes(speech_sec)

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

    # {Seconds/pause}: 13 estadísticas
    add_stats("dys_pause_sec", pauses, dys)

    return dys

# ======================== AUDIO LOOKUP ========================
AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".flac"}

def find_audio_files(audio_base):
    audio_files = {}
    if not audio_base or not os.path.exists(audio_base):
        return audio_files
    for ext in AUDIO_EXTENSIONS:
        pattern = os.path.join(audio_base, f"**/*{ext}")
        for filepath in glob.glob(pattern, recursive=True):
            pid = os.path.splitext(os.path.basename(filepath))[0]
            if pid in audio_files:
                if os.path.getsize(filepath) > os.path.getsize(audio_files[pid]):
                    audio_files[pid] = filepath
            else:
                audio_files[pid] = filepath
    return audio_files

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(
        description="Extrae 28 features DEN+DYS (Paper Version) con ratios correctos y 13 estadísticas"
    )
    parser.add_argument("--cha_dir", required=True, help="Directorio con archivos .CHA")
    parser.add_argument("--word_align_csv", required=True, help="CSV con word alignments (patient_id, word, start_sec, end_sec)")
    parser.add_argument("--output", default="../data/features_den_dys_COMPLETO.csv", help="Archivo CSV de salida")
    parser.add_argument("--audio_base", default=None, help="Carpeta raíz con audios (para validar silencios en gaps)")
    args = parser.parse_args()

    print("="*70)
    print("EXTRACCION FEATURES DEN+DYS - PAPER VERSION (28 features)")
    print("="*70)

    print("\nVerificando modelos spaCy...")
    required_models = ["en_core_web_sm", "es_core_news_sm"]
    missing_models = []
    for model in required_models:
        try:
            spacy.load(model, disable=["ner", "lemmatizer"])
        except Exception:
            missing_models.append(model)
    if missing_models:
        print("\nERROR: Faltan modelos spaCy:")
        for model in missing_models:
            print("  python -m spacy download {}".format(model))
        sys.exit(1)
    print("Modelos spaCy disponibles")

    if not os.path.exists(args.cha_dir):
        print("\nERROR: No existe directorio: {}".format(args.cha_dir))
        sys.exit(1)

    if not os.path.exists(args.word_align_csv):
        print("\nERROR: No existe archivo: {}".format(args.word_align_csv))
        print("Ejecuta primero: 02_alignments/generate_whisperx_alignments.py")
        sys.exit(1)

    print("\nBuscando archivos .CHA...")
    cha_files = glob.glob(os.path.join(args.cha_dir, "**/*.cha"), recursive=True)
    print("Archivos .CHA encontrados: {}".format(len(cha_files)))
    if len(cha_files) == 0:
        print("No se encontraron archivos .CHA")
        sys.exit(1)

    cha_data_dict = {}
    for cha_path in cha_files:
        parsed = parse_cha_file(cha_path)
        if parsed and len(parsed['utterances']) > 0:
            cha_data_dict[parsed['patient_id']] = parsed
    print("Pacientes con .CHA validos: {}".format(len(cha_data_dict)))

    print("\nCargando word alignments...")
    df_align = pd.read_csv(args.word_align_csv)
    print("Word alignments cargados: {}".format(len(df_align)))

    required_cols = {'patient_id', 'word', 'start_sec', 'end_sec'}
    if not required_cols.issubset(df_align.columns):
        print("Faltan columnas: {}".format(required_cols - set(df_align.columns)))
        sys.exit(1)

    df_align['start_sec'] = safe_float_series(df_align['start_sec'])
    df_align['end_sec'] = safe_float_series(df_align['end_sec'])

    audio_map = find_audio_files(args.audio_base) if args.audio_base else {}

    print("\n" + "="*70)
    print("PROCESANDO PACIENTES")
    print("="*70)

    patients = sorted(set(cha_data_dict.keys()) & set(df_align['patient_id'].unique()))
    print("Pacientes con .CHA + alignments: {}".format(len(patients)))
    if len(patients) == 0:
        print("No hay pacientes con ambos datos")
        sys.exit(1)

    results = []
    for i, pid in enumerate(patients, 1):
        if i % 50 == 0 or i == len(patients):
            print("  [{:4d}/{}] {}".format(i, len(patients), pid))

        cha_data = cha_data_dict[pid]
        df_words = df_align[df_align['patient_id'] == pid].copy()

        if not df_words.empty:
            n_words_align = len(df_words)
            n_words_cha_approx = sum(len(utt.split()) for utt in cha_data['utterances'])
            coverage = n_words_align / max(n_words_cha_approx, 1)
            if coverage < 0.3:
                speech_sec = (n_words_cha_approx / 120.0) * 60.0
            else:
                speech_sec = float((df_words['end_sec'] - df_words['start_sec']).sum())
        else:
            speech_sec = 60.0

        den_feat = build_DEN_from_cha(cha_data, speech_sec)
        audio_path = audio_map.get(pid, None)
        dys_feat = build_DYS_from_cha_and_align(cha_data, df_words, speech_sec, audio_path=audio_path)

        row = {'patient_id': pid, 'language': cha_data['language']}
        if 'group' in cha_data and cha_data['group'] is not None:
            row['group'] = cha_data['group']
        row.update(den_feat)
        row.update(dys_feat)
        results.append(row)

    print("\n" + "="*70)
    print("GUARDANDO RESULTADOS")
    print("="*70)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print("Creado directorio: {}".format(output_dir))

    df_out = pd.DataFrame(results).sort_values('patient_id')
    df_out.to_csv(args.output, index=False)

    den_cols = [c for c in df_out.columns if c.startswith('den_')]
    dys_cols = [c for c in df_out.columns if c.startswith('dys_')]

    print("Pacientes procesados: {}".format(len(df_out)))
    print("\nFeatures extraidas:")
    print("  DEN: {}".format(len(den_cols)))
    print("  DYS: {}".format(len(dys_cols)))
    print("  TOTAL: {}".format(len(den_cols) + len(dys_cols)))

    print("\nDistribucion por idioma:")
    print(df_out['language'].value_counts())

    if 'group' in df_out.columns:
        print("\nDistribucion por grupo (si detectado):")
        print(df_out['group'].value_counts(dropna=False))

    print("\nGuardado en: {}".format(args.output))
    print("="*70)

if __name__ == "__main__":
    main()
