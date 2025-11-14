#!/usr/bin/env python3
# aphasia_paper_replication_FINAL.py
# -*- coding: utf-8 -*-
"""
REPLICACIÓN DEL PIPELINE DE ALINEADO Y FEATURES (Le et al., 2018) CON WHISPERX

- Explora múltiples carpetas de audio y .CHA (por idioma) y empareja por patient_id (stem).
- Concatena "chunks" por paciente opcionalmente (p. ej., catalán troceado).
- Limpia y fusiona *PAR: de archivos .CHA; detecta idioma (en/es/ca).
- Controla palabras/segundo (WPS) y recorta texto para evitar desajustes graves.
- Forced alignment con WhisperX:
    * Fallback a CPU si CUDA falla u ocupa.
    * Fallback de align model 'ca' -> 'es' si no hay modelo 'ca'.
    * Segmentación por "chunk_words" para estabilizar el alineado.
- Features acústicas:
    * MFCC 12 + energy -> 13 base; + Δ + ΔΔ => 39 (paper-like).
    * MFB 40 log-mel (paper-like).
    * Deltas seguros para audios cortos (sin errores de 'width').
    * Estandarización (StandardScaler) y guardado opcional por paciente.
- Salidas:
    * ./aphasia_output/word_alignments_ALL.csv
    * ./aphasia_output/processing_metadata.csv
    * (opcional) ./aphasia_output/features/<patient_id>/{mfcc.npy,mfb.npy,scalers.pkl}
"""

import os
import re
import sys
import glob
import math
import pickle
import argparse
import warnings
import tempfile
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Audio/ML
try:
    import soundfile as sf
except Exception:
    sf = None
try:
    import librosa
except Exception:
    librosa = None
from sklearn.preprocessing import StandardScaler

# Torch / WhisperX
try:
    import torch
except Exception:
    torch = None

try:
    import whisperx
    HAS_WHISPERX = True
except Exception:
    HAS_WHISPERX = False


# ==================== PARÁMETROS "PAPER-LIKE" ====================
SAMPLE_RATE = 16000
WINDOW_MS = 25
FRAME_SHIFT_MS = 10
N_MFCC = 12          # 12 + energy -> 13 base
N_MFBS = 40
N_FFT = 512
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_MS / 1000)
HOP_LENGTH = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000)

# Umbrales y controles
DEFAULT_MIN_AUDIO_SEC = 1.5
DEFAULT_MAX_WPS = 3.5
DEFAULT_TARGET_WPS = 2.6
LOW_SCORE_DROP = 0.00  # si quieres filtrar palabras por score, sube este valor
DEFAULT_CHUNK_WORDS = 40
PAD_BETWEEN_CHUNKS_SEC = 0.0

AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg"}

# Defaults ejemplo (ajusta a tus rutas reales)
DEFAULT_AUDIO_DIRS = [
    "/lustre/ific.uv.es/ml/upc150/upc1503/data/audios_completos/English",
    "/lustre/ific.uv.es/ml/upc150/upc1503/data/audios_completos/Spanish",
    "/lustre/ific.uv.es/ml/upc150/upc1503/data/audios_chunks",  # catalán en trozos
]
DEFAULT_CHA_DIRS = [
    "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones/English",
    "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones/Spanish",
]


# ==================== UTILIDADES TEXTO ====================
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ\u00f1\u00d1']+", re.UNICODE)

def tokenize_words(text: str):
    if not text:
        return []
    text = text.replace("_", " ")
    return _WORD_RE.findall(text)

def clean_chat_text(text):
    t = text
    t = re.sub(r'\b\d+_\d+\b', '', t)
    t = re.sub(r'\[[\^:]\]', '', t)
    t = re.sub(r'<[^>]*>', '', t)
    t = re.sub(r'\[[^\]]*\]', '', t)
    t = re.sub(r'xxx|www', '', t)
    t = re.sub(r'&=\w+', '', t)
    t = re.sub(r'&\w+', '', t)  # &uh, &mm
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'\(\.\)', ' ', t)
    t = re.sub(r'\([\d.]+\)', ' ', t)
    t = re.sub(r'\([^)]*\)', ' ', t)
    t = re.sub(r'[+/]', ' ', t)
    t = re.sub(r'[:;]', ' ', t)
    t = re.sub(r"[^\w\s']", ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def extract_par_utterances_from_cha(cha_path):
    if not cha_path or not os.path.exists(cha_path):
        return None, []
    utterances = []
    full_transcript = []
    try:
        with open(cha_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        current = ""
        is_par = False
        for line in lines:
            line = line.rstrip('\n')
            if line.startswith('*PAR:'):
                if current and is_par:
                    clean = clean_chat_text(current)
                    if clean:
                        utterances.append(clean)
                        full_transcript.extend(tokenize_words(clean))
                current = line[5:]
                is_par = True
            elif line.startswith('*'):
                if current and is_par:
                    clean = clean_chat_text(current)
                    if clean:
                        utterances.append(clean)
                        full_transcript.extend(tokenize_words(clean))
                current = ""
                is_par = False
            elif line.startswith('\t') and is_par:
                current += " " + line[1:]
        if current and is_par:
            clean = clean_chat_text(current)
            if clean:
                utterances.append(clean)
                full_transcript.extend(tokenize_words(clean))
    except Exception:
        return None, []
    if not full_transcript:
        return None, []
    return ' '.join(full_transcript).lower(), utterances

def detect_language_from_cha(cha_path):
    try:
        with open(cha_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(5000)
        if '@Languages:' in content:
            m = re.search(r'@Languages:\s*([^\n]+)', content, flags=re.IGNORECASE)
            if m:
                lang = m.group(1).lower()
                if 'spa' in lang or 'es' in lang:
                    return 'es'
                if 'cat' in lang or 'ca' in lang:
                    return 'ca'
                if 'eng' in lang or 'en' in lang:
                    return 'en'
        low = content.lower()
        es = sum(w in low for w in ['hola', 'porque', 'entonces', 'qué'])
        ca = sum(w in low for w in ['com', 'què', 'això', 'molt'])
        if es > 2: return 'es'
        if ca > 2: return 'ca'
    except Exception:
        pass
    return 'en'

def determine_group(patient_id, cha_path):
    group = 'PWA'
    plow = (cha_path or "").lower()
    if 'control' in plow or 'typical' in plow or 'healthy' in plow:
        group = 'Control'
    elif 'aphasia' in plow or 'pwa' in plow:
        group = 'PWA'
    return group


# ==================== INDEXACIÓN / CARGA AUDIO ====================
def index_files(roots, exts):
    idx = {}
    for root in roots:
        if not root or not os.path.exists(root):
            continue
        pattern = os.path.join(root, "**", "*")
        for p in glob.glob(pattern, recursive=True):
            if os.path.isfile(p):
                ext = Path(p).suffix.lower()
                if exts and ext not in exts:
                    continue
                stem = Path(p).stem
                idx.setdefault(stem, []).append(p)
    return idx

def best_audio_choice(paths):
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    return max(paths, key=lambda p: os.path.getsize(p))

def guess_chunk_family(stem, candidates):
    fam = []
    for p in candidates:
        s = Path(p).stem
        if s.startswith(stem) or stem.startswith(s):
            fam.append(p)
    return sorted(fam)

def concat_audios_to_temp(file_list, sr=SAMPLE_RATE, pad_sec=PAD_BETWEEN_CHUNKS_SEC):
    if not file_list:
        return None
    waves = []
    for fp in sorted(file_list):
        try:
            if librosa is None:
                return None
            y, _ = librosa.load(fp, sr=sr, mono=True)
            if y.size == 0:
                continue
            waves.append(y)
            if pad_sec > 0:
                waves.append(np.zeros(int(sr * pad_sec), dtype=np.float32))
        except Exception:
            continue
    if not waves:
        return None
    ycat = np.concatenate(waves).astype(np.float32)
    if len(ycat) < int(sr * DEFAULT_MIN_AUDIO_SEC):
        return None
    fd, tmppath = tempfile.mkstemp(suffix=".wav", prefix="concat_")
    os.close(fd)
    if sf is None:
        return None
    sf.write(tmppath, ycat, sr)
    return tmppath

def find_audio_for_patient(pid, audio_index, merge_chunks=True):
    candidates = audio_index.get(pid, [])
    if candidates:
        if merge_chunks and len(candidates) > 1:
            fam = guess_chunk_family(pid, candidates)
            if len(fam) > 1:
                cat = concat_audios_to_temp(fam)
                if cat:
                    return cat, True
        return best_audio_choice(candidates), False

    rel = []
    for stem, paths in audio_index.items():
        if stem.startswith(pid) or pid.startswith(stem):
            rel.extend(paths)
    if rel:
        if merge_chunks and len(rel) > 1:
            cat = concat_audios_to_temp(rel)
            if cat:
                return cat, True
        return best_audio_choice(rel), False

    return None, False


# ==================== FEATURES ACÚSTICAS ====================
def _safe_deltas(feat):
    T = feat.shape[1]
    if T < 3:
        return np.zeros_like(feat), np.zeros_like(feat)
    width = min(9, T if (T % 2 == 1) else T - 1)
    try:
        d1 = librosa.feature.delta(feat, width=width, order=1)
        d2 = librosa.feature.delta(feat, width=width, order=2)
    except Exception:
        d1 = np.zeros_like(feat)
        d2 = np.zeros_like(feat)
    return d1, d2

def extract_mfcc_features(audio, sr=SAMPLE_RATE):
    if librosa is None:
        return None
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    if len(audio) < max(WINDOW_SIZE + HOP_LENGTH, 1024):
        return None
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WINDOW_SIZE,
        window='hamming', center=True
    )
    energy = librosa.feature.rms(
        y=audio, frame_length=WINDOW_SIZE, hop_length=HOP_LENGTH, center=True
    )
    base = np.vstack([mfcc, energy])
    d1, d2 = _safe_deltas(base)
    full = np.vstack([base, d1, d2])  # 13 * 3 = 39
    return full.T  # (frames, 39)

def extract_mfb_features(audio, sr=SAMPLE_RATE):
    if librosa is None:
        return None
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    if len(audio) < max(WINDOW_SIZE + HOP_LENGTH, 1024):
        return None
    spec = np.abs(librosa.stft(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WINDOW_SIZE, window='hamming', center=True
    ))**2
    mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MFBS, fmin=0, fmax=sr/2)
    mel_spec = np.dot(mel_basis, spec)
    mel_spec = np.log(mel_spec + 1e-10)
    return mel_spec.T  # (frames, 40)


# ==================== CONTROL WPS ====================
def trim_text_by_wps(text: str, duration_sec: float, max_wps=DEFAULT_MAX_WPS, target_wps=DEFAULT_TARGET_WPS):
    words = tokenize_words(text)
    n = len(words)
    if duration_sec <= 0 or n == 0:
        return "", 0, 0.0
    wps = n / max(duration_sec, 1e-6)
    if wps <= max_wps:
        return " ".join(words), n, wps
    n_target = int(math.floor(duration_sec * target_wps))
    n_target = max(1, min(n, n_target))
    trimmed = " ".join(words[:n_target])
    return trimmed, n_target, n_target / duration_sec


# ==================== ALINEADO WHISPERX ====================
def _load_align_model_with_fallback(language_code: str, device: str):
    try:
        return whisperx.load_align_model(language_code=language_code, device=device)
    except Exception:
        if language_code == "ca":
            return whisperx.load_align_model(language_code="es", device=device)
        raise

def perform_whisperx_forced_alignment(audio_path, transcript, language='en',
                                      device='cuda', chunk_words=DEFAULT_CHUNK_WORDS):
    try:
        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000.0
        if duration < DEFAULT_MIN_AUDIO_SEC:
            return []

        def _try_align(dev):
            align_model, metadata = _load_align_model_with_fallback(language, dev)
            words = transcript.split()
            if not words:
                return []
            segments = []
            W = len(words)
            for i in range(0, W, chunk_words):
                cw = words[i:i + chunk_words]
                start_ratio = i / max(W, 1)
                end_ratio = min((i + chunk_words) / max(W, 1), 1.0)
                segments.append({
                    'text': ' '.join(cw),
                    'start': start_ratio * duration,
                    'end': max(end_ratio * duration - 1e-3, 0.0)
                })
            result = {"segments": segments}
            aligned = whisperx.align(
                result["segments"], align_model, metadata, audio, dev,
                return_char_alignments=False, print_progress=False
            )
            return aligned

        # Primer intento
        try:
            dev = device if (device == "cpu" or (torch and torch.cuda.is_available())) else "cpu"
            aligned = _try_align(dev)
        except Exception:
            aligned = _try_align("cpu")

        word_alignments = []
        for seg in aligned.get("segments", []):
            for wi in seg.get("words", []):
                if not wi or not wi.get("word"):
                    continue
                start, end = wi.get('start'), wi.get('end')
                sc = wi.get('score', None)
                if start is None or end is None:
                    continue
                if end <= start:
                    end = start + 0.01
                if sc is not None and sc < LOW_SCORE_DROP:
                    continue
                word_alignments.append({
                    'word': wi['word'].strip().lower(),
                    'start_sec': float(start),
                    'end_sec': float(end),
                    'score': float(sc) if sc is not None else np.nan
                })
        return word_alignments

    except Exception as e:
        print(f"    Error crítico en alignment (tras fallback): {e}")
        return []


# ==================== PROCESADO POR PACIENTE ====================
def process_patient(patient_id, audio_path, cha_path,
                    device='cuda', force_language=None,
                    max_wps=DEFAULT_MAX_WPS, target_wps=DEFAULT_TARGET_WPS,
                    min_audio_sec=DEFAULT_MIN_AUDIO_SEC, chunk_words=DEFAULT_CHUNK_WORDS):
    print(f"\n[{patient_id}] Procesando...")

    transcript, utterances = extract_par_utterances_from_cha(cha_path)
    if not transcript:
        print(" No se encontraron utterances PAR")
        return None

    language = force_language if force_language else detect_language_from_cha(cha_path)
    group = determine_group(patient_id, cha_path)

    # Carga audio para features
    try:
        if librosa is None:
            raise RuntimeError("librosa no disponible para cargar audio.")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f" Error cargando audio: {e}")
        return None

    duration = len(y) / SAMPLE_RATE
    if duration < min_audio_sec:
        print(" Audio demasiado corto; salto este paciente")
        return None

    # Control de WPS
    words_all = tokenize_words(transcript)
    n_words_all = len(words_all)
    wps_all = n_words_all / max(duration, 1e-6)
    text_for_align, n_words_used, wps_used = trim_text_by_wps(
        transcript, duration, max_wps=max_wps, target_wps=target_wps
    )
    print(f" Transcripción: {n_words_all} palabras | Idioma: {language} | Audio: {duration:.2f}s | WPS bruto: {wps_all:.2f} | WPS usado: {wps_used:.2f} ({n_words_used} palabras)")

    # Features (si fallan, seguimos con alineado igualmente)
    mfcc = extract_mfcc_features(y, sr=SAMPLE_RATE)
    mfb  = extract_mfb_features(y,  sr=SAMPLE_RATE)
    n_frames = mfcc.shape[0] if mfcc is not None else 0

    if mfcc is None or mfb is None:
        print(" Advertencia: audio demasiado corto para features; continúo con alineado sin features.")
        mfcc_norm = None
        mfb_norm = None
        mfcc_scaler = None
        mfb_scaler = None
    else:
        mfcc_scaler = StandardScaler().fit(mfcc)
        mfb_scaler  = StandardScaler().fit(mfb)
        mfcc_norm   = mfcc_scaler.transform(mfcc)
        mfb_norm    = mfb_scaler.transform(mfb)

    print(" Realizando forced alignment...")
    word_alignments = perform_whisperx_forced_alignment(
        audio_path, text_for_align, language=language, device=device, chunk_words=chunk_words
    )
    if word_alignments:
        print(f"  Alineadas {len(word_alignments)} palabras")
    else:
        print("  No se pudieron alinear palabras")

    for a in word_alignments:
        a['patient_id'] = patient_id
        a['group'] = group
        a['language'] = language

    return {
        'patient_id': patient_id,
        'group': group,
        'language': language,
        'transcript': transcript,
        'n_words_transcript': n_words_all,
        'n_words_aligned': len(word_alignments),
        'n_utterances': len(utterances),
        'audio_duration': duration,
        'mfcc_features': mfcc_norm,
        'mfb_features': mfb_norm,
        'mfcc_scaler': mfcc_scaler,
        'mfb_scaler': mfb_scaler,
        'word_alignments': word_alignments,
        'n_frames': n_frames
    }


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(
        description="Replicación de alineado y features estilo Le et al. (2018) con WhisperX"
    )
    parser.add_argument("--audio_roots", nargs="+", default=DEFAULT_AUDIO_DIRS,
                        help="Carpetas raíz con audios (recursivo).")
    parser.add_argument("--cha_roots", nargs="+", default=DEFAULT_CHA_DIRS,
                        help="Carpetas raíz con .CHA (recursivo).")
    parser.add_argument("--output_dir", default="./aphasia_output", help="Carpeta de salida")
    parser.add_argument("--output_csv", default="word_alignments_ALL.csv", help="CSV principal de alineamientos")
    parser.add_argument("--device", default=("cuda" if (torch and torch.cuda.is_available()) else "cpu"),
                        choices=["cuda", "cpu"], help="Device preferido para align")
    parser.add_argument("--language", choices=["en", "es", "ca", "auto"],
                        default="auto", help="Idioma (auto=detectar del .CHA)")
    parser.add_argument("--max_patients", type=int, help="Limitar número de pacientes (debug)")
    parser.add_argument("--merge_chunks", action="store_true", help="Concatenar automáticamente múltiples audios por paciente")
    parser.add_argument("--save_features", action="store_true", help="Guardar MFCC/MFB y scalers por paciente")
    parser.add_argument("--min_audio_sec", type=float, default=DEFAULT_MIN_AUDIO_SEC, help="Umbral mínimo de segundos")
    parser.add_argument("--max_wps", type=float, default=DEFAULT_MAX_WPS, help="Corte de WPS para recortar texto")
    parser.add_argument("--target_wps", type=float, default=DEFAULT_TARGET_WPS, help="WPS objetivo tras recorte")
    parser.add_argument("--chunk_words", type=int, default=DEFAULT_CHUNK_WORDS, help="Palabras por segmento de alineado")
    args = parser.parse_args()

    if not HAS_WHISPERX:
        print("ERROR: WhisperX no está instalado.")
        print("Instala: pip install git+https://github.com/m-bain/whisperx.git")
        sys.exit(1)

    if args.device == "cuda" and (not torch or not torch.cuda.is_available()):
        print("CUDA no disponible, usando CPU")
        args.device = "cpu"
    elif args.device == "cuda":
        try:
            print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_concat_dir = os.path.join(args.output_dir, "_tmp_concat")
    os.makedirs(tmp_concat_dir, exist_ok=True)
    tempfile.tempdir = tmp_concat_dir

    print("\nINDEXANDO .CHA...")
    cha_index = index_files(args.cha_roots, exts={".cha"})
    print(f" .CHA únicos: {len(cha_index)}")

    print("INDEXANDO AUDIOS...")
    audio_index = index_files(args.audio_roots, exts=AUDIO_EXTENSIONS)
    print(f" Stems de audio únicos: {len(audio_index)}")

    if not cha_index:
        print("No se han encontrado archivos .CHA en las carpetas indicadas.")
        sys.exit(1)
    if not audio_index:
        print("No se han encontrado audios en las carpetas indicadas.")
        sys.exit(1)

    patients = sorted(list(cha_index.keys()))
    if args.max_patients:
        patients = patients[:args.max_patients]
    print(f"\nTOTAL PACIENTES A PROCESAR: {len(patients)}")

    all_alignments = []
    all_metadata = []
    successful = 0
    failed = 0
    no_audio = 0

    force_lang = (None if args.language == "auto" else args.language)

    global PAD_BETWEEN_CHUNKS_SEC
    PAD_BETWEEN_CHUNKS_SEC = 0.0  # si quieres, ajusta relleno entre chunks concatenados

    for i, pid in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] {pid}")

        cha_paths = cha_index.get(pid, [])
        if not cha_paths:
            print(" No se encontró .CHA")
            failed += 1
            continue
        cha_path = cha_paths[0]

        audio_path, was_concat = find_audio_for_patient(pid, audio_index, merge_chunks=args.merge_chunks)
        if not audio_path or not os.path.exists(audio_path):
            print(" No se encontró audio; salto")
            no_audio += 1
            continue

        try:
            result = process_patient(
                pid, audio_path, cha_path,
                device=args.device, force_language=force_lang,
                max_wps=args.max_wps, target_wps=args.target_wps,
                min_audio_sec=args.min_audio_sec, chunk_words=args.chunk_words
            )
            if result:
                if result['word_alignments']:
                    all_alignments.extend(result['word_alignments'])
                all_metadata.append({
                    'patient_id': pid,
                    'group': result['group'],
                    'language': result['language'],
                    'n_words_transcript': result['n_words_transcript'],
                    'n_words_aligned': result['n_words_aligned'],
                    'n_utterances': result['n_utterances'],
                    'audio_duration': result['audio_duration'],
                    'n_frames': result['n_frames'],
                    'concatenated_chunks': was_concat
                })
                if args.save_features and (result['mfcc_features'] is not None) and (result['mfb_features'] is not None):
                    outp = os.path.join(args.output_dir, 'features', pid)
                    os.makedirs(outp, exist_ok=True)
                    np.save(os.path.join(outp, 'mfcc.npy'), result['mfcc_features'])
                    np.save(os.path.join(outp, 'mfb.npy'), result['mfb_features'])
                    with open(os.path.join(outp, 'scalers.pkl'), 'wb') as f:
                        pickle.dump({'mfcc': result['mfcc_scaler'], 'mfb': result['mfb_scaler']}, f)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f" Error en paciente {pid}: {e}")
            failed += 1

    print("\nGUARDANDO RESULTADOS...")

    output_csv_path = os.path.join(args.output_dir, args.output_csv)
    if all_alignments:
        df_align = pd.DataFrame(all_alignments)
        order = ['patient_id', 'group', 'language', 'word', 'start_sec', 'end_sec', 'score']
        cols = [c for c in order if c in df_align.columns] + [c for c in df_align.columns if c not in order]
        df_align = df_align[cols].sort_values(['patient_id', 'start_sec'])
        df_align.to_csv(output_csv_path, index=False)
        print(f" Alineamientos: {len(df_align)} filas -> {output_csv_path}")
    else:
        print(" No hay alineamientos para guardar.")

    if all_metadata:
        df_meta = pd.DataFrame(all_metadata).sort_values('patient_id')
        meta_path = os.path.join(args.output_dir, 'processing_metadata.csv')
        df_meta.to_csv(meta_path, index=False)
        print(f" Metadata -> {meta_path}")
        try:
            by_lang = df_meta['language'].value_counts().to_dict()
            print(f" Distribución por idioma: {by_lang}")
        except Exception:
            pass

    print("\nRESUMEN")
    print(f" Exitosos: {successful}/{len(patients)}")
    print(f" Fallidos: {failed}/{len(patients)}")
    print(f" Sin audio: {no_audio}/{len(patients)}")
    print(f" Carpeta de salida: {args.output_dir}")
    if all_alignments:
        print(f" CSV principal: {output_csv_path}")


if __name__ == "__main__":
    main()
