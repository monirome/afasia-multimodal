#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracción SIN filtros agresivos.
Acepta TODO lo que tenga timestamps válidos.
"""
import os
import sys
import re
import warnings
warnings.filterwarnings("ignore")

import pylangacq as pla
import pandas as pd

# ==================== CONFIGURACIÓN ====================
DATA_BASE = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025/data"

TRANSCRIPTIONS_DIRS = {
    "en": os.path.join(DATA_BASE, "transcripciones/English"),
    "es": os.path.join(DATA_BASE, "transcripciones/Spanish"),
}

AUDIO_DIRS = {
    "en": os.path.join(DATA_BASE, "audios_completos/English"),
    "es": os.path.join(DATA_BASE, "audios_completos/Spanish"),
}

OUTPUT_CSV = os.path.join(DATA_BASE, "df_utterances_complete.csv")

# ==================== FUNCIONES ====================

def determine_task_type(filename: str, lang: str) -> str:
    """Determina tipo de tarea"""
    fname = filename.lower()
    
    if lang == "en":
        if any(task in fname for task in [
            'cookie', 'window', 'cat', 'flood', 'cinderella', 
            'sandwich', 'important', 'stroke'
        ]):
            return "semi"
        
        basename = fname.replace('.cha', '')
        if basename and basename[-1] == 'a':
            return "free"
        elif basename and basename[-1] in 'bcdefg':
            return "semi"
    
    elif lang == "es":
        if any(task in fname for task in [
            'entrevista', 'interview', 'conversacion', 'libre'
        ]):
            return "free"
        if any(task in fname for task in [
            'descripcion', 'lamina', 'cookie', 'historia'
        ]):
            return "semi"
    
    return "unknown"


def extract_utterances_no_filter(cha_path: str, patient_id: str, lang: str) -> list:
    """
    Extracción SIN FILTROS.
    Acepta TODO lo que tenga timestamps.
    """
    try:
        ds = pla.read_chat(cha_path)
    except Exception as e:
        print(f"    [ERROR] No se pudo leer .cha: {str(e)[:50]}")
        return []
    
    # Metadata
    try:
        header = ds.headers()[0]
        participant = header.get('Participants', {}).get('PAR', {})
        
        sex = participant.get('sex', 'unknown')
        age_str = str(participant.get('age', ''))
        age = age_str.split(';')[0] if ';' in age_str else ''
        
        wab_aq = None
        custom = participant.get('custom', '')
        if custom:
            match = re.search(r'(\d+\.?\d*)', str(custom))
            if match:
                wab_aq = float(match.group(1))
        
        aphasia_type = participant.get('group', 'unknown')
    except Exception:
        sex = age = aphasia_type = 'unknown'
        wab_aq = None
    
    cha_filename = os.path.basename(cha_path)
    utt_type = determine_task_type(cha_filename, lang)
    audio_filename = f"{patient_id}.wav"
    
    # Extraer utterances
    try:
        utterances = ds.utterances(participants="PAR")
    except Exception as e:
        print(f"    [WARN] No se pudieron extraer utterances: {str(e)[:50]}")
        return []
    
    if len(utterances) == 0:
        print(f"    [INFO] .cha sin utterances del participante PAR")
        return []
    
    rows = []
    skipped_no_timestamps = 0
    skipped_negative_duration = 0
    
    for utt in utterances:
        try:
            time_marks = utt.time_marks
            
            # ÚNICO FILTRO: Debe tener timestamps válidos
            if not time_marks or len(time_marks) < 2:
                skipped_no_timestamps += 1
                continue
            
            start_ms, end_ms = time_marks[0], time_marks[1]
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            duration = end_sec - start_sec
            
            # ÚNICO FILTRO: Duración positiva
            if duration <= 0:
                skipped_negative_duration += 1
                continue
            
            # Transcripción RAW (sin limpiar)
            raw_text = utt.tiers.get('PAR', '')
            if '\x15' in raw_text:
                raw_text = raw_text.split('\x15')[0]
            
            rows.append({
                'patient_id': patient_id,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'duration': duration,
                'transcription': raw_text.strip(),
                'utt_type': utt_type,
                'sex': sex,
                'age': age,
                'WAB_AQ': wab_aq,
                'aphasia_type': aphasia_type,
                'lang': lang,
                'audio_path': audio_filename,
            })
        except Exception as e:
            continue
    
    if skipped_no_timestamps > 0 or skipped_negative_duration > 0:
        print(f"    [INFO] Descartadas: {skipped_no_timestamps} sin timestamps, "
              f"{skipped_negative_duration} duración negativa")
    
    return rows


def process_from_audios():
    """Procesa TODOS los audios disponibles"""
    all_rows = []
    
    for lang, audio_dir in AUDIO_DIRS.items():
        if not os.path.isdir(audio_dir):
            continue
        
        print(f"\n{'='*70}")
        print(f"Procesando: {lang.upper()}")
        print('='*70)
        
        wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        print(f"Audios disponibles: {len(wav_files)}")
        
        trans_dir = TRANSCRIPTIONS_DIRS.get(lang)
        if not trans_dir or not os.path.isdir(trans_dir):
            print(f"[ERROR] No existe carpeta de transcripciones")
            continue
        
        patients_with_data = 0
        patients_no_cha = 0
        patients_no_utterances = 0
        total_utterances = 0
        
        for i, wav_file in enumerate(wav_files, 1):
            patient_id = wav_file.replace('.wav', '')
            cha_path = os.path.join(trans_dir, f"{patient_id}.cha")
            
            print(f"\n  [{i:3d}/{len(wav_files)}] {patient_id}")
            
            if not os.path.exists(cha_path):
                print(f"    [SKIP] .cha no encontrado")
                patients_no_cha += 1
                continue
            
            rows = extract_utterances_no_filter(cha_path, patient_id, lang)
            
            if len(rows) == 0:
                print(f"    [SKIP] Sin utterances válidas")
                patients_no_utterances += 1
            else:
                print(f"    [OK] {len(rows)} utterances extraídas")
                all_rows.extend(rows)
                patients_with_data += 1
                total_utterances += len(rows)
        
        print(f"\n{'='*70}")
        print(f"RESUMEN {lang.upper()}:")
        print(f"  Con datos: {patients_with_data}")
        print(f"  Sin .cha: {patients_no_cha}")
        print(f"  Sin utterances: {patients_no_utterances}")
        print(f"  Total utterances: {total_utterances}")
        print('='*70)
    
    return pd.DataFrame(all_rows)


def main():
    print("="*70)
    print("EXTRACCIÓN SIN FILTROS AGRESIVOS")
    print("Acepta TODO lo que tenga timestamps válidos")
    print("="*70)
    
    df = process_from_audios()
    
    if len(df) == 0:
        print("\n[ERROR] No se extrajeron utterances")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"Total utterances: {len(df):,}")
    print(f"Pacientes: {df['patient_id'].nunique()}")
    
    print(f"\nPor idioma:")
    for lang, count in df['lang'].value_counts().items():
        n_patients = df[df['lang'] == lang]['patient_id'].nunique()
        print(f"  {lang.upper()}: {count:,} utterances, {n_patients} pacientes")
    
    print(f"\nPor tipo de tarea:")
    print(df['utt_type'].value_counts())
    
    print(f"\nEstadísticas de duración:")
    print(df['duration'].describe())
    
    print(f"\nTop 10 pacientes por número de utterances:")
    top10 = df.groupby('patient_id').size().sort_values(ascending=False).head(10)
    for pid, count in top10.items():
        lang = df[df['patient_id'] == pid]['lang'].iloc[0]
        print(f"  {pid:20s} ({lang.upper()}): {count:4d} utterances")
    
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\nGuardado en: {OUTPUT_CSV}")
    
    print("\n" + "="*70)
    print("Listo para calcular DYS")
    print("="*70)


if __name__ == "__main__":
    main()