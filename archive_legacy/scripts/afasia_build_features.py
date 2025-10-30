#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cálculo de pausas DYS según Le et al. (2018) - Aphasia
Método: Pausas = silencios entre palabras (gaps) ≥ 150ms
Requiere: word-level timestamps (alineación forzada)
"""
import os
import warnings
warnings.filterwarnings("ignore")

import json
import glob
import numpy as np
import pandas as pd

# ==================== CONFIGURACIÓN ====================
DATA_BASE = "/lustre/ific.uv.es/ml/upc150/upc1503/data"

# Input: CSV con word alignments (patient_id, word, start_sec, end_sec)
CSV_WORD_ALIGNMENTS = os.path.join(DATA_BASE, "word_alignments.csv")

# Alternativamente: carpeta con CSVs por paciente
ALIGNMENTS_DIR = os.path.join(DATA_BASE, "alignments_by_patient")

# Output
OUT_CSV = os.path.join(DATA_BASE, "dys_pauses_le2018_aphasia.csv")

# ==================== PARÁMETROS (Le et al. 2018) ====================
PAUSE_MIN = 0.150    # 150 ms - Umbral mínimo (Le et al. 2018)
PAUSE_LONG = 0.400   # 400 ms - Umbral corta/larga (Le et al. 2018)

# Percentiles a reportar
PERCENTILES = [1, 25, 50, 75, 99]

# ==================== FUNCIONES ====================

def detect_time_unit(df, col='end_sec'):
    """Detecta si timestamps están en ms y convierte a segundos"""
    df_copy = df.copy()
    if df_copy[col].max() > 1000:  # Están en milisegundos
        df_copy['start_sec'] = df_copy['start_sec'] / 1000.0
        df_copy['end_sec'] = df_copy['end_sec'] / 1000.0
    return df_copy


def clean_word_alignments(df):
    """Limpia y valida word alignments"""
    # Convierte a numérico
    df['start_sec'] = pd.to_numeric(df['start_sec'], errors='coerce')
    df['end_sec'] = pd.to_numeric(df['end_sec'], errors='coerce')
    
    # Elimina NaN
    df = df.dropna(subset=['start_sec', 'end_sec'])
    
    # Detecta y corrige unidades
    df = detect_time_unit(df)
    
    # Filtra duraciones inválidas
    df = df[df['end_sec'] > df['start_sec']]
    
    # Ordena por tiempo
    df = df.sort_values(['start_sec', 'end_sec']).reset_index(drop=True)
    
    return df


def compute_pauses_between_words(df_words):
    """
    Calcula pausas como gaps entre palabras consecutivas.
    Le et al. (2018): pausa = silencio entre word_i.end y word_{i+1}.start
    
    Returns:
        list: pausas en segundos (solo las ≥ PAUSE_MIN)
    """
    if len(df_words) < 2:
        return []
    
    pauses = []
    
    for i in range(len(df_words) - 1):
        end_current = df_words.iloc[i]['end_sec']
        start_next = df_words.iloc[i + 1]['start_sec']
        
        gap = start_next - end_current
        
        # Solo gaps positivos (sin solapamiento)
        if gap >= 0:
            pauses.append(gap)
    
    # Filtrar por umbral mínimo
    pauses = [p for p in pauses if p >= PAUSE_MIN]
    
    return pauses


def compute_dys_metrics(pauses, total_words, total_speech_sec, total_window_sec):
    """
    Calcula métricas DYS según Le et al. (2018) - Tabla 4
    
    Args:
        pauses: lista de duraciones de pausas (segundos)
        total_words: número total de palabras
        total_speech_sec: suma de duraciones de palabras (habla efectiva)
        total_window_sec: desde primera palabra hasta última (incluye pausas)
    """
    pauses = np.array(pauses)
    
    num_pauses = len(pauses)
    num_long = int((pauses > PAUSE_LONG).sum())
    num_short = num_pauses - num_long
    
    total_pause_sec = float(pauses.sum()) if num_pauses > 0 else 0.0
    
    # Por minuto
    minutes_speech = max(total_speech_sec / 60.0, 1e-6)
    minutes_window = max(total_window_sec / 60.0, 1e-6)
    
    metrics = {
        # Conteos
        'num_pauses': num_pauses,
        'num_long_pauses': num_long,
        'num_short_pauses': num_short,
        
        # Por minuto (speech = solo habla, window = habla + pausas)
        'pauses_per_min_speech': num_pauses / minutes_speech,
        'long_pauses_per_min_speech': num_long / minutes_speech,
        'short_pauses_per_min_speech': num_short / minutes_speech,
        
        'pauses_per_min_window': num_pauses / minutes_window,
        'long_pauses_per_min_window': num_long / minutes_window,
        'short_pauses_per_min_window': num_short / minutes_window,
        
        # Por palabra
        'pauses_per_word': num_pauses / max(total_words, 1),
        'long_pauses_per_word': num_long / max(total_words, 1),
        'short_pauses_per_word': num_short / max(total_words, 1),
        
        # Estadísticas de duración
        'pause_total_sec': total_pause_sec,
    }
    
    # Percentiles y estadísticas
    if num_pauses > 0:
        q = np.percentile(pauses, PERCENTILES)
        metrics.update({
            'pause_p01': float(q[0]),
            'pause_q1': float(q[1]),
            'pause_median': float(q[2]),
            'pause_q3': float(q[3]),
            'pause_p99': float(q[4]),
            'pause_range_p01_p99': float(q[4] - q[0]),
            'pause_mean': float(np.mean(pauses)),
            'pause_std': float(np.std(pauses, ddof=1)) if num_pauses > 1 else 0.0,
            'pause_iqr': float(q[3] - q[1]),
        })
    else:
        metrics.update({
            'pause_p01': 0.0, 'pause_q1': 0.0, 'pause_median': 0.0,
            'pause_q3': 0.0, 'pause_p99': 0.0, 'pause_range_p01_p99': 0.0,
            'pause_mean': 0.0, 'pause_std': 0.0, 'pause_iqr': 0.0,
        })
    
    # Duración total y palabras
    metrics['total_words'] = int(total_words)
    metrics['total_speech_sec'] = float(total_speech_sec)
    metrics['total_window_sec'] = float(total_window_sec)
    metrics['pause_durations'] = json.dumps([float(p) for p in pauses])
    
    return metrics


def load_word_alignments():
    """Carga word alignments desde CSV maestro o carpeta"""
    
    # Modo 1: CSV maestro
    if os.path.exists(CSV_WORD_ALIGNMENTS):
        print(f"\nUsando CSV maestro: {CSV_WORD_ALIGNMENTS}")
        df = pd.read_csv(CSV_WORD_ALIGNMENTS)
        
        required = ['patient_id', 'word', 'start_sec', 'end_sec']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Faltan columnas requeridas: {required}")
        
        df = clean_word_alignments(df)
        return df
    
    # Modo 2: Carpeta con CSVs por paciente
    elif os.path.exists(ALIGNMENTS_DIR):
        print(f"\nBuscando CSVs en: {ALIGNMENTS_DIR}")
        files = glob.glob(os.path.join(ALIGNMENTS_DIR, "**/*.csv"), recursive=True)
        
        all_data = []
        for filepath in files:
            try:
                df = pd.read_csv(filepath)
                if not set(['word', 'start_sec', 'end_sec']).issubset(df.columns):
                    continue
                
                patient_id = os.path.splitext(os.path.basename(filepath))[0]
                df['patient_id'] = patient_id
                df = clean_word_alignments(df)
                
                if not df.empty:
                    all_data.append(df[['patient_id', 'word', 'start_sec', 'end_sec']])
            except Exception:
                continue
        
        if not all_data:
            raise FileNotFoundError("No se encontraron word alignments válidos")
        
        return pd.concat(all_data, ignore_index=True)
    
    else:
        raise FileNotFoundError(
            f"No existe:\n"
            f"  - CSV maestro: {CSV_WORD_ALIGNMENTS}\n"
            f"  - Carpeta: {ALIGNMENTS_DIR}\n"
            f"Provee word alignments (patient_id, word, start_sec, end_sec)"
        )


def process_patient(patient_id, df_patient):
    """Procesa un paciente"""
    
    if len(df_patient) < 2:
        # Sin suficientes palabras para calcular pausas
        return {
            'patient_id': patient_id,
            'num_pauses': 0,
            'num_long_pauses': 0,
            'num_short_pauses': 0,
            'pauses_per_min_speech': 0.0,
            'pauses_per_min_window': 0.0,
            'pauses_per_word': 0.0,
            'pause_total_sec': 0.0,
            'pause_mean': 0.0,
            'pause_median': 0.0,
            'total_words': len(df_patient),
            'total_speech_sec': 0.0,
            'total_window_sec': 0.0,
            'pause_durations': json.dumps([]),
        }
    
    # Calcular tiempos
    total_words = len(df_patient)
    total_speech_sec = (df_patient['end_sec'] - df_patient['start_sec']).sum()
    total_window_sec = df_patient['end_sec'].max() - df_patient['start_sec'].min()
    
    # Calcular pausas entre palabras
    pauses = compute_pauses_between_words(df_patient)
    
    # Calcular métricas
    metrics = compute_dys_metrics(pauses, total_words, total_speech_sec, total_window_sec)
    metrics['patient_id'] = patient_id
    
    return metrics


def main():
    print("="*70)
    print("CÁLCULO DE PAUSAS DYS - Le et al. (2018) Aphasia")
    print(f"Método: Pausas = gaps entre palabras ≥ {PAUSE_MIN}s")
    print(f"Corta: ≤ {PAUSE_LONG}s | Larga: > {PAUSE_LONG}s")
    print("="*70)
    
    # Cargar word alignments
    try:
        df_all = load_word_alignments()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return
    
    n_patients = df_all['patient_id'].nunique()
    n_words = len(df_all)
    
    print(f"\nPacientes: {n_patients}")
    print(f"Palabras totales: {n_words}")
    
    # Procesar cada paciente
    results = []
    
    print(f"\n{'='*70}")
    print("Procesando pacientes...")
    print('='*70)
    
    for i, (patient_id, group) in enumerate(df_all.groupby('patient_id'), 1):
        metrics = process_patient(patient_id, group)
        results.append(metrics)
        
        if i % 50 == 0 or i == n_patients:
            print(f"  [{i:4d}/{n_patients}] {patient_id}: "
                  f"{metrics['num_pauses']} pausas, "
                  f"{metrics['pauses_per_min_window']:.1f}/min")
    
    # Guardar resultados
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values('patient_id').reset_index(drop=True)
    df_out.to_csv(OUT_CSV, index=False)
    
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    print(f"Pacientes procesados: {len(df_out)}")
    
    # Estadísticas
    cols_stats = ['num_pauses', 'pauses_per_min_window', 'pause_mean', 'pause_median']
    print(f"\nEstadísticas de pausas:")
    print(df_out[cols_stats].describe())
    
    print(f"\nTop 10 pacientes por pausas/min:")
    top10 = df_out.nlargest(10, 'pauses_per_min_window')
    print(top10[['patient_id', 'num_pauses', 'pauses_per_min_window', 
                  'total_words']].to_string(index=False))
    
    print(f"\n\nGuardado en: {OUT_CSV}")
    print("="*70)


if __name__ == "__main__":
    main()