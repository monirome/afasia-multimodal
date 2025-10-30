#!/usr/bin/env python3
# extract_metadata_from_csv.py
# -*- coding: utf-8 -*-
"""
Extrae metadata (QA + idioma) desde CSV AphasiaBank pre-procesado

NOTA: Este script requiere df_aphbank_pos_metrics.csv que debe existir previamente.
Si solo tienes archivos .CHA, usa extract_wab_metadata.py en su lugar.
"""
import os
import sys
import pandas as pd
import numpy as np

# ======================== CONFIGURACION ========================
CSV_APHBANK = "/lustre/ific.uv.es/ml/upc150/upc1503/data/df_aphbank_pos_metrics.csv"
OUTPUT_CSV = "../data/patient_metadata_WAB.csv"

def get_language(row):
    """Detecta idioma desde LLengWAB o path"""
    lleng = row.get('LLengWAB', 0)
    if lleng == 2:
        return 'es'
    elif lleng == 3:
        return 'en'
    
    # Fallback: detectar por path
    path = str(row.get('name_chunk_audio_path', ''))
    if 'aphasiabank_es' in path.lower() or 'spanish' in path.lower():
        return 'es'
    else:
        return 'en'

def main():
    print("="*70)
    print("EXTRACCION METADATA DESDE CSV APHASIABANK")
    print("="*70)
    
    # Verificar que existe el CSV
    if not os.path.exists(CSV_APHBANK):
        print("\nERROR: No existe el archivo:")
        print("   {}".format(CSV_APHBANK))
        print("\nEste CSV debe ser generado previamente.")
        print("Si solo tienes archivos .CHA, usa:")
        print("   python3 extract_wab_metadata.py")
        sys.exit(1)
    
    # Cargar CSV
    print("\nCargando CSV: {}".format(CSV_APHBANK))
    df = pd.read_csv(CSV_APHBANK)
    
    print("Total filas: {}".format(len(df)))
    print("Pacientes unicos: {}".format(df['CIP'].nunique()))
    
    # Verificar columnas requeridas
    required_cols = ['CIP', 'QA']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print("\nERROR: Faltan columnas requeridas: {}".format(missing_cols))
        sys.exit(1)
    
    # Detectar idioma
    print("\nDetectando idioma...")
    df['language'] = df.apply(get_language, axis=1)
    
    # Agrupar por paciente
    print("Agrupando por paciente (CIP)...")
    
    df_patient = df.groupby('CIP').agg({
        'QA': 'first',
        'sex': 'first',
        'Edat': 'first',
        'aphasia_type': 'first',
        'language': 'first'
    }).reset_index()
    
    # Renombrar columnas
    df_patient = df_patient.rename(columns={
        'CIP': 'patient_id',
        'Edat': 'age'
    })
    
    # Filtrar pacientes con QA valido
    df_patient = df_patient[df_patient['QA'].notna()]
    
    print("\n" + "="*70)
    print("RESULTADOS")
    print("="*70)
    print("Pacientes con QA valido: {}".format(len(df_patient)))
    
    if len(df_patient) == 0:
        print("\nERROR: No se encontraron pacientes con QA valido")
        sys.exit(1)
    
    print("\nDistribucion por idioma:")
    print(df_patient['language'].value_counts())
    
    print("\n" + "="*70)
    print("ESTADISTICAS POR IDIOMA")
    print("="*70)
    for lang in df_patient['language'].unique():
        df_lang = df_patient[df_patient['language'] == lang]
        print("\n{}:".format(lang.upper()))
        print("  N: {}".format(len(df_lang)))
        print("  QA: {:.1f} +/- {:.1f}".format(df_lang['QA'].mean(), df_lang['QA'].std()))
        print("  Range: [{:.1f}, {:.1f}]".format(df_lang['QA'].min(), df_lang['QA'].max()))
        
        # Tipos de afasia mas comunes
        if 'aphasia_type' in df_lang.columns and df_lang['aphasia_type'].notna().sum() > 0:
            print("  Tipos mas comunes:")
            top_types = df_lang['aphasia_type'].value_counts().head(3)
            for atype, count in top_types.items():
                print("    - {}: {}".format(atype, count))
    
    # Crear directorio de salida si no existe
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("\nCreado directorio: {}".format(output_dir))
    
    # Guardar
    df_patient.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*70)
    print("Guardado: {}".format(OUTPUT_CSV))
    print("="*70)
    
    # Mostrar ejemplos
    print("\nEjemplos (primeras 5):")
    cols_show = ['patient_id', 'QA', 'language', 'sex', 'age', 'aphasia_type']
    cols_available = [c for c in cols_show if c in df_patient.columns]
    print(df_patient[cols_available].head(5).to_string(index=False))
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()