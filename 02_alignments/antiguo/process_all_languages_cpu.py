#!/usr/bin/env python3
# process_all_languages_cpu.py
# -*- coding: utf-8 -*-
"""
Wrapper para procesar todos los idiomas con forced alignment - VERSION CPU
Ejecuta English y Spanish secuencialmente
"""

import subprocess
import sys
import os

print("="*80)
print("PROCESAMIENTO COMPLETO: ENGLISH + SPANISH (CPU)")
print("Forced Alignment Mode (replica P2FA del paper)")
print("="*80)

# Configuración
BASE_DIR = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025"
SCRIPT = os.path.join(BASE_DIR, "02_alignments/generate_whisperx_alignments.py")
CHA_DIR = "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones"

languages = [
    {
        'name': 'ENGLISH',
        'audio_base': '/lustre/ific.uv.es/ml/upc150/upc1503/data/audios_completos/English',
        'output': os.path.join(BASE_DIR, 'data/word_alignments_EN.csv'),
        'lang_code': 'en'
    },
    {
        'name': 'SPANISH',
        'audio_base': '/lustre/ific.uv.es/ml/upc150/upc1503/data/audios_completos/Spanish',
        'output': os.path.join(BASE_DIR, 'data/word_alignments_ES.csv'),
        'lang_code': 'es'
    }
]

results = []

for i, lang in enumerate(languages, 1):
    print("\n" + "="*80)
    print(f"[{i}/{len(languages)}] PROCESANDO {lang['name']}")
    print("="*80)
    
    cmd = [
        sys.executable,
        SCRIPT,
        '--audio_base', lang['audio_base'],
        '--cha_dir', CHA_DIR,
        '--output', lang['output'],
        '--language', lang['lang_code'],
        '--device', 'cpu'  # ← CAMBIO CRÍTICO: CPU en vez de cuda
    ]
    
    print(f"\nComando: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print(f"\n✓ {lang['name']} completado exitosamente")
            results.append({'lang': lang['name'], 'status': 'SUCCESS'})
        else:
            print(f"\n {lang['name']} falló")
            results.append({'lang': lang['name'], 'status': 'FAILED'})
    
    except Exception as e:
        print(f"\n Error procesando {lang['name']}: {e}")
        results.append({'lang': lang['name'], 'status': 'ERROR', 'error': str(e)})

# Combinar resultados
print("\n" + "="*80)
print("COMBINANDO RESULTADOS")
print("="*80)

try:
    import pandas as pd
    
    dfs = []
    for lang in languages:
        if os.path.exists(lang['output']):
            df = pd.read_csv(lang['output'])
            print(f"\n{lang['name']}:")
            print(f"  Pacientes: {df['patient_id'].nunique()}")
            print(f"  Palabras: {len(df):,}")
            dfs.append(df)
    
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.sort_values(['patient_id', 'start_sec'])
        
        output_all = os.path.join(BASE_DIR, 'data/word_alignments_ALL.csv')
        df_all.to_csv(output_all, index=False)
        
        print("\n" + "="*80)
        print("RESULTADO FINAL")
        print("="*80)
        print(f"✓ Total pacientes: {df_all['patient_id'].nunique()}")
        print(f"✓ Total palabras: {len(df_all):,}")
        print(f"✓ Palabras/paciente: {len(df_all)/df_all['patient_id'].nunique():.1f}")
        print(f"✓ Guardado: {output_all}")
    else:
        print("\n No se generaron archivos para combinar")
        sys.exit(1)

except Exception as e:
    print(f"\n Error combinando resultados: {e}")
    sys.exit(1)

# Resumen final
print("\n" + "="*80)
print("RESUMEN")
print("="*80)
for r in results:
    status_icon = "✓" if r['status'] == 'SUCCESS' else ""
    print(f"{status_icon} {r['lang']}: {r['status']}")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)

sys.exit(0)
