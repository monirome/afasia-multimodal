#!/usr/bin/env python3
# merge_features_with_metadata.py
# -*- coding: utf-8 -*-

"""
Combina features DEN+DYS con metadata EN+ES
"""
import pandas as pd

print("="*70)
print("MERGE: Features DEN+DYS + Metadata EN+ES")
print("="*70)

# Cargar datos
df_features = pd.read_csv('features_den_dys_COMPLETO.csv')
df_metadata = pd.read_csv('metadata_EN_ES_from_CSV.csv')

print(f"\nFeatures: {len(df_features)} registros")
print(f"Metadata: {len(df_metadata)} pacientes")

# Merge
df_final = pd.merge(
    df_features,
    df_metadata[['patient_id', 'QA', 'sex', 'age', 'aphasia_type', 'language']],
    on='patient_id',
    how='inner'
)

print(f"\nDespués del merge: {len(df_final)} pacientes")

# Limpiar NaN en features críticas
initial_len = len(df_final)
df_final = df_final.dropna(subset=['den_light_verbs', 'dys_pause_sec_mean'])
print(f"Después de limpiar NaN: {len(df_final)} pacientes (eliminados: {initial_len - len(df_final)})")

# Reportar distribución
print(f"\n{'='*70}")
print("DISTRIBUCIÓN FINAL POR IDIOMA")
print("="*70)
print(df_final['language'].value_counts())

print(f"\n{'='*70}")
print("ESTADÍSTICAS POR IDIOMA")
print("="*70)
for lang in ['en', 'es']:
    df_lang = df_final[df_final['language'] == lang]
    if len(df_lang) > 0:
        print(f"\n{lang.upper()}:")
        print(f"  N: {len(df_lang)}")
        print(f"  QA: {df_lang['QA'].mean():.1f} ± {df_lang['QA'].std():.1f}")
        print(f"  Range: [{df_lang['QA'].min():.1f}, {df_lang['QA'].max():.1f}]")
        
        # Distribución de severidad
        df_lang['severity'] = pd.cut(
            df_lang['QA'],
            bins=[0, 25, 50, 75, 100],
            labels=['Very Severe', 'Severe', 'Moderate', 'Mild']
        )
        print(f"  Severidad:")
        for sev, count in df_lang['severity'].value_counts().sort_index().items():
            print(f"    - {sev}: {count}")

# Guardar
df_final.to_csv('dataset_FINAL_EN_ES.csv', index=False)
print(f"\n{'='*70}")
print(f"✓ Dataset final guardado: dataset_FINAL_EN_ES.csv")
print(f"  Total: {len(df_final)} pacientes")
print(f"  EN: {len(df_final[df_final['language']=='en'])} pacientes")
print(f"  ES: {len(df_final[df_final['language']=='es'])} pacientes")
print("="*70)
