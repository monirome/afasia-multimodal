#!/usr/bin/env python3
# merge_features_metadata.py
"""Merge DEN+DYS features con metadata WAB"""

import os
import sys
import pandas as pd

# Paths
FEATURES_CSV = "data/features_den_dys_COMPLETO.csv"
METADATA_CSV = "data/patient_metadata_WAB.csv"
OUTPUT_CSV = "data/dataset_FINAL.csv"

def main():
    print("="*70)
    print("MERGE: Features DEN+DYS + Metadata WAB")
    print("="*70)
    
    # Verificar archivos
    if not os.path.exists(FEATURES_CSV):
        print(f"\nERROR: No existe {FEATURES_CSV}")
        print("Ejecuta primero: python3 03_features/build_den_dys.py")
        sys.exit(1)
    
    if not os.path.exists(METADATA_CSV):
        print(f"\nERROR: No existe {METADATA_CSV}")
        print("Ejecuta primero: python3 01_metadata/extract_wab_metadata_FINAL.py")
        sys.exit(1)
    
    # Cargar
    print(f"\nCargando features: {FEATURES_CSV}")
    df_feat = pd.read_csv(FEATURES_CSV)
    print(f"  Features: {len(df_feat)} pacientes")
    
    print(f"\nCargando metadata: {METADATA_CSV}")
    df_meta = pd.read_csv(METADATA_CSV)
    print(f"  Metadata: {len(df_meta)} pacientes")
    
    # Limpiar conflictos (metadata tiene precedencia)
    if 'group' in df_feat.columns:
        df_feat = df_feat.drop(columns=['group'])
        print("  - Eliminada columna 'group' de features (se usa de metadata)")
    
    if 'language' in df_feat.columns:
        df_feat = df_feat.drop(columns=['language'])
        print("  - Eliminada columna 'language' de features (se usa de metadata)")
    
    # Merge
    meta_cols = ['patient_id', 'group', 'QA', 'language', 'sex', 'age', 'aphasia_type']
    meta_cols = [c for c in meta_cols if c in df_meta.columns]
    
    print(f"\nColumnas de metadata a usar: {len(meta_cols)}")
    for col in meta_cols:
        print(f"  - {col}")
    
    df = pd.merge(df_feat, df_meta[meta_cols], on='patient_id', how='inner')
    print(f"\nMerge: {len(df)} pacientes (inner join)")
    
    # Filtrar pacientes sin QA
    n_before = len(df)
    df = df.dropna(subset=['QA'])
    n_after = len(df)
    print(f"Con QA válido: {n_after} ({n_before - n_after} eliminados)")
    
    # Estadísticas
    print("\n" + "="*70)
    print("ESTADÍSTICAS")
    print("="*70)
    
    if 'group' in df.columns:
        print("\nDistribución por grupo:")
        print(df['group'].value_counts(dropna=False))
        
        print("\nEstadísticas por grupo:")
        for grp in ['control', 'pwa']:
            dg = df[(df['group'] == grp) & (df['QA'].notna())]
            if len(dg) > 0:
                print(f"\n  {grp.upper()}:")
                print(f"    N:     {len(dg)}")
                print(f"    QA:    {dg['QA'].mean():.1f} ± {dg['QA'].std():.1f}")
                print(f"    Range: [{dg['QA'].min():.1f}, {dg['QA'].max():.1f}]")
    
    if 'language' in df.columns:
        print("\nDistribución por idioma:")
        print(df['language'].value_counts(dropna=False))
    
    # Contar features
    den_cols = [c for c in df.columns if c.startswith('den_')]
    dys_cols = [c for c in df.columns if c.startswith('dys_')]
    
    print(f"\nFeatures incluidas:")
    print(f"  DEN: {len(den_cols)}")
    print(f"  DYS: {len(dys_cols)}")
    print(f"  Total: {len(den_cols) + len(dys_cols)}")
    
    # Guardar
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Guardado: {OUTPUT_CSV}")
    print(f"  {len(df)} pacientes")
    print(f"  {len(df.columns)} columnas totales")
    print("="*70)

if __name__ == "__main__":
    main()