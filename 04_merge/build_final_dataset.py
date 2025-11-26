#!/usr/bin/env python3
# 04_merge/build_final_dataset.py
"""
Construye dataset final desde dataset_base.csv
VERSIÓN CORREGIDA: Solo mantiene pacientes con TODAS las features
"""

import os
import sys
import pandas as pd
from pathlib import Path

def main():
    print("="*70)
    print("CONSTRUCCIÓN DATASET FINAL (INNER JOINS)")
    print("="*70)
    
    # 1. Cargar base
    base_path = Path("data/dataset_base.csv")
    if not base_path.exists():
        print(f"\nERROR: No existe {base_path}")
        print("Ejecuta primero: python3 03_features/build_dataset_base.py")
        sys.exit(1)
    
    print(f"\nCargando base: {base_path}")
    df = pd.read_csv(base_path)
    print(f"  Base: {len(df)} pacientes")
    
    # 2. Cargar DEN+DYS
    features_path = Path("data/features_den_dys_COMPLETO.csv")
    if not features_path.exists():
        print(f"\nERROR: No existe {features_path}")
        sys.exit(1)
    
    print(f"\nCargando: {features_path}")
    df_features = pd.read_csv(features_path)
    
    # INNER JOIN - Solo pacientes con features DEN+DYS
    df = df.merge(df_features, on='patient_id', how='inner')
    print(f"  Después de merge DEN+DYS: {len(df)} pacientes")
    
    # Limpiar columnas duplicadas
    if 'group_x' in df.columns and 'group_y' in df.columns:
        # Mantener group del dataset_base (más confiable)
        df['group'] = df['group_x']
        df = df.drop(columns=['group_x', 'group_y'])
        print("  - Resuelto conflicto en columna 'group'")
    
    if 'language_x' in df.columns and 'language_y' in df.columns:
        df['language'] = df['language_x']
        df = df.drop(columns=['language_x', 'language_y'])
        print("  - Resuelto conflicto en columna 'language'")
    
    # 3. Agregar LEX (INNER JOIN)
    lex_path = Path("data/lex_features_ALL.csv")
    if lex_path.exists():
        print(f"\nCargando: {lex_path}")
        df_lex = pd.read_csv(lex_path)
        
        lex_cols = ['patient_id'] + [c for c in df_lex.columns if c.startswith('lex_')]
        df_lex = df_lex[lex_cols]
        
        df = df.merge(df_lex, on='patient_id', how='inner')
        print(f"  Después de merge LEX: {len(df)} pacientes")
    else:
        print(f"\n No existe: {lex_path}")
    
    # 4. Agregar POS-LM (LEFT JOIN - opcional)
    poslm_path = Path("data/poslm_features_ALL.csv")
    df_with_poslm = df.copy()
    
    if poslm_path.exists():
        print(f"\nCargando: {poslm_path}")
        df_poslm = pd.read_csv(poslm_path)
        
        poslm_cols = ['patient_id'] + [c for c in df_poslm.columns if c.startswith('poslm_')]
        df_poslm = df_poslm[poslm_cols]
        
        # POS-LM es opcional, así que LEFT JOIN
        df_with_poslm = df.merge(df_poslm, on='patient_id', how='left')
        print(f"  POS-LM agregado: {len(df_with_poslm)} pacientes")
    else:
        print(f"\n No existe: {poslm_path}")
    
    # 5. Filtrar solo pacientes con QA válido
    print("\n" + "="*70)
    print("FILTRADO FINAL")
    print("="*70)
    
    n_before = len(df)
    df = df[df['QA'].notna()].copy()
    df_with_poslm = df_with_poslm[df_with_poslm['QA'].notna()].copy()
    print(f"\nPacientes con QA válido: {len(df)}/{n_before}")
    
    # Verificar NaN en features
    den_cols = [c for c in df.columns if c.startswith('den_')]
    dys_cols = [c for c in df.columns if c.startswith('dys_')]
    lex_cols = [c for c in df.columns if c.startswith('lex_')]
    
    nan_counts = df[den_cols + dys_cols + lex_cols].isna().sum().sum()
    if nan_counts > 0:
        print(f"\n⚠️  ADVERTENCIA: Hay {nan_counts} valores NaN en features")
        print("  Esto degradará el rendimiento del modelo")
    else:
        print("\n✓ No hay valores NaN en features DEN+DYS+LEX")
    
    # 6. Estadísticas finales
    print("\n" + "="*70)
    print("ESTADÍSTICAS FINALES")
    print("="*70)
    
    poslm_cols = [c for c in df_with_poslm.columns if c.startswith('poslm_')]
    
    print(f"\nFeatures totales:")
    print(f"  DEN:   {len(den_cols)}")
    print(f"  DYS:   {len(dys_cols)}")
    print(f"  LEX:   {len(lex_cols)}")
    print(f"  POSLM: {len(poslm_cols)}")
    print(f"  ────────────")
    print(f"  TOTAL: {len(den_cols) + len(dys_cols) + len(lex_cols) + len(poslm_cols)}")
    
    # Distribución por grupo
    if 'group' in df.columns:
        print("\nDistribución por grupo:")
        for grp, count in df['group'].value_counts(dropna=False).items():
            print(f"  {grp}: {count}")
    
    # 7. Guardar
    output_path = Path("data/dataset_FINAL_COMPLETO.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Guardado (sin POS-LM): {output_path}")
    print(f"  {len(df)} pacientes")
    print(f"  {len(df.columns)} columnas")
    
    if len(poslm_cols) > 0:
        output_poslm = Path("data/dataset_FINAL_CON_POSLM.csv")
        df_with_poslm.to_csv(output_poslm, index=False)
        print(f"\n✓ Guardado (con POS-LM): {output_poslm}")
        print(f"  {len(df_with_poslm)} pacientes")
        print(f"  {len(df_with_poslm.columns)} columnas")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()