#!/usr/bin/env python3
# add_lex_features.py
"""Agrega features LEX al dataset final"""

import os
import pandas as pd

def main():
    print("="*70)
    print("AGREGANDO FEATURES LEX")
    print("="*70)
    
    # Cargar dataset base
    base_path = 'data/dataset_FINAL.csv'
    if not os.path.exists(base_path):
        print(f"\nERROR: No existe {base_path}")
        print("Ejecuta primero: python3 merge_features_metadata.py")
        return
    
    df_main = pd.read_csv(base_path)
    print(f"\nDataset base: {len(df_main)} pacientes")
    
    # Buscar archivos LEX
    lex_path = 'data/lex_features_ALL.csv'
    
    if not os.path.exists(lex_path):
        print(f"\n No existe {lex_path}")
        print("  Copiando dataset sin LEX...")
        df_main.to_csv('data/dataset_FINAL_COMPLETO.csv', index=False)
        print(" Guardado sin features LEX")
        return
    
    # Cargar LEX
    df_lex = pd.read_csv(lex_path)
    print(f"LEX features: {len(df_lex)} pacientes")
    
    # Preparar columnas LEX
    lex_cols = ['patient_id'] + [c for c in df_lex.columns if c.startswith('lex_')]
    df_lex = df_lex[lex_cols]
    
    print(f"  Columnas LEX: {len(lex_cols) - 1}")
    for col in lex_cols[1:]:
        print(f"    - {col}")
    
    # Merge (left join para mantener todos los pacientes)
    df_final = df_main.merge(df_lex, on='patient_id', how='left')
    
    # Estadísticas
    n_with_lex = df_final['lex_ttr'].notna().sum()
    print(f"\nPacientes con LEX: {n_with_lex}/{len(df_final)}")
    
    # Contar features totales
    den = len([c for c in df_final.columns if c.startswith('den_')])
    dys = len([c for c in df_final.columns if c.startswith('dys_')])
    lex = len([c for c in df_final.columns if c.startswith('lex_')])
    
    print(f"\nFeatures totales:")
    print(f"  DEN: {den}")
    print(f"  DYS: {dys}")
    print(f"  LEX: {lex}")
    print(f"  Total: {den + dys + lex}")
    
    # Guardar
    df_final.to_csv('data/dataset_FINAL_COMPLETO.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f" Guardado: data/dataset_FINAL_COMPLETO.csv")
    print(f"  {len(df_final)} pacientes")
    print(f"  {len(df_final.columns)} columnas totales")
    print("="*70)

if __name__ == "__main__":
    main()