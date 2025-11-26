#!/usr/bin/env python3
# 04_merge/add_poslm_features.py
"""
Añade features POS-LM al dataset
"""

import os
import sys
import pandas as pd

def main():
    print("="*70)
    print("AGREGANDO FEATURES POS-LM")
    print("="*70)
    
    # Cargar dataset base (DEN+DYS+LEX)
    base_path = 'data/dataset_FINAL_COMPLETO.csv'
    
    if not os.path.exists(base_path):
        print(f"\n No existe: {base_path}")
        print("   Ejecuta primero:")
        print("   1. python 04_merge/merge_features_metadata.py")
        print("   2. python 04_merge/add_lex_features.py")
        sys.exit(1)
    
    df_main = pd.read_csv(base_path)
    print(f"\nDataset base: {len(df_main)} pacientes")
    
    # Buscar POS-LM features
    poslm_path = 'data/poslm_features_ALL.csv'
    
    if not os.path.exists(poslm_path):
        print(f"\n  No existe: {poslm_path}")
        print("   Ejecuta: python 03_features/extract_poslm_features.py")
        print("   Continuando sin POS-LM...")
        return
    
    # Cargar POS-LM
    df_poslm = pd.read_csv(poslm_path)
    print(f"POS-LM features: {len(df_poslm)} pacientes")
    
    # Preparar columnas POS-LM
    poslm_cols = ['patient_id'] + [c for c in df_poslm.columns if c.startswith('poslm_')]
    df_poslm = df_poslm[poslm_cols]
    
    # Contar por método
    kn_cols = [c for c in poslm_cols if 'poslm_kn_' in c]
    bo_cols = [c for c in poslm_cols if 'poslm_bo_' in c]
    lstm_cols = [c for c in poslm_cols if 'poslm_lstm_' in c]
    
    print(f"\nColumnas POS-LM:")
    if kn_cols:
        print(f"  Kneser-Ney: {len(kn_cols)}")
    if bo_cols:
        print(f"  Backoff: {len(bo_cols)}")
    if lstm_cols:
        print(f"  LSTM: {len(lstm_cols)}")
    print(f"  Total: {len(poslm_cols) - 1}")
    
    # Merge (left join)
    df_final = df_main.merge(df_poslm, on='patient_id', how='left')
    
    # Estadísticas
    n_with_poslm = df_final[poslm_cols[1]].notna().sum() if len(poslm_cols) > 1 else 0
    print(f"\nPacientes con POS-LM: {n_with_poslm}/{len(df_final)}")
    
    # Contar features totales
    den = len([c for c in df_final.columns if c.startswith('den_')])
    dys = len([c for c in df_final.columns if c.startswith('dys_')])
    lex = len([c for c in df_final.columns if c.startswith('lex_')])
    poslm = len([c for c in df_final.columns if c.startswith('poslm_')])
    
    print(f"\nFeatures totales:")
    print(f"  DEN:   {den}")
    print(f"  DYS:   {dys}")
    print(f"  LEX:   {lex}")
    print(f"  POSLM: {poslm}")
    print(f"  ────────────")
    print(f"  TOTAL: {den + dys + lex + poslm}")
    
    # Guardar
    output_path = 'data/dataset_FINAL_CON_POSLM.csv'
    df_final.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f" Guardado: {output_path}")
    print(f"  {len(df_final)} pacientes")
    print(f"  {len(df_final.columns)} columnas totales")
    print("="*70)

if __name__ == "__main__":
    main()