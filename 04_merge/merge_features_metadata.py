#!/usr/bin/env python3
# merge_features_metadata.py
# -*- coding: utf-8 -*-
"""
Combina features DEN+DYS con metadata (QA + idioma)
"""
import os
import sys
import pandas as pd

# ======================== CONFIGURACION ========================
FEATURES_CSV = "../data/features_den_dys_COMPLETO.csv"
METADATA_CSV = "../data/patient_metadata_WAB.csv"
OUTPUT_CSV = "../data/dataset_FINAL_EN_ES.csv"

def main():
    print("="*70)
    print("MERGE: Features DEN+DYS + Metadata")
    print("="*70)
    
    # ==================== VERIFICAR ARCHIVOS ====================
    print("\nVerificando archivos de entrada...")
    
    if not os.path.exists(FEATURES_CSV):
        print("\nERROR: No existe: {}".format(FEATURES_CSV))
        print("Ejecuta primero: 03_features/build_den_dys.py")
        sys.exit(1)
    
    if not os.path.exists(METADATA_CSV):
        print("\nERROR: No existe: {}".format(METADATA_CSV))
        print("Ejecuta primero: 01_metadata/extract_metadata_from_csv.py")
        sys.exit(1)
    
    print("Archivos de entrada encontrados")
    
    # ==================== CARGAR DATOS ====================
    print("\nCargando features desde: {}".format(FEATURES_CSV))
    df_features = pd.read_csv(FEATURES_CSV)
    print("  Features: {} pacientes".format(len(df_features)))
    
    print("\nCargando metadata desde: {}".format(METADATA_CSV))
    df_metadata = pd.read_csv(METADATA_CSV)
    print("  Metadata: {} pacientes".format(len(df_metadata)))
    
    # ==================== VERIFICAR COLUMNAS ====================
    print("\nVerificando columnas requeridas...")
    
    required_features = ['patient_id']
    required_metadata = ['patient_id', 'QA', 'language']
    
    missing_feat = [c for c in required_features if c not in df_features.columns]
    missing_meta = [c for c in required_metadata if c not in df_metadata.columns]
    
    if missing_feat:
        print("\nERROR: Faltan columnas en features: {}".format(missing_feat))
        sys.exit(1)
    
    if missing_meta:
        print("\nERROR: Faltan columnas en metadata: {}".format(missing_meta))
        sys.exit(1)
    
    print("Columnas requeridas presentes")
    
    # ==================== LIMPIAR CONFLICTOS ====================
    print("\nLimpiando columnas conflictivas...")
    
    if 'language' in df_features.columns:
        print("  Eliminando 'language' de features (se usara la de metadata)")
        df_features = df_features.drop(columns=['language'])
    
    conflict_cols = ['QA', 'sex', 'age', 'aphasia_type']
    existing_conflicts = [c for c in conflict_cols if c in df_features.columns]
    if existing_conflicts:
        print("  Eliminando columnas conflictivas: {}".format(existing_conflicts))
        df_features = df_features.drop(columns=existing_conflicts)
    
    # ==================== MERGE ====================
    print("\nHaciendo merge por 'patient_id'...")
    
    df_final = pd.merge(
        df_features,
        df_metadata[['patient_id', 'QA', 'sex', 'age', 'aphasia_type', 'language']],
        on='patient_id',
        how='inner'
    )
    
    print("Despues del merge: {} pacientes".format(len(df_final)))
    
    # Verificar resultado
    if 'language' not in df_final.columns:
        print("\nERROR: No se encontro columna 'language' despues del merge")
        sys.exit(1)
    
    if len(df_final) == 0:
        print("\nERROR: El merge resulto en 0 pacientes")
        print("Verifica que los patient_id coincidan entre features y metadata")
        sys.exit(1)
    
    # ==================== LIMPIAR NaN ====================
    print("\nLimpiando valores NaN en features criticas...")
    
    initial_len = len(df_final)
    critical_features = ['den_light_verbs', 'dys_pause_sec_mean']
    
    # Verificar que existen estas features
    missing_critical = [f for f in critical_features if f not in df_final.columns]
    if missing_critical:
        print("\n  Advertencia: No se encontraron features criticas: {}".format(missing_critical))
        critical_features = [f for f in critical_features if f in df_final.columns]
    
    if critical_features:
        df_final = df_final.dropna(subset=critical_features)
        
        if len(df_final) < initial_len:
            print("  Eliminados {} pacientes con NaN".format(initial_len - len(df_final)))
    
    # ==================== REPORTAR DISTRIBUCION ====================
    print("\n" + "="*70)
    print("DISTRIBUCION FINAL POR IDIOMA")
    print("="*70)
    print(df_final['language'].value_counts())
    
    print("\n" + "="*70)
    print("ESTADISTICAS POR IDIOMA")
    print("="*70)
    for lang in df_final['language'].unique():
        df_lang = df_final[df_final['language'] == lang]
        print("\n{}:".format(lang.upper()))
        print("  N: {}".format(len(df_lang)))
        print("  QA: {:.1f} +/- {:.1f}".format(df_lang['QA'].mean(), df_lang['QA'].std()))
        print("  Range: [{:.1f}, {:.1f}]".format(df_lang['QA'].min(), df_lang['QA'].max()))
    
    # ==================== GUARDAR ====================
    print("\n" + "="*70)
    print("GUARDANDO RESULTADO")
    print("="*70)
    
    # Crear directorio de salida si no existe
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Creado directorio: {}".format(output_dir))
    
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print("\nDataset final guardado: {}".format(OUTPUT_CSV))
    print("  Total pacientes: {}".format(len(df_final)))
    
    # Contar features
    den_cols = [c for c in df_final.columns if c.startswith('den_')]
    dys_cols = [c for c in df_final.columns if c.startswith('dys_')]
    print("  Features DEN: {}".format(len(den_cols)))
    print("  Features DYS: {}".format(len(dys_cols)))
    print("  Features TOTAL: {}".format(len(den_cols) + len(dys_cols)))
    
    print("="*70)

if __name__ == "__main__":
    main()