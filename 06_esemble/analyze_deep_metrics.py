import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === CONFIGURACION DE RUTAS ===
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent 

# Archivos de entrada
RESULTS_FILE = PROJECT_ROOT / "outputs/experiments/ENSEMBLE_FINAL/VALIDACION_REAL_NELDER_PONDERATION/validacion_real_5fold.csv"
DATA_FILE = PROJECT_ROOT / "data/dataset_FINAL_CON_POSLM.csv"

# Salida DENTRO de 06_esemble
OUTPUT_DIR = CURRENT_DIR / "outputs_analisis_profundo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("="*70)
    print("ANALISIS CLINICO DE ERRORES (CORREGIDO V2)")
    print(f"Leyendo resultados de: {RESULTS_FILE}")
    print("="*70)

    if not RESULTS_FILE.exists():
        print(f"[ERROR] No encuentro el archivo: {RESULTS_FILE}")
        return

    # 1. Cargar Datos
    print("Cargando datos...")
    df_res = pd.read_csv(RESULTS_FILE)
    df_orig = pd.read_csv(DATA_FILE)

    # 2. Ingeniería de Features (Texto y Audio)
    if 'transcript' in df_orig.columns:
        print("   -> Calculando longitud de transcripciones...")
        # Numero de palabras
        df_orig['calc_word_count'] = df_orig['transcript'].astype(str).apply(lambda x: len(x.split()))
        # Numero de oraciones (estimado por puntuacion)
        df_orig['calc_utterances'] = df_orig['transcript'].astype(str).apply(lambda x: x.count('.') + x.count('?') + x.count('!'))

    # Buscar columnas de interes en el original
    keywords = ['duration', 'total_duration', 'len', 'sec', 'pauses', 'rate', 'fill', 'silence', 'utt']
    audio_cols = [c for c in df_orig.columns if any(k in c.lower() for k in keywords) and pd.api.types.is_numeric_dtype(df_orig[c])]
    
    # Añadir calculadas, metadatos Y LA LLAVE patient_id
    cols_to_add = list(set(audio_cols + ['calc_word_count', 'calc_utterances', 'sex', 'age', 'aphasia_type', 'patient_id']))
    cols_present_in_orig = [c for c in cols_to_add if c in df_orig.columns]
    
    print(f"   -> Variables clinicas encontradas: {len(cols_present_in_orig)}")

    # 3. Merge Inteligente
    cols_in_res = df_res.columns.tolist()
    # Logica: Queremos columnas nuevas OR patient_id (necesario para el join)
    cols_to_merge = [c for c in cols_present_in_orig if c not in cols_in_res or c == 'patient_id']
    
    # Asegurarnos de que patient_id esta en ambos lados antes del merge
    if 'patient_id' not in cols_to_merge:
        print("[AVISO] Forzando inclusion de patient_id...")
        cols_to_merge.append('patient_id')

    df_merge = pd.merge(df_res, df_orig[cols_to_merge], on='patient_id', how='left')
    
    # Calcular error absoluto si no esta
    if 'Error_Abs' not in df_merge.columns:
        df_merge['Error_Abs'] = abs(df_merge['QA_pred'] - df_merge['QA_real'])

    # 4. Generar Graficas
    print("\nGenerando graficas de correlacion...")
    
    # Usamos la lista de variables que queriamos analizar
    cols_to_analyze = [c for c in cols_present_in_orig if c in df_merge.columns and c != 'patient_id']

    for col in cols_to_analyze:
        # Saltamos no numericas para correlacion
        if not pd.api.types.is_numeric_dtype(df_merge[col]): continue
        
        # Correlacion
        corr = df_merge[col].corr(df_merge['Error_Abs'])
        
        # Solo si es relevante (> 0.15) para no llenar de basura
        if abs(corr) > 0.15: 
            print(f"   [ALERTA] '{col}' influye en el error! (Corr: {corr:.2f})")
            
            plt.figure(figsize=(8, 6))
            # Usar QA_real para colorear si existe
            if 'QA_real' in df_merge.columns:
                sns.scatterplot(data=df_merge, x=col, y='Error_Abs', hue='QA_real', palette='viridis', alpha=0.7)
            else:
                sns.scatterplot(data=df_merge, x=col, y='Error_Abs', alpha=0.7)
                
            try:
                sns.regplot(data=df_merge, x=col, y='Error_Abs', scatter=False, color='red', line_kws={'linestyle':'--'})
            except:
                pass

            plt.title(f'Diagnostico: Error vs {col}\n(Correlacion: {corr:.2f})')
            plt.ylabel('Error Absoluto (MAE)')
            plt.xlabel(col)
            plt.grid(True, alpha=0.3)
            
            safe_name = col.replace("/", "_")
            plt.savefig(OUTPUT_DIR / f"DIAGNOSTICO_{safe_name}.png")
            plt.close()

    # 5. Informe CSV
    bad_cases = df_merge[df_merge['Error_Abs'] > 20].sort_values('Error_Abs', ascending=False)
    
    if not bad_cases.empty:
        print(f"\n[INFORME] {len(bad_cases)} pacientes tienen error > 20 puntos.")
        csv_path = OUTPUT_DIR / "INFORME_FALLOS_CRITICOS.csv"
        bad_cases.to_csv(csv_path, index=False)
        print(f"   -> Detalle guardado en: {csv_path}")

    print("\n" + "="*70)
    print(f"Analisis completado. Resultados en:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
