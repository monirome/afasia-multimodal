import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === CONFIGURACION DE RUTAS ===
# 1. Donde estamos ahora (06_esemble)
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent 

# 2. Donde buscar los resultados del Nelder-Mead (input)
#    Nota: Ajusta esto si tu script Nelder guardó en otro sitio. 
#    Por defecto asumimos la ruta del script anterior.
RESULTS_FILE = PROJECT_ROOT / "outputs/experiments/ENSEMBLE_FINAL/VALIDACION_REAL_NELDER_PONDERATION/validacion_real_5fold.csv"
DATA_FILE = PROJECT_ROOT / "data/dataset_FINAL_CON_POSLM.csv"

# 3. Donde guardar los graficos (DENTRO de 06_esemble)
OUTPUT_DIR = CURRENT_DIR / "outputs_analisis_profundo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("="*70)
    print("ANALISIS PROFUNDO DE ERRORES (UTTERANCES, PAUSAS, DURACION)")
    print(f"Guardando resultados en: {OUTPUT_DIR}")
    print("="*70)

    # 1. Cargar Datos
    if not RESULTS_FILE.exists():
        print(f"[ERROR] No encuentro los resultados en: {RESULTS_FILE}")
        print("¿Terminó ya el trabajo de Condor 'job_nelder.sub'?")
        return

    print("Cargando predicciones y dataset original...")
    df_res = pd.read_csv(RESULTS_FILE)
    df_orig = pd.read_csv(DATA_FILE)

    # 2. Ingeniería de Features para el Análisis
    # Vamos a buscar o crear las métricas que te interesan
    
    # A. Calcular Longitud (Palabras) si hay transcript
    if 'transcript' in df_orig.columns:
        print("   -> Calculando conteo de palabras (Word Count)...")
        df_orig['analisis_word_count'] = df_orig['transcript'].astype(str).apply(lambda x: len(x.split()))
        # Estimacion de utterances (basado en signos de puntuacion o saltos)
        df_orig['analisis_utterances'] = df_orig['transcript'].astype(str).apply(lambda x: x.count('.') + x.count('?') + x.count('!'))
    
    # B. Buscar columnas de audio/prosodia existentes
    # Buscamos cualquier columna que suene a lo que pides
    keywords = ['dur', 'len', 'sec', 'time', 'paus', 'rate', 'fill', 'silence', 'utt']
    audio_cols = [c for c in df_orig.columns if any(k in c.lower() for k in keywords) and pd.api.types.is_numeric_dtype(df_orig[c])]
    
    # Añadimos las calculadas manuales
    cols_interest = audio_cols + ['analisis_word_count', 'analisis_utterances', 'sex', 'age', 'aphasia_type']
    # Filtramos las que realmente existen
    cols_interest = [c for c in cols_interest if c in df_orig.columns]
    
    print(f"   -> Variables encontradas para analizar: {len(cols_interest)}")
    print(f"      {cols_interest[:5]} ...")

    # 3. Cruzar con las predicciones
    # Usamos left join para asegurarnos de no perder pacientes
    df_merge = pd.merge(df_res, df_orig[['patient_id'] + cols_interest], on='patient_id', how='left')
    
    # Recalcular error absoluto
    df_merge['Error_Abs'] = abs(df_merge['QA_pred'] - df_merge['QA_real'])
    
    # 4. Generar Gráficas de Correlación
    print("\nGenerando gráficas de diagnóstico...")
    
    numeric_cols = [c for c in cols_interest if pd.api.types.is_numeric_dtype(df_merge[c])]
    
    for col in numeric_cols:
        # Calcular correlacion
        corr = df_merge[col].corr(df_merge['Error_Abs'])
        
        # Solo graficar si hay cierta correlacion (> 0.1 o < -0.1) para no llenar de basura
        if abs(corr) > 0.05: 
            plt.figure(figsize=(10, 6))
            
            # Scatter con color segun severidad real
            scatter = plt.scatter(df_merge[col], df_merge['Error_Abs'], 
                                c=df_merge['QA_real'], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='QA Real (Severidad)')
            
            # Linea de tendencia
            sns.regplot(data=df_merge, x=col, y='Error_Abs', scatter=False, color='red', line_kws={'linestyle':'--'})
            
            plt.title(f'¿Influye "{col}" en el Error? (Corr: {corr:.2f})')
            plt.xlabel(col)
            plt.ylabel('Error Absoluto del Modelo')
            plt.grid(True, alpha=0.3)
            
            # Guardar
            safe_name = col.replace("/", "_").replace(" ", "_")
            plt.savefig(OUTPUT_DIR / f"corr_{safe_name}.png")
            plt.close()
            print(f"   [PLOT] Guardado: corr_{safe_name}.png (Corr: {corr:.2f})")

    # 5. Análisis de "Los Casos Imposibles"
    # Filtramos los errores > 20 puntos
    bad_cases = df_merge[df_merge['Error_Abs'] > 20].copy()
    bad_cases = bad_cases.sort_values('Error_Abs', ascending=False)
    
    if not bad_cases.empty:
        print(f"\n[INSIGHT] Tenemos {len(bad_cases)} pacientes con Error > 20 puntos.")
        
        # Guardar CSV detallado
        cols_export = ['patient_id', 'QA_real', 'QA_pred', 'Error_Abs'] + cols_interest
        bad_cases[cols_export].to_csv(OUTPUT_DIR / "CASOS_CRITICOS_DETALLE.csv", index=False)
        
        # Resumen rapido en pantalla
        print("Top 5 Peores Casos:")
        print(bad_cases[['patient_id', 'QA_real', 'QA_pred', 'Error_Abs']].head(5))
        
        # Comprobar hipótesis de audio corto
        if 'analisis_word_count' in bad_cases.columns:
            avg_words_bad = bad_cases['analisis_word_count'].mean()
            avg_words_all = df_merge['analisis_word_count'].mean()
            print(f"\n   -> Media de palabras en casos con ERROR: {avg_words_bad:.1f}")
            print(f"   -> Media de palabras en casos BIEN:      {avg_words_all:.1f}")
            if avg_words_bad < avg_words_all:
                print("   [CONCLUSION] CONFIRMADO: El modelo falla más en audios cortos.")
            else:
                print("   [CONCLUSION] No parece que la longitud sea el único problema.")

    print("\n" + "="*70)
    print(f"Todo listo. Revisa la carpeta: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()