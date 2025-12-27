#!/usr/bin/env python3
# 06_ensemble/run_ensemble_retrain.py
# -*- coding: utf-8 -*-

import sys
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr

# === CONFIGURACIÓN ===
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/ENSEMBLE_FINAL_RETRAINED_RICH"
DATA_FILE = PROJECT_ROOT / "data/dataset_FINAL_CON_POSLM.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOGGER DUAL (Pantalla + Archivo) ===
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(OUTPUT_DIR / "execution.log")
sys.stderr = DualLogger(OUTPUT_DIR / "execution_errors.log")

# Copia de seguridad del propio script
shutil.copy(__file__, OUTPUT_DIR / "script_source_code.py")

# Importar modelos
try:
    from tabpfn import TabPFNRegressor
    from interpret.glassbox import ExplainableBoostingRegressor
    from catboost import CatBoostRegressor
except ImportError:
    print("[ERROR] Faltan librerias (tabpfn, interpret, catboost).")
    sys.exit(1)

# === HIPERPARÁMETROS GANADORES ===
PARAMS_CATBOOST = {
    'verbose': 0,
    'random_state': 42,
    'iterations': 267,
    'learning_rate': 0.029,
    'depth': 4,
    'l2_leaf_reg': 3.39e-08,
    'border_count': 103
}

PARAMS_EBM = {
    'random_state': 42,
    'n_jobs': 1,
    'interactions': 10,
    'learning_rate': 0.01,
    'max_bins': 128,
    'min_samples_leaf': 5
}

# Configuración
N_FOLDS = 4
CV_STRATEGY = "severity" 

def get_subdataset(patient_id):
    import re
    match = re.match(r"([a-zA-Z]+)", patient_id)
    if match: return match.group(1).lower()
    return "unknown"

def get_severity_bin(score):
    if score < 25: return 0 
    if score < 50: return 1 
    if score < 75: return 2 
    return 3 

def get_severity_label(score):
    if score < 25: return "Very Severe (0-25)"
    if score < 50: return "Severe (25-50)"
    if score < 75: return "Moderate (50-75)"
    return "Mild (75-100)"

def optimize_weights(y_true, pred_matrix):
    def objective(weights):
        w = np.array(weights)
        if w.sum() == 0: return 9999
        w = w / w.sum()
        final_pred = np.dot(pred_matrix, w)
        return mean_absolute_error(y_true, final_pred)
    
    n = pred_matrix.shape[1]
    # Restricciones: suma=1, limites 0-1
    res = minimize(objective, [1.0/n]*n, method='SLSQP', bounds=[(0,1)]*n, constraints={'type':'eq', 'fun': lambda w: 1-sum(w)})
    return res.x / res.x.sum()

def generate_rich_plots(y_true, y_pred, output_path, mae, corr, title_suffix=""):
    # 1. Scatter Plot
    plt.figure(figsize=(10, 9))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='rebeccapurple', s=80, edgecolor='w')
    plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Ideal')
    plt.title(f'Predicción vs Real {title_suffix}\nMAE: {mae:.2f} | Pearson r: {corr:.3f}', fontsize=14)
    plt.xlabel('QA Real (Severidad)', fontsize=12)
    plt.ylabel('QA Predicho (Modelo)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / f"plot_scatter{title_suffix.replace(' ', '_')}.png", dpi=300)
    plt.close()

    # 2. Confusion Matrix (Severidad)
    true_sev = [get_severity_label(x) for x in y_true]
    pred_sev = [get_severity_label(x) for x in y_pred]
    labels = ["Very Severe (0-25)", "Severe (25-50)", "Moderada (50-75)", "Leve (75-100)"]
    labels_short = ["0-25", "25-50", "50-75", "75-100"]
    
    cm = confusion_matrix(true_sev, pred_sev, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_short)
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap='PuBu', values_format='d', ax=plt.gca(), colorbar=False)
    plt.title(f'Matriz de Confusión {title_suffix}', fontsize=14)
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Realidad', fontsize=12)
    plt.savefig(output_path / f"plot_confusion{title_suffix.replace(' ', '_')}.png", dpi=300)
    plt.close()

    # 3. Histograma de Error
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, color='teal', bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Distribución de Errores {title_suffix}', fontsize=14)
    plt.xlabel('Error (Pred - Real)', fontsize=12)
    plt.savefig(output_path / f"plot_errors{title_suffix.replace(' ', '_')}.png", dpi=300)
    plt.close()

def main():
    print("="*70)
    print("ENSEMBLE: RE-ENTRENAMIENTO MAESTRO + VISUALIZACION")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)

    # 1. Cargar Datos
    df = pd.read_csv(DATA_FILE)
    df = df[df['QA'].notna()].copy().reset_index(drop=True)
    
    # FILTRO SOLO PWA
    df = df[df['group'] == 'pwa'].reset_index(drop=True)
    print(f"[INFO] Entrenando SOLO con PWA. Total: {len(df)}")

    y = df['QA'].values
    # Guardamos meta-datos para luego analizar por idioma
    cols_meta = ['patient_id', 'QA', 'group', 'language', 'sex', 'age', 'aphasia_type', 'group_original', 'transcript']
    X = df.drop(columns=[c for c in cols_meta if c in df.columns], errors='ignore')
    
    groups = df['QA'].apply(get_severity_bin).values
    real_groups = df['patient_id'].values # Para no romper pacientes
    
    skf = StratifiedGroupKFold(n_splits=N_FOLDS)
    y_bins = pd.cut(y, bins=5, labels=False) 

    oof_preds_tab = np.zeros(len(y))
    oof_preds_cat = np.zeros(len(y))
    oof_preds_ebm = np.zeros(len(y))
    
    fold = 1
    for train_idx, test_idx in skf.split(X, y_bins, groups=real_groups):
        print(f"\n>>> FOLD {fold}/{N_FOLDS} <<<")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_proc = scaler.transform(imputer.transform(X_test))
        
        # 1. TabPFN
        print("   > TabPFN (64)...", end=" ", flush=True)
        m_tab = TabPFNRegressor(device='cpu', n_estimators=64)
        m_tab.fit(X_train_proc, y_train)
        p_tab = m_tab.predict(X_test_proc)
        oof_preds_tab[test_idx] = p_tab
        print(f"OK (MAE: {mean_absolute_error(y_test, p_tab):.2f})")
        
        # 2. CatBoost
        print("   > CatBoost...", end=" ", flush=True)
        m_cat = CatBoostRegressor(**PARAMS_CATBOOST)
        m_cat.fit(X_train_proc, y_train)
        p_cat = m_cat.predict(X_test_proc)
        oof_preds_cat[test_idx] = p_cat
        print(f"OK")
        
        # 3. EBM
        print("   > EBM...", end=" ", flush=True)
        m_ebm = ExplainableBoostingRegressor(**PARAMS_EBM)
        m_ebm.fit(X_train_proc, y_train)
        p_ebm = m_ebm.predict(X_test_proc)
        oof_preds_ebm[test_idx] = p_ebm
        print(f"OK")
        
        fold += 1

    # === OPTIMIZACIÓN FINAL ===
    print("\n" + "="*50)
    print("RESULTADOS GLOBALES (PWA ONLY)")
    
    pred_matrix = np.column_stack([oof_preds_tab, oof_preds_cat, oof_preds_ebm])
    
    weights = optimize_weights(y, pred_matrix)
    final_preds = np.dot(pred_matrix, weights)
    
    final_mae = mean_absolute_error(y, final_preds)
    corr, _ = pearsonr(y, final_preds)
    
    print("-" * 30)
    print(f"PESOS ÓPTIMOS: Tab={weights[0]:.3f}, Cat={weights[1]:.3f}, EBM={weights[2]:.3f}")
    print(f"MAE GLOBAL:   {final_mae:.4f}")
    print(f"CORRELACIÓN:  {corr:.4f}")
    print("-" * 30)
    
    # === ANÁLISIS POR IDIOMA ===
    df['QA_pred'] = final_preds
    df['Error'] = np.abs(df['QA'] - df['QA_pred'])
    
    print("\n[ANÁLISIS POR IDIOMA]")
    for lang in df['language'].unique():
        subset = df[df['language'] == lang]
        if len(subset) > 0:
            mae_lang = mean_absolute_error(subset['QA'], subset['QA_pred'])
            print(f"   Idioma '{lang}' (n={len(subset)}): MAE = {mae_lang:.4f}")
            
            # Guardar resultados de idiomas pequeños
            if lang in ['es', 'ca']:
                subset.to_csv(OUTPUT_DIR / f"resultados_{lang}.csv", index=False)

    # === GUARDADO Y PLOTS ===
    df.to_csv(OUTPUT_DIR / "ensemble_predictions_full.csv", index=False)
    
    print("\n[GENERANDO GRÁFICAS...]")
    generate_rich_plots(y, final_preds, OUTPUT_DIR, final_mae, corr, title_suffix="")
    
    # Gráfica específica solo para Español si hay datos
    df_es = df[df['language'] == 'es']
    if len(df_es) > 0:
        mae_es = mean_absolute_error(df_es['QA'], df_es['QA_pred'])
        corr_es, _ = pearsonr(df_es['QA'], df_es['QA_pred']) if len(df_es) > 1 else (0,0)
        generate_rich_plots(df_es['QA'], df_es['QA_pred'], OUTPUT_DIR, mae_es, corr_es, title_suffix=" (ESPAÑOL)")

    print(f"PROCESO COMPLETADO. Revisa {OUTPUT_DIR}")

if __name__ == "__main__":
    main()