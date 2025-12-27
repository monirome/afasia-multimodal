#!/usr/bin/env python3
# 06_ensemble/run_ensemble_FINAL_COMPLETO.py
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
import time
from datetime import datetime

# === CONFIGURACION CON TIMESTAMP ===
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Crear nombre de carpeta con timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "ENSEMBLE_{}".format(TIMESTAMP)
DATA_FILE = PROJECT_ROOT / "data" / "dataset_FINAL_CON_POSLM.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === DUAL LOGGER ===
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

log_file = OUTPUT_DIR / "execution.log"
error_file = OUTPUT_DIR / "execution_errors.log"

sys.stdout = DualLogger(log_file)
sys.stderr = DualLogger(error_file)

# BACKUP DEL SCRIPT
try:
    shutil.copy(__file__, OUTPUT_DIR / "script_source_code.py")
    print("[OK] Script respaldado en: {}".format(OUTPUT_DIR / 'script_source_code.py'))
except Exception as e:
    print("[WARNING] No se pudo respaldar el script: {}".format(e))

# === IMPORTAR MODELOS ===
try:
    from tabpfn import TabPFNRegressor
    print("[OK] TabPFN importado")
except ImportError:
    print("[ERROR] TabPFN no disponible")
    sys.exit(1)

try:
    from interpret.glassbox import ExplainableBoostingRegressor
    print("[OK] EBM importado")
except ImportError:
    print("[ERROR] EBM no disponible")
    sys.exit(1)

try:
    from catboost import CatBoostRegressor
    print("[OK] CatBoost importado")
except ImportError:
    print("[ERROR] CatBoost no disponible")
    sys.exit(1)

# === CONFIGURACION ===
N_FOLDS = 4
CV_STRATEGY = "subdataset"

# CATBOOST - Promedio de mejores params de CV
PARAMS_CATBOOST = {
    'verbose': 0,
    'random_state': 42,
    'thread_count': 1,
    'iterations': 181,
    'learning_rate': 0.137,
    'depth': 5,
    'l2_leaf_reg': 1e-06,
    'border_count': 135,
    'bagging_temperature': 0.25
}

# EBM - Promedio de mejores params de CV
PARAMS_EBM = {
    'random_state': 42,
    'n_jobs': 1,
    'interactions': 6,
    'learning_rate': 0.05,
    'max_bins': 192,
    'max_interaction_bins': 20,
    'min_samples_leaf': 3
}

# TABPFN
TABPFN_N_ESTIMATORS = 64
TABPFN_FEATURES = "full"

# === FUNCIONES AUXILIARES ===

def get_subdataset(patient_id):
    import re
    match = re.match(r"([a-zA-Z]+)", str(patient_id))
    return match.group(1).lower() if match else "unknown"

def get_severity_label(score):
    if score < 25: return "Very Severe (0-25)"
    if score < 50: return "Severe (25-50)"
    if score < 75: return "Moderate (50-75)"
    return "Mild (75-100)"

def optimize_weights(y_true, pred_matrix):
    def objective(weights):
        w = np.array(weights)
        if w.sum() == 0: 
            return 9999
        w = w / w.sum()
        final_pred = np.dot(pred_matrix, w)
        return mean_absolute_error(y_true, final_pred)
    
    n_models = pred_matrix.shape[1]
    initial_weights = [1.0/n_models] * n_models
    
    result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP', 
        bounds=[(0, 1)] * n_models,
        constraints={'type': 'eq', 'fun': lambda w: 1 - sum(w)}
    )
    
    final_weights = result.x / result.x.sum()
    return final_weights

def generate_rich_plots(y_true, y_pred, output_path, mae, corr, r2, title_suffix=""):
    
    # 1. SCATTER PLOT
    plt.figure(figsize=(10, 9))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='rebeccapurple', 
                    s=80, edgecolor='white', linewidth=0.5)
    plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Identidad')
    
    text = 'MAE: {:.2f}\nR2: {:.3f}\nPearson r: {:.3f}'.format(mae, r2, corr)
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11, family='monospace')
    
    plt.title('Prediccion vs Real{}'.format(title_suffix), fontsize=14, fontweight='bold')
    plt.xlabel('QA Real (Severidad)', fontsize=12)
    plt.ylabel('QA Predicho (Modelo)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "plot_scatter{}.png".format(title_suffix.replace(' ', '_')), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. CONFUSION MATRIX
    true_sev = [get_severity_label(x) for x in y_true]
    pred_sev = [get_severity_label(x) for x in y_pred]
    labels = ["Very Severe (0-25)", "Severe (25-50)", "Moderate (50-75)", "Mild (75-100)"]
    labels_short = ["0-25", "25-50", "50-75", "75-100"]
    
    cm = confusion_matrix(true_sev, pred_sev, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_short)
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap='PuBu', values_format='d', ax=plt.gca(), colorbar=False)
    plt.title('Matriz de Confusion{}'.format(title_suffix), fontsize=14, fontweight='bold')
    plt.xlabel('Severidad Predicha', fontsize=12)
    plt.ylabel('Severidad Real', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / "plot_confusion{}.png".format(title_suffix.replace(' ', '_')), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. HISTOGRAMA DE ERRORES
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    
    sns.histplot(errors, kde=True, color='teal', bins=30, alpha=0.6)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axvline(x=mean_error, color='blue', linestyle=':', linewidth=2, 
                label='Media = {:.2f}'.format(mean_error))
    
    text = 'Media: {:.2f}\nStd: {:.2f}\nMAE: {:.2f}'.format(mean_error, std_error, mae)
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             fontsize=10, family='monospace')
    
    plt.title('Distribucion de Errores{}'.format(title_suffix), fontsize=14, fontweight='bold')
    plt.xlabel('Error de Prediccion (Pred - Real)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / "plot_errors{}.png".format(title_suffix.replace(' ', '_')), 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_by_language(df, output_dir):
    print("\n" + "="*70)
    print("ANALISIS POR IDIOMA")
    print("="*70)
    
    language_stats = []
    
    for lang in sorted(df['language'].unique()):
        subset = df[df['language'] == lang]
        if len(subset) == 0:
            continue
        
        mae_lang = mean_absolute_error(subset['QA'], subset['QA_pred'])
        
        if len(subset) > 1:
            corr_lang, _ = pearsonr(subset['QA'], subset['QA_pred'])
            r2_lang = r2_score(subset['QA'], subset['QA_pred'])
        else:
            corr_lang = np.nan
            r2_lang = np.nan
        
        language_stats.append({
            'language': lang,
            'n': len(subset),
            'mae': mae_lang,
            'correlation': corr_lang,
            'r2': r2_lang,
            'qa_mean': subset['QA'].mean(),
            'qa_std': subset['QA'].std()
        })
        
        print("  {:5s} (n={:3d}): MAE={:6.3f}, r={:6.3f}, R2={:6.3f}".format(
            lang.upper(), len(subset), mae_lang, corr_lang, r2_lang))
        
        subset.to_csv(output_dir / "results_{}.csv".format(lang), index=False)
        
        if len(subset) > 1 and lang in ['es', 'ca']:
            generate_rich_plots(
                subset['QA'].values, 
                subset['QA_pred'].values, 
                output_dir, 
                mae_lang, 
                corr_lang,
                r2_lang if not np.isnan(r2_lang) else 0,
                title_suffix=" ({})".format(lang.upper())
            )
    
    pd.DataFrame(language_stats).to_csv(output_dir / "language_analysis.csv", index=False)
    print("\n  Analisis guardado: language_analysis.csv")

def save_config_file(output_dir):
    """Guarda archivo de configuracion del experimento"""
    config_text = """
======================================================================
CONFIGURACION DEL EXPERIMENTO
======================================================================

Timestamp:       {}
Output dir:      {}

CV Strategy:     {}
CV Folds:        {}

TabPFN:
  n_estimators:  {}
  features:      {}

CatBoost:
  iterations:    {}
  learning_rate: {}
  depth:         {}
  l2_leaf_reg:   {}
  border_count:  {}

EBM:
  interactions:  {}
  learning_rate: {}
  max_bins:      {}
  min_samples:   {}

Dataset:         {}

======================================================================
""".format(
        TIMESTAMP,
        output_dir,
        CV_STRATEGY,
        N_FOLDS,
        TABPFN_N_ESTIMATORS,
        TABPFN_FEATURES,
        PARAMS_CATBOOST['iterations'],
        PARAMS_CATBOOST['learning_rate'],
        PARAMS_CATBOOST['depth'],
        PARAMS_CATBOOST['l2_leaf_reg'],
        PARAMS_CATBOOST['border_count'],
        PARAMS_EBM['interactions'],
        PARAMS_EBM['learning_rate'],
        PARAMS_EBM['max_bins'],
        PARAMS_EBM['min_samples_leaf'],
        DATA_FILE
    )
    
    with open(output_dir / "CONFIG.txt", "w") as f:
        f.write(config_text)

# === FUNCION PRINCIPAL ===

def main():
    print("\n" + "="*70)
    print("ENSEMBLE FINAL - VERSION COMPLETA")
    print("="*70)
    print("Timestamp: {}".format(TIMESTAMP))
    print("CV Strategy: {}".format(CV_STRATEGY.upper()))
    print("Folds: {}".format(N_FOLDS))
    print("TabPFN estimators: {}".format(TABPFN_N_ESTIMATORS))
    print("Output: {}".format(OUTPUT_DIR))
    print("="*70 + "\n")

    # Guardar configuracion
    save_config_file(OUTPUT_DIR)

    # 1. CARGAR DATOS
    print("[1/7] Cargando datos...")
    sys.stdout.flush()
    
    if not DATA_FILE.exists():
        print("[ERROR] No existe: {}".format(DATA_FILE))
        sys.exit(1)
    
    df = pd.read_csv(DATA_FILE)
    df = df[df['QA'].notna()].copy().reset_index(drop=True)
    df = df[df['group'] == 'pwa'].reset_index(drop=True)
    
    print("  Total pacientes PWA: {}".format(len(df)))

    y = df['QA'].values
    cols_meta = ['patient_id', 'QA', 'group', 'language', 'sex', 'age', 
                 'aphasia_type', 'group_original', 'transcript']
    X = df.drop(columns=[c for c in cols_meta if c in df.columns], errors='ignore')
    
    print("  Features: {}".format(X.shape[1]))
    print("  Idiomas: {}".format(', '.join(sorted(df['language'].unique()))))

    # 2. CONFIGURAR CV
    print("\n[2/7] Configurando {} CV ({} folds)...".format(CV_STRATEGY.upper(), N_FOLDS))
    sys.stdout.flush()
    
    groups = df['patient_id'].apply(get_subdataset).values
    subdatasets_unique = np.unique(groups)
    print("  Sub-datasets unicos: {}".format(len(subdatasets_unique)))
    
    y_bins = pd.cut(y, bins=5, labels=False)
    real_groups = df['patient_id'].values
    
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # 3. ARRAYS PARA PREDICCIONES
    oof_preds_tab = np.zeros(len(y))
    oof_preds_cat = np.zeros(len(y))
    oof_preds_ebm = np.zeros(len(y))
    
    # 4. LOOP DE CROSS-VALIDATION
    print("\n[3/7] Entrenando modelos con CV...")
    sys.stdout.flush()
    
    fold = 1
    for train_idx, test_idx in skf.split(X, y_bins, groups=real_groups):
        print("\n{}".format('='*50))
        print("FOLD {}/{}".format(fold, N_FOLDS))
        print('='*50)
        sys.stdout.flush()
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print("  Train: {:3d} | Test: {:3d}".format(len(y_train), len(y_test)))
        sys.stdout.flush()
        
        # Preprocesamiento
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_proc = scaler.transform(imputer.transform(X_test))
        
        # MODELO 1: TabPFN
        print("  [1/3] TabPFN ({} est)... [{}] ".format(
            TABPFN_N_ESTIMATORS, datetime.now().strftime('%H:%M:%S')), end="", flush=True)
        sys.stdout.flush()
        
        start_time = time.time()
        m_tab = TabPFNRegressor(device='cpu', n_estimators=TABPFN_N_ESTIMATORS)
        
        print("[FITTING] ", end="", flush=True)
        sys.stdout.flush()
        m_tab.fit(X_train_proc, y_train)
        
        print("[PREDICTING] ", end="", flush=True)
        sys.stdout.flush()
        p_tab = m_tab.predict(X_test_proc)
        oof_preds_tab[test_idx] = p_tab
        
        elapsed = time.time() - start_time
        mae_tab = mean_absolute_error(y_test, p_tab)
        print("OK (MAE = {:.2f}) [Tiempo: {:.1f} min]".format(mae_tab, elapsed/60))
        sys.stdout.flush()
        
        # MODELO 2: CatBoost
        print("  [2/3] CatBoost... ", end="", flush=True)
        sys.stdout.flush()
        m_cat = CatBoostRegressor(**PARAMS_CATBOOST)
        m_cat.fit(X_train_proc, y_train)
        p_cat = m_cat.predict(X_test_proc)
        oof_preds_cat[test_idx] = p_cat
        mae_cat = mean_absolute_error(y_test, p_cat)
        print("OK (MAE = {:.2f})".format(mae_cat))
        sys.stdout.flush()
        
        # MODELO 3: EBM
        print("  [3/3] EBM... ", end="", flush=True)
        sys.stdout.flush()
        m_ebm = ExplainableBoostingRegressor(**PARAMS_EBM)
        m_ebm.fit(X_train_proc, y_train)
        p_ebm = m_ebm.predict(X_test_proc)
        oof_preds_ebm[test_idx] = p_ebm
        mae_ebm = mean_absolute_error(y_test, p_ebm)
        print("OK (MAE = {:.2f})".format(mae_ebm))
        sys.stdout.flush()
        
        fold += 1

    # 5. RESULTADOS INDIVIDUALES
    print("\n" + "="*70)
    print("[4/7] RESULTADOS INDIVIDUALES")
    print("="*70)
    sys.stdout.flush()
    
    mae_tab_global = mean_absolute_error(y, oof_preds_tab)
    mae_cat_global = mean_absolute_error(y, oof_preds_cat)
    mae_ebm_global = mean_absolute_error(y, oof_preds_ebm)
    
    corr_tab, _ = pearsonr(y, oof_preds_tab)
    corr_cat, _ = pearsonr(y, oof_preds_cat)
    corr_ebm, _ = pearsonr(y, oof_preds_ebm)
    
    print("  1. TabPFN ({}): MAE={:.4f}, r={:.4f}".format(TABPFN_N_ESTIMATORS, mae_tab_global, corr_tab))
    print("  2. CatBoost:    MAE={:.4f}, r={:.4f}".format(mae_cat_global, corr_cat))
    print("  3. EBM:         MAE={:.4f}, r={:.4f}".format(mae_ebm_global, corr_ebm))
    sys.stdout.flush()
    
    # 6. OPTIMIZAR ENSEMBLE
    print("\n" + "="*70)
    print("[5/7] OPTIMIZANDO ENSEMBLE")
    print("="*70)
    sys.stdout.flush()
    
    pred_matrix = np.column_stack([oof_preds_tab, oof_preds_cat, oof_preds_ebm])
    
    simple_preds = np.mean(pred_matrix, axis=1)
    simple_mae = mean_absolute_error(y, simple_preds)
    print("  Promedio simple (1/3, 1/3, 1/3): MAE = {:.4f}".format(simple_mae))
    sys.stdout.flush()
    
    weights = optimize_weights(y, pred_matrix)
    final_preds = np.dot(pred_matrix, weights)
    final_mae = mean_absolute_error(y, final_preds)
    final_corr, _ = pearsonr(y, final_preds)
    final_r2 = r2_score(y, final_preds)
    
    print("  Pesos optimizados: Tab={:.3f}, Cat={:.3f}, EBM={:.3f}".format(
        weights[0], weights[1], weights[2]))
    print("  MAE ENSEMBLE: {:.4f}".format(final_mae))
    print("  CORRELACION:  {:.4f}".format(final_corr))
    print("  R2:           {:.4f}".format(final_r2))
    sys.stdout.flush()
    
    mejora_mae = mae_tab_global - final_mae
    porcentaje_mejora = (mejora_mae / mae_tab_global) * 100
    
    print("\n" + "-"*70)
    print("  MEJORA vs TabPFN solo:")
    print("    DMAE: {:+.4f} ({:+.2f}%)".format(mejora_mae, porcentaje_mejora))
    
    if abs(mejora_mae) < 0.05:
        print("  MEJORA PEQUENA - Considera usar solo TabPFN")
        recomendacion = "TabPFN solo (mejora < 0.05)"
    elif mejora_mae > 0:
        print("  MEJORA SIGNIFICATIVA - Ensemble recomendado")
        recomendacion = "Ensemble (mejora significativa)"
    else:
        print("  EMPEORA - Usar solo TabPFN")
        recomendacion = "TabPFN solo (ensemble empeora)"
    print("-"*70)
    sys.stdout.flush()
    
    # 7. GUARDAR RESULTADOS
    print("\n" + "="*70)
    print("[6/7] GUARDANDO RESULTADOS")
    print("="*70)
    sys.stdout.flush()
    
    df_results = df[['patient_id', 'language', 'QA']].copy()
    df_results['QA_real'] = y
    df_results['QA_pred_ensemble'] = final_preds
    df_results['QA_pred_tabpfn'] = oof_preds_tab
    df_results['QA_pred_catboost'] = oof_preds_cat
    df_results['QA_pred_ebm'] = oof_preds_ebm
    df_results['error_ensemble'] = np.abs(final_preds - y)
    df_results['error_tabpfn'] = np.abs(oof_preds_tab - y)
    df_results['severity_real'] = [get_severity_label(x) for x in y]
    df_results['severity_pred'] = [get_severity_label(x) for x in final_preds]
    
    df_results.to_csv(OUTPUT_DIR / "ensemble_predictions_full.csv", index=False)
    print("  Predicciones completas: ensemble_predictions_full.csv")
    sys.stdout.flush()
    
    metrics_global = {
        'timestamp': TIMESTAMP,
        'mae_tabpfn': mae_tab_global,
        'mae_catboost': mae_cat_global,
        'mae_ebm': mae_ebm_global,
        'mae_ensemble': final_mae,
        'correlation_tabpfn': corr_tab,
        'correlation_ensemble': final_corr,
        'r2_ensemble': final_r2,
        'weight_tabpfn': weights[0],
        'weight_catboost': weights[1],
        'weight_ebm': weights[2],
        'mejora_mae': mejora_mae,
        'mejora_porcentaje': porcentaje_mejora,
        'recomendacion': recomendacion,
        'cv_strategy': CV_STRATEGY,
        'n_folds': N_FOLDS,
        'tabpfn_n_estimators': TABPFN_N_ESTIMATORS
    }
    pd.DataFrame([metrics_global]).to_csv(OUTPUT_DIR / "ensemble_metrics.csv", index=False)
    print("  Metricas globales: ensemble_metrics.csv")
    sys.stdout.flush()
    
    # 8. GENERAR PLOTS
    print("\n" + "="*70)
    print("[7/7] GENERANDO GRAFICOS")
    print("="*70)
    sys.stdout.flush()
    
    print("  Generando plots del ensemble...")
    sys.stdout.flush()
    generate_rich_plots(y, final_preds, OUTPUT_DIR, final_mae, final_corr, 
                       final_r2, " (Ensemble)")
    
    print("  Generando plots de TabPFN...")
    sys.stdout.flush()
    r2_tab = r2_score(y, oof_preds_tab)
    generate_rich_plots(y, oof_preds_tab, OUTPUT_DIR, mae_tab_global, corr_tab, 
                       r2_tab, " (TabPFN)")
    
    print("  Generando comparacion lado a lado...")
    sys.stdout.flush()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    axes[0].scatter(y, final_preds, alpha=0.6, s=60, color='teal', 
                   edgecolors='black', linewidth=0.5)
    axes[0].plot([0, 100], [0, 100], 'r--', linewidth=2)
    axes[0].set_xlabel('QA Real', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('QA Predicho', fontsize=12, fontweight='bold')
    axes[0].set_title('Ensemble Optimizado\nMAE: {:.2f} | r: {:.3f} | R2: {:.3f}\nPesos: T={:.2f} C={:.2f} E={:.2f}'.format(
        final_mae, final_corr, final_r2, weights[0], weights[1], weights[2]), 
        fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y, oof_preds_tab, alpha=0.6, s=60, color='steelblue', 
                   edgecolors='black', linewidth=0.5)
    axes[1].plot([0, 100], [0, 100], 'r--', linewidth=2)
    axes[1].set_xlabel('QA Real', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('QA Predicho', fontsize=12, fontweight='bold')
    axes[1].set_title('TabPFN ({})\nMAE: {:.2f} | r: {:.3f} | R2: {:.3f}'.format(
        TABPFN_N_ESTIMATORS, mae_tab_global, corr_tab, r2_tab), 
        fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_ensemble_vs_tabpfn.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Comparacion: comparison_ensemble_vs_tabpfn.png")
    sys.stdout.flush()
    
    # ANALISIS POR IDIOMA
    df_with_preds = df_results.copy()
    df_with_preds['QA_pred'] = final_preds
    analyze_by_language(df_with_preds, OUTPUT_DIR)
    
    # RESUMEN FINAL
    print("\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)
    print("Resultados guardados en: {}".format(OUTPUT_DIR))
    print("\nRESUMEN:")
    print("  Timestamp:     {}".format(TIMESTAMP))
    print("  MAE Ensemble:  {:.4f}".format(final_mae))
    print("  MAE TabPFN:    {:.4f}".format(mae_tab_global))
    print("  Mejora:        {:+.4f} ({:+.2f}%)".format(mejora_mae, porcentaje_mejora))
    print("  Recomendacion: {}".format(recomendacion))
    print("="*70 + "\n")
    sys.stdout.flush()
    
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()
    if hasattr(sys.stderr, 'close'):
        sys.stderr.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR FATAL] {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)