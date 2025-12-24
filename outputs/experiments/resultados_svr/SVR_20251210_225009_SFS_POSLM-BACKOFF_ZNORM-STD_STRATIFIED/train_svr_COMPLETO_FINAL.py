#!/usr/bin/env python3
# train_svr_COMPLETO_FINAL.py
# -*- coding: utf-8 -*-
"""
SVR COMPLETO FINAL - TODO INCLUIDO
- CV en PWA (con Control en train)
- Evaluación ESPAÑOL (si existe)
- Evaluación CATALÁN (si existe)
- SHAP + Permutation + Feature groups
- Todas las gráficas con métricas de precisión
- Calibración en TODOS los splits
- NOMBRES DESCRIPTIVOS EN OUTPUT
- EXCLUSIÓN MANUAL/AUTOMÁTICA DE FEATURES
- CV INTERNO CONFIGURABLE (--cv-inner)
- Z-NORMALIZACIÓN CON STATS DE CONTROLES (--znorm-controls) - CORREGIDO
"""

import os
import sys
import datetime
import pathlib
import shutil
import warnings
import argparse
import json
warnings.filterwarnings("ignore")

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Asegurar que se puede importar experiment_logger desde 05_models
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, pearsonr, ttest_ind
from sklearn.inspection import permutation_importance

# Logger de experimentos
from experiment_logger import (
    ExperimentLogger,
    create_experiment_config,
    create_experiment_results,
)

# Intentar importar SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("ADVERTENCIA: SHAP no instalado. Ejecuta: pip install shap")

# ======================== PATHS ========================
PROJECT_BASE = pathlib.Path(__file__).resolve().parent.parent
DATASET_CSV = PROJECT_BASE / "data" / "dataset_FINAL_CON_POSLM.csv"
RESULTS_BASE = PROJECT_BASE / "outputs" / "experiments" / "resultados_svr"

os.makedirs(RESULTS_BASE, exist_ok=True)

# ======================== SEVERITY ========================
SEVERITY_BINS = [0, 25, 50, 75, 100]
SEVERITY_LABELS = ['Very Severe', 'Severe', 'Moderate', 'Mild']

def qa_to_severity(qa_scores):
    return pd.cut(qa_scores, bins=SEVERITY_BINS, labels=SEVERITY_LABELS, include_lowest=True)

# ======================== LOGGER ========================
def set_logger(run_dir):
    log = logging.getLogger("SVR")
    log.setLevel(logging.INFO)
    
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "console.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    
    for h in list(log.handlers):
        log.removeHandler(h)
    log.addHandler(ch)
    log.addHandler(fh)
    
    return log

# ======================== EXTRACT SUBDATASET (NUEVO PARA SPLIT ESTRATIFICADO) ========================
def extract_subdataset_from_patient_id(patient_ids):
    """
    Extrae el nombre del sub-dataset desde patient_id.
    Como en Le et al. 2018: "we withhold 25% of speakers from each sub-dataset"
    
    Ejemplos:
        "adler01a" -> "adler"
        "TCU02a" -> "tcu"
        "kurland12" -> "kurland"
        "scale03a" -> "scale"
    
    Args:
        patient_ids: pandas Series o array de patient_ids
    
    Returns:
        numpy array de strings con subdatasets extraídos (lowercase, sin NaN)
    """
    if isinstance(patient_ids, pd.Series):
        s = patient_ids.astype(str)
    else:
        s = pd.Series(list(patient_ids), dtype=object).astype(str)

    sub = s.str.extract(r'^([A-Za-z]+)', expand=False).str.lower()
    sub = sub.fillna("unknown")
    sub = sub.replace(["nan", "none", "NaN", "None"], "unknown")

    return sub.astype(str).values

# ======================== Z-NORMALIZATION FUNCTIONS ========================
def compute_control_stats(X_control):
    """
    Calcula mean y std de controles para z-normalization.
    
    Args:
        X_control: Features de controles (n_controls, n_features)
    
    Returns:
        mean_control, std_control
    """
    X_control = np.asarray(X_control, dtype=float)
    
    mean_control = np.nanmean(X_control, axis=0)
    std_control = np.nanstd(X_control, axis=0, ddof=1)
    
    # Evitar división por cero
    std_control[std_control == 0] = 1.0
    std_control[np.isnan(std_control)] = 1.0
    
    return mean_control, std_control

def apply_znorm(X, mean, std):
    """
    Aplica z-normalization con mean y std dados.
    
    Args:
        X: Features a normalizar
        mean: Media para restar
        std: Std para dividir
    
    Returns:
        X_normalized
    """
    X = np.asarray(X, dtype=float)
    return (X - mean) / std

# ======================== FEATURES ========================
def get_simple_features():
    den_simple = [
        'den_words_per_min', 'den_phones_per_min', 'den_W', 'den_OCW',
        'den_words_utt_mean', 'den_phones_utt_mean',
        'den_nouns', 'den_verbs', 'den_nouns_per_verb', 'den_noun_ratio',
        'den_light_verbs', 'den_determiners', 'den_demonstratives',
        'den_prepositions', 'den_adjectives', 'den_adverbs',
        'den_pronoun_ratio', 'den_function_words'
    ]
    dys_simple = [
        'dys_fillers_per_min', 'dys_fillers_per_word', 'dys_fillers_per_phone',
        'dys_pauses_per_min', 'dys_long_pauses_per_min', 'dys_short_pauses_per_min',
        'dys_pauses_per_word', 'dys_long_pauses_per_word', 'dys_short_pauses_per_word',
        'dys_pause_sec_mean'
    ]
    lex_simple = ['lex_ttr']
    return den_simple + dys_simple + lex_simple

# ======================== POS-LM HELPERS ========================
POSLM_METHODS = {
    'kneser-ney': 'poslm_kn_',
    'backoff': 'poslm_bo_',
    'lstm': 'poslm_lstm_',
    'all': None
}

def get_poslm_features(df, method='kneser-ney'):
    """
    Obtiene columnas POS-LM según método elegido
    
    Args:
        method: 'kneser-ney', 'backoff', 'lstm', 'all', 'none'
    
    Returns:
        list: columnas POS-LM filtradas
    """
    # Obtener TODAS las columnas POS-LM disponibles
    all_poslm = [c for c in df.columns if c.startswith('poslm_')]
    
    if not all_poslm:
        return []
    
    # Sin POS-LM
    if method == 'none':
        return []
    
    # Todas las disponibles
    if method == 'all':
        return all_poslm
    
    # Filtrar por método específico
    method_prefix_map = {
        'kneser-ney': 'poslm_kn_',
        'backoff': 'poslm_bo_',
        'lstm': 'poslm_lstm_'
    }
    
    if method not in method_prefix_map:
        print(f"  Método POS-LM desconocido: '{method}'. Usando 'all'.")
        return all_poslm
    
    prefix = method_prefix_map[method]
    filtered = [c for c in all_poslm if c.startswith(prefix)]
    
    if not filtered:
        print(f"  No se encontraron columnas con prefijo '{prefix}'")
        print(f"    Columnas POS-LM disponibles:")
        for col in all_poslm[:5]:
            print(f"      - {col}")
        if len(all_poslm) > 5:
            print(f"      ... y {len(all_poslm) - 5} más")
    
    return filtered

# ======================== METRICAS ========================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    try:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
    except Exception:
        pearson_r, pearson_p = np.nan, np.nan
    
    try:
        spearman_rho, spearman_p = spearmanr(y_true, y_pred)
    except Exception:
        spearman_rho, spearman_p = np.nan, np.nan
    
    return {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'Pearson_r': pearson_r, 'Pearson_p': pearson_p,
        'Spearman_rho': spearman_rho, 'Spearman_p': spearman_p
    }

def compute_accuracy_metrics(y_true, y_pred_int):
    errors = np.abs(y_pred_int - y_true)
    return {
        'Acc@1': float(np.mean(errors <= 1)),
        'Acc@5': float(np.mean(errors <= 5)),
        'Acc@10': float(np.mean(errors <= 10)),
        'Exact': float(np.mean(errors == 0))
    }

def compute_severity_accuracy(y_true, y_pred):
    sev_true = qa_to_severity(y_true)
    sev_pred = qa_to_severity(y_pred)
    
    acc = accuracy_score(sev_true, sev_pred)
    cm = confusion_matrix(sev_true, sev_pred, labels=SEVERITY_LABELS)
    report = classification_report(sev_true, sev_pred, labels=SEVERITY_LABELS, output_dict=True, zero_division=0)
    
    return {'severity_accuracy': acc, 'confusion_matrix': cm, 'classification_report': report}

def to_int(pred):
    return np.rint(np.asarray(pred)).clip(0, 100).astype(int)

# ======================== VISUALIZACION ========================
def plot_scatter_with_precision(y_true, y_pred, title, out_png, metrics=None, acc_metrics=None):
    """Scatter plot con métricas de precisión incluidas"""
    if metrics is None:
        metrics = compute_metrics(y_true, y_pred)
    if acc_metrics is None:
        y_pred_int = to_int(y_pred)
        acc_metrics = compute_accuracy_metrics(y_true, y_pred_int)
    
    plt.figure(figsize=(9, 9))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Identidad")
    
    plt.xlabel("QA Real", fontsize=12, fontweight='bold')
    plt.ylabel("QA Predicho", fontsize=12, fontweight='bold')
    
    text = ("MAE:      {:.2f}\nRMSE:     {:.2f}\nR²:       {:.3f}\n"
            "Pearson:  {:.3f}\nSpearman: {:.3f}\n"
            "─────────────\n"
            "Acc@5:    {:.1f}%\nAcc@10:   {:.1f}%").format(
                metrics['MAE'], metrics['RMSE'], metrics['R2'],
                metrics['Pearson_r'], metrics['Spearman_rho'],
                100*acc_metrics['Acc@5'], 100*acc_metrics['Acc@10'])
    
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), 
             fontsize=10, family='monospace')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, title, out_png):
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS,
           ylabel='Severidad Real', xlabel='Severidad Predicha')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "{}\n({:.1f}%)".format(cm[i, j], cm_norm[i, j] * 100),
                   ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black", fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def plot_error_histogram(y_true, y_pred, title, out_png, threshold=5.316):
    """Histograma de errores con línea de partición (como Figura 6 del paper)"""
    errors = np.abs(y_true - y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    
    for i, patch in enumerate(patches):
        if bins[i] <= threshold:
            patch.set_facecolor('mediumseagreen')
        else:
            patch.set_facecolor('coral')
    
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
               label='Umbral (MAE={:.3f})'.format(threshold))
    
    low_errors = errors[errors <= threshold]
    high_errors = errors[errors > threshold]
    
    text_low = "Low Errors:\nN={}\nMAE={:.2f}±{:.2f}".format(
        len(low_errors), np.mean(low_errors), np.std(low_errors))
    text_high = "High Errors:\nN={}\nMAE={:.2f}±{:.2f}".format(
        len(high_errors), np.mean(high_errors), np.std(high_errors))
    
    ax.text(0.15, 0.95, text_low, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='mediumseagreen', alpha=0.3),
            fontsize=9, family='monospace')
    ax.text(0.65, 0.95, text_high, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='coral', alpha=0.3),
            fontsize=9, family='monospace')
    
    ax.set_xlabel('Error de Predicción QA', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    return len(low_errors), len(high_errors)

def analyze_error_groups(y_true, y_pred, df_ids, run_dir, log, threshold=5.316):
    """Análisis de grupos Low/High Error como en Table 11 del paper"""
    errors = np.abs(y_true - y_pred)
    
    low_mask = errors <= threshold
    high_mask = errors > threshold
    
    log.info("\n" + "="*70)
    log.info("ANÁLISIS POR GRUPOS DE ERROR (Threshold={:.3f})".format(threshold))
    log.info("="*70)
    
    low_errors = errors[low_mask]
    high_errors = errors[high_mask]
    low_qa = y_true[low_mask]
    high_qa = y_true[high_mask]
    
    log.info("\nGrupo Low Errors:")
    log.info("  N:          {}".format(len(low_errors)))
    log.info("  MAE:        {:.2f} ± {:.2f}".format(np.mean(low_errors), np.std(low_errors)))
    log.info("  QA real:    {:.1f} ± {:.1f}".format(np.mean(low_qa), np.std(low_qa)))
    
    log.info("\nGrupo High Errors:")
    log.info("  N:          {}".format(len(high_errors)))
    log.info("  MAE:        {:.2f} ± {:.2f}".format(np.mean(high_errors), np.std(high_errors)))
    log.info("  QA real:    {:.1f} ± {:.1f}".format(np.mean(high_qa), np.std(high_qa)))
    
    t_stat, p_value = ttest_ind(low_qa, high_qa, equal_var=False)
    log.info("\nDiferencia en QA real entre grupos:")
    log.info("  t-statistic: {:.3f}".format(t_stat))
    log.info("  p-value:     {:.4f}".format(p_value))
    
    analysis_df = pd.DataFrame({
        'Group': ['Low Errors', 'High Errors'],
        'N': [len(low_errors), len(high_errors)],
        'MAE_mean': [np.mean(low_errors), np.mean(high_errors)],
        'MAE_std': [np.std(low_errors), np.std(high_errors)],
        'QA_mean': [np.mean(low_qa), np.mean(high_qa)],
        'QA_std': [np.std(low_qa), np.std(high_qa)]
    })
    analysis_df.to_csv(run_dir / "error_groups_analysis.csv", index=False)
    
    return analysis_df

# ======================== EVALUACION ========================
def evaluate_split(y_true, y_pred, split_name, run_dir, log, df_ids=None):
    log.info("\n" + "="*70)
    log.info("EVALUACION: {}".format(split_name))
    log.info("="*70)
    
    metrics_cont = compute_metrics(y_true, y_pred)
    log.info("Metricas de regresion:")
    log.info("  MAE:        {:.3f}".format(metrics_cont['MAE']))
    log.info("  RMSE:       {:.3f}".format(metrics_cont['RMSE']))
    log.info("  R²:         {:.3f}".format(metrics_cont['R2']))
    log.info("  Pearson r:  {:.3f} (p={:.4f})".format(metrics_cont['Pearson_r'], metrics_cont['Pearson_p']))
    log.info("  Spearman:   {:.3f} (p={:.4f})".format(metrics_cont['Spearman_rho'], metrics_cont['Spearman_p']))
    
    y_pred_int = to_int(y_pred)
    metrics_int = compute_accuracy_metrics(y_true, y_pred_int)
    log.info("\nAccuracy por distancia:")
    log.info("  Acc@1:  {:.2f}%".format(100*metrics_int['Acc@1']))
    log.info("  Acc@5:  {:.2f}%".format(100*metrics_int['Acc@5']))
    log.info("  Acc@10: {:.2f}%".format(100*metrics_int['Acc@10']))
    log.info("  Exact:  {:.2f}%".format(100*metrics_int['Exact']))
    
    sev_metrics = compute_severity_accuracy(y_true, y_pred_int)
    log.info("\nClasificacion por severidad:")
    log.info("  Accuracy: {:.2f}%".format(100*sev_metrics['severity_accuracy']))
    
    report = sev_metrics['classification_report']
    log.info("\n  Por clase:")
    for label in SEVERITY_LABELS:
        if label in report:
            log.info("    {}: Prec={:.2f}%, Rec={:.2f}%, F1={:.2f}%".format(
                label, 100*report[label]['precision'],
                100*report[label]['recall'], 100*report[label]['f1-score']))
    
    all_metrics = {**metrics_cont, **metrics_int, 'severity_accuracy': sev_metrics['severity_accuracy']}
    pd.DataFrame([all_metrics]).to_csv(run_dir / "{}_metrics.csv".format(split_name), index=False)
    
    pd.DataFrame(sev_metrics['classification_report']).transpose().to_csv(
        run_dir / "{}_severity_report.csv".format(split_name))
    
    pd.DataFrame(sev_metrics['confusion_matrix'], index=SEVERITY_LABELS, columns=SEVERITY_LABELS).to_csv(
        run_dir / "{}_confusion_matrix.csv".format(split_name))
    
    preds_df = pd.DataFrame({
        'QA_real': y_true, 'QA_pred_cont': y_pred, 'QA_pred_int': y_pred_int,
        'severity_real': qa_to_severity(y_true), 'severity_pred': qa_to_severity(y_pred_int),
        'error_cont': y_pred - y_true, 'error_int': y_pred_int - y_true,
        'abs_error_int': np.abs(y_pred_int - y_true)
    })
    
    if df_ids is not None:
        for col in df_ids.columns:
            preds_df[col] = df_ids[col].values
    
    preds_df.to_csv(run_dir / "{}_predictions.csv".format(split_name), index=False)
    
    log.info("\nGenerando graficos...")
    
    plot_scatter_with_precision(y_true, y_pred, "{} - Scatter Plot".format(split_name),
                                run_dir / "{}_scatter.png".format(split_name), 
                                metrics_cont, metrics_int)
    
    plot_confusion_matrix(sev_metrics['confusion_matrix'],
                         "{} - Confusion Matrix".format(split_name),
                         run_dir / "{}_confusion_matrix.png".format(split_name))
    
    log.info("Guardados: {}_*.csv/png".format(split_name))
    
    return metrics_cont, metrics_int, sev_metrics

# ======================== INTERPRETABILIDAD ========================
def compute_feature_importance_permutation(model, X, y, feature_names, run_dir, log, n_repeats=10):
    log.info("\n" + "="*70)
    log.info("PERMUTATION IMPORTANCE")
    log.info("="*70)
    log.info("Calculando (n_repeats={})...".format(n_repeats))
    
    perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42,
                                             scoring='neg_mean_absolute_error', n_jobs=1)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    importance_df.to_csv(run_dir / "feature_importance_permutation.csv", index=False)
    
    log.info("\nTop 20 features:")
    for idx, (i, row) in enumerate(importance_df.head(20).iterrows(), 1):
        log.info("  {:2d}. {:40s} {:.4f} +/- {:.4f}".format(
            idx, row['feature'], row['importance_mean'], row['importance_std']))
    
    plt.figure(figsize=(10, 12))
    top_n = min(30, len(importance_df))
    top_features = importance_df.head(top_n)
    
    plt.barh(range(top_n), top_features['importance_mean'].values, 
             xerr=top_features['importance_std'].values, alpha=0.8, color='steelblue', edgecolor='black')
    plt.yticks(range(top_n), top_features['feature'].values, fontsize=9)
    plt.xlabel('Permutation Importance', fontsize=12, fontweight='bold')
    plt.title('Top {} Features - Permutation Importance'.format(top_n), fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(run_dir / "feature_importance_permutation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("\nGuardado: feature_importance_permutation.csv/png")
    
    return importance_df

def compute_shap_values(model, X_train, X_test, feature_names, run_dir, log, max_samples=100):
    if not SHAP_AVAILABLE:
        log.warning("SHAP no instalado. Saltando análisis SHAP...")
        return None
    
    log.info("\n" + "="*70)
    log.info("SHAP VALUES")
    log.info("="*70)
    log.info("Calculando SHAP values...")
    
    if len(X_train) > max_samples:
        log.info("  Usando {} samples random como background".format(max_samples))
        np.random.seed(42)
        bg_idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_bg = X_train[bg_idx]
    else:
        X_bg = X_train
        log.info("  Usando {} samples como background".format(len(X_bg)))
    
    log.info("  Creando SHAP explainer...")
    explainer = shap.KernelExplainer(model.predict, X_bg)
    
    if len(X_test) > max_samples:
        log.info("  Calculando SHAP para {} samples de test...".format(max_samples))
        np.random.seed(42)
        test_idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_test_sample = X_test[test_idx]
    else:
        log.info("  Calculando SHAP para {} samples de test...".format(len(X_test)))
        X_test_sample = X_test
    
    shap_values = explainer.shap_values(X_test_sample)
    
    pd.DataFrame(shap_values, columns=feature_names).to_csv(run_dir / "shap_values.csv", index=False)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(run_dir / "feature_importance_shap.csv", index=False)
    
    log.info("\nTop 20 features (SHAP):")
    for idx, (i, row) in enumerate(importance_df.head(20).iterrows(), 1):
        log.info("  {:2d}. {:40s} {:.4f}".format(idx, row['feature'], row['importance']))
    
    log.info("\nGenerando graficos SHAP...")
    
    try:
        plt.figure(figsize=(10, 12))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                          show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(run_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 12))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(run_dir / "shap_summary_bar.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        log.info("  Guardado: shap_summary_*.png")
    except Exception as e:
        log.warning("  Error graficos SHAP: {}".format(e))
    
    log.info("Guardado: shap_values.csv, feature_importance_shap.csv")
    
    return importance_df

def analyze_feature_groups(importance_df, run_dir, log, method_name=""):
    if importance_df is None or len(importance_df) == 0:
        return None
    
    log.info("\n" + "="*70)
    log.info("ANALISIS POR GRUPOS{}".format(" ({})".format(method_name) if method_name else ""))
    log.info("="*70)
    
    def get_group(feat):
        if feat.startswith('den_'): 
            return 'DEN'
        elif feat.startswith('dys_'): 
            return 'DYS'
        elif feat.startswith('lex_'): 
            return 'LEX'
        elif feat.startswith('poslm_'):
            return 'POSLM'
        else: 
            return 'OTHER'
    
    importance_df = importance_df.copy()
    importance_df['group'] = importance_df['feature'].apply(get_group)
    
    imp_col = 'importance_mean' if 'importance_mean' in importance_df.columns else 'importance'
    group_stats = importance_df.groupby('group')[imp_col].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
    
    log.info("\nImportancia por grupo:")
    log.info("  {:<8s} {:>10s} {:>10s} {:>8s} {:>10s}".format(
        "Grupo", "Total", "Media", "Count", "% Total"))
    log.info("  " + "-"*60)
    
    total_imp = group_stats['sum'].sum()
    for grp in group_stats.index:
        total = group_stats.loc[grp, 'sum']
        mean = group_stats.loc[grp, 'mean']
        count = int(group_stats.loc[grp, 'count'])
        pct = 100 * total / total_imp if total_imp > 0 else 0
        log.info("  {:<8s} {:>10.4f} {:>10.4f} {:>8d} {:>9.1f}%".format(
            grp, total, mean, count, pct))
    
    plt.figure(figsize=(10, 6))
    colors = {'DEN': 'steelblue', 'DYS': 'coral', 'LEX': 'mediumseagreen', 'POSLM': 'goldenrod', 'OTHER': 'gray'}
    bar_colors = [colors.get(g, 'gray') for g in group_stats.index]
    
    plt.bar(group_stats.index, group_stats['sum'], alpha=0.8, edgecolor='black', linewidth=1.5, color=bar_colors)
    plt.xlabel('Grupo de Features', fontsize=12, fontweight='bold')
    plt.ylabel('Importancia Total', fontsize=12, fontweight='bold')
    
    title = 'Importancia por Grupo de Features'
    if method_name:
        title += ' ({})'.format(method_name)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    filename = "feature_importance_by_group"
    if method_name:
        filename += "_{}".format(method_name.lower().replace(" ", "_"))
    plt.savefig(run_dir / "{}.png".format(filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    group_stats.to_csv(run_dir / "{}.csv".format(filename))
    
    log.info("\nGuardado: {}.csv/png".format(filename))
    
    return group_stats

# ======================== FEATURE SELECTION ========================
def perform_sfs(X_all, y_all, feat_cols, run_dir, log, use_znorm_controls, X_control_all, max_features=40):
    try:
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    except ImportError:
        log.error("mlxtend no instalado: pip install mlxtend")
        sys.exit(1)
    
    log.info("\n" + "="*70)
    log.info("SEQUENTIAL FORWARD SELECTION")
    log.info("="*70)
    
    # ============ FILTRAR FEATURES VACÍAS ============
    valid_mask = ~np.isnan(X_all).all(axis=0)
    n_invalid = (~valid_mask).sum()
    
    if n_invalid > 0:
        log.warning("\n  Encontradas {} features completamente vacías".format(n_invalid))
        log.warning("   Estas features serán excluidas del SFS:")
        invalid_features = []
        for i, (is_valid, feat) in enumerate(zip(valid_mask, feat_cols)):
            if not is_valid:
                invalid_features.append(feat)
                log.warning("     [{}] {}".format(i, feat))
        
        X_all = X_all[:, valid_mask]
        if X_control_all is not None:
            X_control_all = X_control_all[:, valid_mask]
        feat_cols_filtered = [feat_cols[i] for i in range(len(feat_cols)) if valid_mask[i]]
        
        log.info("\n✓ Features válidas para SFS: {} (de {} originales)".format(
            len(feat_cols_filtered), len(feat_cols)))
        
        with open(run_dir / "sfs_excluded_features.txt", "w") as f:
            f.write("FEATURES EXCLUIDAS DE SFS (completamente vacías)\n")
            f.write("="*70 + "\n")
            f.write("Total excluidas: {}\n\n".format(n_invalid))
            for feat in invalid_features:
                f.write("  - {}\n".format(feat))
    else:
        feat_cols_filtered = feat_cols
        log.info("\n✓ Todas las features son válidas para SFS")
    # ================================================
    
    log.info("\nIniciando SFS...")
    
    # Pre-procesar según tipo de normalización
    if use_znorm_controls and X_control_all is not None:
        log.info("  SFS con z-norm de controles")
        
        # Imputar
        imputer = SimpleImputer(strategy="median")
        X_all_imputed = imputer.fit_transform(X_all)
        X_control_imputed = imputer.transform(X_control_all)
        
        # Calcular stats de controles
        mean_ctrl, std_ctrl = compute_control_stats(X_control_imputed)
        
        # Normalizar
        X_all_norm = apply_znorm(X_all_imputed, mean_ctrl, std_ctrl)
        
        # SFS sobre datos normalizados (sin pipeline)
        svr = SVR(C=10, epsilon=0.1, kernel='rbf', gamma='scale')
        sfs = SFS(svr, k_features=(1, max_features), forward=True, floating=False,
                  scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=2)
        sfs.fit(X_all_norm, y_all)
    else:
        log.info("  SFS con StandardScaler")
        
        # Pipeline tradicional
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svr", SVR(C=10, epsilon=0.1, kernel='rbf', gamma='scale'))
        ])
        
        sfs = SFS(pipe, k_features=(1, max_features), forward=True, floating=False,
                  scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=2)
        sfs.fit(X_all, y_all)
    
    best_k = len(sfs.k_feature_names_)
    best_features_idx = list(sfs.k_feature_idx_)
    best_features_list = [feat_cols_filtered[i] for i in best_features_idx]
    best_score = -sfs.k_score_
    
    log.info("\n" + "="*70)
    log.info("RESULTADOS SFS")
    log.info("="*70)
    log.info("Número óptimo: {} features".format(best_k))
    log.info("MAE: {:.3f}".format(best_score))
    
    with open(run_dir / "selected_features_sfs.txt", "w") as f:
        f.write("FEATURES SELECCIONADAS POR SFS\n")
        f.write("="*70 + "\n")
        f.write("Total: {}\nMAE: {:.3f}\n\n".format(best_k, best_score))
        for feat in best_features_list:
            f.write("{}\n".format(feat))
    
    return best_features_idx, best_features_list

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description='SVR Training COMPLETO')
    parser.add_argument('--features', type=str, default='full',
                       choices=['simple', 'full', 'sfs'],
                       help='Feature selection: simple (basic), full (all), sfs (sequential forward selection)')
    parser.add_argument('--poslm-method', type=str, default='none',
                       choices=['none', 'kneser-ney', 'backoff', 'lstm', 'all'],
                       help='POS-LM method: none, kneser-ney, backoff, lstm, all')
    
    # CV interno para GridSearchCV
    parser.add_argument('--cv-inner', type=int, default=5,
                       help='Número de folds para el GridSearchCV interno (búsqueda de hiperparámetros)')
    
    # ============ Z-NORMALIZATION CON STATS DE CONTROLES ============
    parser.add_argument('--znorm-controls', action='store_true',
                       help='Usar z-normalización con estadísticas de controles (como en Le et al. 2018)')
    # ================================================================
    
    # ============ EXCLUIR FEATURES MANUALMENTE ============
    parser.add_argument('--exclude-features', type=str, default=None,
                       help='Archivo con features a excluir (una por línea) o lista separada por comas')
    parser.add_argument('--exclude-empty', action='store_true',
                       help='Excluir automáticamente features completamente vacías')
    # =======================================================
    
    # ============ NOTAS PARA EL EXPERIMENTO (LOGGER) ============
    parser.add_argument('--notes', type=str, default="",
                       help='Notas para el experimento (se guardan en el logger)')
    # =============================================================
    
    args = parser.parse_args()
    
    # ============ NOMBRE DESCRIPTIVO ============
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config_name = args.features.upper()
    if args.poslm_method != 'none':
        poslm_label = args.poslm_method.upper().replace('-', '')
        config_name += f"_POSLM-{poslm_label}"
    else:
        config_name += "_NO-POSLM"
    
    # Añadir indicador de z-norm
    if args.znorm_controls:
        config_name += "_ZNORM-CTRL"
    else:
        config_name += "_ZNORM-STD"
    
    # Añadir indicador de split estratificado
    config_name += "_STRATIFIED"
    
    run_name = f"SVR_{ts}_{config_name}"
    run_dir = pathlib.Path(RESULTS_BASE) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log = set_logger(run_dir)
    log.info("="*70)
    log.info(f"SVR COMPLETO FINAL - {config_name}")
    log.info("="*70)
    log.info(f"Output: {run_dir}")
    log.info("Configuration:")
    log.info(f"  Features:          {args.features}")
    log.info(f"  POS-LM method:     {args.poslm_method}")
    log.info(f"  CV inner folds:    {args.cv_inner}")
    log.info(f"  Z-norm controls:   {args.znorm_controls}")
    log.info(f"  Stratified CV:     True (25% de cada sub-dataset)")
    log.info(f"  Dataset:           {DATASET_CSV.name}")
    log.info(f"  Exclude empty:     {args.exclude_empty}")
    if args.exclude_features:
        log.info(f"  Exclude file/list: {args.exclude_features}")
    if args.notes:
        log.info(f"  Notes:             {args.notes}")
    
    if SHAP_AVAILABLE:
        log.info("  SHAP:              disponible")
    else:
        log.info("  SHAP:              NO disponible (pip install shap)")
    
    # ============ GUARDAR CONFIG ============
    config_info = {
        'timestamp': ts,
        'config_name': config_name,
        'features': args.features,
        'poslm_method': args.poslm_method,
        'cv_inner': args.cv_inner,
        'znorm_controls': args.znorm_controls,
        'stratified_cv': True,
        'exclude_empty': args.exclude_empty,
        'exclude_features': args.exclude_features,
        'output_dir': str(run_dir),
        'dataset': str(DATASET_CSV),
        'shap_available': SHAP_AVAILABLE,
        'notes': args.notes,
    }
    
    with open(run_dir / "CONFIG.txt", "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("CONFIGURACIÓN DEL EXPERIMENTO\n")
        f.write("="*70 + "\n\n")
        f.write(f"Nombre:          {config_name}\n")
        f.write(f"Fecha/hora:      {ts}\n")
        f.write(f"Features:        {args.features}\n")
        f.write(f"POS-LM method:   {args.poslm_method}\n")
        f.write(f"CV inner folds:  {args.cv_inner}\n")
        f.write(f"Z-norm controls: {args.znorm_controls}\n")
        f.write(f"Stratified CV:   True (25% de cada sub-dataset)\n")
        f.write(f"Exclude empty:   {args.exclude_empty}\n")
        if args.exclude_features:
            f.write(f"Exclude file:    {args.exclude_features}\n")
        if args.notes:
            f.write(f"Notas:           {args.notes}\n")
        f.write(f"Dataset:         {config_info['dataset']}\n")
        f.write(f"Output dir:      {config_info['output_dir']}\n")
        f.write(f"SHAP:            {'Disponible' if SHAP_AVAILABLE else 'NO disponible'}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("COMANDO PARA REPLICAR:\n")
        f.write("="*70 + "\n")
        f.write("python3 05_models/train_svr_COMPLETO_FINAL.py \\\n")
        f.write(f"    --features {args.features} \\\n")
        f.write(f"    --poslm-method {args.poslm_method} \\\n")
        f.write(f"    --cv-inner {args.cv_inner}")
        if args.znorm_controls:
            f.write(" \\\n    --znorm-controls")
        if args.exclude_empty:
            f.write(" \\\n    --exclude-empty")
        if args.exclude_features:
            f.write(f" \\\n    --exclude-features {args.exclude_features}")
        if args.notes:
            f.write(f" \\\n    --notes \"{args.notes}\"")
        f.write("\n\n" + "="*70 + "\n")
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_info, f, indent=2)
    
    log.info("\nConfig guardada: CONFIG.txt, config.json")
    
    try:
        shutil.copy2(__file__, run_dir / pathlib.Path(__file__).name)
    except Exception:
        pass
    
    # ==================== CARGAR DATOS ====================
    log.info("\n" + "="*70)
    log.info("CARGANDO DATOS")
    log.info("="*70)
    
    if not DATASET_CSV.exists():
        log.error("No existe: {}".format(DATASET_CSV))
        sys.exit(1)
    
    df = pd.read_csv(DATASET_CSV)
    log.info("Total registros: {}".format(len(df)))
    
    # PARCHE AphasiaBank – Español (TCU02a, TCU04a, TCU06a, TCU10a)
    es_ids = ["TCU02a", "TCU04a", "TCU06a", "TCU10a"]
    if "patient_id" in df.columns:
        try:
            log.info("\nPacientes ES (TCU) antes del parche:")
            log.info(
                "\n%s",
                df.loc[df["patient_id"].isin(es_ids),
                       ["patient_id", "language", "group", "QA"]].to_string(index=False)
            )
        except Exception:
            pass

        mask_fix = df["patient_id"].isin(["TCU06a", "TCU10a"])
        df.loc[mask_fix, "group"] = "pwa"

        try:
            log.info("\nPacientes ES (TCU) después del parche:")
            log.info(
                "\n%s",
                df.loc[df["patient_id"].isin(es_ids),
                       ["patient_id", "language", "group", "QA"]].to_string(index=False)
            )
        except Exception:
            pass
    
    df = df[df['QA'].notna()].copy()
    log.info("Con QA valido: {}".format(len(df)))
    
    has_language = 'language' in df.columns
    subsets_pwa_en = None
    
    if has_language:
        log.info("\nDistribucion por idioma:")
        for lang, count in df['language'].value_counts(dropna=False).items():
            log.info("  {}: {}".format(lang, count))
    
    # ==================== SPLIT ====================
    log.info("\n" + "="*70)
    log.info("SPLIT")
    log.info("="*70)
    
    if 'group' not in df.columns:
        log.warning("\nNo hay columna 'group', usando todos como PWA")
        df_pwa = df.copy()
        df_control = pd.DataFrame()
    else:
        df_pwa = df[df['group'] == 'pwa'].copy()
        df_control = df[df['group'] == 'control'].copy()
        
        log.info("\nSplit:")
        log.info("  PWA:     {}".format(len(df_pwa)))
        log.info("  Control: {}".format(len(df_control)))
    
    if len(df_pwa) < 10:
        log.error("Muy pocos PWA (n={})".format(len(df_pwa)))
        sys.exit(1)
    
    # Splits por idioma
    df_en = pd.DataFrame()
    df_es = pd.DataFrame()
    df_ca = pd.DataFrame()
    
    if has_language:
        langs = df_pwa["language"].astype(str).str.lower()
        
        if (langs == "en").any():
            df_en = df_pwa[langs == "en"].copy()
        else:
            df_en = df_pwa.copy()
        
        mask_es = langs.isin(["es", "spanish"])
        df_es = df_pwa[mask_es].copy() if mask_es.any() else pd.DataFrame()
        
        mask_ca = langs.isin(["ca", "catalan"])
        df_ca = df_pwa[mask_ca].copy() if mask_ca.any() else pd.DataFrame()

        # ============ EXTRAER SUB-DATASETS (NUEVO) ============
        if len(df_en) > 0 and 'patient_id' in df_en.columns:
            subsets_pwa_en = extract_subdataset_from_patient_id(df_en["patient_id"])

            # Sanity check adicional: asegurar strings limpios
            subsets_series = pd.Series(subsets_pwa_en, dtype=object)
            subsets_series = subsets_series.fillna("unknown")
            subsets_series = subsets_series.replace(["nan", "none", "NaN", "None"], "unknown")
            subsets_pwa_en = subsets_series.astype(str).values

            log.info("\nSub-datasets extraídos de patient_id:")
            unique_subsets = np.unique(subsets_pwa_en)
            log.info("  Total sub-datasets únicos: {}".format(len(unique_subsets)))
            log.info("  Sub-datasets: {}".format(', '.join(unique_subsets)))
            
            log.info("\n  Distribución de pacientes por sub-dataset:")
            for subset, count in pd.Series(subsets_pwa_en).value_counts().items():
                log.info("    {}: {} pacientes".format(subset, count))
        else:
            subsets_pwa_en = None
        # ======================================================
        
        log.info("\nPor idioma (PWA):")
        log.info("  EN (train): {}".format(len(df_en)))
        if len(df_es) > 0:
            log.info("  ES (eval):  {}".format(len(df_es)))
        if len(df_ca) > 0:
            log.info("  CA (eval):  {}".format(len(df_ca)))
    else:
        df_en = df_pwa.copy()
        df_es = pd.DataFrame()
        df_ca = pd.DataFrame()
        subsets_pwa_en = None
        log.info("\nSin idioma, usando todos PWA como EN")
    
    # Info para el logger
    df_info = {
        "n_total": len(df),
        "n_pwa": len(df_pwa),
        "n_control": len(df_control),
        "n_en": len(df_en),
        "n_es": len(df_es),
        "n_ca": len(df_ca),
    }
    
    # ==================== FEATURES ====================
    log.info("\n" + "="*70)
    log.info("FEATURES")
    log.info("="*70)

    base_feat_cols = sorted([c for c in df.columns if c.startswith(('den_', 'dys_', 'lex_'))])

    poslm_feat_cols = []
    if args.poslm_method != 'none':
        poslm_feat_cols = get_poslm_features(df, method=args.poslm_method)
        
        if poslm_feat_cols:
            log.info(f"\nPOS-LM método: {args.poslm_method}")
            log.info(f"  Columnas POS-LM encontradas: {len(poslm_feat_cols)}")
        else:
            log.warning(f"\n  No se encontraron columnas POS-LM para '{args.poslm_method}'")
            log.warning("    Columnas disponibles:")
            available = [c for c in df.columns if c.startswith('poslm_')]
            if available:
                prefixes = set('_'.join(c.split('_')[:2]) + '_' for c in available)
                for prefix in sorted(prefixes):
                    count = len([c for c in available if c.startswith(prefix)])
                    log.warning(f"      {prefix}: {count} columnas")
            else:
                log.warning("      (No hay columnas POS-LM en el dataset)")

    all_feat_cols = base_feat_cols + poslm_feat_cols

    # ============ EXCLUIR FEATURES ============
    features_to_exclude = set()
    
    if args.exclude_empty:
        log.info("\n Detectando features vacías...")
        X_temp = df[all_feat_cols].values
        empty_mask = np.isnan(X_temp).all(axis=0)
        empty_features = [all_feat_cols[i] for i, is_empty in enumerate(empty_mask) if is_empty]
        
        if empty_features:
            log.info(f"   Encontradas {len(empty_features)} features vacías:")
            for feat in empty_features:
                log.info(f"     - {feat}")
            features_to_exclude.update(empty_features)
        else:
            log.info("   ✓ No se encontraron features vacías")
    
    if args.exclude_features:
        log.info("\n Procesando exclusiones manuales...")
        
        if ',' in args.exclude_features or not os.path.exists(args.exclude_features):
            manual_exclude = [f.strip() for f in args.exclude_features.split(',')]
            log.info(f"   Desde argumentos: {len(manual_exclude)} features")
        else:
            with open(args.exclude_features, 'r') as f:
                manual_exclude = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            log.info(f"   Desde archivo '{args.exclude_features}': {len(manual_exclude)} features")
        
        invalid = [f for f in manual_exclude if f not in all_feat_cols]
        if invalid:
            log.warning(f"     {len(invalid)} features no encontradas en el dataset:")
            for feat in invalid[:5]:
                log.warning(f"       - {feat}")
            if len(invalid) > 5:
                log.warning(f"       ... y {len(invalid) - 5} más")
        
        valid_exclude = [f for f in manual_exclude if f in all_feat_cols]
        if valid_exclude:
            log.info(f"   ✓ {len(valid_exclude)} features válidas para excluir")
            features_to_exclude.update(valid_exclude)
    
    if features_to_exclude:
        all_feat_cols_original = all_feat_cols.copy()
        all_feat_cols = [f for f in all_feat_cols if f not in features_to_exclude]
        
        log.info("\n RESUMEN DE EXCLUSIONES:")
        log.info(f"   Features originales: {len(all_feat_cols_original)}")
        log.info(f"   Features excluidas:  {len(features_to_exclude)}")
        log.info(f"   Features finales:    {len(all_feat_cols)}")
        
        exclude_file = run_dir / "excluded_features.txt"
        with open(exclude_file, "w") as f:
            f.write("FEATURES EXCLUIDAS\n")
            f.write("="*70 + "\n")
            f.write(f"Total excluidas: {len(features_to_exclude)}\n\n")
            for feat in sorted(features_to_exclude):
                f.write(f"  - {feat}\n")
        log.info(f"   Lista guardada: {exclude_file.name}")
    # ===========================================

    if args.features == 'simple':
        simple_base = [
            'den_words_per_min', 'den_phones_per_min', 'den_W', 'den_OCW',
            'den_words_utt_mean', 'den_phones_utt_mean',
            'den_nouns', 'den_verbs', 'den_nouns_per_verb', 'den_noun_ratio',
            'den_light_verbs', 'den_determiners', 'den_demonstratives',
            'den_prepositions', 'den_adjectives', 'den_adverbs',
            'den_pronoun_ratio', 'den_function_words',
            'dys_fillers_per_min', 'dys_fillers_per_word', 'dys_fillers_per_phone',
            'dys_pauses_per_min', 'dys_long_pauses_per_min', 'dys_short_pauses_per_min',
            'dys_pauses_per_word', 'dys_long_pauses_per_word', 'dys_short_pauses_per_word',
            'dys_pause_sec_mean',
            'lex_ttr'
        ]
        
        feat_cols = [f for f in simple_base if f in df.columns]
        
        if args.poslm_method != 'none' and poslm_feat_cols:
            if args.poslm_method in ['kneser-ney', 'all']:
                feat_cols.extend([f for f in ['poslm_kn_bigram_ce_mean', 'poslm_kn_trigram_ce_mean'] 
                                if f in df.columns])
            if args.poslm_method in ['backoff', 'all']:
                feat_cols.extend([f for f in ['poslm_bo_bigram_ce_mean', 'poslm_bo_trigram_ce_mean'] 
                                if f in df.columns])
            if args.poslm_method in ['lstm', 'all']:
                feat_cols.extend([f for f in ['poslm_lstm_ce_mean', 'poslm_lstm_ppl_mean'] 
                                if f in df.columns])

    elif args.features == 'full':
        feat_cols = all_feat_cols

    elif args.features == 'sfs':
        X_pwa_en_all = df_en[all_feat_cols].values
        y_pwa_en_all = df_en['QA'].values
        
        if len(df_control) > 0:
            X_control_all = df_control[all_feat_cols].values
            y_control_all = df_control['QA'].values
            X_all = np.vstack([X_pwa_en_all, X_control_all])
            y_all = np.concatenate([y_pwa_en_all, y_control_all])
        else:
            X_control_all = None
            X_all = X_pwa_en_all
            y_all = y_pwa_en_all
        
        selected_idx, feat_cols = perform_sfs(X_all, y_all, all_feat_cols, run_dir, log, 
                                              args.znorm_controls, X_control_all)

    den_cols = [f for f in feat_cols if f.startswith('den_')]
    dys_cols = [f for f in feat_cols if f.startswith('dys_')]
    lex_cols = [f for f in feat_cols if f.startswith('lex_')]
    poslm_cols = [f for f in feat_cols if f.startswith('poslm_')]

    log.info("\nFeatures:")
    log.info(f"  DEN:   {len(den_cols)}")
    log.info(f"  DYS:   {len(dys_cols)}")
    log.info(f"  LEX:   {len(lex_cols)}")
    log.info(f"  POSLM: {len(poslm_cols)}")
    log.info(f"  TOTAL: {len(feat_cols)}")

    with open(run_dir / "FEATURES.txt", "w", encoding="utf-8") as f:
        f.write(f"FEATURES ({args.features.upper()})\n")
        f.write("="*70 + "\n")
        f.write(f"Total: {len(feat_cols)}\n\n")
        
        f.write(f"DEN ({len(den_cols)}):\n")
        for c in sorted(den_cols):
            f.write(f"  - {c}\n")
        
        f.write(f"\nDYS ({len(dys_cols)}):\n")
        for c in sorted(dys_cols):
            f.write(f"  - {c}\n")
        
        f.write(f"\nLEX ({len(lex_cols)}):\n")
        for c in sorted(lex_cols):
            f.write(f"  - {c}\n")
        
        if poslm_cols:
            f.write(f"\nPOSLM ({len(poslm_cols)}):\n")
            for c in sorted(poslm_cols):
                f.write(f"  - {c}\n")
    
    # ==================== PREPARAR DATOS ====================
    X_pwa_en = df_en[feat_cols].values
    y_pwa_en = df_en['QA'].values
    groups_pwa_en = df_en['patient_id'].values if 'patient_id' in df_en.columns else np.arange(len(df_en))
    
    if len(df_control) > 0:
        X_control = df_control[feat_cols].values
        y_control = df_control['QA'].values
    else:
        X_control = None
        y_control = None
    
    # ==================== PARAM GRID ====================
    # param_grid = {
    #     "C": [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    #     "epsilon": [1.0, 1e-1, 1e-2, 1e-3],
    #     "kernel": ["rbf", "linear"],
    #     "shrinking": [True, False],
    #     "gamma": ["scale"],
    # }

    param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "epsilon": [0.01, 0.1, 1, 5, 10],
    "kernel": ["rbf"],
    "shrinking": [True],
    "gamma": ["scale", "auto"],
    }

    # ==================== CV EN INGLÉS ====================
    log.info("\n" + "="*70)
    if subsets_pwa_en is not None:
        log.info("CROSS-VALIDATION (EN) - StratifiedGroupKFold por sub-dataset")
        log.info("  * Cada fold tiene 25% de pacientes de CADA sub-dataset")
        cv_splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        split_generator = cv_splitter.split(X_pwa_en, subsets_pwa_en, groups=groups_pwa_en)
    else:
        log.info("CROSS-VALIDATION (EN) - GroupKFold por paciente")
        log.warning("  * No se encontró columna de sub-dataset, usando GroupKFold estándar")
        cv_splitter = GroupKFold(n_splits=4)
        split_generator = cv_splitter.split(X_pwa_en, y_pwa_en, groups=groups_pwa_en)
    
    if args.znorm_controls:
        log.info("Normalización: Z-norm con stats de controles (Le et al. 2018)")
    else:
        log.info("Normalización: StandardScaler (sklearn default)")
    
    log.info("="*70)
    
    cv_results = []
    cv_preds = np.zeros_like(y_pwa_en, dtype=float)
    
    # Lista para guardar stats de cada fold (solo si znorm_controls)
    fold_stats = []
    
    # Lista para guardar distribución de sub-datasets por fold
    subdataset_distributions = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(split_generator, 1):
        log.info("\n  Fold {}/4:".format(fold_idx))
        
        # ============ LOGGING DE SUB-DATASETS POR FOLD (NUEVO) ============
        if subsets_pwa_en is not None:
            train_subsets = subsets_pwa_en[train_idx]
            test_subsets = subsets_pwa_en[test_idx]
            
            train_dist = pd.Series(train_subsets).value_counts().to_dict()
            test_dist = pd.Series(test_subsets).value_counts().to_dict()
            
            log.info("    Train sub-datasets: {}".format(train_dist))
            log.info("    Test sub-datasets:  {}".format(test_dist))
            
            # Guardar para CSV
            for subset in np.unique(subsets_pwa_en):
                subdataset_distributions.append({
                    'fold': fold_idx,
                    'split': 'train',
                    'subdataset': subset,
                    'count': train_dist.get(subset, 0)
                })
                subdataset_distributions.append({
                    'fold': fold_idx,
                    'split': 'test',
                    'subdataset': subset,
                    'count': test_dist.get(subset, 0)
                })
        # ==================================================================
        # ============ PREPARAR DATOS DEL FOLD ============
        if X_control is not None and len(X_control) > 0:
            X_train_fold_raw = np.vstack([X_pwa_en[train_idx], X_control])
            y_train_fold = np.concatenate([y_pwa_en[train_idx], y_control])
            n_control_fold = len(X_control)
        else:
            X_train_fold_raw = X_pwa_en[train_idx]
            y_train_fold = y_pwa_en[train_idx]
            n_control_fold = 0
        
        X_test_fold_raw = X_pwa_en[test_idx]
        y_test_fold = y_pwa_en[test_idx]
        
        # ============ IMPUTAR ============
        imputer = SimpleImputer(strategy="median")
        X_train_fold_imputed = imputer.fit_transform(X_train_fold_raw)
        X_test_fold_imputed = imputer.transform(X_test_fold_raw)
        
        # ============ NORMALIZAR ============
        if args.znorm_controls and n_control_fold > 0:
            # Separar controles DESPUÉS de imputar
            X_control_fold = X_train_fold_imputed[-n_control_fold:]
            
            # Calcular stats SOLO de controles
            mean_fold, std_fold = compute_control_stats(X_control_fold)
            
            # Guardar stats para referencia
            fold_stats.append({
                'fold': fold_idx,
                'mean_control': mean_fold,
                'std_control': std_fold
            })
            
            # Normalizar con stats de controles
            X_train_fold_norm = apply_znorm(X_train_fold_imputed, mean_fold, std_fold)
            X_test_fold_norm = apply_znorm(X_test_fold_imputed, mean_fold, std_fold)
            
        else:
            # StandardScaler normal
            scaler = StandardScaler()
            X_train_fold_norm = scaler.fit_transform(X_train_fold_imputed)
            X_test_fold_norm = scaler.transform(X_test_fold_imputed)
        
        # ============ GRIDSEARCH (solo optimiza SVR) ============
        svr = SVR()
        gs_fold = GridSearchCV(
            svr,
            param_grid,
            cv=args.cv_inner,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=0
        )
        
        gs_fold.fit(X_train_fold_norm, y_train_fold)
        
        # Predicciones
        cv_preds[test_idx] = gs_fold.predict(X_test_fold_norm)
        
        mae_fold = mean_absolute_error(y_test_fold, cv_preds[test_idx])
        log.info("    MAE: {:.3f}".format(mae_fold))
        log.info("    Best params: C={}, epsilon={}, kernel={}".format(
            gs_fold.best_params_['C'],
            gs_fold.best_params_['epsilon'],
            gs_fold.best_params_['kernel']
        ))
        
        cv_results.append({
            'fold': fold_idx,
            'mae': mae_fold,
            'best_C': gs_fold.best_params_['C'],
            'best_epsilon': gs_fold.best_params_['epsilon'],
            'best_kernel': gs_fold.best_params_['kernel']
        })
    
    pd.DataFrame(cv_results).to_csv(run_dir / "cv_results_by_fold.csv", index=False)
    
    # Guardar distribución de sub-datasets por fold
    if subdataset_distributions:
        pd.DataFrame(subdataset_distributions).to_csv(run_dir / "subdataset_distribution.csv", index=False)
        log.info("\nDistribución de sub-datasets guardada: subdataset_distribution.csv")
    
    # Guardar stats de normalización por fold (si se usó znorm_controls)
    if args.znorm_controls and fold_stats:
        # Guardar mean/std de cada fold
        stats_df_list = []
        for stat_dict in fold_stats:
            fold_num = stat_dict['fold']
            mean_arr = stat_dict['mean_control']
            std_arr = stat_dict['std_control']
            
            for i, feat_name in enumerate(feat_cols):
                stats_df_list.append({
                    'fold': fold_num,
                    'feature': feat_name,
                    'mean_control': mean_arr[i],
                    'std_control': std_arr[i]
                })
        
        pd.DataFrame(stats_df_list).to_csv(run_dir / "znorm_stats_by_fold.csv", index=False)
        log.info("\nStats de z-norm guardadas: znorm_stats_by_fold.csv")
    
    metrics_cv_cont, metrics_cv_acc, metrics_cv_sev = evaluate_split(
        y_pwa_en, cv_preds, "CV_PWA", run_dir, log,
        df_ids=df_en[['patient_id']] if 'patient_id' in df_en.columns else None
    )
    
    # ==================== CALIBRACION ====================
    log.info("\n" + "="*70)
    log.info("CALIBRACION")
    log.info("="*70)
    
    calibrator = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
    calibrator.fit(cv_preds, y_pwa_en)
    joblib.dump(calibrator, run_dir / "calibrator.pkl")
    
    cv_preds_cal = calibrator.predict(cv_preds)
    metrics_cv_cal_cont, metrics_cv_cal_acc, metrics_cv_cal_sev = evaluate_split(
        y_pwa_en, cv_preds_cal, "CV_PWA_CALIBRATED", run_dir, log,
        df_ids=df_en[['patient_id']] if 'patient_id' in df_en.columns else None
    )
    
    # ==================== ANÁLISIS DE ERROR ====================
    log.info("\n" + "="*70)
    log.info("ANÁLISIS DE ERROR (Como Figura 6 del paper)")
    log.info("="*70)
    
    plot_error_histogram(
        y_pwa_en, cv_preds_cal,
        "Distribución de Errores de Predicción QA",
        run_dir / "error_histogram.png",
        threshold=5.316
    )
    
    analyze_error_groups(
        y_pwa_en, cv_preds_cal,
        df_en[['patient_id']] if 'patient_id' in df_en.columns else None,
        run_dir, log, threshold=5.316
    )
    
    log.info("\nGuardado: error_histogram.png, error_groups_analysis.csv")
    
    # ==================== MODELO FINAL ====================
    log.info("\n" + "="*70)
    log.info("MODELO FINAL")
    log.info("="*70)
    
    if X_control is not None and len(X_control) > 0:
        X_final_raw = np.vstack([X_pwa_en, X_control])
        y_final = np.concatenate([y_pwa_en, y_control])
        n_control_final = len(X_control)
        log.info("Entrenando con: {} PWA EN + {} Control".format(len(X_pwa_en), len(X_control)))
    else:
        X_final_raw = X_pwa_en
        y_final = y_pwa_en
        n_control_final = 0
        log.info("Entrenando con: {} PWA EN".format(len(X_pwa_en)))
    
    # Imputar
    imputer_final = SimpleImputer(strategy="median")
    X_final_imputed = imputer_final.fit_transform(X_final_raw)
    
    # Normalizar
    if args.znorm_controls and n_control_final > 0:
        X_control_final = X_final_imputed[-n_control_final:]
        mean_final, std_final = compute_control_stats(X_control_final)
        X_final_norm = apply_znorm(X_final_imputed, mean_final, std_final)
        
        # Guardar stats del modelo final
        final_stats_df = pd.DataFrame({
            'feature': feat_cols,
            'mean_control': mean_final,
            'std_control': std_final
        })
        final_stats_df.to_csv(run_dir / "znorm_stats_final_model.csv", index=False)
        log.info("Stats de z-norm del modelo final guardadas: znorm_stats_final_model.csv")
    else:
        scaler_final = StandardScaler()
        X_final_norm = scaler_final.fit_transform(X_final_imputed)
    
    # GridSearch
    svr_final = SVR()
    gs_final = GridSearchCV(
        svr_final,
        param_grid,
        cv=args.cv_inner,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0
    )
    
    gs_final.fit(X_final_norm, y_final)
    
    # Guardar modelo final (necesitamos reconstruir el pipeline completo)
    if args.znorm_controls and n_control_final > 0:
        # Crear un "modelo" que incluya imputer, znorm y svr
        final_model_dict = {
            'imputer': imputer_final,
            'znorm_mean': mean_final,
            'znorm_std': std_final,
            'svr': gs_final.best_estimator_,
            'feature_names': feat_cols,
            'znorm_type': 'controls'
        }
    else:
        final_model_dict = {
            'imputer': imputer_final,
            'scaler': scaler_final,
            'svr': gs_final.best_estimator_,
            'feature_names': feat_cols,
            'znorm_type': 'standard'
        }
    
    joblib.dump(final_model_dict, run_dir / "model_final.pkl")
    pd.DataFrame(gs_final.cv_results_).to_csv(run_dir / "cv_results_full.csv", index=False)
    
    log.info("Mejores parámetros:")
    for key, val in gs_final.best_params_.items():
        log.info("  {}: {}".format(key, val))
    
    # ==================== INTERPRETABILIDAD ====================
    log.info("\n" + "="*70)
    log.info("INTERPRETABILIDAD")
    log.info("="*70)
    
    # Para interpretabilidad, necesitamos un "modelo" que tome X_norm
    # Vamos a crear un wrapper simple
    class ModelWrapper:
        def __init__(self, svr_model):
            self.svr = svr_model
        
        def fit(self, X, y):
            # No hace nada, el modelo ya está entrenado
            return self
        
        def predict(self, X):
            return self.svr.predict(X)
        
        def score(self, X, y):
            # Para compatibilidad con sklearn
            from sklearn.metrics import r2_score
            return r2_score(y, self.predict(X))
        
        def score(self, X, y):
            # Para compatibilidad con sklearn
            from sklearn.metrics import r2_score
            return r2_score(y, self.predict(X))
    
    # Usamos los datos normalizados del modelo final
    model_wrapper = ModelWrapper(gs_final.best_estimator_)
    
    perm_importance = compute_feature_importance_permutation(
        model_wrapper, X_final_norm, y_final, feat_cols, run_dir, log, n_repeats=10
    )
    
    analyze_feature_groups(perm_importance, run_dir, log, method_name="Permutation")
    
    if SHAP_AVAILABLE:
        try:
            # Usar una muestra para SHAP (es muy lento)
            n_samples_shap = min(100, len(X_final_norm))
            np.random.seed(42)
            sample_idx = np.random.choice(len(X_final_norm), n_samples_shap, replace=False)
            
            X_train_shap = X_final_norm[:int(0.8*n_samples_shap)]
            X_test_shap = X_final_norm[int(0.8*n_samples_shap):n_samples_shap]
            
            shap_importance = compute_shap_values(
                model_wrapper, X_train_shap, X_test_shap,
                feat_cols, run_dir, log, max_samples=50
            )
            
            if shap_importance is not None:
                analyze_feature_groups(shap_importance, run_dir, log, method_name="SHAP")
        except Exception as e:
            log.warning("Error en SHAP: {}".format(e))
    
    # ==================== EVALUACIÓN ESPAÑOL ====================
    metrics_es = None
    if len(df_es) > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACION ESPAÑOL")
        log.info("="*70)
        
        X_es_raw = df_es[feat_cols].values
        y_es = df_es['QA'].values
        
        log.info("Evaluando {} pacientes españoles...".format(len(df_es)))
        
        # Pre-procesar igual que en entrenamiento
        X_es_imputed = imputer_final.transform(X_es_raw)
        
        if args.znorm_controls and n_control_final > 0:
            X_es_norm = apply_znorm(X_es_imputed, mean_final, std_final)
        else:
            X_es_norm = scaler_final.transform(X_es_imputed)
        
        preds_es = gs_final.best_estimator_.predict(X_es_norm)
        metrics_es_raw_cont, metrics_es_raw_acc, metrics_es_raw_sev = evaluate_split(
            y_es, preds_es, "EVAL_ES_RAW", run_dir, log,
            df_ids=df_es[['patient_id']] if 'patient_id' in df_es.columns else None
        )
        
        preds_es_cal = calibrator.predict(preds_es)
        metrics_es_cal_cont, metrics_es_cal_acc, metrics_es_cal_sev = evaluate_split(
            y_es, preds_es_cal, "EVAL_ES_CALIBRATED", run_dir, log,
            df_ids=df_es[['patient_id']] if 'patient_id' in df_es.columns else None
        )
        
        metrics_es = {
            "raw_mae": metrics_es_raw_cont["MAE"],
            "cal_mae": metrics_es_cal_cont["MAE"],
        }
    else:
        log.info("\n" + "="*70)
        log.info("EVALUACION ESPAÑOL")
        log.info("="*70)
        log.info("No hay datos en español para evaluar")
    
    # ==================== EVALUACIÓN CATALÁN ====================
    metrics_ca = None
    if len(df_ca) > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACION CATALÁN")
        log.info("="*70)
        
        X_ca_raw = df_ca[feat_cols].values
        y_ca = df_ca['QA'].values
        
        log.info("Evaluando {} pacientes catalanes...".format(len(df_ca)))
        
        # Pre-procesar igual que en entrenamiento
        X_ca_imputed = imputer_final.transform(X_ca_raw)
        
        if args.znorm_controls and n_control_final > 0:
            X_ca_norm = apply_znorm(X_ca_imputed, mean_final, std_final)
        else:
            X_ca_norm = scaler_final.transform(X_ca_imputed)
        
        preds_ca = gs_final.best_estimator_.predict(X_ca_norm)
        metrics_ca_raw_cont, metrics_ca_raw_acc, metrics_ca_raw_sev = evaluate_split(
            y_ca, preds_ca, "EVAL_CA_RAW", run_dir, log,
            df_ids=df_ca[['patient_id']] if 'patient_id' in df_ca.columns else None
        )
        
        preds_ca_cal = calibrator.predict(preds_ca)
        metrics_ca_cal_cont, metrics_ca_cal_acc, metrics_ca_cal_sev = evaluate_split(
            y_ca, preds_ca_cal, "EVAL_CA_CALIBRATED", run_dir, log,
            df_ids=df_ca[['patient_id']] if 'patient_id' in df_ca.columns else None
        )
        
        metrics_ca = {
            "raw_mae": metrics_ca_raw_cont["MAE"],
            "cal_mae": metrics_ca_cal_cont["MAE"],
        }
    else:
        log.info("\n" + "="*70)
        log.info("EVALUACION CATALÁN")
        log.info("="*70)
        log.info("No hay datos en catalán para evaluar")
    
    # ==================== RESUMEN FINAL ====================
    log.info("\n" + "="*70)
    log.info("PROCESO COMPLETADO")
    log.info("="*70)
    log.info("Resultados guardados en: {}".format(run_dir))
    log.info("\nArchivos generados:")
    log.info("  - Configuración: CONFIG.txt, config.json")
    if features_to_exclude:
        log.info("  - Exclusiones: excluded_features.txt")
    log.info("  - Modelo: model_final.pkl")
    log.info("  - Calibrador: calibrator.pkl")
    if args.znorm_controls:
        log.info("  - Stats z-norm: znorm_stats_by_fold.csv, znorm_stats_final_model.csv")
    log.info("  - CV inglés: CV_PWA_*.csv/png")
    log.info("  - CV calibrado: CV_PWA_CALIBRATED_*.csv/png")
    if len(df_es) > 0:
        log.info("  - Español: EVAL_ES_*.csv/png")
    if len(df_ca) > 0:
        log.info("  - Catalán: EVAL_CA_*.csv/png")
    log.info("  - Interpretabilidad: feature_importance_*.csv/png")
    if SHAP_AVAILABLE:
        log.info("  - SHAP: shap_values.csv, shap_summary_*.png")
    else:
        log.info("  - SHAP: NO DISPONIBLE (instalar con: pip install shap)")
    log.info("  - Análisis de error: error_histogram.png, error_groups_analysis.csv")
    
    # ==================== LOGGER DE EXPERIMENTOS ====================
    log.info("\n" + "="*70)
    log.info("REGISTRANDO EXPERIMENTO EN EXPERIMENT_LOGGER")
    log.info("="*70)
    
    metrics_cv_all = metrics_cv_cont.copy()
    metrics_cv_all["severity_accuracy"] = metrics_cv_sev["severity_accuracy"]
    
    metrics_cv_cal_all = metrics_cv_cal_cont.copy()
    metrics_cv_cal_all["severity_accuracy"] = metrics_cv_cal_sev["severity_accuracy"]
    
    exp_config = create_experiment_config(args, param_grid, df_info, feat_cols)
    exp_config["znorm_controls"] = args.znorm_controls
    exp_config["stratified_cv"] = True
    
    exp_results = create_experiment_results(
        metrics_cv_all,
        metrics_cv_cal_all,
        metrics_es=metrics_es,
        metrics_ca=metrics_ca,
        best_params=gs_final.best_params_
    )
    
    exp_logger = ExperimentLogger(log_dir=RESULTS_BASE)
    exp_logger.log_experiment(exp_config, exp_results, run_dir=str(run_dir))
    
    log.info("Experimento registrado en experiments_history.csv/json")
    
    # ==================== COMPARACIÓN CON EL PAPER ====================
    log.info("\n" + "="*70)
    log.info("COMPARACIÓN CON LE ET AL. (2018)")
    log.info("="*70)
    log.info("\nResultados del paper (Table 9, Combined protocol):")
    log.info("  MAE:        8.86 (Oracle) / 9.18 (Auto)")
    log.info("  Correlación: .801 (Oracle) / .799 (Auto)")
    log.info("\nNuestros resultados (CV_PWA_CALIBRATED):")
    try:
        metrics_df = pd.read_csv(run_dir / "CV_PWA_CALIBRATED_metrics.csv")
        our_mae = metrics_df['MAE'].values[0]
        our_corr = metrics_df['Pearson_r'].values[0]
        log.info("  MAE:        {:.2f}".format(our_mae))
        log.info("  Correlación: {:.3f}".format(our_corr))
        
        diff_mae = our_mae - 9.18
        diff_corr = our_corr - 0.799
        log.info("\nDiferencia con el paper:")
        log.info("  ΔMAE:  {:+.2f} ({})".format(
            diff_mae, 
            "mejor" if diff_mae < 0 else "peor" if diff_mae > 0 else "igual"
        ))
        log.info("  ΔCorr: {:+.3f} ({})".format(
            diff_corr,
            "mejor" if diff_corr > 0 else "peor" if diff_corr < 0 else "igual"
        ))
    except Exception:
        log.info("  [Error leyendo métricas]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUMPIDO]")
        sys.exit(1)
    except Exceaption as e:
        print("\n\n[ERROR] {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
