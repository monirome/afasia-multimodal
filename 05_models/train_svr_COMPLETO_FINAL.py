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
"""

import os
import sys
import datetime
import pathlib
import shutil
import warnings
import argparse
warnings.filterwarnings("ignore")

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, pearsonr, ttest_ind
from sklearn.inspection import permutation_importance

# Intentar importar SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("ADVERTENCIA: SHAP no instalado. Ejecuta: pip install shap")

# ======================== PATHS ========================
PROJECT_BASE = pathlib.Path(__file__).resolve().parent.parent
DATASET_CSV = PROJECT_BASE / "data" / "dataset_FINAL_COMPLETO.csv"
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

# ======================== METRICAS ========================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    try:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
    except:
        pearson_r, pearson_p = np.nan, np.nan
    
    try:
        spearman_rho, spearman_p = spearmanr(y_true, y_pred)
    except:
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
    
    # Texto con métricas incluyendo precisión
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
    
    # Histograma
    n, bins, patches = ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Colorear barras según threshold
    for i, patch in enumerate(patches):
        if bins[i] <= threshold:
            patch.set_facecolor('mediumseagreen')
        else:
            patch.set_facecolor('coral')
    
    # Línea de partición
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
               label='Umbral (MAE={:.3f})'.format(threshold))
    
    # Estadísticas por grupo
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
    
    # Dividir en grupos
    low_mask = errors <= threshold
    high_mask = errors > threshold
    
    log.info("\n" + "="*70)
    log.info("ANÁLISIS POR GRUPOS DE ERROR (Threshold={:.3f})".format(threshold))
    log.info("="*70)
    
    # Estadísticas por grupo
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
    
    # Test estadístico
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(low_qa, high_qa, equal_var=False)
    log.info("\nDiferencia en QA real entre grupos:")
    log.info("  t-statistic: {:.3f}".format(t_stat))
    log.info("  p-value:     {:.4f}".format(p_value))
    
    # Guardar análisis
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
    
    # Usar la nueva función con métricas de precisión
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
    
    # Limitar samples para acelerar cálculo
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
    
    # Guardar valores SHAP
    pd.DataFrame(shap_values, columns=feature_names).to_csv(run_dir / "shap_values.csv", index=False)
    
    # Calcular importancia basada en SHAP
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
        # Summary plot beeswarm
        plt.figure(figsize=(10, 12))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                         show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(run_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Summary plot bar
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
        if feat.startswith('den_'): return 'DEN'
        elif feat.startswith('dys_'): return 'DYS'
        elif feat.startswith('lex_'): return 'LEX'
        else: return 'OTHER'
    
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
    colors = {'DEN': 'steelblue', 'DYS': 'coral', 'LEX': 'mediumseagreen', 'OTHER': 'gray'}
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
def perform_sfs(X_all, y_all, feat_cols, run_dir, log, max_features=40):
    try:
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    except ImportError:
        log.error("mlxtend no instalado: pip install mlxtend")
        sys.exit(1)
    
    log.info("\n" + "="*70)
    log.info("SEQUENTIAL FORWARD SELECTION")
    log.info("="*70)
    log.info("\nIniciando SFS...\n")
    
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
    best_features_list = [feat_cols[i] for i in best_features_idx]
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
                       help='Feature selection')
    args = parser.parse_args()
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(RESULTS_BASE) / "SVR_FINAL_{}_{}_features".format(ts, args.features.upper())
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log = set_logger(run_dir)
    log.info("="*70)
    log.info("SVR COMPLETO FINAL - {} FEATURES".format(args.features.upper()))
    log.info("="*70)
    log.info("Output: {}".format(run_dir))
    
    if SHAP_AVAILABLE:
        log.info("SHAP disponible: SI")
    else:
        log.info("SHAP disponible: NO (instalar con: pip install shap)")
    
    try:
        shutil.copy2(__file__, run_dir / pathlib.Path(__file__).name)
    except:
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
    
    df = df[df['QA'].notna()].copy()
    log.info("Con QA valido: {}".format(len(df)))
    
    has_language = 'language' in df.columns
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
        # Inglés para entrenamiento
        df_en = df_pwa[df_pwa['language'] == 'en'].copy() if 'en' in df_pwa['language'].values else df_pwa.copy()
        # Español para evaluación
        df_es = df_pwa[df_pwa['language'] == 'spanish'].copy() if 'spanish' in df_pwa['language'].values else pd.DataFrame()
        # Catalán para evaluación  
        df_ca = df_pwa[df_pwa['language'] == 'catalan'].copy() if 'catalan' in df_pwa['language'].values else pd.DataFrame()
        
        log.info("\nPor idioma (PWA):")
        log.info("  EN (train): {}".format(len(df_en)))
        if len(df_es) > 0:
            log.info("  ES (eval):  {}".format(len(df_es)))
        if len(df_ca) > 0:
            log.info("  CA (eval):  {}".format(len(df_ca)))
    else:
        df_en = df_pwa.copy()
        log.info("\nSin idioma, usando todos PWA como EN")
    
    # ==================== FEATURES ====================
    log.info("\n" + "="*70)
    log.info("FEATURES")
    log.info("="*70)
    
    all_feat_cols = sorted([c for c in df.columns if c.startswith(('den_', 'dys_', 'lex_'))])
    
    if args.features == 'simple':
        feat_cols = [f for f in get_simple_features() if f in df.columns]
    elif args.features == 'full':
        feat_cols = all_feat_cols
    elif args.features == 'sfs':
        # Usar PWA EN + Control para SFS
        X_pwa_en = df_en[all_feat_cols].values
        y_pwa_en = df_en['QA'].values
        
        if len(df_control) > 0:
            X_control_all = df_control[all_feat_cols].values
            y_control_all = df_control['QA'].values
            X_all = np.vstack([X_pwa_en, X_control_all])
            y_all = np.concatenate([y_pwa_en, y_control_all])
        else:
            X_all = X_pwa_en
            y_all = y_pwa_en
        
        selected_idx, feat_cols = perform_sfs(X_all, y_all, all_feat_cols, run_dir, log)
    
    den_cols = [f for f in feat_cols if f.startswith('den_')]
    dys_cols = [f for f in feat_cols if f.startswith('dys_')]
    lex_cols = [f for f in feat_cols if f.startswith('lex_')]
    
    log.info("\nFeatures:")
    log.info("  DEN: {}".format(len(den_cols)))
    log.info("  DYS: {}".format(len(dys_cols)))
    log.info("  LEX: {}".format(len(lex_cols)))
    log.info("  TOTAL: {}".format(len(feat_cols)))
    
    with open(run_dir / "features_used.txt", "w", encoding="utf-8") as f:
        f.write("FEATURES ({})\n".format(args.features.upper()))
        f.write("="*70 + "\n")
        f.write("Total: {}\n\n".format(len(feat_cols)))
        f.write("DEN ({}):\n".format(len(den_cols)))
        for c in den_cols: f.write("  - {}\n".format(c))
        f.write("\nDYS ({}):\n".format(len(dys_cols)))
        for c in dys_cols: f.write("  - {}\n".format(c))
        f.write("\nLEX ({}):\n".format(len(lex_cols)))
        for c in lex_cols: f.write("  - {}\n".format(c))
    
    # ==================== PREPARAR DATOS ====================
    # Solo usamos EN para entrenamiento/CV
    X_pwa_en = df_en[feat_cols].values
    y_pwa_en = df_en['QA'].values
    groups_pwa_en = df_en['patient_id'].values if 'patient_id' in df_en.columns else np.arange(len(df_en))
    
    if len(df_control) > 0:
        X_control = df_control[feat_cols].values
        y_control = df_control['QA'].values
    else:
        X_control = None
        y_control = None
    
    # ==================== MODELO ====================
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])
    
    param_grid = {
        "svr__C": [0.1, 1, 10, 100, 1000],
        "svr__epsilon": [0.01, 0.1, 1, 5, 10],
        "svr__kernel": ["rbf"],
        "svr__gamma": ["scale", "auto"],
    }
    
    # ==================== CV EN INGLÉS ====================
    log.info("\n" + "="*70)
    log.info("CROSS-VALIDATION (EN)")
    log.info("="*70)
    
    gkf = GroupKFold(n_splits=4)
    cv_results = []
    cv_preds = np.zeros_like(y_pwa_en, dtype=float)
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_pwa_en, y_pwa_en, groups=groups_pwa_en), 1):
        log.info("\n  Fold {}/4:".format(fold_idx))
        
        if X_control is not None and len(X_control) > 0:
            X_train_fold = np.vstack([X_pwa_en[train_idx], X_control])
            y_train_fold = np.concatenate([y_pwa_en[train_idx], y_control])
        else:
            X_train_fold = X_pwa_en[train_idx]
            y_train_fold = y_pwa_en[train_idx]
        
        gs_fold = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0)
        gs_fold.fit(X_train_fold, y_train_fold)
        cv_preds[test_idx] = gs_fold.predict(X_pwa_en[test_idx])
        
        mae_fold = mean_absolute_error(y_pwa_en[test_idx], cv_preds[test_idx])
        log.info("    MAE: {:.3f}".format(mae_fold))
        
        cv_results.append({'fold': fold_idx, 'mae': mae_fold})
    
    pd.DataFrame(cv_results).to_csv(run_dir / "cv_results_by_fold.csv", index=False)
    
    evaluate_split(y_pwa_en, cv_preds, "CV_PWA", run_dir, log,
                  df_ids=df_en[['patient_id']] if 'patient_id' in df_en.columns else None)
    
    # ==================== CALIBRACION ====================
    log.info("\n" + "="*70)
    log.info("CALIBRACION")
    log.info("="*70)
    
    calibrator = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
    calibrator.fit(cv_preds, y_pwa_en)
    joblib.dump(calibrator, run_dir / "calibrator.pkl")
    
    cv_preds_cal = calibrator.predict(cv_preds)
    evaluate_split(y_pwa_en, cv_preds_cal, "CV_PWA_CALIBRATED", run_dir, log,
                  df_ids=df_en[['patient_id']] if 'patient_id' in df_en.columns else None)
    
    # ==================== ANÁLISIS DE ERROR (PAPER FIGURA 6) ====================
    log.info("\n" + "="*70)
    log.info("ANÁLISIS DE ERROR (Como Figura 6 del paper)")
    log.info("="*70)
    
    # Histograma de errores
    plot_error_histogram(y_pwa_en, cv_preds_cal, 
                        "Distribución de Errores de Predicción QA",
                        run_dir / "error_histogram.png",
                        threshold=5.316)
    
    # Análisis de grupos de error
    analyze_error_groups(y_pwa_en, cv_preds_cal, 
                        df_en[['patient_id']] if 'patient_id' in df_en.columns else None,
                        run_dir, log, threshold=5.316)
    
    log.info("\nGuardado: error_histogram.png, error_groups_analysis.csv")
    
    # ==================== MODELO FINAL ====================
    log.info("\n" + "="*70)
    log.info("MODELO FINAL")
    log.info("="*70)
    
    if X_control is not None and len(X_control) > 0:
        X_final = np.vstack([X_pwa_en, X_control])
        y_final = np.concatenate([y_pwa_en, y_control])
        log.info("Entrenando con: {} PWA EN + {} Control".format(len(X_pwa_en), len(X_control)))
    else:
        X_final = X_pwa_en
        y_final = y_pwa_en
        log.info("Entrenando con: {} PWA EN".format(len(X_pwa_en)))
    
    gs_final = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0)
    gs_final.fit(X_final, y_final)
    
    best_model = gs_final.best_estimator_
    joblib.dump(best_model, run_dir / "model_final.pkl")
    pd.DataFrame(gs_final.cv_results_).to_csv(run_dir / "cv_results_full.csv", index=False)
    
    log.info("Mejores parámetros:")
    for key, val in gs_final.best_params_.items():
        log.info("  {}: {}".format(key, val))
    
    # ==================== INTERPRETABILIDAD ====================
    log.info("\n" + "="*70)
    log.info("INTERPRETABILIDAD")
    log.info("="*70)
    
    # Permutation importance
    perm_importance = compute_feature_importance_permutation(
        best_model, X_pwa_en, y_pwa_en, feat_cols, run_dir, log, n_repeats=10)
    
    analyze_feature_groups(perm_importance, run_dir, log, method_name="Permutation")
    
    # SHAP
    if SHAP_AVAILABLE:
        try:
            # Usar subset para SHAP
            n_test_shap = min(100, max(int(len(X_pwa_en) * 0.2), 20))
            X_train_shap = X_pwa_en[:-n_test_shap]
            X_test_shap = X_pwa_en[-n_test_shap:]
            
            shap_importance = compute_shap_values(best_model, X_train_shap, X_test_shap, 
                                                 feat_cols, run_dir, log, max_samples=100)
            
            if shap_importance is not None:
                analyze_feature_groups(shap_importance, run_dir, log, method_name="SHAP")
        except Exception as e:
            log.warning("Error en SHAP: {}".format(e))
    
    # ==================== EVALUACIÓN ESPAÑOL ====================
    if len(df_es) > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACION ESPAÑOL")
        log.info("="*70)
        
        X_es = df_es[feat_cols].values
        y_es = df_es['QA'].values
        
        log.info("Evaluando {} pacientes españoles...".format(len(df_es)))
        
        # Raw predictions
        preds_es = best_model.predict(X_es)
        evaluate_split(y_es, preds_es, "EVAL_ES_RAW", run_dir, log,
                      df_ids=df_es[['patient_id']] if 'patient_id' in df_es.columns else None)
        
        # Calibrated predictions
        preds_es_cal = calibrator.predict(preds_es)
        evaluate_split(y_es, preds_es_cal, "EVAL_ES_CALIBRATED", run_dir, log,
                      df_ids=df_es[['patient_id']] if 'patient_id' in df_es.columns else None)
    else:
        log.info("\n" + "="*70)
        log.info("EVALUACION ESPAÑOL")
        log.info("="*70)
        log.info("No hay datos en español para evaluar")
    
    # ==================== EVALUACIÓN CATALÁN ====================
    if len(df_ca) > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACION CATALÁN")
        log.info("="*70)
        
        X_ca = df_ca[feat_cols].values
        y_ca = df_ca['QA'].values
        
        log.info("Evaluando {} pacientes catalanes...".format(len(df_ca)))
        
        # Raw predictions
        preds_ca = best_model.predict(X_ca)
        evaluate_split(y_ca, preds_ca, "EVAL_CA_RAW", run_dir, log,
                      df_ids=df_ca[['patient_id']] if 'patient_id' in df_ca.columns else None)
        
        # Calibrated predictions
        preds_ca_cal = calibrator.predict(preds_ca)
        evaluate_split(y_ca, preds_ca_cal, "EVAL_CA_CALIBRATED", run_dir, log,
                      df_ids=df_ca[['patient_id']] if 'patient_id' in df_ca.columns else None)
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
    log.info("  - Modelo: model_final.pkl")
    log.info("  - Calibrador: calibrator.pkl")
    log.info("  - CV inglés: CV_PWA_*.csv/png")
    log.info("  - CV calibrado: CV_PWA_CALIBRATED_*.csv/png")
    if 'df_es' in locals() and len(df_es) > 0:
        log.info("  - Español: EVAL_ES_*.csv/png")
    if 'df_ca' in locals() and len(df_ca) > 0:
        log.info("  - Catalán: EVAL_CA_*.csv/png")
    log.info("  - Interpretabilidad: feature_importance_*.csv/png")
    if SHAP_AVAILABLE:
        log.info("  - SHAP: shap_values.csv, shap_summary_*.png")
    else:
        log.info("  - SHAP: NO DISPONIBLE (instalar con: pip install shap)")
    log.info("  - Análisis de error: error_histogram.png, error_groups_analysis.csv")
    
    # ==================== EVALUACIÓN ESPAÑOL ====================
    if len(df_es) > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACIÓN ESPAÑOL")
        log.info("="*70)
        
        X_es = df_es[feat_cols].values
        y_es = df_es['QA'].values
        
        log.info("Evaluando {} pacientes españoles...".format(len(df_es)))
        
        # Raw predictions
        preds_es = best_model.predict(X_es)
        evaluate_split(y_es, preds_es, "EVAL_ES_RAW", run_dir, log,
                      df_ids=df_es[['patient_id']] if 'patient_id' in df_es.columns else None)
        
        # Calibrated predictions
        preds_es_cal = calibrator.predict(preds_es)
        evaluate_split(y_es, preds_es_cal, "EVAL_ES_CALIBRATED", run_dir, log,
                      df_ids=df_es[['patient_id']] if 'patient_id' in df_es.columns else None)
    else:
        log.info("\n" + "="*70)
        log.info("EVALUACIÓN ESPAÑOL")
        log.info("="*70)
        log.info("No hay datos en español para evaluar")
    
    # ==================== EVALUACIÓN CATALÁN ====================
    if len(df_ca) > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACIÓN CATALÁN")
        log.info("="*70)
        
        X_ca = df_ca[feat_cols].values
        y_ca = df_ca['QA'].values
        
        log.info("Evaluando {} pacientes catalanes...".format(len(df_ca)))
        
        # Raw predictions
        preds_ca = best_model.predict(X_ca)
        evaluate_split(y_ca, preds_ca, "EVAL_CA_RAW", run_dir, log,
                      df_ids=df_ca[['patient_id']] if 'patient_id' in df_ca.columns else None)
        
        # Calibrated predictions
        preds_ca_cal = calibrator.predict(preds_ca)
        evaluate_split(y_ca, preds_ca_cal, "EVAL_CA_CALIBRATED", run_dir, log,
                      df_ids=df_ca[['patient_id']] if 'patient_id' in df_ca.columns else None)
    else:
        log.info("\n" + "="*70)
        log.info("EVALUACIÓN CATALÁN")
        log.info("="*70)
        log.info("No hay datos en catalán para evaluar")
    
    # ==================== COMPARACIÓN CON EL PAPER ====================
    log.info("\n" + "="*70)
    log.info("COMPARACIÓN CON LE ET AL. (2018)")
    log.info("="*70)
    log.info("\nResultados del paper (Table 9, Combined protocol):")
    log.info("  MAE:        8.86 (Oracle) / 9.18 (Auto)")
    log.info("  Correlación: .801 (Oracle) / .799 (Auto)")
    log.info("\nNuestros resultados (CV_PWA_CALIBRATED):")
    # Leer métricas guardadas
    try:
        metrics_df = pd.read_csv(run_dir / "CV_PWA_CALIBRATED_metrics.csv")
        our_mae = metrics_df['MAE'].values[0]
        our_corr = metrics_df['Pearson_r'].values[0]
        log.info("  MAE:        {:.2f}".format(our_mae))
        log.info("  Correlación: {:.3f}".format(our_corr))
        
        # Calcular diferencia
        diff_mae = our_mae - 9.18
        diff_corr = our_corr - 0.799
        log.info("\nDiferencia con el paper:")
        log.info("  ΔMAE:  {:+.2f} ({})".format(diff_mae, 
                 "mejor" if diff_mae < 0 else "peor" if diff_mae > 0 else "igual"))
        log.info("  ΔCorr: {:+.3f} ({})".format(diff_corr,
                 "mejor" if diff_corr > 0 else "peor" if diff_corr < 0 else "igual"))
    except:
        log.info("  [Error leyendo métricas]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUMPIDO]")
        sys.exit(1)
    except Exception as e:
        print("\n\n[ERROR] {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)