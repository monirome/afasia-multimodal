#!/usr/bin/env python3
# train_svr_COMPLETO_FINAL.py
# -*- coding: utf-8 -*-
"""
VERSIÓN COMPLETA FINAL:
✅ Metodología correcta: Control+PWA en train, solo PWA en eval (Le et al. 2018)
✅ TODAS las gráficas y métricas del script original
✅ SHAP + Permutation Importance
✅ Feature importance por grupos (DEN, DYS, LEX)
✅ Severity classification completa
✅ Isotonic calibration
✅ In-sample diagnostics
"""

import os
import sys
import datetime
import pathlib
import shutil
import warnings
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
from scipy.stats import spearmanr, pearsonr

# ======================== PATHS ========================
PROJECT_BASE = pathlib.Path(__file__).resolve().parent.parent  # Auto-detecta la raíz
DATASET_CSV = PROJECT_BASE / "data" / "dataset_FINAL_COMPLETO.csv"
RESULTS_BASE = PROJECT_BASE / "outputs" / "experiments" / "resultados_svr"
DATASET_CSV = os.path.join(PROJECT_BASE, "data/dataset_FINAL_COMPLETO.csv")
RESULTS_BASE = os.path.join(PROJECT_BASE, "outputs/experiments/resultados_svr")

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
def plot_scatter(y_true, y_pred, title, out_png, metrics=None):
    if metrics is None:
        metrics = compute_metrics(y_true, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Identidad")
    
    plt.xlabel("QA Real", fontsize=12, fontweight='bold')
    plt.ylabel("QA Predicho", fontsize=12, fontweight='bold')
    
    text = ("MAE:      {:.2f}\nRMSE:     {:.2f}\nR2:       {:.3f}\n"
            "Pearson:  {:.3f}\nSpearman: {:.3f}").format(
                metrics['MAE'], metrics['RMSE'], metrics['R2'],
                metrics['Pearson_r'], metrics['Spearman_rho'])
    
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontsize=10, family='monospace')
    
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

# ======================== EVALUACION ========================
def evaluate_split(y_true, y_pred, split_name, run_dir, log, df_ids=None):
    log.info("\n" + "="*70)
    log.info("EVALUACION: {}".format(split_name))
    log.info("="*70)
    
    # Metricas regresion
    metrics_cont = compute_metrics(y_true, y_pred)
    log.info("Metricas de regresion:")
    log.info("  MAE:        {:.3f}".format(metrics_cont['MAE']))
    log.info("  RMSE:       {:.3f}".format(metrics_cont['RMSE']))
    log.info("  R2:         {:.3f}".format(metrics_cont['R2']))
    log.info("  Pearson r:  {:.3f} (p={:.4f})".format(metrics_cont['Pearson_r'], metrics_cont['Pearson_p']))
    log.info("  Spearman:   {:.3f} (p={:.4f})".format(metrics_cont['Spearman_rho'], metrics_cont['Spearman_p']))
    
    # Accuracy por distancia
    y_pred_int = to_int(y_pred)
    metrics_int = compute_accuracy_metrics(y_true, y_pred_int)
    log.info("\nAccuracy por distancia:")
    log.info("  Acc@1:  {:.2f}%".format(100*metrics_int['Acc@1']))
    log.info("  Acc@5:  {:.2f}%".format(100*metrics_int['Acc@5']))
    log.info("  Acc@10: {:.2f}%".format(100*metrics_int['Acc@10']))
    log.info("  Exact:  {:.2f}%".format(100*metrics_int['Exact']))
    
    # Clasificacion por severidad
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
    
    # Guardar metricas
    all_metrics = {**metrics_cont, **metrics_int, 'severity_accuracy': sev_metrics['severity_accuracy']}
    pd.DataFrame([all_metrics]).to_csv(run_dir / "{}_metrics.csv".format(split_name), index=False)
    
    pd.DataFrame(sev_metrics['classification_report']).transpose().to_csv(
        run_dir / "{}_severity_report.csv".format(split_name))
    
    pd.DataFrame(sev_metrics['confusion_matrix'], index=SEVERITY_LABELS, columns=SEVERITY_LABELS).to_csv(
        run_dir / "{}_confusion_matrix.csv".format(split_name))
    
    # Guardar predicciones
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
    
    # Generar graficos
    log.info("\nGenerando graficos...")
    
    plot_scatter(y_true, y_pred, "{} - Scatter Plot".format(split_name),
                run_dir / "{}_scatter.png".format(split_name), metrics_cont)
    
    plot_confusion_matrix(sev_metrics['confusion_matrix'],
                         "{} - Confusion Matrix".format(split_name),
                         run_dir / "{}_confusion_matrix.png".format(split_name))
    
    log.info("Guardados: {}_*.csv/png".format(split_name))
    
    return metrics_cont, metrics_int, sev_metrics

# ======================== INTERPRETABILIDAD ========================
def compute_feature_importance_permutation(model, X, y, feature_names, run_dir, log, n_repeats=10):
    """Permutation Importance"""
    from sklearn.inspection import permutation_importance
    
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
    
    # Grafico
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
    """SHAP Values"""
    try:
        import shap
    except ImportError:
        log.warning("SHAP no instalado: pip install shap")
        return None
    
    log.info("\n" + "="*70)
    log.info("SHAP VALUES")
    log.info("="*70)
    log.info("Calculando SHAP values...")
    
    # Background samples
    if len(X_train) > max_samples:
        log.info("  Usando {} samples random como background".format(max_samples))
        np.random.seed(42)
        bg_idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_bg = X_train[bg_idx]
    else:
        X_bg = X_train
        log.info("  Usando {} samples como background".format(len(X_bg)))
    
    # Explainer
    log.info("  Creando SHAP explainer...")
    explainer = shap.KernelExplainer(model.predict, X_bg)
    
    # Test samples
    if len(X_test) > max_samples:
        log.info("  Calculando SHAP para {} samples de test...".format(max_samples))
        np.random.seed(42)
        test_idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_test_sample = X_test[test_idx]
    else:
        log.info("  Calculando SHAP para {} samples de test...".format(len(X_test)))
        X_test_sample = X_test
    
    shap_values = explainer.shap_values(X_test_sample)
    
    # Guardar
    pd.DataFrame(shap_values, columns=feature_names).to_csv(run_dir / "shap_values.csv", index=False)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(run_dir / "feature_importance_shap.csv", index=False)
    
    log.info("\nTop 20 features (SHAP):")
    for idx, (i, row) in enumerate(importance_df.head(20).iterrows(), 1):
        log.info("  {:2d}. {:40s} {:.4f}".format(idx, row['feature'], row['importance']))
    
    # Graficos
    log.info("\nGenerando graficos SHAP...")
    
    try:
        # Beeswarm
        plt.figure(figsize=(10, 12))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                         show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(run_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Bar
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
    """Analiza importancia por grupos (DEN, DYS, LEX)"""
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
    
    # Grafico
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

# ======================== MAIN ========================
def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(RESULTS_BASE) / "SVR_FINAL_{}".format(ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log = set_logger(run_dir)
    log.info("="*70)
    log.info("SVR COMPLETO - Metodologia Le et al. (2018)")
    log.info("="*70)
    log.info("Output: {}".format(run_dir))
    log.info("Timestamp: {}".format(ts))
    
    try:
        shutil.copy2(__file__, run_dir / pathlib.Path(__file__).name)
    except:
        pass
    
    # ==================== CARGAR DATOS ====================
    log.info("\n" + "="*70)
    log.info("CARGANDO DATOS")
    log.info("="*70)
    
    if not os.path.exists(DATASET_CSV):
        log.error("No existe: {}".format(DATASET_CSV))
        sys.exit(1)
    
    log.info("Cargando: {}".format(DATASET_CSV))
    df = pd.read_csv(DATASET_CSV)
    log.info("Total registros: {}".format(len(df)))
    
    # Verificar QA
    df = df[df['QA'].notna()].copy()
    log.info("Con QA valido: {}".format(len(df)))
    
    # ==================== SPLIT Control vs PWA ====================
    log.info("\n" + "="*70)
    log.info("METODOLOGIA DEL PAPER (Le et al. 2018)")
    log.info("="*70)
    log.info("Paper: 'The remaining data and all Control speakers are used for training'")
    log.info("Paper: 'we withhold 25% of speakers from each sub-dataset in the Aphasia set'")
    log.info("\nINTERPRETACION:")
    log.info("  - Control + PWA → TRAINING (normalizacion, baseline)")
    log.info("  - Solo PWA → EVALUATION (CV, test)")
    
    if 'group' not in df.columns:
        log.warning("\nNo hay columna 'group', usando todos como PWA")
        df_pwa = df.copy()
        df_control = pd.DataFrame()
    else:
        log.info("\nDistribucion original:")
        for grp, count in df['group'].value_counts(dropna=False).items():
            log.info("  {}: {}".format(grp, count))
        
        df_pwa = df[df['group'] == 'pwa'].copy()
        df_control = df[df['group'] == 'control'].copy()
        
        log.info("\nSplit metodologico:")
        log.info("  PWA (para CV y eval):     {}".format(len(df_pwa)))
        log.info("  Control (solo en train):  {}".format(len(df_control)))
        
        if len(df_control) > 0:
            log.info("\nControl estadisticas:")
            log.info("  QA: {:.1f} ± {:.1f}".format(df_control['QA'].mean(), df_control['QA'].std()))
            log.info("  Range: [{:.1f}, {:.1f}]".format(df_control['QA'].min(), df_control['QA'].max()))
    
    if len(df_pwa) < 10:
        log.error("Muy pocos PWA (n={})".format(len(df_pwa)))
        sys.exit(1)
    
    log.info("\nPWA estadisticas:")
    log.info("  N:     {}".format(len(df_pwa)))
    log.info("  QA:    {:.1f} ± {:.1f}".format(df_pwa['QA'].mean(), df_pwa['QA'].std()))
    log.info("  Range: [{:.1f}, {:.1f}]".format(df_pwa['QA'].min(), df_pwa['QA'].max()))
    
    # Distribucion por severidad
    log.info("\nDistribucion por severidad (PWA):")
    sev_dist = qa_to_severity(df_pwa['QA']).value_counts()
    for sev in SEVERITY_LABELS:
        count = sev_dist.get(sev, 0)
        log.info("  {}: {}".format(sev, count))
    
    # ==================== FEATURES ====================
    log.info("\n" + "="*70)
    log.info("FEATURES (34 del paper)")
    log.info("="*70)
    
    den_paper = [
        'den_words_per_min','den_phones_per_min','den_W','den_OCW',
        'den_words_utt_mean','den_phones_utt_mean',
        'den_nouns','den_verbs','den_nouns_per_verb','den_noun_ratio',
        'den_light_verbs','den_determiners','den_demonstratives',
        'den_prepositions','den_adjectives','den_adverbs',
        'den_pronoun_ratio','den_function_words'
    ]
    dys_paper = [
        'dys_fillers_per_min','dys_fillers_per_word','dys_fillers_per_phone',
        'dys_pauses_per_min','dys_long_pauses_per_min','dys_short_pauses_per_min',
        'dys_pauses_per_word','dys_long_pauses_per_word','dys_short_pauses_per_word',
        'dys_pause_sec_mean'
    ]
    lex_paper = [
        'lex_ttr','lex_freq_mean','lex_img_mean','lex_aoa_mean','lex_fam_mean','lex_phones_mean'
    ]
    
    paper_features = den_paper + dys_paper + lex_paper
    feat_cols = [f for f in paper_features if f in df.columns]
    missing = [f for f in paper_features if f not in df.columns]
    
    if missing:
        log.warning("\nFeatures faltantes del paper:")
        for f in missing:
            log.warning("  - {}".format(f))
    
    log.info("\nFeatures seleccionadas:")
    log.info("  DEN: {}".format(len([f for f in feat_cols if f.startswith('den_')])))
    log.info("  DYS: {}".format(len([f for f in feat_cols if f.startswith('dys_')])))
    log.info("  LEX: {}".format(len([f for f in feat_cols if f.startswith('lex_')])))
    log.info("  TOTAL: {} (de 34 del paper)".format(len(feat_cols)))
    
    # Guardar lista
    with open(run_dir / "features_used.txt", "w", encoding="utf-8") as f:
        f.write("FEATURES (Paper - Fraser et al. 2013 / Le et al. 2018)\n")
        f.write("="*70 + "\n")
        f.write("Total: {}\n\n".format(len(feat_cols)))
        
        for prefix in ['DEN', 'DYS', 'LEX']:
            feats = sorted([c for c in feat_cols if c.startswith(prefix.lower() + '_')])
            f.write("{} features ({}):\n".format(prefix, len(feats)))
            for c in feats:
                f.write("  - {}\n".format(c))
            f.write("\n")
    
    # ==================== PREPARAR DATOS ====================
    log.info("\n" + "="*70)
    log.info("PREPARACION DE DATOS")
    log.info("="*70)
    
    # PWA para CV
    X_pwa = df_pwa[feat_cols].values
    y_pwa = df_pwa['QA'].values
    groups_pwa = df_pwa['patient_id'].values if 'patient_id' in df_pwa.columns else np.arange(len(df_pwa))
    
    log.info("PWA (para cross-validation):")
    log.info("  X: {}".format(X_pwa.shape))
    log.info("  y: {}".format(y_pwa.shape))
    log.info("  Grupos unicos: {}".format(len(np.unique(groups_pwa))))
    
    # Control (se agrega al training en cada fold)
    if len(df_control) > 0:
        X_control = df_control[feat_cols].values
        y_control = df_control['QA'].values
        log.info("\nControl (siempre en train de cada fold):")
        log.info("  X: {}".format(X_control.shape))
        log.info("  y: {}".format(y_control.shape))
    else:
        X_control = None
        y_control = None
        log.info("\nNo hay Control samples")
    
    # ==================== MODELO ====================
    log.info("\n" + "="*70)
    log.info("CONFIGURACION DEL MODELO")
    log.info("="*70)
    
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])
    
    log.info("Pipeline:")
    log.info("  1. SimpleImputer (strategy='median')")
    log.info("  2. StandardScaler")
    log.info("  3. SVR")
    
    param_grid = {
        "svr__C": [1, 10, 100],
        "svr__epsilon": [0.1, 1],
        "svr__kernel": ["rbf", "linear"],
        "svr__shrinking": [True, False]
    }
    
    log.info("\nHiperparametros a explorar:")
    total_comb = 1
    for param, values in param_grid.items():
        log.info("  {}: {}".format(param, values))
        total_comb *= len(values)
    log.info("  Total combinaciones: {}".format(total_comb))
    
    # ==================== CROSS-VALIDATION (Control en cada fold train) ====================
    log.info("\n" + "="*70)
    log.info("CROSS-VALIDATION (GroupKFold, Control siempre en train)")
    log.info("="*70)
    log.info("Metodo: GroupKFold(n_splits=4) sobre PWA")
    log.info("En cada fold:")
    log.info("  - Train = PWA_train + Control")
    log.info("  - Test  = PWA_test (solo)")
    
    gkf = GroupKFold(n_splits=4)
    
    cv_results = []
    cv_preds = np.zeros_like(y_pwa, dtype=float)
    
    log.info("\nIniciando cross-validation...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_pwa, y_pwa, groups=groups_pwa), 1):
        log.info("\n  Fold {}/4:".format(fold_idx))
        log.info("    PWA train: {}, test: {}".format(len(train_idx), len(test_idx)))
        
        # Training = PWA_train + Control
        if X_control is not None and len(X_control) > 0:
            X_train_fold = np.vstack([X_pwa[train_idx], X_control])
            y_train_fold = np.concatenate([y_pwa[train_idx], y_control])
            log.info("    + Control: {}".format(len(X_control)))
            log.info("    = Total train: {}".format(len(X_train_fold)))
        else:
            X_train_fold = X_pwa[train_idx]
            y_train_fold = y_pwa[train_idx]
            log.info("    = Total train: {} (solo PWA)".format(len(X_train_fold)))
        
        X_test_fold = X_pwa[test_idx]
        y_test_fold = y_pwa[test_idx]
        
        # Grid search en este fold
        gs_fold = GridSearchCV(
            pipe, param_grid, cv=3, # EL PAPER USA 5-FOLD 
            scoring="neg_mean_absolute_error", 
            n_jobs=1, verbose=0
        )
        gs_fold.fit(X_train_fold, y_train_fold)
        
        # Predecir en test (solo PWA)
        cv_preds[test_idx] = gs_fold.predict(X_test_fold)
        
        mae_fold = mean_absolute_error(y_test_fold, cv_preds[test_idx])
        log.info("    MAE fold: {:.3f}".format(mae_fold))
        log.info("    Mejores params: {}".format(gs_fold.best_params_))
        
        cv_results.append({
            'fold': fold_idx,
            'n_train_pwa': len(train_idx),
            'n_train_control': len(X_control) if X_control is not None else 0,
            'n_test': len(test_idx),
            'mae': mae_fold,
            'best_params': str(gs_fold.best_params_)
        })
    
    pd.DataFrame(cv_results).to_csv(run_dir / "cv_results_by_fold.csv", index=False)
    
    log.info("\n" + "="*70)
    log.info("CROSS-VALIDATION COMPLETO")
    log.info("="*70)
    log.info("MAE promedio CV: {:.3f}".format(np.mean([r['mae'] for r in cv_results])))
    log.info("MAE std CV:      {:.3f}".format(np.std([r['mae'] for r in cv_results])))
    
    # Evaluar predicciones CV
    evaluate_split(y_pwa, cv_preds, "CV_PWA", run_dir, log,
                  df_ids=df_pwa[['patient_id']] if 'patient_id' in df_pwa.columns else None)
    
    # ==================== CALIBRACION ====================
    log.info("\n" + "="*70)
    log.info("CALIBRACION POST-HOC (Isotonic Regression)")
    log.info("="*70)
    
    log.info("Entrenando calibrador en predicciones CV...")
    calibrator = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
    calibrator.fit(cv_preds, y_pwa)
    
    joblib.dump(calibrator, run_dir / "calibrator.pkl")
    log.info("Calibrador guardado: calibrator.pkl")
    
    cv_preds_cal = calibrator.predict(cv_preds)
    evaluate_split(y_pwa, cv_preds_cal, "CV_PWA_CALIBRATED", run_dir, log,
                  df_ids=df_pwa[['patient_id']] if 'patient_id' in df_pwa.columns else None)
    
    # ==================== MODELO FINAL ====================
    log.info("\n" + "="*70)
    log.info("ENTRENAMIENTO MODELO FINAL")
    log.info("="*70)
    
    if X_control is not None and len(X_control) > 0:
        X_final = np.vstack([X_pwa, X_control])
        y_final = np.concatenate([y_pwa, y_control])
        log.info("Training: {} PWA + {} Control = {}".format(
            len(X_pwa), len(X_control), len(X_final)))
    else:
        X_final = X_pwa
        y_final = y_pwa
        log.info("Training: {} PWA (solo)".format(len(X_final)))
    
    log.info("\nEntrenando Grid Search final...")
    gs_final = GridSearchCV(
        pipe, param_grid, cv=3, #EL PAPER USA 5-FOLD
        scoring="neg_mean_absolute_error",
        n_jobs=1, verbose=2
    )
    gs_final.fit(X_final, y_final)
    
    best_model = gs_final.best_estimator_
    
    log.info("\n" + "="*70)
    log.info("MODELO FINAL ENTRENADO")
    log.info("="*70)
    log.info("Mejores hiperparametros:")
    for param, value in gs_final.best_params_.items():
        log.info("  {}: {}".format(param, value))
    log.info("Mejor MAE (CV interno): {:.3f}".format(-gs_final.best_score_))
    
    joblib.dump(best_model, run_dir / "model_final.pkl")
    log.info("\nModelo guardado: model_final.pkl")
    
    pd.DataFrame(gs_final.cv_results_).to_csv(run_dir / "cv_results_full.csv", index=False)
    
    # ==================== INTERPRETABILIDAD ====================
    log.info("\n" + "="*70)
    log.info("ANALISIS DE INTERPRETABILIDAD")
    log.info("="*70)
    
    # 1. Permutation Importance (siempre)
    perm_importance = compute_feature_importance_permutation(
        best_model, X_pwa, y_pwa, feat_cols, run_dir, log, n_repeats=10
    )
    
    # 2. Analisis por grupos (Permutation)
    analyze_feature_groups(perm_importance, run_dir, log, method_name="Permutation")
    
    # 3. SHAP (si esta instalado)
    try:
        n_test_shap = max(int(len(X_pwa) * 0.2), 20)
        X_train_shap = X_pwa[:-n_test_shap]
        X_test_shap = X_pwa[-n_test_shap:]
        
        shap_importance = compute_shap_values(
            best_model, X_train_shap, X_test_shap, 
            feat_cols, run_dir, log, max_samples=100
        )
        
        if shap_importance is not None:
            analyze_feature_groups(shap_importance, run_dir, log, method_name="SHAP")
    
    except Exception as e:
        log.warning("No se pudo calcular SHAP: {}".format(e))
        log.warning("Para instalar: pip install shap")
    
    # ==================== IN-SAMPLE PWA ====================
    log.info("\n" + "="*70)
    log.info("IN-SAMPLE PWA (diagnostico)")
    log.info("="*70)
    
    log.info("Predicciones in-sample sobre PWA...")
    
    preds_in = best_model.predict(X_pwa)
    evaluate_split(y_pwa, preds_in, "IN_SAMPLE_PWA_RAW", run_dir, log,
                  df_ids=df_pwa[['patient_id']] if 'patient_id' in df_pwa.columns else None)
    
    preds_in_cal = calibrator.predict(preds_in)
    evaluate_split(y_pwa, preds_in_cal, "IN_SAMPLE_PWA_CALIBRATED", run_dir, log,
                  df_ids=df_pwa[['patient_id']] if 'patient_id' in df_pwa.columns else None)
    
    # ==================== RESUMEN FINAL ====================
    log.info("\n" + "="*70)
    log.info("RESUMEN FINAL")
    log.info("="*70)
    
    log.info("\nDirectorio de resultados: {}".format(run_dir))
    
    log.info("\nPacientes procesados:")
    log.info("  PWA (para CV y eval):     {}".format(len(df_pwa)))
    if X_control is not None:
        log.info("  Control (solo en train):  {}".format(len(df_control)))
    log.info("  TOTAL:                    {}".format(len(df_pwa) + (len(df_control) if X_control is not None else 0)))
    
    log.info("\nFeatures utilizadas:")
    log.info("  DEN: {}".format(len([f for f in feat_cols if f.startswith('den_')])))
    log.info("  DYS: {}".format(len([f for f in feat_cols if f.startswith('dys_')])))
    log.info("  LEX: {}".format(len([f for f in feat_cols if f.startswith('lex_')])))
    log.info("  TOTAL: {} (de 34 del paper)".format(len(feat_cols)))
    
    try:
        cv_metrics = pd.read_csv(run_dir / "CV_PWA_CALIBRATED_metrics.csv").iloc[0]
        log.info("\nMetricas CV (calibrado):")
        log.info("  MAE:                {:.3f}".format(cv_metrics['MAE']))
        log.info("  RMSE:               {:.3f}".format(cv_metrics['RMSE']))
        log.info("  R2:                 {:.3f}".format(cv_metrics['R2']))
        log.info("  Pearson r:          {:.3f}".format(cv_metrics['Pearson_r']))
        log.info("  Spearman rho:       {:.3f}".format(cv_metrics['Spearman_rho']))
        log.info("  Acc@5:              {:.1f}%".format(100*cv_metrics['Acc@5']))
        log.info("  Severity Accuracy:  {:.1f}%".format(100*cv_metrics['severity_accuracy']))
    except:
        pass
    
    log.info("\n" + "="*70)
    log.info("PROCESO COMPLETADO EXITOSAMENTE")
    log.info("="*70)
    log.info("Timestamp: {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log.info("Resultados en: {}".format(run_dir))
    log.info("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUMPIDO] Proceso cancelado")
        sys.exit(1)
    except Exception as e:
        print("\n\n[ERROR FATAL] {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
