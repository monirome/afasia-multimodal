#!/usr/bin/env python3
# train_svr_den_dys.py
# -*- coding: utf-8 -*-
"""
SVR - Prediccion WAB-AQ con features DEN+DYS (Le et al. 2018)
Pipeline completo con GridSearchCV, calibracion y evaluacion exhaustiva
"""

import os
import sys
import re
import datetime
import pathlib
import shutil
import warnings
warnings.filterwarnings("ignore")

# FIX: stdout sin buffer + limitar hilos BLAS para evitar bloqueos
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional

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

# ======================== CONFIGURACION PATHS ========================

# Ruta base del proyecto
PROJECT_BASE = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025"

# Carpeta de datos
DATA_BASE = os.path.join(PROJECT_BASE, "data")

# Dataset final con features + metadata
CSV_FINAL = os.path.join(DATA_BASE, "dataset_FINAL_EN_ES.csv")

# Carpeta donde se guardaran los resultados del entrenamiento
RESULTS_BASE = os.path.join(PROJECT_BASE, "resultados_svr")

# Crear la carpeta si no existe
os.makedirs(RESULTS_BASE, exist_ok=True)

# ======================== SEVERIDAD DE AFASIA ========================
SEVERITY_BINS = [0, 25, 50, 75, 100]
SEVERITY_LABELS = ['Very Severe', 'Severe', 'Moderate', 'Mild']

def qa_to_severity(qa_scores):
    """
    Convierte scores QA a categorias de severidad
    
    Args:
        qa_scores: array de scores QA (0-100)
    
    Returns:
        array de categorias de severidad
    """
    return pd.cut(
        qa_scores, 
        bins=SEVERITY_BINS, 
        labels=SEVERITY_LABELS,
        include_lowest=True
    )

# ======================== LOGGER ========================
def set_logger(run_dir: pathlib.Path) -> logging.Logger:
    """
    Configura logger con salida a consola + archivo
    """
    log = logging.getLogger("SVR_DEN_DYS_COMPLETE")
    log.setLevel(logging.INFO)
    
    # Formato con timestamp
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    
    # Handler archivo
    fh = logging.FileHandler(
        run_dir / "console.log",
        mode="w",
        encoding="utf-8"
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    
    # Limpiar handlers anteriores
    for h in list(log.handlers):
        log.removeHandler(h)
    
    log.addHandler(ch)
    log.addHandler(fh)
    
    return log

# ======================== UTILIDADES ========================
def to_num(s: pd.Series) -> pd.Series:
    """Convierte Series a numerico con NaN para errores"""
    return pd.to_numeric(s, errors="coerce")

def to_int(pred):
    """Convierte predicciones a enteros en rango [0, 100]"""
    return np.rint(np.asarray(pred)).clip(0, 100).astype(int)

# ======================== METRICAS ========================
def compute_metrics(y_true, y_pred):
    """
    Calcula metricas completas de evaluacion
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
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
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_rho': spearman_rho,
        'Spearman_p': spearman_p
    }

def compute_accuracy_metrics(y_true, y_pred_int):
    """
    Calcula metricas de accuracy para predicciones enteras
    """
    y_true = np.asarray(y_true)
    y_pred_int = np.asarray(y_pred_int)
    
    errors = np.abs(y_pred_int - y_true)
    
    return {
        'Acc@1': float(np.mean(errors <= 1)),
        'Acc@5': float(np.mean(errors <= 5)),
        'Acc@10': float(np.mean(errors <= 10)),
        'Exact': float(np.mean(errors == 0))
    }

def compute_severity_accuracy(y_true, y_pred):
    """
    Calcula accuracy de clasificacion por severidad
    
    Args:
        y_true: valores reales de QA
        y_pred: predicciones de QA
    
    Returns:
        dict con accuracy y metricas por clase
    """
    # Convertir a categorias de severidad
    sev_true = qa_to_severity(y_true)
    sev_pred = qa_to_severity(y_pred)
    
    # Accuracy general
    acc = accuracy_score(sev_true, sev_pred)
    
    # Matriz de confusion
    cm = confusion_matrix(sev_true, sev_pred, labels=SEVERITY_LABELS)
    
    # Report por clase
    report = classification_report(
        sev_true, sev_pred, 
        labels=SEVERITY_LABELS,
        output_dict=True,
        zero_division=0
    )
    
    return {
        'severity_accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': report
    }

# ======================== VISUALIZACION ========================
def plot_scatter(y_true, y_pred, title, out_png, metrics=None):
    """
    Genera scatter plot con linea identidad y metricas
    """
    if metrics is None:
        metrics = compute_metrics(y_true, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Linea identidad
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Identidad")
    
    # Etiquetas
    plt.xlabel("QA Real (0-100)", fontsize=12, fontweight='bold')
    plt.ylabel("QA Predicho (0-100)", fontsize=12, fontweight='bold')
    
    # Texto con metricas
    text = ("MAE:      {:.2f}\n"
            "RMSE:     {:.2f}\n"
            "R2:       {:.3f}\n"
            "Pearson:  {:.3f}\n"
            "Spearman: {:.3f}").format(
                metrics['MAE'],
                metrics['RMSE'],
                metrics['R2'],
                metrics['Pearson_r'],
                metrics['Spearman_rho']
            )
    
    plt.text(0.05, 0.95, text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
             fontsize=10,
             family='monospace')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def plot_residuals(y_true, y_pred, title, out_png):
    """
    Genera plots de analisis de residuos
    """
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuos vs predicciones
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel("QA Predicho", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Residuos (Pred - Real)", fontsize=11, fontweight='bold')
    axes[0].set_title("Residuos vs Predicciones", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Histograma de residuos
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel("Residuos", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Frecuencia", fontsize=11, fontweight='bold')
    axes[1].set_title("Distribucion de Residuos", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Anadir estadisticas al histograma
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    axes[1].text(0.05, 0.95, 
                "Media = {:.2f}\nStd = {:.2f}".format(mean_res, std_res),
                transform=axes[1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                fontsize=10,
                family='monospace')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def plot_error_distribution(y_true, y_pred_int, title, out_png):
    """
    Genera distribucion de errores absolutos
    """
    errors = np.abs(y_pred_int - y_true)
    
    plt.figure(figsize=(10, 6))
    
    # Histograma
    n, bins, patches = plt.hist(
        errors,
        bins=range(0, int(errors.max()) + 2),
        edgecolor='black',
        alpha=0.7,
        color='lightcoral'
    )
    
    plt.xlabel("Error Absoluto (puntos QA)", fontsize=12, fontweight='bold')
    plt.ylabel("Frecuencia", fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Lineas verticales en umbrales
    for x, label in [(1, "+-1"), (5, "+-5"), (10, "+-10")]:
        plt.axvline(x=x, color='blue', linestyle='--', alpha=0.6, linewidth=2)
        plt.text(x, plt.ylim()[1] * 0.9, label, 
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Estadisticas
    pct_1 = 100 * np.mean(errors <= 1)
    pct_5 = 100 * np.mean(errors <= 5)
    pct_10 = 100 * np.mean(errors <= 10)
    
    stats_text = ("Acc@1:  {:.1f}%\n"
                 "Acc@5:  {:.1f}%\n"
                 "Acc@10: {:.1f}%").format(pct_1, pct_5, pct_10)
    
    plt.text(0.98, 0.97, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=10,
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, title, out_png):
    """
    Genera matriz de confusion para clasificacion de severidad
    
    Args:
        cm: matriz de confusion (numpy array)
        title: titulo del grafico
        out_png: ruta de salida PNG
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalizar por filas (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Etiquetas
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=SEVERITY_LABELS,
           yticklabels=SEVERITY_LABELS,
           ylabel='Severidad Real',
           xlabel='Severidad Predicha')
    
    # Rotar etiquetas
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Anadir valores en cada celda
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "{}\n({:.1f}%)".format(cm[i, j], cm_norm[i, j] * 100),
                   ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black",
                   fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

# ======================== EVALUACION ========================
def evaluate_split(y_true, y_pred, split_name, run_dir, log, df_ids=None):
    """
    Evaluacion completa de un split con todas las metricas y graficos
    """
    log.info("\n{}".format("="*70))
    log.info("EVALUACION: {}".format(split_name))
    log.info("="*70)
    
    # Metricas continuas (regresion)
    metrics_cont = compute_metrics(y_true, y_pred)
    log.info("Metricas de regresion:")
    log.info("  MAE:        {:.3f}".format(metrics_cont['MAE']))
    log.info("  RMSE:       {:.3f}".format(metrics_cont['RMSE']))
    log.info("  R2:         {:.3f}".format(metrics_cont['R2']))
    log.info("  Pearson r:  {:.3f} (p={:.4f})".format(
        metrics_cont['Pearson_r'], metrics_cont['Pearson_p']))
    log.info("  Spearman:   {:.3f} (p={:.4f})".format(
        metrics_cont['Spearman_rho'], metrics_cont['Spearman_p']))
    
    # Metricas de accuracy por distancia
    y_pred_int = to_int(y_pred)
    metrics_int = compute_accuracy_metrics(y_true, y_pred_int)
    log.info("\nMetricas de accuracy por distancia:")
    log.info("  Acc@1:  {:.2f}%".format(100*metrics_int['Acc@1']))
    log.info("  Acc@5:  {:.2f}%".format(100*metrics_int['Acc@5']))
    log.info("  Acc@10: {:.2f}%".format(100*metrics_int['Acc@10']))
    log.info("  Exact:  {:.2f}%".format(100*metrics_int['Exact']))
    
    # Metricas de clasificacion por severidad
    sev_metrics = compute_severity_accuracy(y_true, y_pred_int)
    log.info("\nMetricas de clasificacion por severidad:")
    log.info("  Accuracy: {:.2f}%".format(100*sev_metrics['severity_accuracy']))
    
    # Accuracy por clase
    report = sev_metrics['classification_report']
    log.info("\n  Accuracy por clase:")
    for label in SEVERITY_LABELS:
        if label in report:
            log.info("    {}: Precision={:.2f}%, Recall={:.2f}%, F1={:.2f}%".format(
                label,
                100*report[label]['precision'],
                100*report[label]['recall'],
                100*report[label]['f1-score']
            ))
    
    # Guardar metricas en CSV
    all_metrics = {**metrics_cont, **metrics_int}
    all_metrics['severity_accuracy'] = sev_metrics['severity_accuracy']
    
    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(run_dir / "{}_metrics.csv".format(split_name), index=False)
    log.info("\nMetricas guardadas: {}_metrics.csv".format(split_name))
    
    # Guardar classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(run_dir / "{}_severity_report.csv".format(split_name))
    log.info("Report de severidad guardado: {}_severity_report.csv".format(split_name))
    
    # Guardar matriz de confusion
    cm_df = pd.DataFrame(
        sev_metrics['confusion_matrix'],
        index=SEVERITY_LABELS,
        columns=SEVERITY_LABELS
    )
    cm_df.to_csv(run_dir / "{}_confusion_matrix.csv".format(split_name))
    log.info("Matriz de confusion guardada: {}_confusion_matrix.csv".format(split_name))
    
    # Guardar predicciones
    preds_df = pd.DataFrame({
        'QA_real': y_true,
        'QA_pred_cont': y_pred,
        'QA_pred_int': y_pred_int,
        'severity_real': qa_to_severity(y_true),
        'severity_pred': qa_to_severity(y_pred_int),
        'error_cont': y_pred - y_true,
        'error_int': y_pred_int - y_true,
        'abs_error_int': np.abs(y_pred_int - y_true)
    })
    
    # Anadir IDs si estan disponibles
    if df_ids is not None:
        for col in df_ids.columns:
            preds_df[col] = df_ids[col].values
    
    preds_df.to_csv(run_dir / "{}_predictions.csv".format(split_name), index=False)
    log.info("Predicciones guardadas: {}_predictions.csv".format(split_name))
    
    # Generar graficos
    log.info("\nGenerando graficos...")
    
    plot_scatter(
        y_true, y_pred,
        "{} - Scatter Plot".format(split_name),
        run_dir / "{}_scatter.png".format(split_name),
        metrics_cont
    )
    
    plot_residuals(
        y_true, y_pred,
        "{} - Analisis de Residuos".format(split_name),
        run_dir / "{}_residuals.png".format(split_name)
    )
    
    plot_error_distribution(
        y_true, y_pred_int,
        "{} - Distribucion de Errores".format(split_name),
        run_dir / "{}_errors.png".format(split_name)
    )
    
    plot_confusion_matrix(
        sev_metrics['confusion_matrix'],
        "{} - Matriz de Confusion (Severidad)".format(split_name),
        run_dir / "{}_confusion_matrix.png".format(split_name)
    )
    
    log.info("Graficos guardados: {}_*.png".format(split_name))
    
    return metrics_cont, metrics_int, sev_metrics

# ======================== MAIN ========================
def main():
    """Pipeline completo de entrenamiento y evaluacion"""
    
    # ==================== SETUP ====================
    # Crear directorio de resultados
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(RESULTS_BASE) / "SVR_DEN_DYS_{}".format(ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    log = set_logger(run_dir)
    log.info("="*70)
    log.info("INICIO: Prediccion WAB-AQ con features DEN+DYS")
    log.info("Metodologia: Le et al. (2018)")
    log.info("="*70)
    log.info("Directorio de resultados: {}".format(run_dir))
    log.info("Timestamp: {}".format(ts))
    
    # Copiar este script a resultados
    try:
        shutil.copy2(__file__, run_dir / pathlib.Path(__file__).name)
        log.info("Script copiado a resultados")
    except Exception as e:
        log.warning("No se pudo copiar script: {}".format(e))
    
    # ==================== CARGAR DATOS ====================
    log.info("\n" + "="*70)
    log.info("CARGANDO DATOS")
    log.info("="*70)
    
    # Verificar archivo
    if not os.path.exists(CSV_FINAL):
        log.error("No existe archivo: {}".format(CSV_FINAL))
        log.error("El archivo debe contener features DEN+DYS + metadata (QA, language)")
        sys.exit(1)
    
    # Cargar dataset final
    log.info("Cargando dataset desde: {}".format(CSV_FINAL))
    df = pd.read_csv(CSV_FINAL)
    log.info("Dataset cargado: {} pacientes".format(len(df)))
    
    # Adaptar nombre de columna de idioma
    if 'language' in df.columns and 'lang' not in df.columns:
        df['lang'] = df['language']
        log.info("Columna 'language' renombrada a 'lang'")
    
    # Filtrar pacientes con QA
    df = df.dropna(subset=['QA'])
    log.info("Pacientes con QA valido: {}".format(len(df)))
    
    if len(df) < 10:
        log.error("Insuficientes pacientes para entrenamiento (n={} < 10)".format(len(df)))
        sys.exit(1)
    
    # ==================== PREPARAR FEATURES ====================
    log.info("\n" + "="*70)
    log.info("PREPARANDO FEATURES")
    log.info("="*70)
    
    # Identificar columnas de features
    den_cols = [c for c in df.columns if c.startswith('den_')]
    dys_cols = [c for c in df.columns if c.startswith('dys_')]
    feat_cols = den_cols + dys_cols
    
    log.info("Features detectadas:")
    log.info("  DEN: {}".format(len(den_cols)))
    log.info("  DYS: {}".format(len(dys_cols)))
    log.info("  TOTAL: {}".format(len(feat_cols)))
    
    # Filtrar solo columnas numericas
    feat_cols_numeric = df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(feat_cols_numeric) < len(feat_cols):
        log.warning("Se filtraron {} features no numericas".format(
            len(feat_cols) - len(feat_cols_numeric)))
        feat_cols = feat_cols_numeric
    
    if len(feat_cols) == 0:
        log.error("No hay features numericas disponibles")
        sys.exit(1)
    
    log.info("Features numericas finales: {}".format(len(feat_cols)))
    
    # Guardar lista de features usadas
    with open(run_dir / "features_used.txt", "w", encoding="utf-8") as f:
        f.write("FEATURES UTILIZADAS\n")
        f.write("{}\n".format("="*70))
        f.write("Total: {}\n\n".format(len(feat_cols)))
        
        f.write("DEN features ({}):\n".format(
            len([c for c in feat_cols if c.startswith('den_')])))
        for c in sorted([c for c in feat_cols if c.startswith('den_')]):
            f.write("  - {}\n".format(c))
        
        f.write("\nDYS features ({}):\n".format(
            len([c for c in feat_cols if c.startswith('dys_')])))
        for c in sorted([c for c in feat_cols if c.startswith('dys_')]):
            f.write("  - {}\n".format(c))
    
    log.info("Lista de features guardada: features_used.txt")
    
    # ==================== SPLITS POR IDIOMA ====================
    log.info("\n" + "="*70)
    log.info("PREPARANDO SPLITS POR IDIOMA")
    log.info("="*70)
    
    # Detectar idioma
    if 'lang' not in df.columns:
        log.warning("Columna 'lang' no encontrada. Asignando 'en' por defecto")
        df['lang'] = 'en'
    
    # Splits
    df_en = df[df['lang'] == 'en'].reset_index(drop=True)
    df_es = df[df['lang'] == 'es'].reset_index(drop=True)
    df_ca = df[df['lang'] == 'ca'].reset_index(drop=True)
    
    n_en, n_es, n_ca = len(df_en), len(df_es), len(df_ca)
    
    log.info("Distribucion por idioma:")
    log.info("  EN (train):               {} pacientes".format(n_en))
    log.info("  ES (internal validation): {} pacientes".format(n_es))
    log.info("  CA (external test):       {} pacientes".format(n_ca))
    
    if n_en < 4:
        log.error("Se requieren >=4 pacientes EN para GroupKFold (n_en={})".format(n_en))
        sys.exit(1)
    
    # Preparar datos EN para entrenamiento
    X_en = df_en[feat_cols].values
    y_en = df_en['QA'].values
    g_en = df_en['patient_id'].values
    
    log.info("\nDatos de entrenamiento (EN):")
    log.info("  X shape: {}".format(X_en.shape))
    log.info("  y shape: {}".format(y_en.shape))
    log.info("  Grupos unicos: {}".format(len(np.unique(g_en))))
    log.info("  QA range: [{:.1f}, {:.1f}]".format(y_en.min(), y_en.max()))
    log.info("  QA mean +/- std: {:.2f} +/- {:.2f}".format(y_en.mean(), y_en.std()))
    
    # Distribucion de severidad
    log.info("\nDistribucion de severidad (EN):")
    sev_dist = qa_to_severity(y_en).value_counts()
    for sev, count in sev_dist.items():
        log.info("  {}: {}".format(sev, count))
    
    # ==================== MODELO ====================
    log.info("\n" + "="*70)
    log.info("CONFIGURACION Y ENTRENAMIENTO DEL MODELO")
    log.info("="*70)
    
    # Pipeline: Imputer -> Scaler -> SVR
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])
    
    log.info("Pipeline:")
    log.info("  1. SimpleImputer (strategy='median')")
    log.info("  2. StandardScaler")
    log.info("  3. SVR")
    
    # Grid de hiperparametros (Le et al. 2018)
    param_grid = {
        "svr__C": [1, 10, 100],
        "svr__epsilon": [0.1, 1],
        "svr__kernel": ["rbf", "linear"],
        "svr__shrinking": [True, False]
    }
    
    log.info("\nHiperparametros a explorar:")
    total_combinations = 1
    for param, values in param_grid.items():
        log.info("  {}: {}".format(param, values))
        total_combinations *= len(values)
    log.info("  Total combinaciones: {}".format(total_combinations))
    
    # GridSearchCV con GroupKFold (speaker-independent)
    gkf = GroupKFold(n_splits=4)
    log.info("\nCross-validation: GroupKFold (n_splits=4)")
    log.info("  Objetivo: speaker-independent evaluation")
    log.info("  Metrica: negative MAE")
    
    log.info("\nIniciando GridSearchCV...")
    log.info("  (esto puede tomar varios minutos)")
    
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        verbose=2,
        return_train_score=True
    )
    
    gs.fit(X_en, y_en, groups=g_en)
    
    best_model = gs.best_estimator_
    
    log.info("\n{}".format("="*70))
    log.info("GridSearchCV COMPLETADO")
    log.info("="*70)
    log.info("Mejores hiperparametros:")
    for param, value in gs.best_params_.items():
        log.info("  {}: {}".format(param, value))
    log.info("Mejor MAE (CV): {:.3f}".format(-gs.best_score_))
    
    # Guardar modelo y resultados de CV
    joblib.dump(best_model, run_dir / "model_best_cv.pkl")
    log.info("Modelo guardado: model_best_cv.pkl")
    
    cv_results = pd.DataFrame(gs.cv_results_)
    cv_results.to_csv(run_dir / "cv_results_full.csv", index=False)
    log.info("Resultados CV guardados: cv_results_full.csv")
    
    # ==================== PREDICCIONES CV (OUT-OF-FOLD) ====================
    log.info("\n" + "="*70)
    log.info("PREDICCIONES CROSS-VALIDATION (OUT-OF-FOLD)")
    log.info("="*70)
    
    # Generar predicciones out-of-fold para cada fold
    cv_preds = np.zeros_like(y_en, dtype=float)
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_en, y_en, groups=g_en), 1):
        log.info("  Fold {}/4: train={}, test={}".format(
            fold_idx, len(train_idx), len(test_idx)))
        
        model_fold = clone(best_model)
        model_fold.fit(X_en[train_idx], y_en[train_idx])
        cv_preds[test_idx] = model_fold.predict(X_en[test_idx])
    
    log.info("Predicciones out-of-fold completadas")
    
    # Evaluar CV
    evaluate_split(
        y_en, cv_preds, "CV_EN", run_dir, log,
        df_ids=df_en[['patient_id']]
    )
    
    # ==================== CALIBRACION ====================
    log.info("\n" + "="*70)
    log.info("CALIBRACION POST-HOC (Isotonic Regression)")
    log.info("="*70)
    
    log.info("Entrenando calibrador en predicciones CV...")
    calibrator = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
    calibrator.fit(cv_preds, y_en)
    
    joblib.dump(calibrator, run_dir / "calibrator.pkl")
    log.info("Calibrador guardado: calibrator.pkl")
    
    # Evaluar CV calibrado
    cv_preds_cal = calibrator.predict(cv_preds)
    evaluate_split(
        y_en, cv_preds_cal, "CV_EN_CALIBRATED", run_dir, log,
        df_ids=df_en[['patient_id']]
    )
    
    # ==================== MODELO FINAL ====================
    log.info("\n" + "="*70)
    log.info("ENTRENAMIENTO MODELO FINAL (TODO EN)")
    log.info("="*70)
    
    log.info("Reentrenando modelo en TODO el set EN...")
    best_model.fit(X_en, y_en)
    
    joblib.dump(best_model, run_dir / "model_final_all_EN.pkl")
    log.info("Modelo final guardado: model_final_all_EN.pkl")
    
    # ==================== EVALUACION EN_IN (IN-SAMPLE) ====================
    log.info("\n" + "="*70)
    log.info("EVALUACION IN-SAMPLE (EN)")
    log.info("="*70)
    
    log.info("Predicciones in-sample (diagnostico, no evaluacion real)...")
    
    # Raw predictions
    preds_en_in = best_model.predict(X_en)
    evaluate_split(
        y_en, preds_en_in, "EN_IN_RAW", run_dir, log,
        df_ids=df_en[['patient_id']]
    )
    
    # Calibrated predictions
    preds_en_in_cal = calibrator.predict(preds_en_in)
    evaluate_split(
        y_en, preds_en_in_cal, "EN_IN_CALIBRATED", run_dir, log,
        df_ids=df_en[['patient_id']]
    )
    
    # ==================== EVALUACION ES (INTERNAL VALIDATION) ====================
    if n_es > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACION INTERNAL VALIDATION (ES)")
        log.info("="*70)
        
        X_es = df_es[feat_cols].values
        y_es = df_es['QA'].values
        
        log.info("Evaluando en {} pacientes ES...".format(n_es))
        
        # Raw predictions
        preds_es = best_model.predict(X_es)
        evaluate_split(
            y_es, preds_es, "INT_ES_RAW", run_dir, log,
            df_ids=df_es[['patient_id']]
        )
        
        # Calibrated predictions
        preds_es_cal = calibrator.predict(preds_es)
        evaluate_split(
            y_es, preds_es_cal, "INT_ES_CALIBRATED", run_dir, log,
            df_ids=df_es[['patient_id']]
        )
    else:
        log.warning("No hay pacientes ES para evaluacion")
    
    # ==================== EVALUACION CA (EXTERNAL TEST) ====================
    if n_ca > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACION EXTERNAL TEST (CA)")
        log.info("="*70)
        
        X_ca = df_ca[feat_cols].values
        y_ca = df_ca['QA'].values
        
        log.info("Evaluando en {} pacientes CA...".format(n_ca))
        
        # Raw predictions
        preds_ca = best_model.predict(X_ca)
        evaluate_split(
            y_ca, preds_ca, "EXT_CA_RAW", run_dir, log,
            df_ids=df_ca[['patient_id']]
        )
        
        # Calibrated predictions
        preds_ca_cal = calibrator.predict(preds_ca)
        evaluate_split(
            y_ca, preds_ca_cal, "EXT_CA_CALIBRATED", run_dir, log,
            df_ids=df_ca[['patient_id']]
        )
    else:
        log.warning("No hay pacientes CA para evaluacion")
    
    # ==================== RESUMEN FINAL ====================
    log.info("\n" + "="*70)
    log.info("RESUMEN FINAL")
    log.info("="*70)
    
    log.info("\nDirectorio de resultados: {}".format(run_dir))
    
    log.info("\nPacientes procesados:")
    log.info("  EN (train):               {}".format(n_en))
    log.info("  ES (internal validation): {}".format(n_es))
    log.info("  CA (external test):       {}".format(n_ca))
    log.info("  TOTAL:                    {}".format(n_en + n_es + n_ca))
    
    log.info("\nFeatures utilizadas:")
    log.info("  DEN: {}".format(len([c for c in feat_cols if c.startswith('den_')])))
    log.info("  DYS: {}".format(len([c for c in feat_cols if c.startswith('dys_')])))
    log.info("  TOTAL: {}".format(len(feat_cols)))
    
    # Leer y mostrar metricas clave
    try:
        cv_metrics = pd.read_csv(run_dir / "CV_EN_CALIBRATED_metrics.csv").iloc[0]
        log.info("\nMetricas CV (calibrado):")
        log.info("  MAE:                {:.3f}".format(cv_metrics['MAE']))
        log.info("  R2:                 {:.3f}".format(cv_metrics['R2']))
        log.info("  Spearman:           {:.3f}".format(cv_metrics['Spearman_rho']))
        log.info("  Acc@5:              {:.1f}%".format(100*cv_metrics['Acc@5']))
        log.info("  Severity Accuracy:  {:.1f}%".format(100*cv_metrics['severity_accuracy']))
    except:
        pass
    
    # ==================== FINALIZACION ====================
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
        print("\n\n[INTERRUMPIDO] Proceso cancelado por el usuario")
        sys.exit(1)
    except Exception as e:
        print("\n\n[ERROR FATAL] {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
