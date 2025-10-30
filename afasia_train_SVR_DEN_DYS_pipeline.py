#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
SVR - Predicción WAB-AQ con features DEN+DYS (Le et al. 2018)
VERSIÓN ADAPTADA para dataset_FINAL_den_dys_wab.csv
==============================================================================
Pipeline completo:
- Carga features DEN+DYS + metadata desde CSV único
- Entrena SVR con GridSearchCV + GroupKFold (speaker-independent)
- Calibración post-hoc con Isotonic Regression
- Evaluación exhaustiva: CV, IN-SAMPLE, INT (ES), EXT (CA)
- Métricas: MAE, RMSE, R², Pearson, Spearman, Acc@1/5/10, Exact
- Visualizaciones: scatter, residuos, distribución de errores
- Logs detallados + archivos de resultados
==============================================================================
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
matplotlib.use("Agg")  # FIX: backend sin X para servidores
import matplotlib.pyplot as plt
from typing import Optional

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, pearsonr

# ======================== CONFIGURACIÓN PATHS ========================
DATA_BASE = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025"

# Dataset final con features + metadata
CSV_FINAL = os.path.join(DATA_BASE, "dataset_FINAL_den_dys_wab.csv")

# Output base
RESULTS_BASE = os.path.join(DATA_BASE, "resultados_svr")

# ======================== LOGGER ========================
def set_logger(run_dir: pathlib.Path) -> logging.Logger:
    """
    Configura logger con salida a consola + archivo
    
    Args:
        run_dir: directorio donde guardar console.log
    
    Returns:
        Logger configurado
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
    """Convierte Series a numérico con NaN para errores"""
    return pd.to_numeric(s, errors="coerce")

def to_int(pred):
    """Convierte predicciones a enteros en rango [0, 100]"""
    return np.rint(np.asarray(pred)).clip(0, 100).astype(int)

# ======================== MÉTRICAS ========================
def compute_metrics(y_true, y_pred):
    """
    Calcula métricas completas de evaluación
    
    Args:
        y_true: valores reales
        y_pred: predicciones
    
    Returns:
        dict con MAE, RMSE, R², Pearson, Spearman
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
    Calcula métricas de accuracy para predicciones enteras
    
    Args:
        y_true: valores reales (enteros)
        y_pred_int: predicciones redondeadas a enteros
    
    Returns:
        dict con Acc@1, Acc@5, Acc@10, Exact
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

# ======================== VISUALIZACIÓN ========================
def plot_scatter(y_true, y_pred, title, out_png, metrics=None):
    """
    Genera scatter plot con línea identidad y métricas
    
    Args:
        y_true: valores reales
        y_pred: predicciones
        title: título del gráfico
        out_png: ruta de salida PNG
        metrics: dict opcional con métricas pre-calculadas
    """
    if metrics is None:
        metrics = compute_metrics(y_true, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Línea identidad
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Identidad")
    
    # Etiquetas
    plt.xlabel("QA Real (0-100)", fontsize=12, fontweight='bold')
    plt.ylabel("QA Predicho (0-100)", fontsize=12, fontweight='bold')
    
    # Texto con métricas
    text = (f"MAE:      {metrics['MAE']:.2f}\n"
            f"RMSE:     {metrics['RMSE']:.2f}\n"
            f"R2:       {metrics['R2']:.3f}\n"
            f"Pearson:  {metrics['Pearson_r']:.3f}\n"
            f"Spearman: {metrics['Spearman_rho']:.3f}")
    
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
    Genera plots de análisis de residuos
    
    Args:
        y_true: valores reales
        y_pred: predicciones
        title: título general
        out_png: ruta de salida PNG
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
    axes[1].set_title("Distribución de Residuos", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Añadir estadísticas al histograma
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    axes[1].text(0.05, 0.95, 
                f"Media = {mean_res:.2f}\nStd = {std_res:.2f}",
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
    Genera distribución de errores absolutos
    
    Args:
        y_true: valores reales
        y_pred_int: predicciones enteras
        title: título del gráfico
        out_png: ruta de salida PNG
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
    
    # Líneas verticales en umbrales
    for x, label in [(1, "±1"), (5, "±5"), (10, "±10")]:
        plt.axvline(x=x, color='blue', linestyle='--', alpha=0.6, linewidth=2)
        plt.text(x, plt.ylim()[1] * 0.9, label, 
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Estadísticas
    pct_1 = 100 * np.mean(errors <= 1)
    pct_5 = 100 * np.mean(errors <= 5)
    pct_10 = 100 * np.mean(errors <= 10)
    
    stats_text = (f"Acc@1:  {pct_1:.1f}%\n"
                 f"Acc@5:  {pct_5:.1f}%\n"
                 f"Acc@10: {pct_10:.1f}%")
    
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

# ======================== EVALUACIÓN ========================
def evaluate_split(y_true, y_pred, split_name, run_dir, log, df_ids=None):
    """
    Evaluación completa de un split con todas las métricas y gráficos
    
    Args:
        y_true: valores reales
        y_pred: predicciones continuas
        split_name: nombre del split (e.g., 'CV_EN', 'INT_ES')
        run_dir: directorio de salida
        log: logger
        df_ids: DataFrame opcional con IDs de pacientes
    
    Returns:
        tuple (metrics_cont, metrics_int)
    """
    log.info(f"\n{'='*70}")
    log.info(f"EVALUACIÓN: {split_name}")
    log.info('='*70)
    
    # Métricas continuas
    metrics_cont = compute_metrics(y_true, y_pred)
    log.info(f"Métricas continuas:")
    log.info(f"  MAE:        {metrics_cont['MAE']:.3f}")
    log.info(f"  RMSE:       {metrics_cont['RMSE']:.3f}")
    log.info(f"  R2:         {metrics_cont['R2']:.3f}")
    log.info(f"  Pearson r:  {metrics_cont['Pearson_r']:.3f} (p={metrics_cont['Pearson_p']:.4f})")
    log.info(f"  Spearman:   {metrics_cont['Spearman_rho']:.3f} (p={metrics_cont['Spearman_p']:.4f})")
    
    # Métricas enteras
    y_pred_int = to_int(y_pred)
    metrics_int = compute_accuracy_metrics(y_true, y_pred_int)
    log.info(f"\nMétricas discretas (enteros [0-100]):")
    log.info(f"  Acc@1:  {100*metrics_int['Acc@1']:.2f}%")
    log.info(f"  Acc@5:  {100*metrics_int['Acc@5']:.2f}%")
    log.info(f"  Acc@10: {100*metrics_int['Acc@10']:.2f}%")
    log.info(f"  Exact:  {100*metrics_int['Exact']:.2f}%")
    
    # Guardar métricas en CSV
    all_metrics = {**metrics_cont, **metrics_int}
    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(run_dir / f"{split_name}_metrics.csv", index=False)
    log.info(f"Métricas guardadas: {split_name}_metrics.csv")
    
    # Guardar predicciones
    preds_df = pd.DataFrame({
        'QA_real': y_true,
        'QA_pred_cont': y_pred,
        'QA_pred_int': y_pred_int,
        'error_cont': y_pred - y_true,
        'error_int': y_pred_int - y_true,
        'abs_error_int': np.abs(y_pred_int - y_true)
    })
    
    # Añadir IDs si están disponibles
    if df_ids is not None:
        for col in df_ids.columns:
            preds_df[col] = df_ids[col].values
    
    preds_df.to_csv(run_dir / f"{split_name}_predictions.csv", index=False)
    log.info(f"Predicciones guardadas: {split_name}_predictions.csv")
    
    # Generar gráficos
    log.info(f"Generando gráficos...")
    
    plot_scatter(
        y_true, y_pred,
        f"{split_name} - Scatter Plot",
        run_dir / f"{split_name}_scatter.png",
        metrics_cont
    )
    
    plot_residuals(
        y_true, y_pred,
        f"{split_name} - Análisis de Residuos",
        run_dir / f"{split_name}_residuals.png"
    )
    
    plot_error_distribution(
        y_true, y_pred_int,
        f"{split_name} - Distribución de Errores",
        run_dir / f"{split_name}_errors.png"
    )
    
    log.info(f"Gráficos guardados: {split_name}_*.png")
    
    return metrics_cont, metrics_int

# ======================== MAIN ========================
def main():
    """Pipeline completo de entrenamiento y evaluación"""
    
    # ==================== SETUP ====================
    # Crear directorio de resultados
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(RESULTS_BASE) / f"SVR_DEN_DYS_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    log = set_logger(run_dir)
    log.info("="*70)
    log.info("INICIO: Predicción WAB-AQ con features DEN+DYS")
    log.info("Metodología: Le et al. (2018)")
    log.info("="*70)
    log.info(f"Directorio de resultados: {run_dir}")
    log.info(f"Timestamp: {ts}")
    
    # Copiar este script a resultados
    try:
        shutil.copy2(__file__, run_dir / pathlib.Path(__file__).name)
        log.info(f"Script copiado a resultados")
    except Exception as e:
        log.warning(f"No se pudo copiar script: {e}")
    
    # ==================== CARGAR DATOS ====================
    log.info("\n" + "="*70)
    log.info("CARGANDO DATOS")
    log.info("="*70)
    
    # Verificar archivo
    if not os.path.exists(CSV_FINAL):
        log.error(f"No existe archivo: {CSV_FINAL}")
        log.error("El archivo debe contener features DEN+DYS + metadata (QA, language)")
        sys.exit(1)
    
    # Cargar dataset final
    log.info(f"Cargando dataset desde: {CSV_FINAL}")
    df = pd.read_csv(CSV_FINAL)
    log.info(f"Dataset cargado: {len(df)} pacientes")
    
    # Adaptar nombre de columna de idioma
    if 'language' in df.columns and 'lang' not in df.columns:
        df['lang'] = df['language']
        log.info("Columna 'language' renombrada a 'lang'")
    
    # Filtrar pacientes con QA
    df = df.dropna(subset=['QA'])
    log.info(f"Pacientes con QA válido: {len(df)}")
    
    if len(df) < 10:
        log.error(f"Insuficientes pacientes para entrenamiento (n={len(df)} < 10)")
        sys.exit(1)
    
    # ==================== PREPARAR FEATURES ====================
    log.info("\n" + "="*70)
    log.info("PREPARANDO FEATURES")
    log.info("="*70)
    
    # Identificar columnas de features
    den_cols = [c for c in df.columns if c.startswith('den_')]
    dys_cols = [c for c in df.columns if c.startswith('dys_')]
    feat_cols = den_cols + dys_cols
    
    log.info(f"Features detectadas:")
    log.info(f"  DEN: {len(den_cols)}")
    log.info(f"  DYS: {len(dys_cols)}")
    log.info(f"  TOTAL: {len(feat_cols)}")
    
    # Filtrar solo columnas numéricas
    feat_cols_numeric = df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(feat_cols_numeric) < len(feat_cols):
        log.warning(f"Se filtraron {len(feat_cols) - len(feat_cols_numeric)} features no numéricas")
        feat_cols = feat_cols_numeric
    
    if len(feat_cols) == 0:
        log.error("No hay features numéricas disponibles")
        sys.exit(1)
    
    log.info(f"Features numéricas finales: {len(feat_cols)}")
    
    # Guardar lista de features usadas
    with open(run_dir / "features_used.txt", "w", encoding="utf-8") as f:
        f.write(f"FEATURES UTILIZADAS\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total: {len(feat_cols)}\n\n")
        
        f.write(f"DEN features ({len([c for c in feat_cols if c.startswith('den_')])}):\n")
        for c in sorted([c for c in feat_cols if c.startswith('den_')]):
            f.write(f"  - {c}\n")
        
        f.write(f"\nDYS features ({len([c for c in feat_cols if c.startswith('dys_')])}):\n")
        for c in sorted([c for c in feat_cols if c.startswith('dys_')]):
            f.write(f"  - {c}\n")
    
    log.info(f"Lista de features guardada: features_used.txt")
    
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
    
    log.info(f"Distribución por idioma:")
    log.info(f"  EN (train):               {n_en} pacientes")
    log.info(f"  ES (internal validation): {n_es} pacientes")
    log.info(f"  CA (external test):       {n_ca} pacientes")
    
    if n_en < 4:
        log.error(f"Se requieren ≥4 pacientes EN para GroupKFold (n_en={n_en})")
        sys.exit(1)
    
    # Preparar datos EN para entrenamiento
    X_en = df_en[feat_cols].values
    y_en = df_en['QA'].values
    g_en = df_en['patient_id'].values  # Grupos para GroupKFold
    
    log.info(f"\nDatos de entrenamiento (EN):")
    log.info(f"  X shape: {X_en.shape}")
    log.info(f"  y shape: {y_en.shape}")
    log.info(f"  Grupos únicos: {len(np.unique(g_en))}")
    log.info(f"  QA range: [{y_en.min():.1f}, {y_en.max():.1f}]")
    log.info(f"  QA mean ± std: {y_en.mean():.2f} ± {y_en.std():.2f}")
    
    # ==================== MODELO ====================
    log.info("\n" + "="*70)
    log.info("CONFIGURACIÓN Y ENTRENAMIENTO DEL MODELO")
    log.info("="*70)
    
    # Pipeline: Imputer → Scaler → SVR
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])
    
    log.info("Pipeline:")
    log.info("  1. SimpleImputer (strategy='median')")
    log.info("  2. StandardScaler")
    log.info("  3. SVR")
    
    # Grid de hiperparámetros (Le et al. 2018)
    param_grid = {
        "svr__C": [1, 10, 100],
        "svr__epsilon": [0.1, 1],
        "svr__kernel": ["rbf", "linear"],
        "svr__shrinking": [True, False]
    }
    
    log.info(f"\nHiperparámetros a explorar:")
    total_combinations = 1
    for param, values in param_grid.items():
        log.info(f"  {param}: {values}")
        total_combinations *= len(values)
    log.info(f"  Total combinaciones: {total_combinations}")
    
    # GridSearchCV con GroupKFold (speaker-independent)
    gkf = GroupKFold(n_splits=4)
    log.info(f"\nCross-validation: GroupKFold (n_splits=4)")
    log.info(f"  Objetivo: speaker-independent evaluation")
    log.info(f"  Métrica: negative MAE")
    
    log.info(f"\nIniciando GridSearchCV...")
    log.info(f"  (esto puede tomar varios minutos)")
    
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=1,  # Sin paralelización para evitar cuelgues
        verbose=2,
        return_train_score=True
    )
    
    gs.fit(X_en, y_en, groups=g_en)
    
    best_model = gs.best_estimator_
    
    log.info(f"\n{'='*70}")
    log.info(f"GridSearchCV COMPLETADO")
    log.info('='*70)
    log.info(f"Mejores hiperparámetros:")
    for param, value in gs.best_params_.items():
        log.info(f"  {param}: {value}")
    log.info(f"Mejor MAE (CV): {-gs.best_score_:.3f}")
    
    # Guardar modelo y resultados de CV
    joblib.dump(best_model, run_dir / "model_best_cv.pkl")
    log.info(f"Modelo guardado: model_best_cv.pkl")
    
    cv_results = pd.DataFrame(gs.cv_results_)
    cv_results.to_csv(run_dir / "cv_results_full.csv", index=False)
    log.info(f"Resultados CV guardados: cv_results_full.csv")
    
    # ==================== PREDICCIONES CV (OUT-OF-FOLD) ====================
    log.info("\n" + "="*70)
    log.info("PREDICCIONES CROSS-VALIDATION (OUT-OF-FOLD)")
    log.info("="*70)
    
    # Generar predicciones out-of-fold para cada fold
    cv_preds = np.zeros_like(y_en, dtype=float)
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_en, y_en, groups=g_en), 1):
        log.info(f"  Fold {fold_idx}/4: train={len(train_idx)}, test={len(test_idx)}")
        
        model_fold = clone(best_model)
        model_fold.fit(X_en[train_idx], y_en[train_idx])
        cv_preds[test_idx] = model_fold.predict(X_en[test_idx])
    
    log.info("Predicciones out-of-fold completadas")
    
    # Evaluar CV
    evaluate_split(
        y_en, cv_preds, "CV_EN", run_dir, log,
        df_ids=df_en[['patient_id']]
    )
    
    # ==================== CALIBRACIÓN ====================
    log.info("\n" + "="*70)
    log.info("CALIBRACIÓN POST-HOC (Isotonic Regression)")
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
    
    # ==================== EVALUACIÓN EN_IN (IN-SAMPLE) ====================
    log.info("\n" + "="*70)
    log.info("EVALUACIÓN IN-SAMPLE (EN)")
    log.info("="*70)
    
    log.info("Predicciones in-sample (diagnóstico, no evaluación real)...")
    
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
    
    # ==================== EVALUACIÓN ES (INTERNAL VALIDATION) ====================
    if n_es > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACIÓN INTERNAL VALIDATION (ES)")
        log.info("="*70)
        
        X_es = df_es[feat_cols].values
        y_es = df_es['QA'].values
        
        log.info(f"Evaluando en {n_es} pacientes ES...")
        
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
        log.warning("No hay pacientes ES para evaluación")
    
    # ==================== EVALUACIÓN CA (EXTERNAL TEST) ====================
    if n_ca > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACIÓN EXTERNAL TEST (CA)")
        log.info("="*70)
        
        X_ca = df_ca[feat_cols].values
        y_ca = df_ca['QA'].values
        
        log.info(f"Evaluando en {n_ca} pacientes CA...")
        
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
        log.warning("No hay pacientes CA para evaluación")
    
    # ==================== RESUMEN FINAL ====================
    log.info("\n" + "="*70)
    log.info("RESUMEN FINAL")
    log.info("="*70)
    
    log.info(f"\nDirectorio de resultados: {run_dir}")
    
    log.info(f"\nPacientes procesados:")
    log.info(f"  EN (train):               {n_en}")
    log.info(f"  ES (internal validation): {n_es}")
    log.info(f"  CA (external test):       {n_ca}")
    log.info(f"  TOTAL:                    {n_en + n_es + n_ca}")
    
    log.info(f"\nFeatures utilizadas:")
    log.info(f"  DEN: {len([c for c in feat_cols if c.startswith('den_')])}")
    log.info(f"  DYS: {len([c for c in feat_cols if c.startswith('dys_')])}")
    log.info(f"  TOTAL: {len(feat_cols)}")
    
    # Leer y mostrar métricas clave
    try:
        cv_metrics = pd.read_csv(run_dir / "CV_EN_CALIBRATED_metrics.csv").iloc[0]
        log.info(f"\nMétricas CV (calibrado):")
        log.info(f"  MAE:        {cv_metrics['MAE']:.3f}")
        log.info(f"  R2:         {cv_metrics['R2']:.3f}")
        log.info(f"  Spearman:   {cv_metrics['Spearman_rho']:.3f}")
        log.info(f"  Acc@5:      {100*cv_metrics['Acc@5']:.1f}%")
    except:
        pass
    
    # ==================== FINALIZACIÓN ====================
    log.info("\n" + "="*70)
    log.info("PROCESO COMPLETADO EXITOSAMENTE")
    log.info("="*70)
    log.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Resultados en: {run_dir}")
    log.info("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUMPIDO] Proceso cancelado por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
