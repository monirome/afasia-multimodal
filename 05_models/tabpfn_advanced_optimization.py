#!/usr/bin/env python3
# 05_models/tabpfn_advanced_optimization.py
# -*- coding: utf-8 -*-
"""
TABPFN ADVANCED OPTIMIZATION & EXPLAINABILITY

FEATURES:
- Optimizacion de n_estimators (16, 32, 64, 128, 256)
- Ensemble con multiples seeds para reducir varianza
- Explicabilidad avanzada: LIME, SHAP, Permutation Importance
- Analisis de incertidumbre (prediccion + intervalo de confianza)
- Feature importance por grupos (DEN, DYS, LEX, POSLM)
- Comparacion automatica de todas las configuraciones
"""

import os
import sys
import datetime
import pathlib
import warnings
import json
warnings.filterwarnings("ignore")

import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Importar TabPFN
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    print("ERROR: TabPFN no disponible - pip install tabpfn")
    sys.exit(1)

# LIME
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("WARNING: LIME no disponible - pip install lime")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP no disponible - pip install shap")

# ======================== PATHS ========================
PROJECT_BASE = pathlib.Path(__file__).resolve().parent.parent
DATASET_CSV = PROJECT_BASE / "data" / "dataset_FINAL_CON_POSLM.csv"
RESULTS_BASE = PROJECT_BASE / "outputs" / "experiments" / "TABPFN_OPTIMIZATION"

os.makedirs(RESULTS_BASE, exist_ok=True)

# ======================== CONFIGURACION ========================
EXPERIMENTS_CONFIG = {
    'n_estimators_list': [16, 32, 64, 128, 256],
    'ensemble_seeds': [42, 123, 456, 789, 1011],
    'cv_strategy': 'subdataset',
    'cv_folds': 4,
    'lime_samples': 50,
    'shap_samples': 100,
}

# ======================== LOGGER ========================
def set_logger(run_dir):
    log = logging.getLogger("TabPFN_Optimization")
    log.setLevel(logging.INFO)
    
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", 
                           datefmt="%Y-%m-%d %H:%M:%S")
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "optimization.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    
    for h in list(log.handlers):
        log.removeHandler(h)
    log.addHandler(ch)
    log.addHandler(fh)
    
    return log

# ======================== UTILIDADES ========================
def extract_subdataset_from_patient_id(patient_ids):
    """Extrae sub-dataset desde patient_id"""
    if isinstance(patient_ids, pd.Series):
        s = patient_ids.astype(str)
    else:
        s = pd.Series(list(patient_ids), dtype=object).astype(str)
    
    sub = s.str.extract(r'^([A-Za-z]+)', expand=False).str.lower()
    sub = sub.fillna("unknown")
    return sub.astype(str).values

def compute_metrics(y_true, y_pred):
    """Calcula metricas de regresion"""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    try:
        pearson_r, _ = pearsonr(y_true, y_pred)
    except:
        pearson_r = np.nan
    
    return {'MAE': mae, 'R2': r2, 'Pearson_r': pearson_r}

# ======================== TABPFN CON DIFERENTES N_ESTIMATORS ========================
def train_tabpfn_with_n_estimators(X_train, y_train, X_test, y_test, 
                                   n_estimators, fold_idx, log):
    """Entrena TabPFN con n_estimators especifico"""
    log.info(f"      n_estimators={n_estimators}...", end="")
    
    model = TabPFNRegressor(device='cpu', n_estimators=n_estimators)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    log.info(f" MAE={mae:.3f}")
    
    return model, y_pred, mae

# ======================== ENSEMBLE CON MULTIPLES SEEDS ========================
def train_ensemble_seeds(X_train, y_train, X_test, y_test, n_estimators, 
                        seeds, fold_idx, log):
    """Entrena ensemble de TabPFN con diferentes seeds"""
    log.info(f"      Ensemble (n_est={n_estimators}, seeds={len(seeds)})...")
    
    predictions = []
    
    for seed in seeds:
        np.random.seed(seed)
        
        model = TabPFNRegressor(device='cpu', n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
    
    # Promedio de predicciones
    y_pred_mean = np.mean(predictions, axis=0)
    y_pred_std = np.std(predictions, axis=0)
    
    mae = mean_absolute_error(y_test, y_pred_mean)
    
    log.info(f"        MAE={mae:.3f} (std={y_pred_std.mean():.2f})")
    
    return y_pred_mean, y_pred_std, mae

# ======================== LIME EXPLAINER ========================
def explain_with_lime(model, X_train, X_test, feature_names, run_dir, log, 
                     num_samples=50):
    """Explica predicciones con LIME"""
    if not LIME_AVAILABLE:
        log.warning("LIME no disponible")
        return None
    
    log.info("\n  LIME - Explicaciones locales...")
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode='regression',
        random_state=42
    )
    
    # Explicar primeras num_samples predicciones
    n_explain = min(num_samples, len(X_test))
    
    all_importances = []
    
    for i in range(n_explain):
        exp = explainer.explain_instance(
            X_test[i],
            model.predict,
            num_features=len(feature_names)
        )
        
        # Extraer importancias
        importance_dict = dict(exp.as_list())
        
        # Convertir a feature names
        feature_importances = {}
        for feat_name in feature_names:
            # LIME devuelve intervalos, buscar coincidencia
            matched_importance = 0.0
            for key, val in importance_dict.items():
                if feat_name in key:
                    matched_importance = abs(val)
                    break
            feature_importances[feat_name] = matched_importance
        
        all_importances.append(feature_importances)
    
    # Promedio de importancias
    importance_df = pd.DataFrame(all_importances)
    mean_importances = importance_df.mean().sort_values(ascending=False)
    
    lime_df = pd.DataFrame({
        'feature': mean_importances.index,
        'importance': mean_importances.values
    })
    
    lime_df.to_csv(run_dir / "lime_importances.csv", index=False)
    
    log.info(f"    Top 10 features (LIME):")
    for idx, (feat, imp) in enumerate(mean_importances.head(10).items(), 1):
        log.info(f"      {idx:2d}. {feat:40s} {imp:.4f}")
    
    # Grafico
    plt.figure(figsize=(10, 12))
    top_features = lime_df.head(30)
    plt.barh(range(len(top_features)), top_features['importance'].values, 
             alpha=0.8, color='coral', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=9)
    plt.xlabel('LIME Importance', fontsize=12, fontweight='bold')
    plt.title('Top 30 Features - LIME', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "lime_importances.png", dpi=150)
    plt.close()
    
    return lime_df

# ======================== SHAP MEJORADO ========================
def explain_with_shap(model, X_train, X_test, feature_names, run_dir, log,
                     max_samples=100):
    """Explica con SHAP (optimizado)"""
    if not SHAP_AVAILABLE:
        log.warning("SHAP no disponible")
        return None
    
    log.info("\n  SHAP - Valores de Shapley...")
    
    # Background data (sample)
    n_bg = min(max_samples, len(X_train))
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), n_bg, replace=False)
    X_bg = X_train[bg_idx]
    
    log.info(f"    Background: {n_bg} samples")
    
    # Explicar test data (sample)
    n_test = min(max_samples, len(X_test))
    test_idx = np.random.choice(len(X_test), n_test, replace=False)
    X_test_sample = X_test[test_idx]
    
    log.info(f"    Explicando: {n_test} samples")
    
    # SHAP KernelExplainer
    explainer = shap.KernelExplainer(model.predict, X_bg)
    shap_values = explainer.shap_values(X_test_sample)
    
    # Guardar
    pd.DataFrame(shap_values, columns=feature_names).to_csv(
        run_dir / "shap_values.csv", index=False)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(run_dir / "shap_importances.csv", index=False)
    
    log.info(f"    Top 10 features (SHAP):")
    for idx, (i, row) in enumerate(importance_df.head(10).iterrows(), 1):
        log.info(f"      {idx:2d}. {row['feature']:40s} {row['importance']:.4f}")
    
    # Graficos
    try:
        # Beeswarm
        plt.figure(figsize=(10, 12))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                         show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(run_dir / "shap_beeswarm.png", dpi=150)
        plt.close()
        
        # Bar
        plt.figure(figsize=(10, 12))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names,
                         plot_type="bar", show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(run_dir / "shap_bar.png", dpi=150)
        plt.close()
    except Exception as e:
        log.warning(f"    Error graficos SHAP: {e}")
    
    return importance_df

# ======================== ANALISIS POR GRUPOS ========================
def analyze_by_groups(importance_df, run_dir, log, method_name=""):
    """Analiza feature importance por grupos"""
    if importance_df is None or len(importance_df) == 0:
        return None
    
    def get_group(feat):
        if feat.startswith('den_'): return 'DEN'
        elif feat.startswith('dys_'): return 'DYS'
        elif feat.startswith('lex_'): return 'LEX'
        elif feat.startswith('poslm_'): return 'POSLM'
        else: return 'OTHER'
    
    importance_df = importance_df.copy()
    importance_df['group'] = importance_df['feature'].apply(get_group)
    
    group_stats = importance_df.groupby('group')['importance'].agg(
        ['sum', 'mean', 'count']).sort_values('sum', ascending=False)
    
    log.info(f"\n  Feature groups ({method_name}):")
    total_imp = group_stats['sum'].sum()
    
    for grp in group_stats.index:
        total = group_stats.loc[grp, 'sum']
        count = int(group_stats.loc[grp, 'count'])
        pct = 100 * total / total_imp if total_imp > 0 else 0
        log.info(f"    {grp:8s}: {total:8.4f} ({count:3d} feats, {pct:5.1f}%)")
    
    # Grafico
    plt.figure(figsize=(10, 6))
    colors = {'DEN': 'steelblue', 'DYS': 'coral', 'LEX': 'mediumseagreen', 
              'POSLM': 'goldenrod', 'OTHER': 'gray'}
    bar_colors = [colors.get(g, 'gray') for g in group_stats.index]
    
    plt.bar(group_stats.index, group_stats['sum'], alpha=0.8, 
           edgecolor='black', linewidth=1.5, color=bar_colors)
    plt.xlabel('Feature Group', fontsize=12, fontweight='bold')
    plt.ylabel('Total Importance', fontsize=12, fontweight='bold')
    
    title = f'Feature Importance by Group'
    if method_name:
        title += f' ({method_name})'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filename = f"groups_{method_name.lower().replace(' ', '_')}" if method_name else "groups"
    plt.savefig(run_dir / f"{filename}.png", dpi=150)
    plt.close()
    
    group_stats.to_csv(run_dir / f"{filename}.csv")
    
    return group_stats

# ======================== ANALISIS DE INCERTIDUMBRE ========================
def uncertainty_analysis(predictions_list, y_test, run_dir, log):
    """Analiza incertidumbre de predicciones (ensemble)"""
    log.info("\n  Analisis de incertidumbre...")
    
    # predictions_list: lista de predicciones de diferentes seeds
    predictions_array = np.array(predictions_list)
    
    mean_pred = predictions_array.mean(axis=0)
    std_pred = predictions_array.std(axis=0)
    
    # Intervalo de confianza 95%
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred
    
    # Cuantas predicciones caen dentro del IC
    within_ci = ((y_test >= ci_lower) & (y_test <= ci_upper)).mean()
    
    log.info(f"    Incertidumbre promedio (std): {std_pred.mean():.2f}")
    log.info(f"    Predicciones dentro IC 95%: {within_ci*100:.1f}%")
    
    # Guardar
    uncertainty_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred_mean': mean_pred,
        'y_pred_std': std_pred,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'within_ci': (y_test >= ci_lower) & (y_test <= ci_upper)
    })
    
    uncertainty_df.to_csv(run_dir / "uncertainty_analysis.csv", index=False)
    
    # Grafico de incertidumbre
    plt.figure(figsize=(12, 6))
    
    sorted_idx = np.argsort(std_pred)
    
    plt.errorbar(range(len(y_test)), mean_pred[sorted_idx], 
                yerr=1.96*std_pred[sorted_idx], fmt='o', alpha=0.5,
                label='Prediccion (IC 95%)', color='steelblue')
    plt.scatter(range(len(y_test)), y_test[sorted_idx], 
               color='red', alpha=0.7, s=30, label='Real')
    
    plt.xlabel('Paciente (ordenado por incertidumbre)', fontsize=12)
    plt.ylabel('QA Score', fontsize=12)
    plt.title('Incertidumbre de Predicciones', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "uncertainty_plot.png", dpi=150)
    plt.close()
    
    return uncertainty_df

# ======================== MAIN ========================
def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE / f"OPTIMIZATION_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log = set_logger(run_dir)
    
    log.info("="*70)
    log.info("TABPFN ADVANCED OPTIMIZATION & EXPLAINABILITY")
    log.info("="*70)
    log.info(f"Output: {run_dir}")
    log.info("\nExperimentos:")
    log.info(f"  n_estimators: {EXPERIMENTS_CONFIG['n_estimators_list']}")
    log.info(f"  ensemble seeds: {EXPERIMENTS_CONFIG['ensemble_seeds']}")
    log.info(f"  CV strategy: {EXPERIMENTS_CONFIG['cv_strategy']}")
    log.info(f"  CV folds: {EXPERIMENTS_CONFIG['cv_folds']}")
    
    # Guardar config
    with open(run_dir / "config.json", "w") as f:
        json.dump(EXPERIMENTS_CONFIG, f, indent=2)
    
    # CARGAR DATOS
    log.info("\n" + "="*70)
    log.info("CARGANDO DATOS")
    log.info("="*70)
    
    df = pd.read_csv(DATASET_CSV)
    df = df[df['QA'].notna()].copy()
    
    # Filtrar PWA + EN
    df_pwa = df[df['group'] == 'pwa'].copy()
    df_en = df_pwa[df_pwa['language'].str.lower() == 'en'].copy()
    
    log.info(f"PWA EN: {len(df_en)}")
    
    # Features
    feat_cols = sorted([c for c in df.columns 
                       if c.startswith(('den_', 'dys_', 'lex_', 'poslm_'))])
    
    log.info(f"Features: {len(feat_cols)}")
    
    # Preparar datos
    X = df_en[feat_cols].values
    y = df_en['QA'].values
    groups = df_en['patient_id'].values
    
    # Sub-datasets
    subsets = extract_subdataset_from_patient_id(df_en['patient_id'])
    
    # CV
    log.info("\n" + "="*70)
    log.info(f"CROSS-VALIDATION ({EXPERIMENTS_CONFIG['cv_strategy']})")
    log.info("="*70)
    
    cv_splitter = StratifiedGroupKFold(
        n_splits=EXPERIMENTS_CONFIG['cv_folds'], 
        shuffle=True, 
        random_state=42
    )
    
    # DICCIONARIO PARA GUARDAR RESULTADOS
    all_results = []
    
    # OOF predictions para cada configuracion
    oof_predictions = {}
    
    # ======================== EXPERIMENTOS ========================
    log.info("\n" + "="*70)
    log.info("EXPERIMENTOS")
    log.info("="*70)
    
    # 1. EXPERIMENTOS CON DIFERENTES N_ESTIMATORS
    log.info("\n[1/2] Probando diferentes n_estimators...")
    
    for n_est in EXPERIMENTS_CONFIG['n_estimators_list']:
        log.info(f"\n  n_estimators = {n_est}")
        
        oof_preds = np.zeros_like(y, dtype=float)
        
        fold_maes = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(
            cv_splitter.split(X, subsets, groups=groups), 1):
            
            log.info(f"    Fold {fold_idx}/{EXPERIMENTS_CONFIG['cv_folds']}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Imputar
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
            # Normalizar
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Entrenar
            model, y_pred, mae = train_tabpfn_with_n_estimators(
                X_train, y_train, X_test, y_test, 
                n_est, fold_idx, log
            )
            
            oof_preds[test_idx] = y_pred
            fold_maes.append(mae)
        
        # Metricas globales
        metrics = compute_metrics(y, oof_preds)
        
        log.info(f"\n  RESULTADOS n_estimators={n_est}:")
        log.info(f"    MAE:      {metrics['MAE']:.4f}")
        log.info(f"    R2:       {metrics['R2']:.4f}")
        log.info(f"    Pearson:  {metrics['Pearson_r']:.4f}")
        
        all_results.append({
            'experiment': f'single_n{n_est}',
            'n_estimators': n_est,
            'n_seeds': 1,
            'mae': metrics['MAE'],
            'r2': metrics['R2'],
            'pearson': metrics['Pearson_r']
        })
        
        oof_predictions[f'single_n{n_est}'] = oof_preds
    
    # 2. ENSEMBLE CON MULTIPLES SEEDS
    log.info("\n[2/2] Ensemble con multiples seeds...")
    
    # Probar con los mejores n_estimators
    best_n_estimators = [64, 128]
    
    for n_est in best_n_estimators:
        log.info(f"\n  Ensemble: n_estimators={n_est}, seeds={len(EXPERIMENTS_CONFIG['ensemble_seeds'])}")
        
        oof_preds_mean = np.zeros_like(y, dtype=float)
        oof_preds_std = np.zeros_like(y, dtype=float)
        
        all_fold_predictions = []
        
        fold_maes = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(
            cv_splitter.split(X, subsets, groups=groups), 1):
            
            log.info(f"    Fold {fold_idx}/{EXPERIMENTS_CONFIG['cv_folds']}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Imputar
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
            # Normalizar
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Ensemble
            y_pred_mean, y_pred_std, mae = train_ensemble_seeds(
                X_train, y_train, X_test, y_test,
                n_est, EXPERIMENTS_CONFIG['ensemble_seeds'],
                fold_idx, log
            )
            
            oof_preds_mean[test_idx] = y_pred_mean
            oof_preds_std[test_idx] = y_pred_std
            fold_maes.append(mae)
        
        # Metricas globales
        metrics = compute_metrics(y, oof_preds_mean)
        
        log.info(f"\n  RESULTADOS Ensemble n_estimators={n_est}:")
        log.info(f"    MAE:      {metrics['MAE']:.4f}")
        log.info(f"    R2:       {metrics['R2']:.4f}")
        log.info(f"    Pearson:  {metrics['Pearson_r']:.4f}")
        log.info(f"    Std mean: {oof_preds_std.mean():.2f}")
        
        all_results.append({
            'experiment': f'ensemble_n{n_est}',
            'n_estimators': n_est,
            'n_seeds': len(EXPERIMENTS_CONFIG['ensemble_seeds']),
            'mae': metrics['MAE'],
            'r2': metrics['R2'],
            'pearson': metrics['Pearson_r'],
            'uncertainty_mean': oof_preds_std.mean()
        })
        
        oof_predictions[f'ensemble_n{n_est}'] = oof_preds_mean
    
    # GUARDAR RESULTADOS
    log.info("\n" + "="*70)
    log.info("GUARDANDO RESULTADOS")
    log.info("="*70)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(run_dir / "all_experiments.csv", index=False)
    
    log.info("\nResultados:")
    log.info(results_df.to_string(index=False))
    
    # Mejor configuracion
    best_idx = results_df['mae'].idxmin()
    best_config = results_df.loc[best_idx]
    
    log.info("\n" + "="*70)
    log.info("MEJOR CONFIGURACION")
    log.info("="*70)
    log.info(f"Experimento: {best_config['experiment']}")
    log.info(f"n_estimators: {best_config['n_estimators']}")
    log.info(f"n_seeds: {best_config['n_seeds']}")
    log.info(f"MAE: {best_config['mae']:.4f}")
    log.info(f"R2: {best_config['r2']:.4f}")
    log.info(f"Pearson: {best_config['pearson']:.4f}")
    
    # EXPLICABILIDAD CON MEJOR MODELO
    log.info("\n" + "="*70)
    log.info("EXPLICABILIDAD - MEJOR MODELO")
    log.info("="*70)
    
    # Re-entrenar mejor modelo para explicabilidad
    best_n_est = int(best_config['n_estimators'])
    
    # Usar primer fold para explicabilidad
    train_idx, test_idx = next(cv_splitter.split(X, subsets, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Preprocesar
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Entrenar
    model_best = TabPFNRegressor(device='cpu', n_estimators=best_n_est)
    model_best.fit(X_train, y_train)
    
    # LIME
    if LIME_AVAILABLE:
        lime_df = explain_with_lime(
            model_best, X_train, X_test, feat_cols, run_dir, log,
            num_samples=EXPERIMENTS_CONFIG['lime_samples']
        )
        
        if lime_df is not None:
            analyze_by_groups(lime_df, run_dir, log, method_name="LIME")
    
    # SHAP
    if SHAP_AVAILABLE:
        shap_df = explain_with_shap(
            model_best, X_train, X_test, feat_cols, run_dir, log,
            max_samples=EXPERIMENTS_CONFIG['shap_samples']
        )
        
        if shap_df is not None:
            analyze_by_groups(shap_df, run_dir, log, method_name="SHAP")
    
    # PERMUTATION IMPORTANCE
    log.info("\n  Permutation Importance...")
    from sklearn.inspection import permutation_importance
    
    perm_imp = permutation_importance(
        model_best, X_test, y_test, 
        n_repeats=10, random_state=42, 
        scoring='neg_mean_absolute_error'
    )
    
    perm_df = pd.DataFrame({
        'feature': feat_cols,
        'importance': perm_imp.importances_mean
    }).sort_values('importance', ascending=False)
    
    perm_df.to_csv(run_dir / "permutation_importances.csv", index=False)
    
    log.info(f"    Top 10 features (Permutation):")
    for idx, (i, row) in enumerate(perm_df.head(10).iterrows(), 1):
        log.info(f"      {idx:2d}. {row['feature']:40s} {row['importance']:.4f}")
    
    analyze_by_groups(perm_df, run_dir, log, method_name="Permutation")
    
    # GRAFICO COMPARATIVO
    log.info("\n" + "="*70)
    log.info("GRAFICO COMPARATIVO")
    log.info("="*70)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(results_df))
    colors = ['steelblue' if 'single' in exp else 'coral' 
             for exp in results_df['experiment']]
    
    bars = ax.bar(x_pos, results_df['mae'], alpha=0.8, 
                  edgecolor='black', linewidth=1.5, color=colors)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['experiment'], rotation=45, ha='right')
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title('Comparacion de Configuraciones TabPFN', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Marcar el mejor
    best_bar = bars[best_idx]
    best_bar.set_color('green')
    best_bar.set_alpha(1.0)
    
    plt.tight_layout()
    plt.savefig(run_dir / "comparison.png", dpi=150)
    plt.close()
    
    log.info(f"Grafico guardado: comparison.png")
    
    log.info("\n" + "="*70)
    log.info("PROCESO COMPLETADO")
    log.info("="*70)
    log.info(f"Resultados en: {run_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)