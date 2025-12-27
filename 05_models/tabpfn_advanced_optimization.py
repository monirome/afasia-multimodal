#!/usr/bin/env python3
# 05_models/tabpfn_advanced_optimization.py
# -*- coding: utf-8 -*-
"""
TABPFN ADVANCED OPTIMIZATION - VERSIÓN "MONSTRUO" COMPLETA
Combina la optimización avanzada con la metodología rigurosa universal.

FEATURES:
- Optimización de n_estimators y Ensemble Seeds.
- Metodología Universal: 
    * CV por Severidad o Subdataset.
    * Filtrado POS-LM.
    * Selección de Características (SelectKBest) dentro del fold.
- Visualización TOTAL:
    * Scatter Plots con métricas de precisión.
    * Matrices de Confusión.
    * Histogramas de Error.
    * Gráfico de Barras Comparativo de Experimentos.
- Explicabilidad: SHAP, LIME, Permutation Importance.
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.feature_selection import SelectKBest, mutual_info_regression

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
    # --- Hiperparámetros ---
    'n_estimators_list': [16, 32, 64, 128], # Lista completa
    'ensemble_seeds': [42, 123, 456, 789, 1011],
    
    # --- Metodología (Alineada) ---
    'cv_strategy': 'subdataset',  # 'subdataset' (Paper) o 'severity' (Tu Récord)
    'poslm_method': 'backoff',    # 'none', 'kneser-ney', 'backoff', 'all'
    'feature_selection': 'kbest', # 'kbest' o 'full'
    'k_features': 40,             # Número de features a mantener
    
    # --- Configuración General ---
    'cv_folds': 4,
    'lime_samples': 50,
    'shap_samples': 100,
}

# ======================== LOGGER ========================
def set_logger(run_dir):
    log = logging.getLogger("TabPFN_Opt")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "optimization.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    # Limpiar handlers previos
    if log.hasHandlers(): log.handlers.clear()
    log.addHandler(ch)
    log.addHandler(fh)
    return log

# ======================== UTILIDADES DE METODOLOGÍA ========================
def extract_subdataset_from_patient_id(patient_ids):
    """Extrae sub-dataset desde patient_id"""
    if isinstance(patient_ids, pd.Series): s = patient_ids.astype(str)
    else: s = pd.Series(list(patient_ids), dtype=object).astype(str)
    sub = s.str.extract(r'^([A-Za-z]+)', expand=False).str.lower()
    return sub.fillna("unknown").astype(str).values

def qa_to_severity_numeric(qa_scores):
    """Convierte QA a bins numéricos (0-3)"""
    bins = [0, 25, 50, 75, 100]
    labels = [0, 1, 2, 3] 
    return pd.cut(qa_scores, bins=bins, labels=labels, include_lowest=True).astype(int).values

def get_poslm_features(df, method='backoff'):
    """Filtra columnas POS-LM"""
    all_poslm = [c for c in df.columns if c.startswith('poslm_')]
    if method == 'none': return []
    if method == 'all': return all_poslm
    prefixes = {'kneser-ney': 'poslm_kn_', 'backoff': 'poslm_bo_', 'lstm': 'poslm_lstm_'}
    prefix = prefixes.get(method, 'poslm_')
    return [c for c in all_poslm if c.startswith(prefix)]

# ======================== VISUALIZACIÓN RICA (DEL UNIVERSAL) ========================
SEVERITY_LABELS = ['Very Severe', 'Severe', 'Moderate', 'Mild']
SEVERITY_BINS = [0, 25, 50, 75, 100]

def qa_to_severity(qa_scores):
    return pd.cut(qa_scores, bins=SEVERITY_BINS, labels=SEVERITY_LABELS, include_lowest=True)

def to_int(pred):
    return np.rint(np.asarray(pred)).clip(0, 100).astype(int)

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try: p_r, _ = pearsonr(y_true, y_pred)
    except: p_r = np.nan
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Pearson_r': p_r}

def compute_accuracy_metrics(y_true, y_pred_int):
    errors = np.abs(y_pred_int - y_true)
    return {
        'Acc@1': float(np.mean(errors <= 1)),
        'Acc@5': float(np.mean(errors <= 5)),
        'Acc@10': float(np.mean(errors <= 10))
    }

def compute_severity_accuracy(y_true, y_pred):
    sev_true = qa_to_severity(y_true)
    sev_pred = qa_to_severity(y_pred)
    acc = accuracy_score(sev_true, sev_pred)
    cm = confusion_matrix(sev_true, sev_pred, labels=SEVERITY_LABELS)
    return {'severity_accuracy': acc, 'confusion_matrix': cm}

def plot_scatter_with_precision(y_true, y_pred, title, out_png, metrics=None):
    if metrics is None: metrics = compute_metrics(y_true, y_pred)
    acc_metrics = compute_accuracy_metrics(y_true, to_int(y_pred))
    
    plt.figure(figsize=(9, 9))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Identidad")
    
    text = (f"MAE:      {metrics['MAE']:.2f}\nR²:       {metrics['R2']:.3f}\n"
            f"Pearson:  {metrics['Pearson_r']:.3f}\n─────────────\n"
            f"Acc@5:    {100*acc_metrics['Acc@5']:.1f}%")
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontsize=10, family='monospace')
    plt.xlabel("QA Real"); plt.ylabel("QA Predicho"); plt.title(title, fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_confusion_matrix(cm, title, out_png):
    fig, ax = plt.subplots(figsize=(8, 7))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS,
           ylabel='Severidad Real', xlabel='Severidad Predicha')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                   ha="center", va="center", color="white" if cm_norm[i, j] > thresh else "black")
    ax.set_title(title, fontweight='bold'); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_error_histogram(y_true, y_pred, title, out_png, threshold=5.316):
    errors = np.abs(y_true - y_pred)
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    for i, patch in enumerate(patches):
        patch.set_facecolor('mediumseagreen' if bins[i] <= threshold else 'coral')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Umbral ({threshold:.2f})')
    plt.xlabel('Error Absoluto'); plt.ylabel('Frecuencia'); plt.title(title, fontweight='bold')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# ======================== ENTRENAMIENTO ========================
def train_tabpfn_with_n_estimators(X_train, y_train, X_test, y_test, n_estimators, log):
    import time; start = time.time()
    model = TabPFNRegressor(device='cpu', n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    elapsed = time.time() - start
    log.info(f"      n={n_estimators}: MAE={mae:.3f} (t={elapsed:.1f}s)")
    return model, y_pred, mae

def train_ensemble_seeds(X_train, y_train, X_test, y_test, n_estimators, seeds, log):
    log.info(f"      Ensemble (n={n_estimators}, seeds={len(seeds)})")
    preds = []
    for seed_idx, seed in enumerate(seeds, 1):
        np.random.seed(seed)
        model = TabPFNRegressor(device='cpu', n_estimators=n_estimators)
        model.fit(X_train, y_train)
        preds.append(model.predict(X_test))
        log.info(f"        seed {seed_idx}: OK")
    
    y_pred_mean = np.mean(preds, axis=0)
    y_pred_std = np.std(preds, axis=0)
    mae = mean_absolute_error(y_test, y_pred_mean)
    log.info(f"        MAE={mae:.3f} (std_mean={y_pred_std.mean():.2f})")
    return y_pred_mean, y_pred_std, mae

# ======================== EXPLICABILIDAD ========================
def explain_with_lime(model, X_train, X_test, feature_names, run_dir, log, num_samples=50):
    if not LIME_AVAILABLE: return None
    log.info("\n  LIME - Explicaciones locales...")
    if X_train.shape[1] != len(feature_names): return None # Safety check

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, mode='regression', random_state=42)
    all_importances = []
    
    for i in range(min(num_samples, len(X_test))):
        try:
            exp = explainer.explain_instance(X_test[i], model.predict, num_features=len(feature_names))
            importance_dict = dict(exp.as_list())
            feats_imp = {}
            for fname in feature_names:
                imp = 0.0
                for k, v in importance_dict.items():
                    if fname in k: imp = abs(v); break
                feats_imp[fname] = imp
            all_importances.append(feats_imp)
        except: pass

    if not all_importances: return None
    
    imp_df = pd.DataFrame(all_importances)
    mean_imp = imp_df.mean().sort_values(ascending=False)
    lime_df = pd.DataFrame({'feature': mean_imp.index, 'importance': mean_imp.values})
    lime_df.to_csv(run_dir / "lime_importances.csv", index=False)
    
    plt.figure(figsize=(10, 12))
    top = lime_df.head(30)
    plt.barh(range(len(top)), top['importance'], color='coral', edgecolor='black')
    plt.yticks(range(len(top)), top['feature'], fontsize=9)
    plt.title('Top 30 Features - LIME', fontweight='bold'); plt.gca().invert_yaxis()
    plt.tight_layout(); plt.savefig(run_dir / "lime_importances.png"); plt.close()
    return lime_df

def explain_with_shap(model, X_train, X_test, feature_names, run_dir, log, max_samples=100):
    if not SHAP_AVAILABLE: return None
    log.info("\n  SHAP - Valores de Shapley...")
    
    bg_idx = np.random.choice(len(X_train), min(max_samples, len(X_train)), replace=False)
    explainer = shap.KernelExplainer(model.predict, X_train[bg_idx])
    
    test_idx = np.random.choice(len(X_test), min(max_samples, len(X_test)), replace=False)
    shap_values = explainer.shap_values(X_test[test_idx])
    
    pd.DataFrame(shap_values, columns=feature_names).to_csv(run_dir / "shap_values.csv", index=False)
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': np.abs(shap_values).mean(axis=0)}).sort_values('importance', ascending=False)
    imp_df.to_csv(run_dir / "shap_importances.csv", index=False)
    
    try:
        plt.figure(figsize=(10, 12))
        shap.summary_plot(shap_values, X_test[test_idx], feature_names=feature_names, show=False)
        plt.tight_layout(); plt.savefig(run_dir / "shap_beeswarm.png"); plt.close()
    except: pass
    return imp_df

def analyze_by_groups(importance_df, run_dir, log, method_name=""):
    if importance_df is None or len(importance_df) == 0: return None
    def get_group(f):
        if f.startswith('den_'): return 'DEN'
        elif f.startswith('dys_'): return 'DYS'
        elif f.startswith('lex_'): return 'LEX'
        elif f.startswith('poslm_'): return 'POSLM'
        else: return 'OTHER'
    
    df = importance_df.copy()
    df['group'] = df['feature'].apply(get_group)
    stats = df.groupby('group')['importance'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
    stats.to_csv(run_dir / f"groups_{method_name}.csv")
    
    plt.figure(figsize=(10, 6))
    colors = {'DEN': 'steelblue', 'DYS': 'coral', 'LEX': 'mediumseagreen', 'POSLM': 'goldenrod', 'OTHER': 'gray'}
    bar_colors = [colors.get(g, 'gray') for g in stats.index]
    plt.bar(stats.index, stats['sum'], color=bar_colors, edgecolor='black')
    plt.title(f'Feature Importance Group ({method_name})', fontweight='bold')
    plt.tight_layout(); plt.savefig(run_dir / f"groups_{method_name}.png"); plt.close()
    return stats

# ======================== MAIN ========================
def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE / f"OPTIMIZATION_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = set_logger(run_dir)
    cfg = EXPERIMENTS_CONFIG
    
    log.info("="*70); log.info(f"TABPFN UNIVERSAL OPTIMIZATION | Strategy: {cfg['cv_strategy'].upper()}"); log.info("="*70)
    with open(run_dir / "config.json", "w") as f: json.dump(cfg, f, indent=2)

    # 1. CARGAR DATOS
    df = pd.read_csv(DATASET_CSV)
    df = df[df['QA'].notna()].copy()
    # Filtro PWA + EN (Igual que Universal)
    df = df[(df['group'] == 'pwa') & (df['language'] == 'en')].reset_index(drop=True)
    log.info(f"Datos PWA (EN): {len(df)}")
    
    # Selección Features
    base_cols = [c for c in df.columns if c.startswith(('den_', 'dys_', 'lex_'))]
    poslm_cols = get_poslm_features(df, cfg['poslm_method'])
    all_feats = sorted(base_cols + poslm_cols)
    
    X = df[all_feats].values
    y = df['QA'].values
    patient_ids = df['patient_id'].values
    
    log.info(f"Features iniciales: {len(all_feats)} (POS-LM: {cfg['poslm_method']})")
    
    # 2. CV STRATEGY
    if cfg['cv_strategy'] == 'subdataset':
        groups = extract_subdataset_from_patient_id(df['patient_id']) 
        splitter = StratifiedGroupKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=42)
    else: # severity
        groups = qa_to_severity_numeric(y)
        splitter = StratifiedGroupKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=42)

    all_results = []
    
    # 3. EXPERIMENTOS N_ESTIMATORS
    log.info("\n[1/2] Probando n_estimators...")
    oof_predictions = {} # Guardar OOF para graficar luego
    
    for n_est in cfg['n_estimators_list']:
        log.info(f"\n  n_estimators = {n_est}")
        oof_preds = np.zeros_like(y, dtype=float)
        
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, groups, groups=patient_ids), 1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            # --- PIPELINE ---
            imputer = SimpleImputer(strategy='median')
            X_tr = imputer.fit_transform(X_tr); X_te = imputer.transform(X_te)
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
            
            # Feature Selection (KBest)
            if cfg['feature_selection'] == 'kbest':
                sel = SelectKBest(score_func=mutual_info_regression, k=cfg['k_features'])
                X_tr = sel.fit_transform(X_tr, y_tr); X_te = sel.transform(X_te)
            
            model, p, _ = train_tabpfn_with_n_estimators(X_tr, y_tr, X_te, y_te, n_est, log)
            oof_preds[test_idx] = p
            
        met = compute_metrics(y, oof_preds)
        log.info(f"  >> GLOBAL n={n_est}: MAE={met['MAE']:.4f}, R2={met['R2']:.4f}")
        all_results.append({'experiment': f'single_n{n_est}', 'n_estimators': n_est, **met})
        oof_predictions[f'single_n{n_est}'] = oof_preds

    # 4. ENSEMBLE SEEDS
    log.info("\n[2/2] Probando Ensemble Seeds...")
    best_config = min(all_results, key=lambda x: x['MAE'])
    best_n = best_config['n_estimators']
    
    oof_preds_ens = np.zeros_like(y, dtype=float)
    
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, groups, groups=patient_ids), 1):
        log.info(f"  Fold {fold}")
        # Repetir Pipeline
        imputer = SimpleImputer(strategy='median')
        X_tr = imputer.fit_transform(X[train_idx]); X_te = imputer.transform(X[test_idx])
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
        
        if cfg['feature_selection'] == 'kbest':
            sel = SelectKBest(score_func=mutual_info_regression, k=cfg['k_features'])
            X_tr = sel.fit_transform(X_tr, y[train_idx]); X_te = sel.transform(X_te)
            
        p, _, _ = train_ensemble_seeds(X_tr, y[train_idx], X_te, y[test_idx], best_n, cfg['ensemble_seeds'], log)
        oof_preds_ens[test_idx] = p

    met_ens = compute_metrics(y, oof_preds_ens)
    log.info(f"  >> GLOBAL ENSEMBLE: MAE={met_ens['MAE']:.4f}")
    all_results.append({'experiment': f'ensemble_n{best_n}', 'n_estimators': best_n, **met_ens})
    oof_predictions['ensemble_final'] = oof_preds_ens
    
    # 5. GENERAR GRÁFICAS RICAS (DEL MEJOR MODELO y ENSEMBLE)
    log.info("\nGenerando gráficas completas...")
    
    # Guardar métricas y gráficas para el ENSEMBLE
    plot_scatter_with_precision(y, oof_preds_ens, "Ensemble Final - Scatter", run_dir / "ensemble_scatter.png", metrics=met_ens)
    sev_acc = compute_severity_accuracy(y, oof_preds_ens)
    plot_confusion_matrix(sev_acc['confusion_matrix'], "Ensemble Final - Confusion", run_dir / "ensemble_confusion.png")
    plot_error_histogram(y, oof_preds_ens, "Ensemble Final - Errores", run_dir / "ensemble_errors.png")
    
    # Guardar análisis de errores (CSV)
    errors = np.abs(y - oof_preds_ens)
    df_err = pd.DataFrame({'patient_id': patient_ids, 'QA_real': y, 'QA_pred': oof_preds_ens, 'Error': errors})
    df_err.to_csv(run_dir / "ensemble_error_analysis.csv", index=False)

    # 6. GRAFICO COMPARATIVO DE BARRAS (DEL SCRIPT ORIGINAL)
    log.info("\nGenerando gráfico comparativo...")
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(run_dir / "summary_results.csv", index=False)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(res_df['experiment'], res_df['MAE'], color='steelblue', edgecolor='black')
    # Resaltar el mejor
    best_idx = res_df['MAE'].idxmin()
    bars[best_idx].set_color('mediumseagreen')
    
    plt.title('Comparativa MAE por Configuración', fontweight='bold')
    plt.ylabel('MAE (Menor es mejor)'); plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(run_dir / "comparison_bar.png"); plt.close()

    # 7. EXPLICABILIDAD FINAL
    log.info("\n" + "="*70); log.info("EXPLICABILIDAD FINAL"); log.info("="*70)
    
    train_idx, test_idx = next(splitter.split(X, groups, groups=patient_ids))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    # Pipeline para nombres
    current_feats = np.array(all_feats)
    imputer = SimpleImputer(strategy='median')
    X_tr = imputer.fit_transform(X_tr); X_te = imputer.transform(X_te)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
    
    if cfg['feature_selection'] == 'kbest':
        sel = SelectKBest(score_func=mutual_info_regression, k=cfg['k_features'])
        X_tr = sel.fit_transform(X_tr, y_tr); X_te = sel.transform(X_te)
        current_feats = current_feats[sel.get_support()]
    
    log.info(f"Entrenando modelo final para SHAP/LIME con {len(current_feats)} features...")
    model = TabPFNRegressor(device='cpu', n_estimators=best_n)
    model.fit(X_tr, y_tr)
    
    if LIME_AVAILABLE:
        lime_df = explain_with_lime(model, X_tr, X_te, current_feats, run_dir, log)
        analyze_by_groups(lime_df, run_dir, log, "LIME")
    
    if SHAP_AVAILABLE:
        shap_df = explain_with_shap(model, X_tr, X_te, current_feats, run_dir, log)
        analyze_by_groups(shap_df, run_dir, log, "SHAP")

    log.info(f"PROCESO COMPLETADO. Resultados en {run_dir}")

if __name__ == "__main__":
    main()