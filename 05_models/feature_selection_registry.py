#!/usr/bin/env python3
# feature_selection_registry.py
# -*- coding: utf-8 -*-
"""
FEATURE SELECTION REGISTRY - Centralized feature selection methods

This module provides a central location for all feature selection methods.

Usage:
    from feature_selection_registry import perform_feature_selection
    
    feat_cols, selection_info = perform_feature_selection(
        method='kbest',
        df_en=df_en,
        all_feat_cols=all_feat_cols,
        run_dir=run_dir,
        log=log,
        args=args
    )
"""

import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# ======================== MAIN FUNCTION ========================

def perform_feature_selection(method, df_en, all_feat_cols, run_dir, log, 
                              base_model=None, args=None, df_control=None):
    """
    Realiza feature selection según el método especificado.
    
    Args:
        method (str): Método de selección ('full', 'simple', 'sfs', 'kbest', 'rfe', 'importance')
        df_en (DataFrame): Dataset de entrenamiento (PWA inglés)
        all_feat_cols (list): Lista completa de features disponibles
        run_dir (Path): Directorio para guardar outputs
        log (Logger): Logger para mensajes
        base_model: Modelo base (necesario para 'sfs', 'rfe', 'importance')
        args: ArgumentParser args (contiene poslm_method, model, etc.)
        df_control (DataFrame): Dataset de controles (opcional, para SFS con z-norm)
    
    Returns:
        tuple: (feat_cols, selection_info)
            - feat_cols: Lista de features seleccionadas
            - selection_info: Dict con información sobre la selección
    """
    
    if method == 'simple':
        return _select_simple(df_en, all_feat_cols, run_dir, log, args)
    
    elif method == 'full':
        return _select_full(all_feat_cols, run_dir, log)
    
    elif method == 'sfs':
        return _select_sfs(df_en, all_feat_cols, run_dir, log, base_model, args, df_control)
    
    elif method == 'kbest':
        return _select_kbest(df_en, all_feat_cols, run_dir, log, k_features=40)
    
    elif method == 'rfe':
        return _select_rfe(df_en, all_feat_cols, run_dir, log, n_features=40)
    
    elif method == 'importance':
        return _select_importance(df_en, all_feat_cols, run_dir, log, args, n_features=40)
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

# ======================== SIMPLE (SUBSET MANUAL) ========================

def _select_simple(df_en, all_feat_cols, run_dir, log, args):
    """Selección manual de features básicas"""
    
    log.info("\n" + "="*70)
    log.info("FEATURE SELECTION - SIMPLE (MANUAL SUBSET)")
    log.info("="*70)
    
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
    
    feat_cols = [f for f in simple_base if f in df_en.columns]
    
    # Añadir POS-LM si aplica
    if args and args.poslm_method != 'none':
        poslm_features = []
        if args.poslm_method in ['kneser-ney', 'all']:
            poslm_features.extend(['poslm_kn_bigram_ce_mean', 'poslm_kn_trigram_ce_mean'])
        if args.poslm_method in ['backoff', 'all']:
            poslm_features.extend(['poslm_bo_bigram_ce_mean', 'poslm_bo_trigram_ce_mean'])
        if args.poslm_method in ['lstm', 'all']:
            poslm_features.extend(['poslm_lstm_ce_mean', 'poslm_lstm_ppl_mean'])
        
        feat_cols.extend([f for f in poslm_features if f in df_en.columns])
    
    log.info(f"Features seleccionadas: {len(feat_cols)}")
    
    _save_features_list(feat_cols, run_dir / "selected_features_simple.txt", 
                       "SIMPLE (MANUAL SUBSET)")
    
    selection_info = {
        'method': 'simple',
        'n_features': len(feat_cols),
        'selection_time_seconds': 0,
    }
    
    return feat_cols, selection_info

# ======================== FULL (TODAS) ========================

def _select_full(all_feat_cols, run_dir, log):
    """Usar todas las features disponibles"""
    
    log.info("\n" + "="*70)
    log.info("FEATURE SELECTION - FULL (ALL FEATURES)")
    log.info("="*70)
    log.info(f"Usando todas las features: {len(all_feat_cols)}")
    
    _save_features_list(all_feat_cols, run_dir / "selected_features_full.txt", "FULL")
    
    selection_info = {
        'method': 'full',
        'n_features': len(all_feat_cols),
        'selection_time_seconds': 0,
    }
    
    return all_feat_cols, selection_info

# ======================== SFS (SEQUENTIAL FORWARD SELECTION) ========================

def _select_sfs(df_en, all_feat_cols, run_dir, log, base_model, args, df_control):
    """Sequential Forward Selection (optimizado)"""
    
    try:
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    except ImportError:
        log.error("mlxtend no instalado: pip install mlxtend")
        sys.exit(1)
    
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
    from sklearn.svm import SVR
    import time
    
    start_time = time.time()
    
    log.info("\n" + "="*70)
    log.info("FEATURE SELECTION - SFS (OPTIMIZADO)")
    log.info("="*70)
    
    X_pwa_en_all = df_en[all_feat_cols].values
    y_pwa_en_all = df_en['QA'].values
    
    # === PRE-FILTRADO ===
    log.info("\nPRE-FILTRADO:")
    
    # 1. Eliminar >30% NaN
    nan_ratio = np.isnan(X_pwa_en_all).mean(axis=0)
    valid_mask = nan_ratio < 0.3
    log.info(f"  Paso 1: Features con <30% NaN: {valid_mask.sum()} de {len(all_feat_cols)}")
    
    X_temp_clean = X_pwa_en_all[:, valid_mask]
    all_feat_cols_filtered = [all_feat_cols[i] for i in range(len(all_feat_cols)) if valid_mask[i]]
    
    # 2. Imputar y eliminar varianza cero
    imputer_pre = SimpleImputer(strategy='median')
    X_temp_imputed = imputer_pre.fit_transform(X_temp_clean)
    
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X_temp_imputed)
    variance_mask = selector.get_support()
    log.info(f"  Paso 2: Features con varianza >0.01: {variance_mask.sum()}")
    
    X_temp_var = X_temp_imputed[:, variance_mask]
    all_feat_cols_filtered = [all_feat_cols_filtered[i] for i in range(len(all_feat_cols_filtered)) 
                              if variance_mask[i]]
    
    # 3. SelectKBest top 80
    if len(all_feat_cols_filtered) > 80:
        log.info(f"  Paso 3: Reduciendo de {len(all_feat_cols_filtered)} a 80 con SelectKBest...")
        
        selector_kbest = SelectKBest(score_func=f_regression, k=80)
        selector_kbest.fit(X_temp_var, y_pwa_en_all)
        kbest_mask = selector_kbest.get_support()
        
        all_feat_cols = [all_feat_cols_filtered[i] for i in range(len(all_feat_cols_filtered)) 
                        if kbest_mask[i]]
        log.info(f"  Resultado: {len(all_feat_cols)} features para SFS")
    else:
        all_feat_cols = all_feat_cols_filtered
        log.info(f"  Resultado: {len(all_feat_cols)} features (no se redujo)")
    
    _save_features_list(all_feat_cols, run_dir / "prefiltered_features_sfs.txt", 
                       "PRE-FILTRADAS PARA SFS")
    
    # Reconstruir datos
    X_pwa_en_all = df_en[all_feat_cols].values
    y_pwa_en_all = df_en['QA'].values
    
    if df_control is not None and len(df_control) > 0:
        X_control_all = df_control[all_feat_cols].values
        y_control_all = df_control['QA'].values
        X_all = np.vstack([X_pwa_en_all, X_control_all])
        y_all = np.concatenate([y_pwa_en_all, y_control_all])
    else:
        X_control_all = None
        X_all = X_pwa_en_all
        y_all = y_pwa_en_all
    
    # === MODELO OPTIMIZADO ===
    log.info("\nCONFIGURACION DE MODELO PARA SFS:")
    
    if args.model.lower() == "catboost":
        log.info("  Usando SVR (rápido)")
        sfs_base_model = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    
    elif args.model.lower() == "lgbm":
        from lightgbm import LGBMRegressor
        log.info("  Usando LightGBM con parámetros fijos")
        sfs_base_model = LGBMRegressor(
            num_leaves=31, learning_rate=0.1, n_estimators=100,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=-1
        )
    
    elif args.model.lower() == "xgb":
        from xgboost import XGBRegressor
        log.info("  Usando XGBoost con parámetros fijos")
        sfs_base_model = XGBRegressor(
            max_depth=5, learning_rate=0.1, n_estimators=100,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', device='cpu', random_state=42, verbosity=0, n_jobs=-1
        )
    
    else:
        log.info(f"  Usando {args.model.upper()} original")
        sfs_base_model = clone(base_model)
    
    # === EJECUTAR SFS ===
    max_features = 20
    log.info(f"\nEjecutando SFS (max_features={max_features}, cv=3)...")
    
    # Filtrar features vacías
    valid_mask = ~np.isnan(X_all).all(axis=0)
    n_invalid = (~valid_mask).sum()
    
    if n_invalid > 0:
        log.warning(f"  Excluyendo {n_invalid} features completamente vacías")
        X_all = X_all[:, valid_mask]
        if X_control_all is not None:
            X_control_all = X_control_all[:, valid_mask]
        all_feat_cols = [all_feat_cols[i] for i in range(len(all_feat_cols)) if valid_mask[i]]
    
    # Pre-procesar
    if args.znorm_controls and X_control_all is not None:
        log.info("  Con z-norm de controles")
        from train_model_UNIVERSAL import compute_control_stats, apply_znorm
        
        imputer = SimpleImputer(strategy="median")
        X_all_imputed = imputer.fit_transform(X_all)
        X_control_imputed = imputer.transform(X_control_all)
        
        mean_ctrl, std_ctrl = compute_control_stats(X_control_imputed)
        X_all_norm = apply_znorm(X_all_imputed, mean_ctrl, std_ctrl)
        
        sfs = SFS(sfs_base_model, k_features=(1, max_features), forward=True, floating=False,
                  scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, verbose=2)
        sfs.fit(X_all_norm, y_all)
    else:
        log.info("  Con StandardScaler")
        
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", sfs_base_model)
        ])
        
        sfs = SFS(pipe, k_features=(1, max_features), forward=True, floating=False,
                  scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, verbose=2)
        sfs.fit(X_all, y_all)
    
    # Resultados
    best_k = len(sfs.k_feature_names_)
    best_features_idx = list(sfs.k_feature_idx_)
    feat_cols = [all_feat_cols[i] for i in best_features_idx]
    best_score = -sfs.k_score_
    
    elapsed_time = time.time() - start_time
    
    log.info("\nRESULTADOS SFS:")
    log.info(f"  Features seleccionadas: {best_k}")
    log.info(f"  MAE: {best_score:.3f}")
    log.info(f"  Tiempo: {elapsed_time/60:.1f} minutos")
    
    _save_features_list(feat_cols, run_dir / "selected_features_sfs.txt", 
                       f"SFS (MAE={best_score:.3f})")
    
    selection_info = {
        'method': 'sfs',
        'n_features': best_k,
        'sfs_mae': best_score,
        'sfs_max_features': max_features,
        'sfs_cv_folds': 3,
        'selection_time_seconds': elapsed_time,
    }
    
    return feat_cols, selection_info

# ======================== KBEST (MUTUAL INFORMATION) ========================

def _select_kbest(df_en, all_feat_cols, run_dir, log, k_features=40):
    """SelectKBest con Mutual Information"""
    
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    import time
    
    start_time = time.time()
    
    log.info("\n" + "="*70)
    log.info("FEATURE SELECTION - SELECTKBEST (MUTUAL INFORMATION)")
    log.info("="*70)
    
    X_pwa_en_all = df_en[all_feat_cols].values
    y_pwa_en_all = df_en['QA'].values
    
    # Imputar
    log.info("Paso 1: Imputando valores faltantes...")
    imputer_kbest = SimpleImputer(strategy='median')
    X_imputed = imputer_kbest.fit_transform(X_pwa_en_all)
    
    # Seleccionar
    log.info(f"Paso 2: Seleccionando top {k_features} features con mutual_info_regression...")
    
    selector = SelectKBest(score_func=mutual_info_regression, k=k_features)
    selector.fit(X_imputed, y_pwa_en_all)
    
    selected_mask = selector.get_support()
    feat_cols = [all_feat_cols[i] for i in range(len(all_feat_cols)) if selected_mask[i]]
    
    # Guardar scores
    scores_df = pd.DataFrame({
        'feature': all_feat_cols,
        'score': selector.scores_,
        'selected': selected_mask
    }).sort_values('score', ascending=False)
    
    scores_df.to_csv(run_dir / "kbest_scores.csv", index=False)
    
    elapsed_time = time.time() - start_time
    
    log.info(f"\nResultado: {len(feat_cols)} features seleccionadas")
    log.info(f"Tiempo: {elapsed_time:.1f} segundos")
    log.info("\nTop 20 features por score:")
    for idx, (i, row) in enumerate(scores_df[scores_df['selected']].head(20).iterrows(), 1):
        log.info(f"  {idx:2d}. {row['feature']:40s} {row['score']:.4f}")
    
    _save_features_list(feat_cols, run_dir / "selected_features_kbest.txt", 
                       f"SELECTKBEST (k={k_features})")
    
    selection_info = {
        'method': 'kbest',
        'n_features': len(feat_cols),
        'kbest_k': k_features,
        'kbest_score_func': 'mutual_info_regression',
        'selection_time_seconds': elapsed_time,
    }
    
    return feat_cols, selection_info

# ======================== RFE (RECURSIVE FEATURE ELIMINATION) ========================

def _select_rfe(df_en, all_feat_cols, run_dir, log, n_features=40):
    """Recursive Feature Elimination"""
    
    from sklearn.feature_selection import RFE, SelectKBest, f_regression
    from sklearn.svm import SVR
    import time
    
    start_time = time.time()
    
    log.info("\n" + "="*70)
    log.info("FEATURE SELECTION - RFE (RECURSIVE FEATURE ELIMINATION)")
    log.info("="*70)
    
    X_pwa_en_all = df_en[all_feat_cols].values
    y_pwa_en_all = df_en['QA'].values
    
    # Pre-filtrar a 80
    log.info("Pre-filtrado a 80 features con SelectKBest...")
    
    imputer_pre = SimpleImputer(strategy='median')
    X_imputed = imputer_pre.fit_transform(X_pwa_en_all)
    
    selector_pre = SelectKBest(score_func=f_regression, k=80)
    selector_pre.fit(X_imputed, y_pwa_en_all)
    premask = selector_pre.get_support()
    
    all_feat_cols_pre = [all_feat_cols[i] for i in range(len(all_feat_cols)) if premask[i]]
    X_pre = X_imputed[:, premask]
    
    log.info(f"Pre-filtrado: {len(all_feat_cols_pre)} features")
    
    # RFE con SVR
    log.info(f"\nEjecutando RFE para seleccionar {n_features} features...")
    rfe_model = SVR(kernel="rbf", C=10.0, epsilon=0.1)
    
    rfe = RFE(estimator=rfe_model, n_features_to_select=n_features, step=5, verbose=1)
    rfe.fit(X_pre, y_pwa_en_all)
    
    rfe_mask = rfe.get_support()
    feat_cols = [all_feat_cols_pre[i] for i in range(len(all_feat_cols_pre)) if rfe_mask[i]]
    
    # Guardar ranking
    ranking_df = pd.DataFrame({
        'feature': all_feat_cols_pre,
        'ranking': rfe.ranking_,
        'selected': rfe_mask
    }).sort_values('ranking')
    
    ranking_df.to_csv(run_dir / "rfe_ranking.csv", index=False)
    
    elapsed_time = time.time() - start_time
    
    log.info(f"\nResultado: {len(feat_cols)} features seleccionadas")
    log.info(f"Tiempo: {elapsed_time/60:.1f} minutos")
    log.info("\nTop 20 features (ranking=1):")
    for idx, (i, row) in enumerate(ranking_df[ranking_df['selected']].head(20).iterrows(), 1):
        log.info(f"  {idx:2d}. {row['feature']:40s} rank={row['ranking']}")
    
    _save_features_list(feat_cols, run_dir / "selected_features_rfe.txt", 
                       f"RFE (n={n_features})")
    
    selection_info = {
        'method': 'rfe',
        'n_features': len(feat_cols),
        'rfe_n_features_to_select': n_features,
        'rfe_step': 5,
        'selection_time_seconds': elapsed_time,
    }
    
    return feat_cols, selection_info

# ======================== IMPORTANCE (MODEL-BASED) ========================

def _select_importance(df_en, all_feat_cols, run_dir, log, args, n_features=40):
    """Feature selection basada en importancia del modelo"""
    
    import time
    
    start_time = time.time()
    
    log.info("\n" + "="*70)
    log.info("FEATURE SELECTION - MODEL IMPORTANCE")
    log.info("="*70)
    
    X_pwa_en_all = df_en[all_feat_cols].values
    y_pwa_en_all = df_en['QA'].values
    
    # Imputar y normalizar
    imputer_imp = SimpleImputer(strategy='median')
    X_imputed = imputer_imp.fit_transform(X_pwa_en_all)
    
    scaler_imp = StandardScaler()
    X_scaled = scaler_imp.fit_transform(X_imputed)
    
    # Entrenar modelo
    log.info(f"Entrenando {args.model.upper()} para obtener feature importance...")
    
    if args.model.lower() == 'lgbm':
        from lightgbm import LGBMRegressor
        model_imp = LGBMRegressor(
            num_leaves=31, learning_rate=0.1, n_estimators=200,
            random_state=42, verbose=-1, n_jobs=-1
        )
    elif args.model.lower() == 'xgb':
        from xgboost import XGBRegressor
        model_imp = XGBRegressor(
            max_depth=5, learning_rate=0.1, n_estimators=200,
            tree_method='hist', device='cpu', random_state=42, n_jobs=-1, verbosity=0
        )
    elif args.model.lower() == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model_imp = RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        )
    elif args.model.lower() == 'catboost':
        from catboost import CatBoostRegressor
        model_imp = CatBoostRegressor(
            iterations=200, learning_rate=0.1, depth=6,
            random_state=42, verbose=0, thread_count=-1
        )
    else:
        # Fallback a SelectKBest
        log.warning(f"{args.model.upper()} no tiene feature_importances_, usando SelectKBest")
        return _select_kbest(df_en, all_feat_cols, run_dir, log, k_features=n_features)
    
    model_imp.fit(X_scaled, y_pwa_en_all)
    
    # Obtener importancias
    importances = model_imp.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': all_feat_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(run_dir / "feature_importances.csv", index=False)
    
    # Seleccionar top N
    feat_cols = importance_df.head(n_features)['feature'].tolist()
    
    elapsed_time = time.time() - start_time
    
    log.info(f"\nResultado: {len(feat_cols)} features seleccionadas")
    log.info(f"Tiempo: {elapsed_time:.1f} segundos")
    log.info("\nTop 20 features por importancia:")
    for idx, (i, row) in enumerate(importance_df.head(20).iterrows(), 1):
        log.info(f"  {idx:2d}. {row['feature']:40s} {row['importance']:.4f}")
    
    _save_features_list(feat_cols, run_dir / "selected_features_importance.txt", 
                       f"{args.model.upper()} IMPORTANCE (n={n_features})")
    
    selection_info = {
        'method': 'importance',
        'n_features': len(feat_cols),
        'importance_model': args.model,
        'importance_n_estimators': 200,
        'selection_time_seconds': elapsed_time,
    }
    
    return feat_cols, selection_info

# ======================== HELPER FUNCTIONS ========================

def _save_features_list(feat_cols, filepath, title):
    """Guarda lista de features en archivo"""
    with open(filepath, "w") as f:
        f.write(f"FEATURES SELECCIONADAS - {title}\n")
        f.write("="*70 + "\n")
        f.write(f"Total: {len(feat_cols)}\n\n")
        for feat in sorted(feat_cols):
            f.write(f"{feat}\n")

def list_available_methods():
    """Lista métodos disponibles"""
    return ['simple', 'full', 'sfs', 'kbest', 'rfe', 'importance']

def get_method_info(method):
    """Información sobre un método"""
    info = {
        'simple': {
            'name': 'Simple (Manual Subset)',
            'description': 'Subset manual de ~30 features básicas',
            'time': 'Instantáneo',
            'best_for': 'Baseline rápido, interpretabilidad'
        },
        'full': {
            'name': 'Full (All Features)',
            'description': 'Usa todas las features disponibles',
            'time': 'Instantáneo',
            'best_for': 'Máximo rendimiento, sin reducción'
        },
        'sfs': {
            'name': 'Sequential Forward Selection',
            'description': 'Selección greedy forward con CV',
            'time': '2-3 horas (optimizado)',
            'best_for': 'Mejor calidad, búsqueda exhaustiva'
        },
        'kbest': {
            'name': 'SelectKBest (Mutual Information)',
            'description': 'Selección univariada por correlación',
            'time': '5-10 minutos',
            'best_for': 'Rápido, buen balance calidad/tiempo'
        },
        'rfe': {
            'name': 'Recursive Feature Elimination',
            'description': 'Eliminación recursiva con modelo',
            'time': '15-30 minutos',
            'best_for': 'Buena calidad, más rápido que SFS'
        },
        'importance': {
            'name': 'Model-Based Importance',
            'description': 'Importancia nativa del modelo (tree-based)',
            'time': '5 minutos',
            'best_for': 'Muy rápido, requiere modelo tree-based'
        }
    }
    return info.get(method.lower(), None)