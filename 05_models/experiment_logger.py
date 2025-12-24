#!/usr/bin/env python3
# 05_models/experiment_logger.py
# -*- coding: utf-8 -*-
"""
Sistema de logging de experimentos UNIVERSAL para tracking de configuraciones y resultados.

SOPORTA:
- SVR (Support Vector Regression)
- LightGBM (Gradient Boosting)
- XGBoost
- CatBoost
- Random Forest
- Elastic Net
- Cualquier modelo sklearn-compatible

FEATURE SELECTION:
- simple, full, sfs, kbest, rfe, importance

HPO METHODS:
- GridSearch (exhaustivo)
- Optuna (bayesiano) - CON GUARDADO DE SEARCH SPACE

Se integra con:
- train_svr_COMPLETO_FINAL.py (legacy, solo SVR)
- train_model_UNIVERSAL.py (nuevo, multi-modelo)
- feature_selection_registry.py (nuevo, feature selection modular)
- hyperparameter_optimizer.py (nuevo, GridSearch vs Optuna)
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """Logger universal para trackear experimentos de cualquier modelo"""
    
    def __init__(self, log_dir=None):
        """
        Si no se pasa log_dir, escribe en:
        <PROJECT_BASE>/outputs/experiments/resultados_svr (o resultados_modelos)
        """
        if log_dir is None:
            project_base = Path(__file__).resolve().parent.parent
            self.log_dir = project_base / "outputs" / "experiments" / "resultados_svr"
        else:
            self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "experiments_history.csv"
        self.json_path = self.log_dir / "experiments_history.json"
    
    def log_experiment(self, config: dict, results: dict, run_dir: str):
        """
        Registra un experimento en el CSV y JSON de historial.
        
        Args:
            config: Diccionario con configuracion del experimento
            results: Diccionario con metricas de resultados
            run_dir: Carpeta donde se guardaron los resultados
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # ============ DETECTAR TIPO DE MODELO ============
        model_type = config.get("model", "svr").lower()
        
        # Crear registro BASE (común para todos los modelos)
        record = {
            "timestamp": timestamp,
            "experiment_name": Path(run_dir).name,
            "run_dir": str(run_dir),
            
            # ============ MODELO ============
            "model_type": model_type,
            
            # Configuracion
            "features_mode": config.get("features", "unknown"),
            "poslm_method": config.get("poslm_method", "none"),
            "n_features": config.get("n_features", 0),
            
            # Feature selection info
            "feature_selection_method": config.get("feature_selection_method", config.get("features", "unknown")),
            "feature_selection_time_sec": config.get("feature_selection_time", 0),
            
            # Info específica por método de feature selection
            "sfs_mae": config.get("sfs_mae", None),
            "sfs_max_features": config.get("sfs_max_features", None),
            "sfs_cv_folds": config.get("sfs_cv_folds", None),
            "kbest_k": config.get("kbest_k", None),
            "kbest_score_func": config.get("kbest_score_func", None),
            "rfe_n_features": config.get("rfe_n_features", None),
            "rfe_step": config.get("rfe_step", None),
            "importance_model": config.get("importance_model", None),
            "importance_n_estimators": config.get("importance_n_estimators", None),
            
            # HPO info (MEJORADO)
            "hpo_method": config.get("hpo_method", "gridsearch"),
            "optuna_trials": config.get("optuna_trials", None),
            "optuna_search_space_file": config.get("optuna_search_space_file", None),  # NUEVO
            
            # CV config
            "cv_folds_outer": config.get("cv_folds_outer", 4),
            "cv_folds_inner": config.get("cv_folds_inner", 5),
            "znorm_controls": config.get("znorm_controls", False),
            "stratified_cv": config.get("stratified_cv", False),
            
            # Dataset info
            "n_patients_total": config.get("n_patients_total", 0),
            "n_pwa": config.get("n_pwa", 0),
            "n_control": config.get("n_control", 0),
            "n_en": config.get("n_en", 0),
            "n_es": config.get("n_es", 0),
            "n_ca": config.get("n_ca", 0),
            
            # Resultados CV (sin calibrar)
            "cv_mae": results.get("cv_mae", None),
            "cv_rmse": results.get("cv_rmse", None),
            "cv_r2": results.get("cv_r2", None),
            "cv_pearson": results.get("cv_pearson", None),
            "cv_spearman": results.get("cv_spearman", None),
            "cv_severity_acc": results.get("cv_severity_acc", None),
            
            # Resultados CV (calibrado)
            "cv_cal_mae": results.get("cv_cal_mae", None),
            "cv_cal_rmse": results.get("cv_cal_rmse", None),
            "cv_cal_r2": results.get("cv_cal_r2", None),
            "cv_cal_pearson": results.get("cv_cal_pearson", None),
            "cv_cal_spearman": results.get("cv_cal_spearman", None),
            "cv_cal_severity_acc": results.get("cv_cal_severity_acc", None),
            
            # Resultados ES (si hay)
            "es_mae": results.get("es_mae", None),
            "es_cal_mae": results.get("es_cal_mae", None),
            
            # Resultados CA (si hay)
            "ca_mae": results.get("ca_mae", None),
            "ca_cal_mae": results.get("ca_cal_mae", None),
            
            # Features seleccionadas (si SFS)
            "selected_features": str(config.get("selected_features", [])),
            
            # Notas
            "notes": config.get("notes", ""),
        }
        
        # ============ HIPERPARÁMETROS ESPECÍFICOS POR MODELO ============
        if model_type == "svr":
            # Param grid SVR (soporta ambos formatos)
            record["param_grid_C"] = str(config.get("C_values", config.get("C", [])))
            record["param_grid_epsilon"] = str(config.get("epsilon_values", config.get("epsilon", [])))
            record["param_grid_kernel"] = str(config.get("kernel_values", config.get("kernel", [])))
            record["param_grid_shrinking"] = str(config.get("shrinking_values", config.get("shrinking", [])))
            record["param_grid_gamma"] = str(config.get("gamma_values", config.get("gamma", [])))
            
            # Mejores parámetros SVR
            record["best_C"] = results.get("best_C", None)
            record["best_epsilon"] = results.get("best_epsilon", None)
            record["best_kernel"] = results.get("best_kernel", None)
            record["best_gamma"] = results.get("best_gamma", None)
            record["best_shrinking"] = results.get("best_shrinking", None)
            
            # Placeholders para otros modelos
            record["best_num_leaves"] = None
            record["best_learning_rate"] = None
            record["best_n_estimators"] = None
            record["best_min_child_samples"] = None
            record["best_max_depth"] = None
            
        elif model_type == "lgbm":
            # Param grid LightGBM
            record["param_grid_num_leaves"] = str(config.get("num_leaves", []))
            record["param_grid_learning_rate"] = str(config.get("learning_rate", []))
            record["param_grid_n_estimators"] = str(config.get("n_estimators", []))
            record["param_grid_min_child_samples"] = str(config.get("min_child_samples", []))
            
            # Mejores parámetros LightGBM
            record["best_num_leaves"] = results.get("best_num_leaves", None)
            record["best_learning_rate"] = results.get("best_learning_rate", None)
            record["best_n_estimators"] = results.get("best_n_estimators", None)
            record["best_min_child_samples"] = results.get("best_min_child_samples", None)
            
            # Placeholders para SVR
            record["param_grid_C"] = None
            record["param_grid_epsilon"] = None
            record["param_grid_kernel"] = None
            record["param_grid_shrinking"] = None
            record["param_grid_gamma"] = None
            record["best_C"] = None
            record["best_epsilon"] = None
            record["best_kernel"] = None
            record["best_gamma"] = None
            record["best_shrinking"] = None
            record["best_max_depth"] = None
        
        elif model_type == "xgb":
            # Param grid XGBoost
            record["param_grid_max_depth"] = str(config.get("max_depth", []))
            record["param_grid_learning_rate"] = str(config.get("learning_rate", []))
            record["param_grid_n_estimators"] = str(config.get("n_estimators", []))
            
            # Mejores parámetros XGBoost
            record["best_max_depth"] = results.get("best_max_depth", None)
            record["best_learning_rate"] = results.get("best_learning_rate", None)
            record["best_n_estimators"] = results.get("best_n_estimators", None)
            
            # Placeholders
            record["param_grid_C"] = None
            record["param_grid_epsilon"] = None
            record["param_grid_kernel"] = None
            record["param_grid_shrinking"] = None
            record["param_grid_gamma"] = None
            record["best_C"] = None
            record["best_epsilon"] = None
            record["best_kernel"] = None
            record["best_gamma"] = None
            record["best_shrinking"] = None
            record["best_num_leaves"] = None
            record["best_min_child_samples"] = None
        
        elif model_type == "catboost":
            # Param grid CatBoost
            record["param_grid_depth"] = str(config.get("depth", []))
            record["param_grid_learning_rate"] = str(config.get("learning_rate", []))
            record["param_grid_iterations"] = str(config.get("iterations", []))
            
            # Mejores parámetros CatBoost
            record["best_depth"] = results.get("best_depth", None)
            record["best_learning_rate"] = results.get("best_learning_rate", None)
            record["best_iterations"] = results.get("best_iterations", None)
            
            # Placeholders
            record["param_grid_C"] = None
            record["param_grid_epsilon"] = None
            record["param_grid_kernel"] = None
            record["param_grid_shrinking"] = None
            record["param_grid_gamma"] = None
            record["best_C"] = None
            record["best_epsilon"] = None
            record["best_kernel"] = None
            record["best_gamma"] = None
            record["best_shrinking"] = None
            record["best_num_leaves"] = None
            record["best_min_child_samples"] = None
            record["best_max_depth"] = None
            record["best_n_estimators"] = None
        
        else:
            # Modelo desconocido - poner todo a None
            record["param_grid_C"] = None
            record["param_grid_epsilon"] = None
            record["param_grid_kernel"] = None
            record["param_grid_shrinking"] = None
            record["param_grid_gamma"] = None
            record["best_C"] = None
            record["best_epsilon"] = None
            record["best_kernel"] = None
            record["best_gamma"] = None
            record["best_shrinking"] = None
            record["best_num_leaves"] = None
            record["best_learning_rate"] = None
            record["best_n_estimators"] = None
            record["best_min_child_samples"] = None
            record["best_max_depth"] = None
        
        # Guardar en CSV
        self._append_to_csv(record)
        
        # Guardar en JSON (historial completo)
        self._append_to_json(record)
        
        print(f"\n[ExperimentLogger] Experimento registrado en:")
        print(f"  CSV: {self.csv_path}")
        print(f"  JSON: {self.json_path}")
        
        return record
    
    def _append_to_csv(self, record: dict):
        """Añade registro al CSV"""
        df_new = pd.DataFrame([record])
        
        if self.csv_path.exists():
            df_existing = pd.read_csv(self.csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(self.csv_path, index=False)
    
    def _append_to_json(self, record: dict):
        """Añade registro al JSON"""
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(record)
        
        with open(self.json_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_history(self) -> pd.DataFrame:
        """Devuelve el historial como DataFrame"""
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame()
    
    def get_best_experiment(self, metric="cv_cal_mae", minimize=True, model_type=None, 
                           feature_selection=None, hpo_method=None) -> dict:
        """
        Devuelve el mejor experimento segun una metrica.
        
        Args:
            metric: Métrica a optimizar (ej: 'cv_cal_mae')
            minimize: True si menor es mejor, False si mayor es mejor
            model_type: Opcional, filtrar por tipo de modelo ('svr', 'lgbm', etc.)
            feature_selection: Opcional, filtrar por método de feature selection
            hpo_method: Opcional, filtrar por método HPO ('gridsearch', 'optuna')
        """
        df = self.get_history()
        if df.empty:
            return {}
        
        # Filtrar por modelo si se especifica
        if model_type is not None:
            df = df[df['model_type'] == model_type.lower()]
            if df.empty:
                return {}
        
        # Filtrar por feature selection si se especifica
        if feature_selection is not None:
            df = df[df['feature_selection_method'] == feature_selection.lower()]
            if df.empty:
                return {}
        
        # Filtrar por HPO method si se especifica
        if hpo_method is not None:
            df = df[df['hpo_method'] == hpo_method.lower()]
            if df.empty:
                return {}
        
        if minimize:
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()
        
        return df.loc[best_idx].to_dict()
    
    def print_summary(self, model_type=None, feature_selection=None, hpo_method=None):
        """
        Imprime resumen de todos los experimentos.
        
        Args:
            model_type: Opcional, filtrar por tipo de modelo
            feature_selection: Opcional, filtrar por método de feature selection
            hpo_method: Opcional, filtrar por método HPO
        """
        df = self.get_history()
        if df.empty:
            print("No hay experimentos registrados.")
            return
        
        # Filtrar por modelo si se especifica
        if model_type is not None:
            df = df[df['model_type'] == model_type.lower()]
            if df.empty:
                print(f"No hay experimentos registrados para modelo '{model_type}'.")
                return
        
        # Filtrar por feature selection si se especifica
        if feature_selection is not None:
            df = df[df['feature_selection_method'] == feature_selection.lower()]
            if df.empty:
                print(f"No hay experimentos registrados para feature selection '{feature_selection}'.")
                return
        
        # Filtrar por HPO si se especifica
        if hpo_method is not None:
            df = df[df['hpo_method'] == hpo_method.lower()]
            if df.empty:
                print(f"No hay experimentos registrados para HPO '{hpo_method}'.")
                return
        
        print("\n" + "="*80)
        title = "RESUMEN DE EXPERIMENTOS"
        if model_type:
            title += f" - {model_type.upper()}"
        if feature_selection:
            title += f" - {feature_selection.upper()}"
        if hpo_method:
            title += f" - {hpo_method.upper()}"
        print(title)
        print("="*80)
        print(f"Total experimentos: {len(df)}")
        
        # Mostrar modelos disponibles
        if 'model_type' in df.columns:
            print(f"\nModelos disponibles:")
            for model, count in df['model_type'].value_counts().items():
                print(f"  {model.upper()}: {count} experimentos")
        
        # Mostrar métodos de feature selection disponibles
        if 'feature_selection_method' in df.columns:
            print(f"\nMétodos de feature selection disponibles:")
            for method, count in df['feature_selection_method'].value_counts().items():
                print(f"  {method}: {count} experimentos")
        
        # Mostrar métodos HPO disponibles
        if 'hpo_method' in df.columns:
            print(f"\nMétodos de HPO disponibles:")
            for method, count in df['hpo_method'].value_counts().items():
                print(f"  {method}: {count} experimentos")
        
        # Mejor por MAE calibrado
        if 'cv_cal_mae' in df.columns and df['cv_cal_mae'].notna().any():
            best = df.loc[df['cv_cal_mae'].idxmin()]
            print(f"\nMejor experimento (MAE calibrado):")
            print(f"  Nombre: {best['experiment_name']}")
            print(f"  Modelo: {best.get('model_type', 'svr').upper()}")
            print(f"  Feature Selection: {best.get('feature_selection_method', 'unknown')}")
            print(f"  HPO: {best.get('hpo_method', 'gridsearch')}")
            print(f"  MAE: {best['cv_cal_mae']:.3f}")
            print(f"  Pearson: {best['cv_cal_pearson']:.3f}")
            print(f"  Features: {best['features_mode']}")
            print(f"  POS-LM: {best['poslm_method']}")
            if best.get('feature_selection_time_sec', 0) > 0:
                print(f"  Tiempo selección: {best['feature_selection_time_sec']/60:.1f} min")
        
        print("\n" + "-"*80)
        print("Todos los experimentos:")
        print("-"*80)
        
        cols_display = ['timestamp', 'experiment_name', 'model_type', 'feature_selection_method',
                       'hpo_method', 'features_mode', 'poslm_method', 'cv_cal_mae', 'cv_cal_pearson']
        cols_available = [c for c in cols_display if c in df.columns]
        
        print(df[cols_available].to_string(index=False))
        print("="*80)
    
    def compare_models(self, metric='cv_cal_mae'):
        """
        Compara rendimiento entre diferentes modelos.
        
        Args:
            metric: Métrica para comparar (ej: 'cv_cal_mae', 'cv_cal_pearson')
        """
        df = self.get_history()
        if df.empty:
            print("No hay experimentos registrados.")
            return
        
        if 'model_type' not in df.columns:
            print("No hay información de tipo de modelo.")
            return
        
        print("\n" + "="*80)
        print(f"COMPARACIÓN DE MODELOS - Métrica: {metric}")
        print("="*80)
        
        summary = df.groupby('model_type')[metric].agg(['mean', 'std', 'min', 'max', 'count'])
        summary = summary.sort_values('mean')
        
        print("\nEstadísticas por modelo:")
        print(summary.to_string())
        
        # Mejor modelo
        best_model = summary['mean'].idxmin() if 'mae' in metric.lower() else summary['mean'].idxmax()
        print(f"\nMejor modelo según {metric}: {best_model.upper()}")
        print("="*80)
    
    def compare_hpo_methods(self, metric='cv_cal_mae', model_type=None):
        """
        Compara rendimiento entre GridSearch y Optuna.
        
        Args:
            metric: Métrica para comparar
            model_type: Opcional, filtrar por modelo específico
        """
        df = self.get_history()
        if df.empty:
            print("No hay experimentos registrados.")
            return
        
        if 'hpo_method' not in df.columns:
            print("No hay información de HPO method.")
            return
        
        # Filtrar por modelo si se especifica
        if model_type is not None:
            df = df[df['model_type'] == model_type.lower()]
            if df.empty:
                print(f"No hay experimentos para el modelo '{model_type}'.")
                return
        
        print("\n" + "="*80)
        title = f"COMPARACIÓN DE HPO METHODS - Métrica: {metric}"
        if model_type:
            title += f" (Modelo: {model_type.upper()})"
        print(title)
        print("="*80)
        
        summary = df.groupby('hpo_method')[metric].agg(['mean', 'std', 'min', 'max', 'count'])
        summary = summary.sort_values('mean')
        
        print("\nEstadísticas por método:")
        print(summary.to_string())
        
        # Mejor método
        best_method = summary['mean'].idxmin() if 'mae' in metric.lower() else summary['mean'].idxmax()
        print(f"\nMejor método según {metric}: {best_method}")
        print("="*80)
    
    def compare_feature_selection(self, metric='cv_cal_mae', model_type=None):
        """
        Compara rendimiento entre diferentes métodos de feature selection.
        
        Args:
            metric: Métrica para comparar
            model_type: Opcional, filtrar por modelo específico
        """
        df = self.get_history()
        if df.empty:
            print("No hay experimentos registrados.")
            return
        
        if 'feature_selection_method' not in df.columns:
            print("No hay información de feature selection.")
            return
        
        # Filtrar por modelo si se especifica
        if model_type is not None:
            df = df[df['model_type'] == model_type.lower()]
            if df.empty:
                print(f"No hay experimentos para el modelo '{model_type}'.")
                return
        
        print("\n" + "="*80)
        title = f"COMPARACIÓN DE FEATURE SELECTION - Métrica: {metric}"
        if model_type:
            title += f" (Modelo: {model_type.upper()})"
        print(title)
        print("="*80)
        
        summary = df.groupby('feature_selection_method')[metric].agg(['mean', 'std', 'min', 'max', 'count'])
        summary = summary.sort_values('mean')
        
        # Añadir tiempo promedio si disponible
        if 'feature_selection_time_sec' in df.columns:
            time_summary = df.groupby('feature_selection_method')['feature_selection_time_sec'].mean()
            summary['avg_time_min'] = time_summary / 60
        
        print("\nEstadísticas por método:")
        print(summary.to_string())
        
        # Mejor método
        best_method = summary['mean'].idxmin() if 'mae' in metric.lower() else summary['mean'].idxmax()
        print(f"\nMejor método según {metric}: {best_method}")
        
        if 'avg_time_min' in summary.columns:
            print(f"Tiempo promedio: {summary.loc[best_method, 'avg_time_min']:.1f} min")
        
        print("="*80)


# ============ FUNCIONES HELPER PARA USO EN SCRIPTS ============

def create_experiment_config(args, param_grid, df_info, feat_cols, selection_info=None):
    """
    Crea diccionario de configuracion desde los argumentos y datos.
    
    COMPATIBLE CON:
    - train_svr_COMPLETO_FINAL.py (legacy)
    - train_model_UNIVERSAL.py (nuevo)
    - feature_selection_registry.py (nuevo)
    - hyperparameter_optimizer.py (nuevo)
    
    Args:
        args: ArgumentParser args
        param_grid: Diccionario del grid search
        df_info: Diccionario con info del dataset
        feat_cols: Lista de features usadas
        selection_info: Diccionario con info de feature selection (nuevo)
    """
    config = {
        "features": args.features,
        "poslm_method": args.poslm_method,
        "n_features": len(feat_cols),
        "selected_features": feat_cols if args.features == 'sfs' else [],
        
        # CV
        "cv_folds_outer": 4,
        "cv_folds_inner": getattr(args, 'cv_inner', 5),
        
        # Normalización
        "znorm_controls": getattr(args, 'znorm_controls', False),
        "stratified_cv": getattr(args, 'stratified_cv', True),
        
        # HPO (MEJORADO - CON SEARCH SPACE FILE)
        "hpo_method": getattr(args, 'hpo_method', 'gridsearch'),
        "optuna_trials": getattr(args, 'optuna_trials', None) if getattr(args, 'hpo_method', 'gridsearch') == 'optuna' else None,
        "optuna_search_space_file": None,  # Se actualizará después si hay archivo
        
        # Dataset
        "n_patients_total": df_info.get("n_total", 0),
        "n_pwa": df_info.get("n_pwa", 0),
        "n_control": df_info.get("n_control", 0),
        "n_en": df_info.get("n_en", 0),
        "n_es": df_info.get("n_es", 0),
        "n_ca": df_info.get("n_ca", 0),
        
        "notes": getattr(args, 'notes', ""),
    }
    
    # ============ AÑADIR MODELO SI EXISTE ============
    if hasattr(args, 'model'):
        config["model"] = args.model
    else:
        # Legacy: asumir SVR
        config["model"] = "svr"
    
    # ============ AÑADIR PARAM GRID ============
    # Soportar formato directo (sin prefijo)
    for key, val in param_grid.items():
        # Remover prefijo "svr__" si existe
        clean_key = key.replace("svr__", "").replace("lgbm__", "").replace("xgb__", "")
        config[clean_key] = val
    
    # ============ FEATURE SELECTION INFO ============
    if selection_info:
        config['feature_selection_method'] = selection_info.get('method', args.features)
        config['feature_selection_time'] = selection_info.get('selection_time_seconds', 0)
        
        # Info específica por método
        if selection_info.get('method') == 'sfs':
            config['sfs_mae'] = selection_info.get('sfs_mae')
            config['sfs_max_features'] = selection_info.get('sfs_max_features')
            config['sfs_cv_folds'] = selection_info.get('sfs_cv_folds')
        elif selection_info.get('method') == 'kbest':
            config['kbest_k'] = selection_info.get('kbest_k')
            config['kbest_score_func'] = selection_info.get('kbest_score_func')
        elif selection_info.get('method') == 'rfe':
            config['rfe_n_features'] = selection_info.get('rfe_n_features_to_select')
            config['rfe_step'] = selection_info.get('rfe_step')
        elif selection_info.get('method') == 'importance':
            config['importance_model'] = selection_info.get('importance_model')
            config['importance_n_estimators'] = selection_info.get('importance_n_estimators')
    
    return config


def create_experiment_results(metrics_cv, metrics_cv_cal, metrics_es=None, 
                              metrics_ca=None, best_params=None):
    """
    Crea diccionario de resultados desde las metricas.
    
    COMPATIBLE CON:
    - train_svr_COMPLETO_FINAL.py (legacy)
    - train_model_UNIVERSAL.py (nuevo)
    """
    results = {
        # CV sin calibrar
        "cv_mae": metrics_cv.get("MAE"),
        "cv_rmse": metrics_cv.get("RMSE"),
        "cv_r2": metrics_cv.get("R2"),
        "cv_pearson": metrics_cv.get("Pearson_r"),
        "cv_spearman": metrics_cv.get("Spearman_rho"),
        "cv_severity_acc": metrics_cv.get("severity_accuracy"),
        
        # CV calibrado
        "cv_cal_mae": metrics_cv_cal.get("MAE"),
        "cv_cal_rmse": metrics_cv_cal.get("RMSE"),
        "cv_cal_r2": metrics_cv_cal.get("R2"),
        "cv_cal_pearson": metrics_cv_cal.get("Pearson_r"),
        "cv_cal_spearman": metrics_cv_cal.get("Spearman_rho"),
        "cv_cal_severity_acc": metrics_cv_cal.get("severity_accuracy"),
    }
    
    # ES
    if metrics_es:
        results["es_mae"] = metrics_es.get("raw_mae")
        results["es_cal_mae"] = metrics_es.get("cal_mae")
    
    # CA
    if metrics_ca:
        results["ca_mae"] = metrics_ca.get("raw_mae")
        results["ca_cal_mae"] = metrics_ca.get("cal_mae")
    
    # Best params (remover prefijos si existen)
    if best_params:
        for key, val in best_params.items():
            clean_key = key.replace("svr__", "").replace("lgbm__", "").replace("xgb__", "")
            results[f"best_{clean_key}"] = val
    
    return results


if __name__ == "__main__":
    # Test del logger
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', action='store_true', help='Mostrar resumen')
    parser.add_argument('--compare', action='store_true', help='Comparar modelos')
    parser.add_argument('--compare-fs', action='store_true', help='Comparar feature selection')
    parser.add_argument('--compare-hpo', action='store_true', help='Comparar HPO methods')
    parser.add_argument('--model', type=str, default=None, help='Filtrar por modelo')
    parser.add_argument('--fs', type=str, default=None, help='Filtrar por feature selection')
    parser.add_argument('--hpo', type=str, default=None, help='Filtrar por HPO method')
    args = parser.parse_args()
    
    logger = ExperimentLogger()
    
    if args.summary:
        logger.print_summary(model_type=args.model, feature_selection=args.fs, hpo_method=args.hpo)
    elif args.compare:
        logger.compare_models()
    elif args.compare_fs:
        logger.compare_feature_selection(model_type=args.model)
    elif args.compare_hpo:
        logger.compare_hpo_methods(model_type=args.model)
    else:
        print("Uso:")
        print("  python experiment_logger.py --summary")
        print("  python experiment_logger.py --summary --model svr")
        print("  python experiment_logger.py --summary --fs kbest")
        print("  python experiment_logger.py --summary --hpo optuna")
        print("  python experiment_logger.py --compare")
        print("  python experiment_logger.py --compare-fs")
        print("  python experiment_logger.py --compare-hpo")
        print("  python experiment_logger.py --compare-fs --model lgbm")