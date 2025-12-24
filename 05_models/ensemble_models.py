#!/usr/bin/env python3
# 05_models/ensemble_models.py
# -*- coding: utf-8 -*-
"""
ENSEMBLE MODELS - Combina múltiples modelos para mejores predicciones

QUÉ ES UN ENSEMBLE:
- Combina predicciones de varios modelos
- Reduce errores aleatorios
- Mejora robustez y generalización

TIPOS IMPLEMENTADOS:
1. Simple Average: Promedio simple de predicciones
2. Weighted Average: Promedio ponderado por MAE
3. Stacking: Meta-modelo aprende a combinar
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# ======================== ENSEMBLE SIMPLE (AVERAGING) ========================

class SimpleEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble más simple: promedio de predicciones.
    
    Ejemplo:
        Si tienes 3 modelos que predicen [10, 12, 11]
        La predicción final será: (10 + 12 + 11) / 3 = 11
    """
    
    def __init__(self, models):
        """
        Args:
            models: Lista de modelos ya entrenados
        """
        self.models = models
    
    def fit(self, X, y):
        """No hace nada, los modelos ya están entrenados"""
        return self
    
    def predict(self, X):
        """Promedio simple de todas las predicciones"""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)


# ======================== WEIGHTED ENSEMBLE (MEJOR) ========================

class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble ponderado: modelos con mejor MAE tienen más peso.
    
    Ejemplo:
        Modelo A: MAE = 10.0 → peso = 1/10 = 0.10
        Modelo B: MAE = 12.0 → peso = 1/12 = 0.08
        Modelo C: MAE = 15.0 → peso = 1/15 = 0.07
        
        Normalizado:
        Peso_A = 0.10 / (0.10 + 0.08 + 0.07) = 0.40
        Peso_B = 0.32
        Peso_C = 0.28
        
        Predicción = 0.40*Pred_A + 0.32*Pred_B + 0.28*Pred_C
    """
    
    def __init__(self, models, weights=None):
        """
        Args:
            models: Lista de modelos ya entrenados
            weights: Pesos opcionales (si None, se calculan automáticamente)
        """
        self.models = models
        self.weights = weights
        
        if self.weights is not None:
            # Normalizar pesos
            self.weights = np.array(self.weights)
            self.weights = self.weights / self.weights.sum()
    
    def fit(self, X, y):
        """No hace nada, los modelos ya están entrenados"""
        return self
    
    def predict(self, X):
        """Promedio ponderado de predicciones"""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.average(predictions, axis=0, weights=self.weights)
    
    def calculate_weights_from_mae(self, X_val, y_val):
        """
        Calcula pesos óptimos basados en MAE de validación.
        
        Args:
            X_val: Features de validación
            y_val: Targets de validación
        """
        maes = []
        for model in self.models:
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            maes.append(mae)
        
        # Peso inversamente proporcional al MAE
        inv_maes = 1.0 / np.array(maes)
        self.weights = inv_maes / inv_maes.sum()
        
        return self.weights, maes


# ======================== STACKING ENSEMBLE (AVANZADO) ========================

class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking: Un meta-modelo aprende cómo combinar predicciones.
    
    Funcionamiento:
    1. Modelos base hacen predicciones
    2. Meta-modelo usa esas predicciones como features
    3. Meta-modelo aprende la mejor combinación
    
    Ventaja: Puede aprender relaciones no-lineales entre modelos
    """
    
    def __init__(self, base_models, meta_model=None):
        """
        Args:
            base_models: Lista de modelos base
            meta_model: Modelo que combina (default: Ridge)
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
    
    def fit(self, X, y):
        """
        Entrena el meta-modelo.
        Los base_models deben estar ya entrenados.
        """
        # Generar predicciones de los modelos base
        meta_features = self._get_meta_features(X)
        
        # Entrenar meta-modelo
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """Predicción usando stacking"""
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def _get_meta_features(self, X):
        """Genera features para el meta-modelo (predicciones de base models)"""
        predictions = []
        for model in self.base_models:
            predictions.append(model.predict(X))
        return np.column_stack(predictions)


# ======================== FUNCIONES HELPER ========================

def load_models_from_directories(model_dirs, model_filename='model_final.pkl'):
    """
    Carga modelos desde múltiples directorios.
    
    Args:
        model_dirs: Lista de paths a directorios con modelos
        model_filename: Nombre del archivo del modelo
    
    Returns:
        Lista de modelos cargados
    """
    models = []
    model_info = []
    
    for model_dir in model_dirs:
        model_path = Path(model_dir) / model_filename
        
        if not model_path.exists():
            print(f"  No encontrado: {model_path}")
            continue
        
        try:
            model_dict = joblib.load(model_path)
            models.append(model_dict['model'])
            
            info = {
                'path': str(model_dir),
                'model_type': model_dict.get('model_type', 'unknown'),
                'hpo_method': model_dict.get('hpo_method', 'unknown'),
            }
            model_info.append(info)
            
            print(f" Cargado: {Path(model_dir).name}")
            
        except Exception as e:
            print(f" Error cargando {model_path}: {e}")
    
    return models, model_info


def create_ensemble_from_experiment_history(
    csv_path,
    top_n=3,
    ensemble_type='weighted',
    metric='cv_cal_mae',
    model_type=None,
    base_results_dir=None
):
    """
    Crea ensemble automáticamente desde experiments_history.csv
    
    Args:
        csv_path: Path al CSV de historial
        top_n: Número de mejores modelos a incluir
        ensemble_type: 'simple', 'weighted', o 'stacking'
        metric: Métrica para seleccionar mejores modelos
        model_type: Filtrar por tipo de modelo (None = todos)
        base_results_dir: Directorio base donde están los modelos
    
    Returns:
        ensemble, model_info
    """
    # Leer historial
    df = pd.read_csv(csv_path)
    
    # Filtrar por modelo si se especifica
    if model_type is not None:
        df = df[df['model_type'] == model_type.lower()]
    
    # Ordenar por métrica y tomar top N
    df_sorted = df.sort_values(metric).head(top_n)
    
    print("\n" + "="*70)
    print(f"CREANDO ENSEMBLE - Top {top_n} modelos por {metric}")
    print("="*70)
    
    # Mostrar modelos seleccionados
    print("\nModelos seleccionados:")
    for idx, (i, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"  {idx}. {row['experiment_name']}")
        print(f"     {metric}: {row[metric]:.3f}")
        print(f"     Modelo: {row['model_type'].upper()}")
    
    # Construir paths
    if base_results_dir is None:
        # Intentar inferir del run_dir del primer modelo
        first_run_dir = Path(df_sorted.iloc[0]['run_dir'])
        base_results_dir = first_run_dir.parent
    
    model_dirs = [Path(base_results_dir) / row['experiment_name'] 
                  for _, row in df_sorted.iterrows()]
    
    # Cargar modelos
    print("\nCargando modelos...")
    models, model_info = load_models_from_directories(model_dirs)
    
    if len(models) < 2:
        raise ValueError(f"Se necesitan al menos 2 modelos, solo se cargaron {len(models)}")
    
    # Crear ensemble
    print(f"\nCreando ensemble tipo '{ensemble_type}'...")
    
    if ensemble_type == 'simple':
        ensemble = SimpleEnsemble(models)
        print(" Ensemble simple creado (promedio igual para todos)")
        
    elif ensemble_type == 'weighted':
        # Pesos basados en MAE del CSV
        maes = df_sorted[metric].values
        inv_maes = 1.0 / maes
        weights = inv_maes / inv_maes.sum()
        
        ensemble = WeightedEnsemble(models, weights)
        
        print(" Ensemble ponderado creado")
        print("\nPesos asignados:")
        for i, (weight, mae) in enumerate(zip(weights, maes), 1):
            print(f"  Modelo {i}: peso={weight:.3f} (MAE={mae:.3f})")
        
    elif ensemble_type == 'stacking':
        ensemble = StackingEnsemble(models)
        print(" Ensemble stacking creado (meta-modelo: Ridge)")
        print("  IMPORTANTE: Debes llamar a ensemble.fit(X_val, y_val) antes de usar")
        
    else:
        raise ValueError(f"ensemble_type desconocido: {ensemble_type}")
    
    print("="*70)
    
    return ensemble, model_info


def evaluate_ensemble(ensemble, X, y, models_info=None):
    """
    Evalúa el ensemble y compara con modelos individuales.
    
    Args:
        ensemble: Ensemble entrenado
        X: Features de evaluación
        y: Targets de evaluación
        models_info: Info de modelos (opcional)
    
    Returns:
        Dict con métricas
    """
    from scipy.stats import pearsonr, spearmanr
    
    # Predicción del ensemble
    y_pred_ensemble = ensemble.predict(X)
    
    # Métricas del ensemble
    mae_ensemble = mean_absolute_error(y, y_pred_ensemble)
    pearson_ensemble, _ = pearsonr(y, y_pred_ensemble)
    spearman_ensemble, _ = spearmanr(y, y_pred_ensemble)
    
    print("\n" + "="*70)
    print("EVALUACIÓN DEL ENSEMBLE")
    print("="*70)
    
    print(f"\n ENSEMBLE:")
    print(f"   MAE:      {mae_ensemble:.3f}")
    print(f"   Pearson:  {pearson_ensemble:.3f}")
    print(f"   Spearman: {spearman_ensemble:.3f}")
    
    # Comparar con modelos individuales si están disponibles
    if hasattr(ensemble, 'models'):
        print(f"\n MODELOS INDIVIDUALES:")
        
        individual_maes = []
        for i, model in enumerate(ensemble.models, 1):
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            pearson, _ = pearsonr(y, y_pred)
            individual_maes.append(mae)
            
            print(f"\n   Modelo {i}:")
            print(f"      MAE:      {mae:.3f}")
            print(f"      Pearson:  {pearson:.3f}")
        
        # Calcular mejora
        best_individual_mae = min(individual_maes)
        improvement = best_individual_mae - mae_ensemble
        improvement_pct = (improvement / best_individual_mae) * 100
        
        print(f"\n MEJORA:")
        print(f"   Mejor modelo individual: MAE = {best_individual_mae:.3f}")
        print(f"   Ensemble:                MAE = {mae_ensemble:.3f}")
        print(f"   Mejora:                  {improvement:.3f} ({improvement_pct:+.1f}%)")
    
    print("="*70)
    
    return {
        'mae_ensemble': mae_ensemble,
        'pearson_ensemble': pearson_ensemble,
        'spearman_ensemble': spearman_ensemble,
    }


# ======================== GUARDAR/CARGAR ENSEMBLE ========================

def save_ensemble(ensemble, filepath, models_info=None):
    """
    Guarda un ensemble entrenado.
    
    Args:
        ensemble: Ensemble a guardar
        filepath: Path donde guardar
        models_info: Información adicional (opcional)
    """
    ensemble_dict = {
        'ensemble': ensemble,
        'ensemble_type': type(ensemble).__name__,
        'n_models': len(ensemble.models) if hasattr(ensemble, 'models') else None,
        'models_info': models_info,
    }
    
    if hasattr(ensemble, 'weights') and ensemble.weights is not None:
        ensemble_dict['weights'] = ensemble.weights
    
    joblib.dump(ensemble_dict, filepath)
    print(f" Ensemble guardado: {filepath}")


def load_ensemble(filepath):
    """
    Carga un ensemble guardado.
    
    Args:
        filepath: Path del ensemble
    
    Returns:
        ensemble, models_info
    """
    ensemble_dict = joblib.load(filepath)
    ensemble = ensemble_dict['ensemble']
    models_info = ensemble_dict.get('models_info', None)
    
    print(f" Ensemble cargado: {filepath}")
    print(f"  Tipo: {ensemble_dict['ensemble_type']}")
    print(f"  N modelos: {ensemble_dict['n_models']}")
    
    if 'weights' in ensemble_dict:
        print(f"  Pesos: {ensemble_dict['weights']}")
    
    return ensemble, models_info