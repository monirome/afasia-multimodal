#!/usr/bin/env python3
# 05_models/hyperparameter_optimizer.py

"""
HYPERPARAMETER OPTIMIZER - Versión expandida con Optuna
Soporta GridSearch y Optuna para todos los modelos
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score

# Verificar si Optuna está disponible
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: Optuna no disponible - pip install optuna")

# =============================================================================
# OPTUNA SEARCH SPACES
# =============================================================================

def get_optuna_search_space(model_name):
    """Retorna función de search space para Optuna según el modelo"""
    
    model_name = model_name.lower()
    
    # =========================================================================
    # SVR
    # =========================================================================
    if model_name == 'svr':
        def objective(trial, model, X, y, cv):
            params = {
                'C': trial.suggest_float('C', 0.1, 100, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }
            model.set_params(**params)
            return -np.mean(cross_val_score(model, X, y, cv=cv, 
                                           scoring='neg_mean_absolute_error', n_jobs=-1))
        return objective
    
    # =========================================================================
    # LIGHTGBM
    # =========================================================================
    elif model_name == 'lgbm':
        def objective(trial, model, X, y, cv):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
            }
            model.set_params(**params)
            return -np.mean(cross_val_score(model, X, y, cv=cv,
                                           scoring='neg_mean_absolute_error', n_jobs=-1))
        return objective
    
    # =========================================================================
    # XGBOOST
    # =========================================================================
    elif model_name == 'xgb':
        def objective(trial, model, X, y, cv):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10)
            }
            model.set_params(**params)
            return -np.mean(cross_val_score(model, X, y, cv=cv,
                                           scoring='neg_mean_absolute_error', n_jobs=-1))
        return objective
    
    # =========================================================================
    # CATBOOST
    # =========================================================================
    elif model_name == 'catboost':
        def objective(trial, model, X, y, cv):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000, step=500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0, 1.0)
            }
            model.set_params(**params)
            return -np.mean(cross_val_score(model, X, y, cv=cv,
                                           scoring='neg_mean_absolute_error', n_jobs=-1))
        return objective
    
    # =========================================================================
    # RANDOM FOREST
    # =========================================================================
    elif model_name == 'rf':
        def objective(trial, model, X, y, cv):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
            model.set_params(**params)
            return -np.mean(cross_val_score(model, X, y, cv=cv,
                                           scoring='neg_mean_absolute_error', n_jobs=-1))
        return objective
    
    # =========================================================================
    # ELASTIC NET
    # =========================================================================
    elif model_name == 'elasticnet':
        def objective(trial, model, X, y, cv):
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9)
            }
            model.set_params(**params)
            return -np.mean(cross_val_score(model, X, y, cv=cv,
                                           scoring='neg_mean_absolute_error', n_jobs=-1))
        return objective
    
    # =========================================================================
    # NGBOOST (NUEVO)
    # =========================================================================
    elif model_name == 'ngboost':
        def objective(trial, model, X, y, cv):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'minibatch_frac': trial.suggest_float('minibatch_frac', 0.5, 1.0),
                'col_sample': trial.suggest_float('col_sample', 0.6, 1.0)
            }
            model.set_params(**params)
            return -np.mean(cross_val_score(model, X, y, cv=cv,
                                           scoring='neg_mean_absolute_error', n_jobs=-1))
        return objective
    
    # =========================================================================
    # EBM (NUEVO)
    # =========================================================================
    elif model_name == 'ebm':
        def objective(trial, model, X, y, cv):
            params = {
                'max_bins': trial.suggest_int('max_bins', 128, 512, step=128),
                'max_interaction_bins': trial.suggest_int('max_interaction_bins', 16, 64, step=16),
                'interactions': trial.suggest_int('interactions', 5, 30),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10)
            }
            model.set_params(**params)
            return -np.mean(cross_val_score(model, X, y, cv=cv,
                                           scoring='neg_mean_absolute_error', n_jobs=-1))
        return objective
    
    # =========================================================================
    # TABNET (NUEVO)
    # =========================================================================
    elif model_name == 'tabnet':
        def objective(trial, model, X, y, cv):
            params = {
                'n_d': trial.suggest_int('n_d', 8, 64, step=8),
                'n_a': trial.suggest_int('n_a', 8, 64, step=8),
                'n_steps': trial.suggest_int('n_steps', 3, 10),
                'gamma': trial.suggest_float('gamma', 1.0, 2.0),
                'lambda_sparse': trial.suggest_float('lambda_sparse', 0.0001, 0.01, log=True)
            }
            model.set_params(**params)
            # TabNet requiere manejo especial
            try:
                return -np.mean(cross_val_score(model, X, y, cv=cv,
                                               scoring='neg_mean_absolute_error', n_jobs=1))
            except:
                return float('inf')
        return objective
    
    # =========================================================================
    # SYMBOLIC REGRESSION (NUEVO)
    # =========================================================================
    elif model_name == 'symreg':
        def objective(trial, model, X, y, cv):
            params = {
                'population_size': trial.suggest_int('population_size', 500, 3000, step=500),
                'generations': trial.suggest_int('generations', 10, 100, step=10),
                'tournament_size': trial.suggest_int('tournament_size', 20, 100),
                'stopping_criteria': trial.suggest_float('stopping_criteria', 0.001, 0.1, log=True),
                'p_crossover': trial.suggest_float('p_crossover', 0.5, 0.95),
                'p_subtree_mutation': trial.suggest_float('p_subtree_mutation', 0.01, 0.2),
                'p_hoist_mutation': trial.suggest_float('p_hoist_mutation', 0.01, 0.1),
                'p_point_mutation': trial.suggest_float('p_point_mutation', 0.01, 0.1),
                'max_samples': trial.suggest_float('max_samples', 0.7, 1.0)
            }
            model.set_params(**params)
            return -np.mean(cross_val_score(model, X, y, cv=cv,
                                           scoring='neg_mean_absolute_error', n_jobs=-1))
        return objective
    
    # =========================================================================
    # TABPFN (Sin hiperparámetros)
    # =========================================================================
    elif model_name == 'tabpfn':
        def objective(trial, model, X, y, cv):
            # TabPFN no tiene hiperparámetros tunables
            return -np.mean(cross_val_score(model, X, y, cv=cv,
                                           scoring='neg_mean_absolute_error', n_jobs=-1))
        return objective
    
    else:
        raise ValueError(f"Modelo '{model_name}' no tiene search space definido")

# =============================================================================
# FUNCIÓN PARA OBTENER DESCRIPCIÓN DEL SEARCH SPACE
# =============================================================================

def get_optuna_search_space_description(model_name):
    """Retorna descripción legible del search space para logging"""
    
    spaces = {
        'svr': {
            'C': 'float(0.1, 100) log-scale',
            'epsilon': 'float(0.01, 1.0) log-scale',
            'gamma': "categorical(['scale', 'auto'])"
        },
        'lgbm': {
            'n_estimators': 'int(500, 2000) step=500',
            'learning_rate': 'float(0.01, 0.2) log-scale',
            'num_leaves': 'int(20, 150)',
            'max_depth': 'int(3, 20)',
            'min_child_samples': 'int(10, 100)',
            'subsample': 'float(0.6, 1.0)',
            'colsample_bytree': 'float(0.6, 1.0)',
            'reg_alpha': 'float(0, 1.0)',
            'reg_lambda': 'float(0, 1.0)'
        },
        'xgb': {
            'n_estimators': 'int(500, 2000) step=500',
            'learning_rate': 'float(0.01, 0.2) log-scale',
            'max_depth': 'int(3, 15)',
            'min_child_weight': 'int(1, 10)',
            'subsample': 'float(0.6, 1.0)',
            'colsample_bytree': 'float(0.6, 1.0)',
            'gamma': 'float(0, 0.5)',
            'reg_alpha': 'float(0, 1.0)',
            'reg_lambda': 'float(1, 10)'
        },
        'catboost': {
            'iterations': 'int(500, 2000) step=500',
            'learning_rate': 'float(0.01, 0.2) log-scale',
            'depth': 'int(4, 10)',
            'l2_leaf_reg': 'float(1, 10)',
            'bagging_temperature': 'float(0, 1.0)',
            'random_strength': 'float(0, 1.0)'
        },
        'rf': {
            'n_estimators': 'int(100, 500) step=100',
            'max_depth': 'int(5, 30)',
            'min_samples_split': 'int(2, 20)',
            'min_samples_leaf': 'int(1, 10)',
            'max_features': "categorical(['sqrt', 'log2'])"
        },
        'elasticnet': {
            'alpha': 'float(0.001, 10.0) log-scale',
            'l1_ratio': 'float(0.1, 0.9)'
        },
        'ngboost': {
            'n_estimators': 'int(500, 2000) step=500',
            'learning_rate': 'float(0.01, 0.2) log-scale',
            'minibatch_frac': 'float(0.5, 1.0)',
            'col_sample': 'float(0.6, 1.0)'
        },
        'ebm': {
            'max_bins': 'int(128, 512) step=128',
            'max_interaction_bins': 'int(16, 64) step=16',
            'interactions': 'int(5, 30)',
            'learning_rate': 'float(0.01, 0.1) log-scale',
            'min_samples_leaf': 'int(2, 10)'
        },
        'tabnet': {
            'n_d': 'int(8, 64) step=8',
            'n_a': 'int(8, 64) step=8',
            'n_steps': 'int(3, 10)',
            'gamma': 'float(1.0, 2.0)',
            'lambda_sparse': 'float(0.0001, 0.01) log-scale'
        },
        'symreg': {
            'population_size': 'int(500, 3000) step=500',
            'generations': 'int(10, 100) step=10',
            'tournament_size': 'int(20, 100)',
            'stopping_criteria': 'float(0.001, 0.1) log-scale',
            'p_crossover': 'float(0.5, 0.95)',
            'p_subtree_mutation': 'float(0.01, 0.2)',
            'p_hoist_mutation': 'float(0.01, 0.1)',
            'p_point_mutation': 'float(0.01, 0.1)',
            'max_samples': 'float(0.7, 1.0)'
        },
        'tabpfn': {
            'note': 'TabPFN no tiene hiperparámetros tunables'
        }
    }
    
    return spaces.get(model_name.lower(), {})

# =============================================================================
# FUNCIÓN PRINCIPAL DE OPTIMIZACIÓN
# =============================================================================

def optimize_hyperparameters(model, X, y, cv, model_name, method='gridsearch',
                            param_grid=None, n_trials=50, n_jobs=-1, verbose=False):
    """
    Optimiza hiperparámetros usando GridSearch o Optuna
    
    Args:
        model: Modelo sklearn
        X, y: Datos
        cv: CV strategy
        model_name: Nombre del modelo ('catboost', 'xgb', etc)
        method: 'gridsearch' o 'optuna'
        param_grid: Para GridSearch
        n_trials: Para Optuna
        n_jobs: Paralelismo
        verbose: Logs
    
    Returns:
        best_model: Modelo con mejores params (y ENTRENADO)
        best_params: Dict de mejores params
        best_score: Mejor MAE
        n_trials_used: Número de combinaciones probadas
    """
    
    if method == 'gridsearch':
        # GridSearch exhaustivo
        if param_grid is None or len(param_grid) == 0:
            # === CORRECCIÓN: ENTRENAR AUNQUE NO HAYA PARAMS ===
            # Esto es vital para TabPFN que tiene param_grid vacío
            model.fit(X, y)
            # ==================================================
            return model, {}, float('inf'), 1
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=n_jobs, verbose=1 if verbose else 0
        )
        
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_
        n_trials_used = len(grid_search.cv_results_['mean_test_score'])
        
        return best_model, best_params, best_score, n_trials_used
    
    elif method == 'optuna':
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna no está disponible")
        
        # Obtener función objetivo
        objective_func = get_optuna_search_space(model_name)
        
        # Crear estudio Optuna
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimizar
        study.optimize(
            lambda trial: objective_func(trial, model, X, y, cv),
            n_trials=n_trials,
            n_jobs=1,  # CV ya usa paralelismo
            show_progress_bar=verbose
        )
        
        # Mejores parámetros
        best_params = study.best_params
        best_score = study.best_value
        
        # Entrenar modelo con mejores params
        model.set_params(**best_params)
        model.fit(X, y)
        
        return model, best_params, best_score, n_trials
    
    else:
        raise ValueError(f"Método '{method}' no reconocido. Usa 'gridsearch' o 'optuna'")

