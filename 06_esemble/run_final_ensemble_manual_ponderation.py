#!/usr/bin/env python3
# 06_esemble/run_final_ensemble_manual_ponderation.py

import sys
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# === CONFIGURACION ===
# Rutas relativas asumiendo ejecucion desde codigos_julio2025
BASE_OUTPUT = Path("outputs/experiments/ENSEMBLE_FINAL/VALIDACION_REAL")
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
DATA_FILE = Path("data/dataset_FINAL_CON_POSLM.csv")

# Configurar Log Dual (Archivo + Pantalla)
log_file = BASE_OUTPUT / "execution.log"

class DualLogger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger()
sys.stderr = DualLogger()

# Copia de seguridad del script
current_script = Path(__file__)
backup_script = BASE_OUTPUT / "script_used.py"
try:
    shutil.copy(current_script, backup_script)
except:
    pass

# Importar modelos
try:
    from tabpfn import TabPFNRegressor
    from interpret.glassbox import ExplainableBoostingRegressor
    from catboost import CatBoostRegressor
except ImportError:
    print("[ERROR] Faltan librerias. Instala: tabpfn interpret catboost seaborn")
    sys.exit(1)

# PESOS DEL ENSEMBLE
WEIGHTS = [0.50, 0.25, 0.25]  # TabPFN, EBM, CatBoost
N_FOLDS = 5
RANDOM_STATE = 42

def get_severity(score):
    if score < 25: return "Muy Severa (0-25)"
    if score < 50: return "Severa (25-50)"
    if score < 75: return "Moderada (50-75)"
    return "Leve (75-100)"

def generate_plots(df_results, feat_importance_df, output_dir, mae_score):
    print("   Generando graficas completas...")
    y_true = df_results['QA_real']
    y_pred = df_results['QA_pred']
    
    # 1. Scatter Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='#2c3e50')
    plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Ideal')
    plt.xlabel('QA Real')
    plt.ylabel('QA Predicho (Ensemble)')
    plt.title(f'Validacion Cruzada Real (MAE: {mae_score:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "plot_scatter_REAL.png")
    plt.close()

    # 2. Analisis por Idioma
    if 'language' in df_results.columns:
        plt.figure(figsize=(8, 6))
        mae_lang = df_results.groupby('language').apply(
            lambda x: mean_absolute_error(x['QA_real'], x['QA_pred'])
        ).reset_index(name='MAE')
        
        mae_lang = mae_lang.sort_values('MAE')
        
        sns.barplot(data=mae_lang, x='language', y='MAE', palette='viridis')
        plt.title('Error Medio (MAE) por Idioma')
        plt.ylim(0, max(mae_lang['MAE']) + 5)
        
        for index, row in mae_lang.iterrows():
            plt.text(index, row['MAE'] + 0.2, f"{row['MAE']:.2f}", ha='center')
            
        plt.savefig(output_dir / "plot_mae_language_REAL.png")
        plt.close()

    # 3. Matriz de Confusion
    true_sev = [get_severity(x) for x in y_true]
    pred_sev = [get_severity(x) for x in y_pred]
    labels = ["Muy Severa (0-25)", "Severa (25-50)", "Moderada (50-75)", "Leve (75-100)"]
    
    cm = confusion_matrix(true_sev, pred_sev, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0-25", "25-50", "50-75", "75-100"])
    
    plt.figure(figsize=(9, 9))
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), colorbar=False)
    plt.title('Matriz de Confusion (Clasificacion Severidad)')
    plt.savefig(output_dir / "plot_confusion_matrix_REAL.png")
    plt.close()

    # 4. Feature Importance
    if feat_importance_df is not None:
        plt.figure(figsize=(10, 12))
        avg_imp = feat_importance_df.groupby('feature')['importance'].mean().reset_index()
        avg_imp = avg_imp.sort_values('importance', ascending=False).head(30)
        
        sns.barplot(data=avg_imp, y='feature', x='importance', color='teal')
        plt.title('Importancia de Variables en el Ensemble (Top 30)')
        plt.xlabel('Impacto en el Error (Cuanto sube el MAE si falta la variable)')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_feature_importance_REAL.png")
        plt.close()

def predict_ensemble_pack(X_input, models_pack, transformers_pack, weights):
    """Predice usando los 3 modelos y sus transformaciones"""
    m_tab, m_ebm, m_cat = models_pack
    imputer, scaler, selector = transformers_pack
    
    X_imp = imputer.transform(X_input)
    X_sc = scaler.transform(X_imp)
    
    # 1. TabPFN
    X_sel = selector.transform(X_sc)
    p_tab = m_tab.predict(X_sel)
    
    # 2. EBM
    p_ebm = m_ebm.predict(X_sc)
    
    # 3. CatBoost
    p_cat = m_cat.predict(X_sc)
    
    return (p_tab * weights[0]) + (p_ebm * weights[1]) + (p_cat * weights[2])

def main():
    print("-" * 70)
    print("EJECUTANDO VALIDACION CIENTIFICA (5-FOLD) - ENSEMBLE FINAL")
    print(f"Salida: {BASE_OUTPUT}")
    print("-" * 70)

    print("Cargando dataset...")
    if not DATA_FILE.exists():
        print(f"[ERROR] No encuentro {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    df = df[df['QA'].notna()].copy().reset_index(drop=True)
    y = df['QA'].values
    cols_meta = ['patient_id', 'QA', 'group', 'language', 'sex', 'age', 'aphasia_type', 'group_original', 'transcript']
    X = df.drop(columns=[c for c in cols_meta if c in df.columns], errors='ignore')
    feature_names = X.columns.tolist()

    final_predictions = np.zeros(len(y))
    y_bins = pd.cut(y, bins=5, labels=False)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_importances = []

    fold = 1
    for train_index, test_index in skf.split(X, y_bins):
        print(f"\n[INFO] PROCESANDO FOLD {fold}/{N_FOLDS}...")
        
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = y[train_index], y[test_index]
        
        # --- Preprocesado ---
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train_np = imputer.fit_transform(X_train)
        X_train_np = scaler.fit_transform(X_train_np)
        
        # Feature Selection para TabPFN
        selector = SelectKBest(score_func=mutual_info_regression, k=40)
        selector.fit(X_train_np, y_train)
        X_train_sel = selector.transform(X_train_np)
        
        # --- Entrenamiento ---
        print("  Entrenando TabPFN...", end=" ")
        try:
            model_tab = TabPFNRegressor(device='cpu')
        except:
            model_tab = TabPFNRegressor()
        model_tab.fit(X_train_sel, y_train)
        print("OK")
        
        print("  Entrenando EBM...", end=" ")
        model_ebm = ExplainableBoostingRegressor(random_state=RANDOM_STATE, n_jobs=1)
        model_ebm.fit(X_train_np, y_train)
        print("OK")
        
        print("  Entrenando CatBoost...", end=" ")
        model_cat = CatBoostRegressor(verbose=0, random_state=RANDOM_STATE, depth=6, iterations=500)
        model_cat.fit(X_train_np, y_train)
        print("OK")
        
        models_pack = (model_tab, model_ebm, model_cat)
        transformers_pack = (imputer, scaler, selector)

        # --- Prediccion ---
        fold_pred = predict_ensemble_pack(X_test, models_pack, transformers_pack, WEIGHTS)
        final_predictions[test_index] = fold_pred
        
        mae_fold = mean_absolute_error(y_test, fold_pred)
        print(f"  [RESULT] MAE Fold {fold}: {mae_fold:.3f}")

        # --- Importancia de Variables ---
        print("  Calculando importancia de variables (Permutation)...")
        cb_imps = model_cat.get_feature_importance()
        top_indices = np.argsort(cb_imps)[-30:] 
        
        baseline_error = mean_absolute_error(y_test, fold_pred)
        
        for idx in top_indices:
            col_name = feature_names[idx]
            original_values = X_test[col_name].values.copy()
            
            X_test[col_name] = np.random.permutation(X_test[col_name].values)
            perm_pred = predict_ensemble_pack(X_test, models_pack, transformers_pack, WEIGHTS)
            perm_mae = mean_absolute_error(y_test, perm_pred)
            
            importance = perm_mae - baseline_error
            fold_importances.append({'feature': col_name, 'importance': importance})
            
            X_test[col_name] = original_values

        fold += 1

    # --- FINALIZACION ---
    print("-" * 70)
    mae_total = mean_absolute_error(y, final_predictions)
    from scipy.stats import pearsonr
    corr, _ = pearsonr(y, final_predictions)
    
    df_res = df[['patient_id', 'QA', 'language', 'aphasia_type']].copy()
    df_res['QA_real'] = y
    df_res['QA_pred'] = final_predictions
    df_res['Error'] = df_res['QA_pred'] - df_res['QA_real']
    df_res.to_csv(BASE_OUTPUT / "validacion_real_5fold.csv", index=False)
    
    df_imp = pd.DataFrame(fold_importances)
    df_imp.to_csv(BASE_OUTPUT / "ensemble_importances.csv", index=False)

    generate_plots(df_res, df_imp, BASE_OUTPUT, mae_total)
    
    print(f"RESULTADO CIENTIFICO FINAL (CV):")
    print(f"  MAE REAL: {mae_total:.4f}")
    print(f"  CORRELACION: {corr:.4f}")
    print("-" * 70)
    print(f"Archivos guardados en: {BASE_OUTPUT}")

if __name__ == "__main__":
    main()