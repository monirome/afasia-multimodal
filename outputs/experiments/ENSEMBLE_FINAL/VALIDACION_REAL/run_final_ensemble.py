import sys
import os
import shutil
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Configuracion de rutas
BASE_OUTPUT = Path("outputs/experiments/ENSEMBLE_FINAL/VALIDACION_REAL")
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

# Configuracion de Log (Dual: Pantalla y Archivo)
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

# Backup del propio script
current_script = Path(__file__)
backup_script = BASE_OUTPUT / current_script.name
try:
    shutil.copy(current_script, backup_script)
    print(f"[INFO] Script copiado a: {backup_script}")
except Exception as e:
    print(f"[WARN] No se pudo hacer backup del script: {e}")

# Importar modelos
try:
    from tabpfn import TabPFNRegressor
    from interpret.glassbox import ExplainableBoostingRegressor
    from catboost import CatBoostRegressor
except ImportError:
    print("[ERROR] Faltan librerias. Instala tabpfn interpret catboost seaborn")
    sys.exit(1)

# Pesos del Ensemble
WEIGHTS = [0.50, 0.25, 0.25]  # TabPFN, EBM, CatBoost
N_FOLDS = 5
RANDOM_STATE = 42

def get_severity(score):
    if score < 25: return "Muy Severa (0-25)"
    if score < 50: return "Severa (25-50)"
    if score < 75: return "Moderada (50-75)"
    return "Leve (75-100)"

def generate_plots(df_results, feat_importance_df, output_dir, mae_score):
    print("   Generando graficas...")
    y_true = df_results['QA_real']
    y_pred = df_results['QA_pred']
    
    # 1. Scatter Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='#2c3e50')
    plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Ideal')
    plt.xlabel('QA Real')
    plt.ylabel('QA Predicho (CV)')
    plt.title(f'Validacion Real 5-Fold (MAE: {mae_score:.2f})')
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
        
        sns.barplot(data=mae_lang, x='language', y='MAE', palette='viridis')
        plt.title('Error Medio (MAE) por Idioma')
        
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
    plt.title('Matriz de Confusion (Severidad)')
    plt.savefig(output_dir / "plot_confusion_matrix_REAL.png")
    plt.close()

    # 4. Feature Importance (Estilo Clasico)
    if feat_importance_df is not None:
        plt.figure(figsize=(10, 12))
        # Agrupar por variable y calcular media de importancia entre folds
        avg_imp = feat_importance_df.groupby('feature')['importance'].mean().reset_index()
        avg_imp = avg_imp.sort_values('importance', ascending=False).head(25)
        
        sns.barplot(data=avg_imp, y='feature', x='importance', color='#2c3e50')
        plt.title('Variables mas Importantes del Ensemble (Top 25)')
        plt.xlabel('Incremento del Error (MAE) al permutar')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_feature_importance_REAL.png")
        plt.close()

def predict_ensemble(X_in, models, transformers, weights):
    """Funcion auxiliar para predecir con el ensemble dentro de un loop"""
    # Desempaquetar
    model_tab, model_ebm, model_cat = models
    imputer, scaler, selector = transformers
    
    # Pipeline comun
    X_imp = imputer.transform(X_in)
    X_sc = scaler.transform(X_imp)
    
    # 1. TabPFN (usa selector)
    X_sel = selector.transform(X_sc)
    p_tab = model_tab.predict(X_sel)
    
    # 2. EBM (usa scaled)
    p_ebm = model_ebm.predict(X_sc)
    
    # 3. CatBoost (usa scaled)
    p_cat = model_cat.predict(X_sc)
    
    # Ponderacion
    return (p_tab * weights[0]) + (p_ebm * weights[1]) + (p_cat * weights[2])

def main():
    print("="*70)
    print("EJECUTANDO VALIDACION CIENTIFICA (5-FOLD CROSS-VALIDATION)")
    print(f"Resultados en: {BASE_OUTPUT}")
    print("="*70)

    print("Cargando dataset...")
    try:
        df = pd.read_csv("data/dataset_FINAL_CON_POSLM.csv")
    except FileNotFoundError:
        print("[ERROR] No se encuentra data/dataset_FINAL_CON_POSLM.csv")
        return

    df = df[df['QA'].notna()].copy().reset_index(drop=True)
    y = df['QA'].values
    cols_meta = ['patient_id', 'QA', 'group', 'language', 'sex', 'age', 'aphasia_type', 'group_original', 'transcript']
    X = df.drop(columns=[c for c in cols_meta if c in df.columns], errors='ignore')
    
    final_predictions = np.zeros(len(y))
    y_bins = pd.cut(y, bins=5, labels=False)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Para guardar importancia de variables
    fold_importances = []

    fold = 1
    for train_index, test_index in skf.split(X, y_bins):
        print(f"\n[INFO] PROCESANDO FOLD {fold}/{N_FOLDS}...")
        
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = y[train_index], y[test_index]
        
        # Preprocesado
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train_np = imputer.fit_transform(X_train)
        X_train_np = scaler.fit_transform(X_train_np)
        
        # Select Features (TabPFN)
        selector = SelectKBest(score_func=mutual_info_regression, k=40)
        selector.fit(X_train_np, y_train)
        X_train_sel = selector.transform(X_train_np)
        
        # --- ENTRENAMIENTO ---
        
        # 1. TabPFN
        print("  Entrenando TabPFN...", end=" ")
        try:
            model_tab = TabPFNRegressor(device='cpu')
        except:
            model_tab = TabPFNRegressor()
        model_tab.fit(X_train_sel, y_train)
        print("OK")
        
        # 2. EBM
        print("  Entrenando EBM...", end=" ")
        model_ebm = ExplainableBoostingRegressor(random_state=RANDOM_STATE, n_jobs=1)
        model_ebm.fit(X_train_np, y_train)
        print("OK")
        
        # 3. CatBoost
        print("  Entrenando CatBoost...", end=" ")
        model_cat = CatBoostRegressor(verbose=0, random_state=RANDOM_STATE, depth=6, iterations=500)
        model_cat.fit(X_train_np, y_train)
        print("OK")
        
        # Agrupar objetos
        models_pack = (model_tab, model_ebm, model_cat)
        transformers_pack = (imputer, scaler, selector)

        # --- PREDICCION FOLD ---
        fold_pred = predict_ensemble(X_test, models_pack, transformers_pack, WEIGHTS)
        final_predictions[test_index] = fold_pred
        
        baseline_mae = mean_absolute_error(y_test, fold_pred)
        print(f"  [RESULT] MAE Fold {fold}: {baseline_mae:.3f}")

        # --- IMPORTANCIA DE VARIABLES (Permutation Importance) ---
        # Solo calculamos las top 50 variables para no tardar una eternidad con TabPFN
        print("  Calculando importancia de variables (esto puede tardar)...")
        # Usamos las features crudas para permutar
        feature_names = X.columns.tolist()
        
        # Seleccion rapida: Usamos la importancia de CatBoost para filtrar que variables permutar
        # (Permutar todas tarda mucho, permutar las que el arbol usa es mas eficiente)
        cb_imps = model_cat.get_feature_importance()
        top_indices = np.argsort(cb_imps)[-30:] # Solo las top 30 candidatas
        
        for idx in top_indices:
            col_name = feature_names[idx]
            
            # Guardar original
            original_col = X_test[col_name].values.copy()
            
            # Permutar (Shuffle)
            X_test[col_name] = np.random.permutation(X_test[col_name].values)
            
            # Predecir con ruido
            perm_pred = predict_ensemble(X_test, models_pack, transformers_pack, WEIGHTS)
            perm_mae = mean_absolute_error(y_test, perm_pred)
            
            # Importancia = Cuanto empeora el error
            importance_score = perm_mae - baseline_mae
            fold_importances.append({'feature': col_name, 'importance': importance_score, 'fold': fold})
            
            # Restaurar original
            X_test[col_name] = original_col

        fold += 1

    # --- RESULTADOS FINALES ---
    print(f"\n{'='*70}")
    mae_total = mean_absolute_error(y, final_predictions)
    from scipy.stats import pearsonr
    corr, _ = pearsonr(y, final_predictions)
    
    # Guardar CSV Predicciones
    df_res = df[['patient_id', 'QA', 'language', 'aphasia_type']].copy()
    df_res['QA_real'] = y
    df_res['QA_pred'] = final_predictions
    df_res['Error_Abs'] = abs(df_res['QA_pred'] - df_res['QA_real'])
    df_res.to_csv(BASE_OUTPUT / "validacion_real_5fold.csv", index=False)
    
    # Guardar CSV Importancia
    df_imp = pd.DataFrame(fold_importances)
    df_imp.to_csv(BASE_OUTPUT / "ensemble_feature_importance.csv", index=False)

    # Generar Graficas
    generate_plots(df_res, df_imp, BASE_OUTPUT, mae_total)
    
    print(f"RESULTADO FINAL (5-FOLD CV):")
    print(f"  MAE REAL:         {mae_total:.4f}")
    print(f"  CORRELACION REAL: {corr:.4f}")
    print(f"{'='*70}")
    print(f"Archivos guardados en: {BASE_OUTPUT}")

if __name__ == "__main__":
    main()  