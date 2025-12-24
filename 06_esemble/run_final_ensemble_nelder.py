import sys
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# === CONFIGURACION DE RUTAS INTELIGENTE ===
# Detectamos donde esta el script y subimos un nivel para encontrar la raiz
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Definimos rutas
BASE_OUTPUT = PROJECT_ROOT / "outputs/experiments/ENSEMBLE_FINAL/VALIDACION_REAL_NELDER_PONDERATION"
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

DATA_FILE = PROJECT_ROOT / "data/dataset_FINAL_CON_POSLM.csv"

# Logging
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

# Backup del script
shutil.copy(Path(__file__), BASE_OUTPUT / "script_used.py")

# Importar modelos
try:
    from tabpfn import TabPFNRegressor
    from interpret.glassbox import ExplainableBoostingRegressor
    from catboost import CatBoostRegressor
except ImportError:
    print("[ERROR] Faltan librerias")
    sys.exit(1)

N_FOLDS = 5
RANDOM_STATE = 42

def optimize_ensemble_weights(y_true, pred_matrix):
    print("   -> Buscando la combinacion matematica perfecta (SLSQP)...")
    def objective(weights):
        final_pred = np.dot(pred_matrix, weights)
        return mean_absolute_error(y_true, final_pred)
    
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = ((0, 1), (0, 1), (0, 1))
    initial_weights = [0.33, 0.33, 0.34]
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def get_severity(score):
    if score < 25: return "Muy Severa (0-25)"
    if score < 50: return "Severa (25-50)"
    if score < 75: return "Moderada (50-75)"
    return "Leve (75-100)"

def generate_plots(df_results, feat_importance_df, output_dir, mae_score, weights):
    y_true = df_results['QA_real']
    y_pred = df_results['QA_pred']
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='#2c3e50')
    plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Ideal')
    title_str = f'Validacion Optimizada (MAE: {mae_score:.3f})\n'
    title_str += f'Pesos: Tab={weights[0]:.2f}, EBM={weights[1]:.2f}, Cat={weights[2]:.2f}'
    plt.title(title_str, fontsize=10)
    plt.xlabel('QA Real')
    plt.ylabel('QA Predicho')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "plot_scatter_REAL.png")
    plt.close()

    true_sev = [get_severity(x) for x in y_true]
    pred_sev = [get_severity(x) for x in y_pred]
    labels = ["Muy Severa (0-25)", "Severa (25-50)", "Moderada (50-75)", "Leve (75-100)"]
    
    cm = confusion_matrix(true_sev, pred_sev, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0-25", "25-50", "50-75", "75-100"])
    plt.figure(figsize=(9, 9))
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), colorbar=False)
    plt.title('Matriz de Confusion')
    plt.savefig(output_dir / "plot_confusion_matrix_REAL.png")
    plt.close()

    if feat_importance_df is not None:
        plt.figure(figsize=(10, 12))
        avg_imp = feat_importance_df.groupby('feature')['importance'].mean().reset_index()
        avg_imp = avg_imp.sort_values('importance', ascending=False).head(30)
        sns.barplot(data=avg_imp, y='feature', x='importance', color='teal')
        plt.title('Importancia de Variables (Ensemble Optimizado)')
        plt.tight_layout()
        plt.savefig(output_dir / "plot_feature_importance_REAL.png")
        plt.close()

def main():
    print("-" * 70)
    print("EJECUTANDO VALIDACION CIENTIFICA CON OPTIMIZACION DE PESOS")
    print(f"Script ubicado en: {CURRENT_DIR}")
    print("-" * 70)

    if not DATA_FILE.exists():
        print(f"[ERROR] No encuentro {DATA_FILE}")
        return
    df = pd.read_csv(DATA_FILE)
    df = df[df['QA'].notna()].copy().reset_index(drop=True)
    y = df['QA'].values
    cols_meta = ['patient_id', 'QA', 'group', 'language', 'sex', 'age', 'aphasia_type', 'group_original', 'transcript']
    X = df.drop(columns=[c for c in cols_meta if c in df.columns], errors='ignore')
    feature_names = X.columns.tolist()

    raw_preds = {'tab': np.zeros(len(y)), 'ebm': np.zeros(len(y)), 'cat': np.zeros(len(y))}
    y_bins = pd.cut(y, bins=5, labels=False)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_importances = []

    fold = 1
    for train_index, test_index in skf.split(X, y_bins):
        print(f"\n[INFO] PROCESANDO FOLD {fold}/{N_FOLDS}...")
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = y[train_index], y[test_index]
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_np = scaler.transform(imputer.transform(X_test))
        
        selector = SelectKBest(score_func=mutual_info_regression, k=40)
        X_train_sel = selector.fit_transform(X_train_np, y_train)
        X_test_sel = selector.transform(X_test_np)
        
        # 1. TabPFN
        try: m1 = TabPFNRegressor(device='cpu')
        except: m1 = TabPFNRegressor()
        m1.fit(X_train_sel, y_train)
        p_tab = m1.predict(X_test_sel)
        
        # 2. EBM
        m2 = ExplainableBoostingRegressor(random_state=RANDOM_STATE, n_jobs=1)
        m2.fit(X_train_np, y_train)
        p_ebm = m2.predict(X_test_np)
        
        # 3. CatBoost
        m3 = CatBoostRegressor(verbose=0, random_state=RANDOM_STATE, depth=6, iterations=500)
        m3.fit(X_train_np, y_train)
        p_cat = m3.predict(X_test_np)
        
        raw_preds['tab'][test_index] = p_tab
        raw_preds['ebm'][test_index] = p_ebm
        raw_preds['cat'][test_index] = p_cat
        
        print("  Calculando importancia variables...")
        baseline_pred = (p_tab + p_ebm + p_cat) / 3
        baseline_err = mean_absolute_error(y_test, baseline_pred)
        cb_imps = m3.get_feature_importance()
        top_idx = np.argsort(cb_imps)[-20:] 
        
        for idx in top_idx:
            col = feature_names[idx]
            orig_vals = X_test.iloc[:, idx].values.copy()
            X_test.iloc[:, idx] = np.random.permutation(orig_vals)
            
            X_perm_imp = imputer.transform(X_test)
            X_perm_sc = scaler.transform(X_perm_imp)
            X_perm_sel = selector.transform(X_perm_sc)
            
            p1_p = m1.predict(X_perm_sel)
            p2_p = m2.predict(X_perm_sc)
            p3_p = m3.predict(X_perm_sc)
            
            imp = mean_absolute_error(y_test, (p1_p+p2_p+p3_p)/3) - baseline_err
            fold_importances.append({'feature': col, 'importance': imp})
            X_test.iloc[:, idx] = orig_vals
        fold += 1

    print("\n" + "="*50)
    print("OPTIMIZANDO PESOS DEL ENSEMBLE...")
    pred_matrix = np.column_stack([raw_preds['tab'], raw_preds['ebm'], raw_preds['cat']])
    
    base_mae = mean_absolute_error(y, (pred_matrix[:,0]*0.5 + pred_matrix[:,1]*0.25 + pred_matrix[:,2]*0.25))
    print(f"MAE Pesos Manuales (0.50, 0.25, 0.25): {base_mae:.4f}")
    
    opt_weights = optimize_ensemble_weights(y, pred_matrix)
    final_preds = np.dot(pred_matrix, opt_weights)
    final_mae = mean_absolute_error(y, final_preds)
    
    print(f"MAE OPTIMIZADO: {final_mae:.4f}")
    print(f"PESOS IDEALES: Tab={opt_weights[0]:.3f}, EBM={opt_weights[1]:.3f}, Cat={opt_weights[2]:.3f}")
    print("="*50)

    df_res = df[['patient_id', 'QA', 'language', 'aphasia_type']].copy()
    df_res['QA_real'] = y
    df_res['QA_pred'] = final_preds
    df_res['Error_Abs'] = abs(df_res['QA_pred'] - df_res['QA_real'])
    
    df_res.to_csv(BASE_OUTPUT / "validacion_real_5fold.csv", index=False)
    pd.DataFrame(fold_importances).to_csv(BASE_OUTPUT / "ensemble_importances.csv", index=False)
    
    generate_plots(df_res, pd.DataFrame(fold_importances), BASE_OUTPUT, final_mae, opt_weights)
    print(f"FIN. Resultados guardados en: {BASE_OUTPUT}")

if __name__ == "__main__":
    main()
