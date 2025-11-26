import pandas as pd

# ===============================
# Datos completos de todos los experimentos
# ===============================
experiments = {
    'ID': ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2', 'PAPER'],
    'Config': [
        'SIMPLE_NO-POSLM',
        'FULL_NO-POSLM', 
        'SFS_NO-POSLM',
        'FULL_POSLM-KN',
        'SFS_POSLM-KN',
        'FULL_POSLM-BO',
        'SFS_POSLM-BO',
        'FULL_POSLM-ALL',
        'SFS_POSLM-ALL',
        'Le et al. 2018 (Auto)'
    ],
    'Features': ['simple', 'full', 'sfs', 'full', 'sfs', 'full', 'sfs', 'full', 'sfs', 'reference'],
    'POS_LM': ['none', 'none', 'none', 'kneser-ney', 'kneser-ney', 'backoff', 'backoff', 'all', 'all', 'reference'],
    'Total_Features': [29, 64, 12, 94, 28, 94, 28, 124, 38, None],
    'DEN': [18, 42, 8, 42, 11, 42, 11, 42, 11, None],
    'DYS': [10, 22, 4, 22, 6, 22, 6, 22, 6, None],
    'LEX': [1, 0, 0, 0, 0, 0, 0, 0, 0, None],
    'POSLM': [0, 0, 0, 30, 11, 30, 11, 60, 21, None],
    
    # Métricas Cross-Validation (sin calibrar)
    'CV_MAE': [13.540, 15.078, 12.863, 15.489, 12.992, 15.522, 13.000, 15.631, 12.761, None],
    'CV_RMSE': [18.562, 20.185, 17.767, 21.915, 18.203, 22.002, 18.209, 22.216, 17.933, None],
    'CV_R2': [0.390, 0.278, 0.441, 0.149, 0.413, 0.143, 0.413, 0.126, 0.430, None],
    'CV_Pearson': [0.659, 0.558, 0.686, 0.492, 0.678, 0.487, 0.678, 0.477, 0.688, None],
    'CV_Spearman': [0.660, 0.558, 0.668, 0.550, 0.673, 0.547, 0.673, 0.539, 0.676, None],
    
    # Métricas Calibradas
    'Calibrated_MAE': [12.859, 14.663, 12.098, 15.035, 11.892, 15.066, 11.918, 15.255, 11.873, 9.18],
    'Calibrated_RMSE': [17.081, 18.854, 16.099, 19.592, 16.131, 19.601, 16.144, 19.855, 16.118, None],
    'Calibrated_R2': [0.483, 0.370, 0.541, 0.320, 0.539, 0.319, 0.538, 0.302, 0.540, None],
    'Calibrated_Pearson': [0.695, 0.609, 0.735, 0.566, 0.734, 0.565, 0.734, 0.549, 0.735, 0.799],
    'Calibrated_Spearman': [0.680, 0.582, 0.686, 0.575, 0.696, 0.576, 0.696, 0.569, 0.701, None],
    
    # Accuracy metrics
    'Acc_10': [51.31, 42.24, 52.74, 38.19, 53.46, 39.14, 52.98, 40.10, 54.18, None],
    'Severity_Acc': [56.56, 49.64, 59.67, 49.40, 63.25, 50.12, 63.25, 51.07, 62.77, None],
    
    # Comparación con paper
    'Delta_MAE_vs_Paper': [3.68, 5.48, 2.92, 5.85, 2.71, 5.89, 2.74, 6.08, 2.69, 0.0],
    'Delta_Pearson_vs_Paper': [-0.104, -0.190, -0.064, -0.233, -0.065, -0.234, -0.065, -0.250, -0.064, 0.0],
    
    # Hiperparámetros SVR
    'SVR_Kernel': ['rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', 'rbf', None],
    'SVR_C': [10, 10, 10, 10, 10, 10, 10, 10, 10, None],
    'SVR_Epsilon': [5, 5, 0.01, 5, 0.01, 5, 0.01, 1, 0.01, None],
    'SVR_Gamma': ['auto', 'auto', 'scale', 'auto', 'scale', 'auto', 'scale', 'auto', 'scale', None],
    
    # Evaluación español (4 pacientes TCU)
    'ES_MAE_raw': [23.88, 18.10, 19.05, 22.21, 19.51, 22.00, 19.30, 20.41, 19.38, None],
    'ES_MAE_calibrated': [18.46, 15.59, 14.93, 17.90, 14.88, 17.74, 14.69, 16.47, 14.61, None],
    'ES_R2_calibrated': [-3.77, -2.03, -1.57, -3.54, -1.54, -3.47, -1.45, -2.71, -1.42, None],
    'ES_Pearson_calibrated': [0.135, 0.082, 0.207, 0.136, 0.252, 0.138, 0.267, 0.165, 0.275, None],
    
    # Información adicional
    'Dataset': ['AphasiaBank', 'AphasiaBank', 'AphasiaBank', 'AphasiaBank', 'AphasiaBank', 
                'AphasiaBank', 'AphasiaBank', 'AphasiaBank', 'AphasiaBank', 'AphasiaBank'],
    'N_samples_train': [419, 419, 419, 419, 419, 419, 419, 419, 419, 401],
    'N_samples_eval_ES': [4, 4, 4, 4, 4, 4, 4, 4, 4, None],
    'CV_Strategy': ['GroupKFold-4', 'GroupKFold-4', 'GroupKFold-4', 'GroupKFold-4', 'GroupKFold-4',
                    'GroupKFold-4', 'GroupKFold-4', 'GroupKFold-4', 'GroupKFold-4', 'Unknown'],
    'Excluded_Features': ['lex_ttr', 'lex_ttr', 'lex_ttr', 'lex_ttr', 'lex_ttr', 
                          'lex_ttr', 'lex_ttr', 'lex_ttr', 'lex_ttr', None],
    'Notes': [
        'Baseline con 29 features simples',
        'Full features sin selección',
        'SFS sin POS-LM - mejor sin POS-LM',
        'Full con KN - overfitting',
        'SFS + KN - segundo mejor overall',
        'Full con Backoff - overfitting',
        'SFS + Backoff - equivalente a B2',
        'Full con KN+BO - más overfitting',
        'SFS + ALL - mejor overall',
        'Paper de referencia con ASR automático'
    ]
}

# ===============================
# Crear DataFrame
# ===============================
df = pd.DataFrame(experiments)

# ===============================
# Patch para fila "PAPER"
# ===============================
paper_mask = df["ID"] == "PAPER"

# Features del paper: 43 medidas (Tabla 4)
df.loc[paper_mask, "Total_Features"] = 43
df.loc[paper_mask, "DEN"] = 18
df.loc[paper_mask, "DYS"] = 10
df.loc[paper_mask, "LEX"] = 6
df.loc[paper_mask, "POSLM"] = 2

# Métricas Combined + Auto (Tabla 9)
df.loc[paper_mask, "CV_MAE"] = 9.18
df.loc[paper_mask, "CV_Pearson"] = 0.799

# Métricas Combined + Calibrated
df.loc[paper_mask, "Calibrated_MAE"] = 9.24
df.loc[paper_mask, "Calibrated_Pearson"] = 0.786

# Información de dataset y CV
df.loc[paper_mask, "Dataset"] = "AphasiaBank (EN only)"
df.loc[paper_mask, "N_samples_train"] = 348
df.loc[paper_mask, "CV_Strategy"] = "Speaker-independent 4-fold CV"

# Notas actualizadas
df.loc[paper_mask, "Notes"] = (
    "Le et al. (2018), Combined protocol, Auto vs Calibrated. "
    "SVR con features transcript-based y ASR automático."
)

# ===============================
# Guardar CSV
# ===============================
output_path = 'resultados_experimentos_svr_completo.csv'
df.to_csv(output_path, index=False)

print(f"  DataFrame creado con {len(df)} experimentos y {len(df.columns)} columnas")
print(f"\nColumnas incluidas:")
for col in df.columns:
    print(f"  - {col}")

print(f"\n  CSV guardado en: {output_path}")
