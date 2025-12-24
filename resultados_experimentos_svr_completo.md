### Resultados de Experimentos Completos (Actualizado a Dic 2025)

| ID | Modelo | Configuración | Features | POS_LM | N_Feat | CV_MAE (Raw) | CV_RMSE | CV_Pearson | **Calibrated MAE** | Calibrated Pearson | SVR_C / Params | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **TabPFN** | **KBEST_POSLM** | **kbest** | **backoff** | **40** | 11.06 | 15.16 | **0.777** | **10.27** | **0.803** | *N_ens=Auto* | **Ganador Absoluto.** Transformer Pre-entrenado. |
| **2** | **EBM** | **FULL_POSLM** | full | backoff | 108 | 12.02 | 15.96 | 0.747 | **11.21** | 0.779 | *Interactions=10* | **2º Mejor.** Modelo Explicable (Glassbox). |
| **3** | **CatBoost** | **FULL_POSLM** | full | backoff | 108 | 11.95 | 16.47 | 0.728 | **11.44** | 0.751 | *Depth=6* | Muy robusto, base del Ensemble. |
| 4 | XGBoost | KBEST_POSLM | kbest | backoff | 40 | 12.28 | 16.88 | 0.713 | 11.53 | 0.749 | *Eta=0.05* | Similar a CatBoost pero un poco por debajo. |
| 5 | LGBM | FULL_POSLM | full | backoff | 108 | 12.22 | 16.25 | 0.735 | 11.50 | 0.771 | *Leaves=31* | Buen rendimiento calibrado. |
| 6 | SVR | SFS_POSLM | sfs | backoff | 21 | 12.43 | 18.66 | 0.655 | 12.13 | 0.685 | C=100, eps=1 | El mejor SVR, pero superado por árboles y DL. |
| 7 | SVR | KBEST_NO-POSLM | kbest | none | 40 | 13.25 | 18.63 | 0.645 | 12.41 | 0.683 | C=10, eps=5 | Referencia sin datos lingüísticos (solo acústicos). |