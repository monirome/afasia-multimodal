#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================
[SVR]  Predicción WAB-AQ (0–100) con Support-Vector Regression
==============================================================
• Entrada : métricas por *chunk*  (df_aphbank_pos_metrics.csv, df_catalan_pos_metrics.csv)
• Salida  : predicción AQ agregada por paciente (CIP)

Pasos
-----
1. Etiquetado de idioma por fila  (en / es / ca)
2. Selección de ~30 features léxicas + acústicas (ver CANDIDATE_COLS).
3. Agregación (media) a nivel paciente.
4. Splits:
     – TRAIN / VAL  : pacientes en  (80/20, GroupShuffle)
     – TEST-INT     : pacientes es
     – TEST-EXT     : pacientes ca
5. Pipeline  Imputer→Scaler→SVR  +  GridSearchCV (MAE).
6. Logs, CSV y gráficas en   results/SVR_<timestamp>/

Autor: ChatGPT • 2025-08-08
"""

# -------------------------------------------------------------------
# 1 · Imports y configuración base
# -------------------------------------------------------------------
import os, sys, datetime, shutil, pathlib, warnings; warnings.filterwarnings("ignore")
import logging, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn.impute         import SimpleImputer
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.svm             import SVR
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics         import mean_absolute_error, r2_score
from scipy.stats             import spearmanr

# -------------------------------------------------------------------
# 2 · Rutas a CSV
# -------------------------------------------------------------------
DATA_BASE   = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025/data"
CSV_APH_ES  = os.path.join(DATA_BASE, "df_aphbank_pos_metrics.csv")   # en + es
CSV_CAT     = os.path.join(DATA_BASE, "df_catalan_pos_metrics.csv")   # ca

# -------------------------------------------------------------------
# 3 · Selección de features
# -------------------------------------------------------------------
CANDIDATE_COLS = [
    "words_per_min", "words_per_utt", "W", "OCW",
    "nouns", "verbs", "nouns_per_verb", "function_words",
    "diversidad_lexica",                               # ≈ TTR
    "F0semitoneFrom27.5Hz_sma3nz_amean", "loudness_sma3_amean",
    *[f"mfcc{i}_mean" for i in range(1, 14)]
]

# -------------------------------------------------------------------
# 4 · Helpers
# -------------------------------------------------------------------
def set_logger(run_dir: pathlib.Path) -> logging.Logger:
    """Crea logger que escribe en consola y en file."""
    log = logging.getLogger("SVR")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    # Consola (stdout)
    h1 = logging.StreamHandler(sys.stdout); h1.setFormatter(fmt); h1.setLevel(logging.INFO)
    # Fichero completo
    h2 = logging.FileHandler(run_dir / "console.log", mode="w"); h2.setFormatter(fmt); h2.setLevel(logging.INFO)
    log.addHandler(h1); log.addHandler(h2)
    return log

def lang_from_row(r) -> str:
    p = str(r.get("name_chunk_audio_path", ""))
    if "aphasiabank_en" in p or r.get("LLengWAB", 0) == 3: return "en"
    if "aphasiabank_es" in p or r.get("LLengWAB", 0) == 2: return "es"
    return "ca"

def agg_patient(grp, feat_cols):
    return grp[feat_cols].astype(float).mean(numeric_only=True)

def scatter_plot(y_true, y_pred, title, out_png):
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    rho,_ = spearmanr(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.scatter(y_true, y_pred, alpha=.7)
    mn,mx = float(min(y_true)), float(max(y_true))
    plt.plot([mn,mx],[mn,mx],"r--")
    plt.title(f"{title}\nMAE {mae:.2f} | R² {r2:.2f} | ρ {rho:.2f}")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    return mae, r2, rho

# -------------------------------------------------------------------
# 5 · Main
# -------------------------------------------------------------------
if __name__ == "__main__":

    # 5.1 Carpeta de resultados con nombre claro
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = pathlib.Path("results") / f"SVR_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 5.2 Logger
    log = set_logger(run_dir)
    log.info("-" * 60)
    log.info("Inicio pipeline Support-Vector Regression")

    # 5.3 Copia del script para trazabilidad
    shutil.copy2(__file__, run_dir / pathlib.Path(__file__).name)

    # 5.4 Carga de CSV
    dfs=[]
    if os.path.isfile(CSV_APH_ES):
        dfs.append(pd.read_csv(CSV_APH_ES, encoding="utf-8"))
    if os.path.isfile(CSV_CAT):
        dfs.append(pd.read_csv(CSV_CAT,  encoding="utf-8"))
    if not dfs:
        log.error("No se encontraron los CSV esperados.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["QA","CIP"])          # imprescindibles
    df["lang"] = df.apply(lang_from_row, axis=1)

    # 5.5 Features disponibles
    feat_cols = [c for c in CANDIDATE_COLS if c in df.columns]
    if miss := sorted(set(CANDIDATE_COLS) - set(feat_cols)):
        log.warning(f"Features ausentes y NO usadas: {miss}")

    # 5.5 bis Guardar las variables usadas en este experimento
    with open(run_dir / "features_used.txt", "w", encoding="utf-8") as f:
        f.write("Dimensiones usadas en este modelo:\n")
        f.write(f"Total: {len(feat_cols)}\n\n")
        for col in feat_cols:
            f.write(f"- {col}\n")

    # 5.6 Agregar por paciente
    df_pat = df.groupby("CIP").apply(agg_patient, feat_cols=feat_cols).reset_index()
    df_pat["QA"]   = df.groupby("CIP")["QA"].first().values
    df_pat["lang"] = df.groupby("CIP")["lang"].first().values

    n_en,n_es,n_ca = (df_pat.lang=="en").sum(), (df_pat.lang=="es").sum(), (df_pat.lang=="ca").sum()
    log.info(f"Pacientes  en:{n_en}  es:{n_es}  ca:{n_ca}")
    if n_en==0:
        log.error("Sin pacientes en inglés → no se puede entrenar.")
        sys.exit(1)

    # 5.7 Splits
    df_en, df_es, df_ca = (df_pat[df_pat.lang==l].reset_index(drop=True) for l in ("en","es","ca"))
    if len(df_en) < 3:
        df_tr, df_val = df_en, pd.DataFrame(columns=df_en.columns)
    else:
        gss = GroupShuffleSplit(test_size=0.20, random_state=42)
        tr_idx,val_idx = next(gss.split(df_en, groups=df_en.CIP))
        df_tr, df_val  = df_en.iloc[tr_idx], df_en.iloc[val_idx]

    df_tint, df_text = df_es, df_ca

    pd.concat([
        pd.DataFrame({"CIP":df_tr.CIP,   "split":"train"}),
        pd.DataFrame({"CIP":df_val.CIP,  "split":"val"}),
        pd.DataFrame({"CIP":df_tint.CIP, "split":"test_int"}),
        pd.DataFrame({"CIP":df_text.CIP, "split":"test_ext"}),
    ]).to_csv(run_dir/"cip_split.csv", index=False)

    log.info(f"Train {len(df_tr)} | Val {len(df_val)} | Test-int {len(df_tint)} | Test-ext {len(df_text)}")

    # 5.8 Matrices
    X_tr, y_tr = df_tr[feat_cols].values, df_tr["QA"].values
    X_val, y_val = df_val[feat_cols].values, df_val["QA"].values
    X_int, y_int = df_tint[feat_cols].values, df_tint["QA"].values
    X_ext, y_ext = df_text[feat_cols].values, df_text["QA"].values

    # 5.9 Pipeline + GridSearch
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svr",     SVR())
    ])
    param_grid = {
        "svr__C":[1,10,100],
        "svr__epsilon":[0.1,1],
        "svr__kernel":["rbf"],
        "svr__shrinking":[True]
    }
    log.info("GridSearchCV (SVR)…")
    gs = GridSearchCV(pipe, param_grid, cv=5,
                      scoring="neg_mean_absolute_error",
                      n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)
    log.info(f"Mejor parámetro: {gs.best_params_}")
    joblib.dump(gs.best_estimator_, run_dir/"best_svr.pkl")

    # 5.10 Evaluaciones
    def eval_split(name, X, y, cip):
        if len(y)==0: return
        pred = gs.best_estimator_.predict(X).clip(0,100)
        mae,r2,rho = scatter_plot(y, pred, f"{name.upper()}", run_dir/f"{name}_scatter.png")
        pd.DataFrame({"CIP":cip, "QA":y, "pred":pred, "err":pred-y}).to_csv(run_dir/f"{name}_preds.csv",index=False)
        log.info(f"{name.upper():8s}  MAE={mae:.2f}  R²={r2:.2f}  ρ={rho:.2f}")

    eval_split("train", X_tr, y_tr, df_tr.CIP)
    eval_split("val",   X_val, y_val, df_val.CIP)
    eval_split("int",   X_int, y_int, df_tint.CIP)
    eval_split("ext",   X_ext, y_ext, df_text.CIP)

    # 5.11 Guardar scaler (solo TRAIN)
    joblib.dump(StandardScaler().fit(X_tr), run_dir/"scaler.pkl")

    log.info("Proceso completado — archivos en %s", run_dir)
