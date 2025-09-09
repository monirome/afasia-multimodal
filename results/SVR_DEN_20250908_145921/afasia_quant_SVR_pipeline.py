#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================
[SVR–DEN]  Predicción WAB-AQ (0–100) con features del grupo DEN
==============================================================
Features del paper (Table 4 – Information Density 1–18) + Tabla 3
(13 estadísticas) aplicadas a {Words/utt} (y {Phones/utt} si existe).

Pipeline:
  1) Etiquetado de idioma por fila (en/es/ca)
  2) Construcción de features DEN a nivel PACIENTE (CIP):
       - Escalares (ratios) → media ponderada por #palabras o duración
       - {Words/utt} → 13 estadísticas (Tabla 3)
       - {Phones/utt} → 13 estadísticas (si hay columna)
  3) Splits
       - TRAIN/VAL: inglés (80/20, GroupShuffle a nivel CIP)
       - TEST-INT : castellano
       - TEST-EXT : catalán
  4) Modelado: Pipeline(Imputer→Scaler→SVR) + GridSearchCV (MAE)
  5) Resultados: scatter + preds.csv para TRAIN, VAL, INT, EXT

"""

import os, sys, datetime, pathlib, shutil, warnings; warnings.filterwarnings("ignore")
import logging, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from typing import Optional

# ----------------------------- RUTAS ---------------------------------
DATA_BASE  = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025/data"
CSV_APH_ES = os.path.join(DATA_BASE, "df_aphbank_pos_metrics.csv")   # en + es
CSV_CAT    = os.path.join(DATA_BASE, "df_catalan_pos_metrics.csv")   # ca

# ----------------------------- LOGGER --------------------------------
def set_logger(run_dir: pathlib.Path) -> logging.Logger:
    log = logging.getLogger("SVR_DEN")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    fh = logging.FileHandler(run_dir/"console.log", mode="w"); fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    if log.handlers: 
        for h in list(log.handlers): log.removeHandler(h)
    log.addHandler(ch); log.addHandler(fh)
    return log

# ---------------------- UTILIDADES / AGREGADORES ---------------------
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def wmean(values: pd.Series, weights: Optional[pd.Series] = None) -> float:
    v = to_num(values)
    if weights is None:
        return float(np.nanmean(v))
    w = to_num(weights)
    mask = ~(v.isna() | w.isna())
    if not mask.any(): 
        return float(np.nanmean(v))
    return float(np.average(v[mask].values, weights=w[mask].values))

def stats13(series: pd.Series, prefix: str) -> dict:
    s = to_num(series).dropna()
    if len(s) == 0:
        return {f"{prefix}_q1":np.nan, f"{prefix}_q2":np.nan, f"{prefix}_q3":np.nan,
                f"{prefix}_iqr12":np.nan, f"{prefix}_iqr23":np.nan, f"{prefix}_iqr13":np.nan,
                f"{prefix}_p01":np.nan, f"{prefix}_p99":np.nan, f"{prefix}_range01_99":np.nan,
                f"{prefix}_mean":np.nan, f"{prefix}_std":np.nan, f"{prefix}_skew":np.nan, f"{prefix}_kurt":np.nan}
    q1 = s.quantile(0.25); q2 = s.quantile(0.50); q3 = s.quantile(0.75)
    p01 = s.quantile(0.01); p99 = s.quantile(0.99)
    return {
        f"{prefix}_q1": float(q1),
        f"{prefix}_q2": float(q2),
        f"{prefix}_q3": float(q3),
        f"{prefix}_iqr12": float(q2 - q1),
        f"{prefix}_iqr23": float(q3 - q2),
        f"{prefix}_iqr13": float(q3 - q1),
        f"{prefix}_p01": float(p01),
        f"{prefix}_p99": float(p99),
        f"{prefix}_range01_99": float(p99 - p01),
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_std": float(s.std(ddof=0)),
        f"{prefix}_skew": float(s.skew()),
        f"{prefix}_kurt": float(s.kurt())
    }

def lang_from_row(r) -> str:
    p = str(r.get("name_chunk_audio_path", ""))
    if "aphasiabank_en" in p or r.get("LLengWAB", 0) == 3: return "en"
    if "aphasiabank_es" in p or r.get("LLengWAB", 0) == 2: return "es"
    return "ca"

def build_DEN_for_patient(grp: pd.DataFrame, log: logging.Logger) -> pd.Series:
    # pesos
    w_words = to_num(grp.get("num_palabras", pd.Series(index=grp.index)))
    w_min   = to_num(grp.get("Duración", pd.Series(index=grp.index))) / 60.0 if "Duración" in grp.columns else None

    out = {}

    # 1 Words/min
    if "words_per_min" in grp.columns:
        out["den_words_per_min"] = wmean(grp["words_per_min"], w_min)
    # 2 Phones/min  (si existe)
    if "phones_per_min" in grp.columns:
        out["den_phones_per_min"] = wmean(grp["phones_per_min"], w_min)

    # 3 W (Words / (Words + Interjections))
    if "W" in grp.columns:
        out["den_W"] = wmean(grp["W"], w_words)

    # 4 OCW (Open / Open+Closed)
    if "OCW" in grp.columns:
        out["den_OCW"] = wmean(grp["OCW"], w_words)

    # 5 {Words/utt}  → 13 estadísticas
    if "words_per_utt" in grp.columns:
        out.update(stats13(grp["words_per_utt"], "den_words_utt"))

    # 6 {Phones/utt}  → 13 estadísticas (si existe)
    if "phones_per_utt" in grp.columns:
        out.update(stats13(grp["phones_per_utt"], "den_phones_utt"))

    # Ratios POS 7–18 (ponderados por #palabras)
    def add_ratio(col, key):
        if col in grp.columns: out[key] = wmean(grp[col], w_words)

    add_ratio("nouns",           "den_nouns")
    add_ratio("verbs",           "den_verbs")
    add_ratio("nouns_per_verb",  "den_nouns_per_verb")
    # 10 noun ratio = nouns / (nouns + verbs)
    if "nouns" in grp.columns and "verbs" in grp.columns:
        num = wmean(grp["nouns"], w_words); den = wmean(grp["nouns"] + grp["verbs"], w_words)
        out["den_noun_ratio"] = float(num/den) if den and np.isfinite(den) and den>0 else np.nan
    add_ratio("light_verbs",     "den_light_verbs")
    add_ratio("determiners",     "den_determiners")
    add_ratio("demonstratives",  "den_demonstratives")
    add_ratio("prepositions",    "den_prepositions")
    add_ratio("adjectives",      "den_adjectives")
    add_ratio("adverbs",         "den_adverbs")
    add_ratio("pronoun_ratio",   "den_pronoun_ratio")
    add_ratio("function_words",  "den_function_words")

    return pd.Series(out, dtype="float64")

def scatter(y_true, y_pred, title, out_png):
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    rho,_ = spearmanr(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.scatter(y_true, y_pred, alpha=.7)
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.title(f"{title}\nMAE {mae:.2f} | R² {r2:.2f} | ρ {rho:.2f}")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    return mae, r2, rho

# ------------------------------- MAIN --------------------------------
if __name__ == "__main__":

    # Carpeta resultados + logger
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path("results") / f"SVR_DEN_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = set_logger(run_dir)
    log.info("------------------------------------------------------------")
    log.info("Inicio pipeline SVR con features DEN (paper)")

    # Copia del script (trazabilidad)
    try:
        shutil.copy2(__file__, run_dir / pathlib.Path(__file__).name)
    except Exception:
        pass

    # Cargar datos
    dfs=[]
    if os.path.isfile(CSV_APH_ES): dfs.append(pd.read_csv(CSV_APH_ES, encoding="utf-8"))
    if os.path.isfile(CSV_CAT):    dfs.append(pd.read_csv(CSV_CAT,    encoding="utf-8"))
    if not dfs:
        log.error("No se encontraron CSV de entrada."); sys.exit(1)

    df = pd.concat(dfs, ignore_index=True).dropna(subset=["QA","CIP"])
    df["lang"] = df.apply(lang_from_row, axis=1)

    # Añadir derivadas por chunk si faltan
    if "nouns" in df.columns and "verbs" in df.columns and "noun_ratio" not in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["noun_ratio"] = (to_num(df["nouns"]) / to_num(df["nouns"]).add(to_num(df["verbs"]), fill_value=np.nan))
    if "words_per_utt" not in df.columns and "promedio_palabras_por_frase" in df.columns:
        df["words_per_utt"] = to_num(df["promedio_palabras_por_frase"])

    # Construcción de features DEN por paciente
    log.info("Agregando features DEN a nivel paciente …")
    df_den = df.groupby("CIP").apply(build_DEN_for_patient, log=log).reset_index()
    # Añadir QA y lang heredados del primer chunk del paciente
    df_den["QA"]   = df.groupby("CIP")["QA"].first().values
    df_den["lang"] = df.groupby("CIP")["lang"].first().values

    # Reportar cuántas columnas DEN se construyeron
    den_cols = [c for c in df_den.columns if c.startswith("den_")]
    log.info("Total features DEN construidas: %d", len(den_cols))
    with open(run_dir/"features_used.txt","w",encoding="utf-8") as f:
        f.write(f"Features DEN usadas ({len(den_cols)}):\n")
        for c in den_cols: f.write(f"- {c}\n")

    # Conteos por idioma
    n_en, n_es, n_ca = (df_den.lang=="en").sum(), (df_den.lang=="es").sum(), (df_den.lang=="ca").sum()
    log.info("Pacientes  en:%d  es:%d  ca:%d", n_en, n_es, n_ca)
    if n_en == 0:
        log.error("Sin pacientes en inglés → no se puede entrenar."); sys.exit(1)

    # Splits
    df_en  = df_den[df_den.lang=="en"].reset_index(drop=True)
    df_es  = df_den[df_den.lang=="es"].reset_index(drop=True)
    df_ca  = df_den[df_den.lang=="ca"].reset_index(drop=True)

    if len(df_en) < 3:
        df_tr, df_val = df_en, pd.DataFrame(columns=df_en.columns)
    else:
        gss = GroupShuffleSplit(test_size=0.20, random_state=42)
        tr_idx, val_idx = next(gss.split(df_en, groups=df_en.CIP))
        df_tr, df_val = df_en.iloc[tr_idx], df_en.iloc[val_idx]

    df_int, df_ext = df_es, df_ca

    pd.concat([
        pd.DataFrame({"CIP":df_tr.CIP,  "split":"train"}),
        pd.DataFrame({"CIP":df_val.CIP, "split":"val"}),
        pd.DataFrame({"CIP":df_int.CIP, "split":"test_int"}),
        pd.DataFrame({"CIP":df_ext.CIP, "split":"test_ext"}),
    ]).to_csv(run_dir/"cip_split.csv", index=False)

    log.info("Train %d | Val %d | Test-int %d | Test-ext %d",
             len(df_tr), len(df_val), len(df_int), len(df_ext))

    # Matrices
    X_tr, y_tr = df_tr[den_cols].values, df_tr["QA"].values
    X_val, y_val = df_val[den_cols].values, df_val["QA"].values
    X_int, y_int = df_int[den_cols].values, df_int["QA"].values
    X_ext, y_ext = df_ext[den_cols].values, df_ext["QA"].values

    # Pipeline + GridSearch
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svr",     SVR())
    ])
    grid = {
        "svr__C":[1,10,100],
        "svr__epsilon":[0.1,1],
        "svr__kernel":["rbf"],
        "svr__shrinking":[True]
    }
    log.info("GridSearchCV (SVR)…")
    gs = GridSearchCV(pipe, grid, cv=5, scoring="neg_mean_absolute_error",
                      n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_tr)
    joblib.dump(gs.best_estimator_, run_dir/"best_svr.pkl")
    log.info("Mejor set de hiperparámetros: %s", gs.best_params_)

    # Evaluación y gráficas (cuatro splits)
    def evaluate(name, X, y, cip):
        if len(y)==0: return
        preds = gs.best_estimator_.predict(X).clip(0,100)
        mae, r2, rho = scatter(y, preds, name.upper(), run_dir/f"{name}_scatter.png")
        pd.DataFrame({"CIP":cip, "QA":y, "pred":preds, "err":preds-y})\
          .to_csv(run_dir/f"{name}_preds.csv", index=False)
        log.info("%-8s → MAE %.2f | R² %.2f | ρ %.2f", name.upper(), mae, r2, rho)

    evaluate("train", X_tr,  y_tr,  df_tr.CIP)
    evaluate("val",   X_val, y_val, df_val.CIP)
    evaluate("int",   X_int, y_int, df_int.CIP)
    evaluate("ext",   X_ext, y_ext, df_ext.CIP)

    # Scaler aparte (entrenado solo con TRAIN, por si quieres inspeccionarlo)
    joblib.dump(StandardScaler().fit(X_tr), run_dir/"scaler.pkl")

    log.info("Proceso completado — resultados en %s", run_dir)
