#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================
[SVR–DEN]  Predicción WAB-AQ (0–100) con features del grupo DEN
==============================================================
Replica del protocolo del paper para selección de modelo:
  • CV EN (GroupKFold=4) a nivel PACIENTE (CIP) con SVR.
  • Reentreno en TODO EN y evaluación en ES (INT) y CA (EXT).
  • Extra: LOOCV dentro de CA para ver estabilidad con pocos pacientes.
  • Además: un split EN Train/Val (GroupShuffle 80/20) para comparar
    con tu estilo anterior (gráficas train/val con muchas “bolitas”).

Construcción de features DEN (Table 4 – Information Density 1–18):
  1) Ratios escalables: medias ponderadas por nº palabras o duración
  2) {Words/utt}  → 13 estadísticas (Tabla 3)
  3) {Phones/utt} → 13 estadísticas (si existen columnas)

Entradas
--------
  • data/df_aphbank_pos_metrics.csv    (en + es)
  • data/df_catalan_pos_metrics.csv     (ca)

Salidas en results/SVR_DEN_<timestamp>/
---------------------------------------
  • console.log, script copiado, features_used.txt
  • cv_en_preds.csv, cv_en_scatter.png
  • en_train_preds.csv, en_train_scatter.png
  • en_val_preds.csv,   en_val_scatter.png
  • int_preds.csv,      int_scatter.png
  • ext_preds.csv,      ext_scatter.png
  • ca_loocv_preds.csv, ca_loocv_scatter.png
  • best_svr_cvEN.pkl,  final_model_EN_all.pkl, scaler_EN_all.pkl
"""

import os, sys, datetime, pathlib, shutil, warnings; warnings.filterwarnings("ignore")
import logging, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Optional

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from scipy.stats import spearmanr

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
    # limpiar handlers previos si re-ejecutas en intérprete
    for h in list(log.handlers):
        log.removeHandler(h)
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
        keys = ["q1","q2","q3","iqr12","iqr23","iqr13","p01","p99","range01_99","mean","std","skew","kurt"]
        return {f"{prefix}_{k}": np.nan for k in keys}
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

def build_DEN_for_patient(grp: pd.DataFrame) -> pd.Series:
    # pesos
    w_words = to_num(grp.get("num_palabras", pd.Series(index=grp.index)))
    w_min   = (to_num(grp["Duración"])/60.0) if "Duración" in grp.columns else None

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
        if col in grp.columns:
            out[key] = wmean(grp[col], w_words)

    add_ratio("nouns",           "den_nouns")
    add_ratio("verbs",           "den_verbs")
    add_ratio("nouns_per_verb",  "den_nouns_per_verb")
    # 10 noun ratio = nouns / (nouns + verbs)
    if "nouns" in grp.columns and "verbs" in grp.columns:
        num = wmean(grp["nouns"], w_words)
        den = wmean(grp["nouns"] + grp["verbs"], w_words)
        out["den_noun_ratio"] = float(num/den) if (den and np.isfinite(den) and den>0) else np.nan
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

    # Derivadas por chunk si faltan
    if "words_per_utt" not in df.columns and "promedio_palabras_por_frase" in df.columns:
        df["words_per_utt"] = to_num(df["promedio_palabras_por_frase"])

    # Construcción de features DEN por paciente
    log.info("Agregando features DEN a nivel paciente …")
    df_den = df.groupby("CIP").apply(build_DEN_for_patient).reset_index()
    # Añadir QA y lang heredados del primer chunk del paciente
    df_den["QA"]   = df.groupby("CIP")["QA"].first().values
    df_den["lang"] = df.groupby("CIP")["lang"].first().values

    # Reportar cuántas columnas DEN se construyeron
    den_cols = [c for c in df_den.columns if c.startswith("den_")]
    log.info("Total features DEN construidas: %d", len(den_cols))
    if not den_cols:
        log.error("No se han construido features DEN. Revisa columnas de entrada.")
        sys.exit(1)
    with open(run_dir/"features_used.txt","w",encoding="utf-8") as f:
        f.write(f"Features DEN usadas ({len(den_cols)}):\n")
        for c in den_cols: f.write(f"- {c}\n")

    # Conteos por idioma
    n_en, n_es, n_ca = (df_den.lang=="en").sum(), (df_den.lang=="es").sum(), (df_den.lang=="ca").sum()
    log.info("Pacientes  en:%d  es:%d  ca:%d", n_en, n_es, n_ca)
    if n_en == 0:
        log.error("Sin pacientes en inglés → no se puede entrenar."); sys.exit(1)

    # ======================== SELECCIÓN DE MODELO (PAPER) ===================
    # CV solo EN
    df_en = df_den[df_den.lang=="en"].reset_index(drop=True)
    X_en, y_en, g_en = df_en[den_cols].values, df_en["QA"].values, df_en["CIP"].values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svr",     SVR())
    ])
    grid = {
        "svr__C":[1,10,100],
        "svr__epsilon":[0.1,1],
        "svr__kernel":["rbf","linear"],
        "svr__shrinking":[True]
    }

    gkf = GroupKFold(n_splits=4)
    log.info("GridSearchCV (SVR) solo EN con GroupKFold=4 …")
    gs_en = GridSearchCV(pipe, grid, cv=gkf, scoring="neg_mean_absolute_error",
                         n_jobs=-1, verbose=0)
    gs_en.fit(X_en, y_en, groups=g_en)
    best_en = gs_en.best_estimator_
    joblib.dump(best_en, run_dir/"best_svr_cvEN.pkl")
    log.info("Mejor params CV-EN: %s", gs_en.best_params_)

    # predicciones cross-val en EN (muchas bolitas)
    cv_pred_en = np.zeros_like(y_en, dtype=float)
    for tr, te in gkf.split(X_en, y_en, groups=g_en):
        est = clone(best_en)
        est.fit(X_en[tr], y_en[tr])
        cv_pred_en[te] = est.predict(X_en[te])
    m,r,ro = scatter(y_en, cv_pred_en, "CV_EN", run_dir/"cv_en_scatter.png")
    pd.DataFrame({"CIP":df_en.CIP, "QA":y_en, "pred":cv_pred_en, "err":cv_pred_en-y_en})\
      .to_csv(run_dir/"cv_en_preds.csv", index=False)
    log.info("CV-EN     → MAE %.2f | R² %.2f | ρ %.2f", m, r, ro)

    # ======================= EN TRAIN/VAL (COMPARABLE A ANTES) ==============
    if len(df_en) >= 3:
        gss = GroupShuffleSplit(test_size=0.20, random_state=42)
        tr_idx, val_idx = next(gss.split(X_en, y_en, groups=g_en))
        df_en_tr, df_en_val = df_en.iloc[tr_idx], df_en.iloc[val_idx]
        X_tr, y_tr = df_en_tr[den_cols].values, df_en_tr["QA"].values
        X_val, y_val = df_en_val[den_cols].values, df_en_val["QA"].values

        est_tv = clone(best_en).fit(X_tr, y_tr)
        # TRAIN
        pred_tr = est_tv.predict(X_tr).clip(0,100)
        m,r,ro = scatter(y_tr, pred_tr, "EN_TRAIN", run_dir/"en_train_scatter.png")
        pd.DataFrame({"CIP":df_en_tr.CIP, "QA":y_tr, "pred":pred_tr, "err":pred_tr-y_tr})\
          .to_csv(run_dir/"en_train_preds.csv", index=False)
        log.info("EN_TRAIN  → MAE %.2f | R² %.2f | ρ %.2f", m, r, ro)
        # VAL
        pred_val = est_tv.predict(X_val).clip(0,100)
        m,r,ro = scatter(y_val, pred_val, "EN_VAL", run_dir/"en_val_scatter.png")
        pd.DataFrame({"CIP":df_en_val.CIP, "QA":y_val, "pred":pred_val, "err":pred_val-y_val})\
          .to_csv(run_dir/"en_val_preds.csv", index=False)
        log.info("EN_VAL    → MAE %.2f | R² %.2f | ρ %.2f", m, r, ro)

    # ======================= TRANSFER: ENTRENAR EN TODO EN ==================
    best_en.fit(X_en, y_en)
    joblib.dump(best_en, run_dir/"final_model_EN_all.pkl")
    joblib.dump(best_en.named_steps["scaler"], run_dir/"scaler_EN_all.pkl")

    # INT y EXT
    df_es = df_den[df_den.lang=="es"].reset_index(drop=True)
    df_ca = df_den[df_den.lang=="ca"].reset_index(drop=True)

    def eval_split_fixed(name, df_split):
        if len(df_split)==0:
            log.info("%s vacío", name.upper()); return
        X = df_split[den_cols].values; y = df_split["QA"].values
        p = best_en.predict(X).clip(0,100)
        m,r,ro = scatter(y, p, name.upper(), run_dir/f"{name}_scatter.png")
        pd.DataFrame({"CIP":df_split.CIP, "QA":y, "pred":p, "err":p-y})\
          .to_csv(run_dir/f"{name}_preds.csv", index=False)
        log.info("%-8s → MAE %.2f | R² %.2f | ρ %.2f", name.upper(), m, r, ro)

    eval_split_fixed("int", df_es)
    eval_split_fixed("ext", df_ca)

    # ======================= CA-LOOCV (estabilidad interna) ================
    if len(df_ca) >= 2:
        X_ca, y_ca, g_ca = df_ca[den_cols].values, df_ca["QA"].values, df_ca["CIP"].values
        logo = LeaveOneGroupOut()
        loocv_pred = np.zeros_like(y_ca, dtype=float)
        for tr, te in logo.split(X_ca, y_ca, groups=g_ca):
            est = clone(best_en)
            est.fit(X_ca[tr], y_ca[tr])
            loocv_pred[te] = est.predict(X_ca[te])
        m,r,ro = scatter(y_ca, loocv_pred, "CA_LOOCV", run_dir/"ca_loocv_scatter.png")
        pd.DataFrame({"CIP":df_ca.CIP, "QA":y_ca, "pred":loocv_pred, "err":loocv_pred-y_ca})\
          .to_csv(run_dir/"ca_loocv_preds.csv", index=False)
        log.info("CA_LOOCV  → MAE %.2f | R² %.2f | ρ %.2f", m, r, ro)

    log.info("Proceso completado — resultados en %s", run_dir)
