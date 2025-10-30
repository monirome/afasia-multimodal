#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================
[SVR–DEN · PAPER-LIKE]  Predicción WAB-AQ (0–100) con features DEN
==============================================================
Replica del protocolo del paper, adaptado a tu dataset (sin grupo control):
  • Selección de modelo: GroupKFold=4 en EN (speaker-independent).
  • Reentreno final en TODO EN.
  • Evaluación en ES (INT) y CA (EXT).
  • Calibración post-hoc de salidas con Isotonic Regression:
      - El calibrador se entrena en EN usando predicciones out-of-fold
        (CV_EN) vs. objetivos reales (sin fuga de información).
      - Se aplica a las predicciones ES/CA (RAW → CAL).
  • Salidas acotadas a [0, 100] como en el paper.

Features DEN (Table 4, 1–18):
  1-4  : Words/min, Phones/min (si existe), W, OCW  → medias ponderadas
  5-6  : {Words/utt}, {Phones/utt} → 13 estadísticas (Tabla 3)
  7-18 : ratios POS (nouns, verbs, etc.) → medias ponderadas
Ponderaciones: #palabras (ratios) y duración en minutos (rates).

Entradas
--------
  • data/df_aphbank_pos_metrics.csv    (en + es)
  • data/df_catalan_pos_metrics.csv     (ca)

Salidas en results/SVR_DEN_PAPER_<timestamp>/
---------------------------------------------
  • console.log, script copiado, features_used.txt
  • cv_en_preds.csv, cv_en_scatter.png
  • en_in_raw_preds.csv, en_in_raw_scatter.png
  • en_in_cal_preds.csv, en_in_cal_scatter.png
  • int_raw_preds.csv,  int_raw_scatter.png
  • ext_raw_preds.csv,  ext_raw_scatter.png
  • int_cal_preds.csv,  int_cal_scatter.png
  • ext_cal_preds.csv,  ext_cal_scatter.png
  • best_svr_cvEN.pkl,  final_model_EN_all.pkl, calibrator_en.pkl
"""

import os, sys, datetime, pathlib, shutil, warnings; warnings.filterwarnings("ignore")
import logging, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Optional

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr

# ----------------------------- RUTAS ---------------------------------
DATA_BASE  = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025/data"
CSV_APH_ES = os.path.join(DATA_BASE, "df_aphbank_pos_metrics.csv")   # en + es
CSV_CAT    = os.path.join(DATA_BASE, "df_catalan_pos_metrics.csv")   # ca

# ----------------------------- LOGGER --------------------------------
def set_logger(run_dir: pathlib.Path) -> logging.Logger:
    log = logging.getLogger("SVR_DEN_PAPER")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    fh = logging.FileHandler(run_dir/"console.log", mode="w"); fh.setFormatter(fmt); fh.setLevel(logging.INFO)
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
    # 2 Phones/min (si existe)
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

    # 6 {Phones/utt} → 13 estadísticas (si existe)
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
    run_dir = pathlib.Path("results") / f"SVR_DEN_PAPER_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = set_logger(run_dir)
    log.info("------------------------------------------------------------")
    log.info("Inicio pipeline SVR con features DEN (protocolo paper)")

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

    # Subconjuntos por idioma
    df_en = df_den[df_den.lang=="en"].reset_index(drop=True)
    df_es = df_den[df_den.lang=="es"].reset_index(drop=True)
    df_ca = df_den[df_den.lang=="ca"].reset_index(drop=True)
    n_en, n_es, n_ca = len(df_en), len(df_es), len(df_ca)
    log.info("Pacientes  en:%d  es:%d  ca:%d", n_en, n_es, n_ca)
    if n_en < 4:
        log.error("Se requieren ≥4 pacientes EN para GroupKFold=4."); sys.exit(1)

    X_en, y_en, g_en = df_en[den_cols].values, df_en["QA"].values, df_en["CIP"].values

    # Pipeline + Grid para SVR (como en el paper: RBF/linear, C, epsilon, shrinking)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svr",     SVR())
    ])
    grid = {
        "svr__C":        [1, 10, 100],
        "svr__epsilon":  [0.1, 1],
        "svr__kernel":   ["rbf", "linear"],
        "svr__shrinking":[True, False],
    }

    # Selección de modelo: CV EN speaker-independent (GroupKFold=4)
    gkf = GroupKFold(n_splits=4)
    log.info("GridSearchCV (SVR) solo EN con GroupKFold=4 …")
    gs_en = GridSearchCV(pipe, grid, cv=gkf, scoring="neg_mean_absolute_error",
                         n_jobs=-1, verbose=0)
    gs_en.fit(X_en, y_en, groups=g_en)
    best_en = gs_en.best_estimator_
    joblib.dump(best_en, run_dir/"best_svr_cvEN.pkl")
    log.info("Mejor params CV-EN: %s", gs_en.best_params_)

    # Predicciones out-of-fold en EN (para gráfica y para calibración)
    cv_pred_en = np.zeros_like(y_en, dtype=float)
    for tr, te in gkf.split(X_en, y_en, groups=g_en):
        est = clone(best_en)
        est.fit(X_en[tr], y_en[tr])
        cv_pred_en[te] = est.predict(X_en[te])
    m,r,ro = scatter(y_en, cv_pred_en, "CV_EN", run_dir/"cv_en_scatter.png")
    pd.DataFrame({"CIP":df_en.CIP, "QA":y_en, "pred":cv_pred_en, "err":cv_pred_en-y_en})\
      .to_csv(run_dir/"cv_en_preds.csv", index=False)
    log.info("CV-EN     → MAE %.2f | R² %.2f | ρ %.2f", m, r, ro)

    # Reentreno en TODO EN (modelo final)
    best_en.fit(X_en, y_en)
    joblib.dump(best_en, run_dir/"final_model_EN_all.pkl")

    # -------------------- Calibración post-hoc (Isotonic) -------------------
    # Se ajusta sobre (pred_CV_EN → y_en), y se aplica en INT/EXT.
    iso = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
    iso.fit(cv_pred_en, y_en)
    joblib.dump(iso, run_dir/"calibrator_en.pkl")

    # -------------------- EN_in (diagnóstico in-sample) ---------------------
    # Predicción del modelo final entrenado con TODO EN sobre el propio EN.
    # Útil para analizar bias/varianza. NO es métrica de generalización.
    p_en_in_raw = best_en.predict(X_en).clip(0, 100)
    mae_in_raw, r2_in_raw, rho_in_raw = scatter(
        y_en, p_en_in_raw, "EN_IN_RAW", run_dir/"en_in_raw_scatter.png"
    )
    pd.DataFrame({
        "CIP": df_en.CIP, "QA": y_en, "pred": p_en_in_raw, "err": p_en_in_raw - y_en
    }).to_csv(run_dir/"en_in_raw_preds.csv", index=False)
    log.info("EN_IN (RAW) → MAE %.2f | R² %.2f | ρ %.2f", mae_in_raw, r2_in_raw, rho_in_raw)

    # Opcional: aplicar calibración isotónica también a EN_in (diagnóstico)
    p_en_in_cal = iso.predict(p_en_in_raw).clip(0, 100)
    mae_in_cal, r2_in_cal, rho_in_cal = scatter(
        y_en, p_en_in_cal, "EN_IN_CAL", run_dir/"en_in_cal_scatter.png"
    )
    pd.DataFrame({
        "CIP": df_en.CIP, "QA": y_en, "pred_cal": p_en_in_cal, "err_cal": p_en_in_cal - y_en
    }).to_csv(run_dir/"en_in_cal_preds.csv", index=False)
    log.info("EN_IN (CAL) → MAE %.2f | R² %.2f | ρ %.2f", mae_in_cal, r2_in_cal, rho_in_cal)

    # -------------------- Evaluación en INT (ES) y EXT (CA) -----------------
    def eval_and_save(tag, df_split):
        if len(df_split)==0:
            log.info("%s vacío", tag.upper()); return
        X = df_split[den_cols].values; y = df_split["QA"].values

        # RAW
        p_raw = best_en.predict(X).clip(0,100)
        m1,r1,ro1 = scatter(y, p_raw, f"{tag.upper()}_RAW", run_dir/f"{tag}_raw_scatter.png")
        pd.DataFrame({"CIP":df_split.CIP, "QA":y, "pred":p_raw, "err":p_raw-y})\
          .to_csv(run_dir/f"{tag}_raw_preds.csv", index=False)
        log.info("%-10s (RAW) → MAE %.2f | R² %.2f | ρ %.2f", tag.upper(), m1, r1, ro1)

        # CAL (isotónica)
        p_cal = iso.predict(p_raw).clip(0,100)
        m2,r2,ro2 = scatter(y, p_cal, f"{tag.upper()}_CAL", run_dir/f"{tag}_cal_scatter.png")
        pd.DataFrame({"CIP":df_split.CIP, "QA":y, "pred_cal":p_cal, "err_cal":p_cal-y})\
          .to_csv(run_dir/f"{tag}_cal_preds.csv", index=False)
        log.info("%-10s (CAL) → MAE %.2f | R² %.2f | ρ %.2f", tag.upper(), m2, r2, ro2)

    eval_and_save("int", df_es)
    eval_and_save("ext", df_ca)

    log.info("Proceso completado — resultados en %s", run_dir)
