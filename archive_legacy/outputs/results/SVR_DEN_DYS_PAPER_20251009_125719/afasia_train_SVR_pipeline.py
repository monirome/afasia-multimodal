#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================
[SVR–DEN+DYS · PAPER-LIKE]  Predicción WAB-AQ (0–100)
==============================================================
... (docstring intacto)
"""
import os, sys, re, datetime, pathlib, shutil, warnings; warnings.filterwarnings("ignore")
# FIX: stdout sin buffer + limitar hilos BLAS para evitar bloqueos en HTCondor
os.environ.setdefault("PYTHONUNBUFFERED","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

import logging, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")  # FIX: backend sin X
import matplotlib.pyplot as plt
from typing import Optional, List

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
import json

# ----------------------------- RUTAS ---------------------------------
DATA_BASE   = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025/data"
CSV_APH_ES      = os.path.join(DATA_BASE, "df_aphbank_pos_metrics.csv")
CSV_APH_ES_DYS  = os.path.join(DATA_BASE, "df_aphbank_pos_metrics_with_dys.csv")
CSV_CAT         = os.path.join(DATA_BASE, "df_catalan_pos_metrics.csv")
CSV_CAT_DYS     = os.path.join(DATA_BASE, "df_catalan_pos_metrics_with_dys.csv")
DYS_PAT_PATH    = os.path.join(DATA_BASE, "dys_pauses_by_patient.csv")

# ----------------------------- LOGGER --------------------------------
def set_logger(run_dir: pathlib.Path) -> logging.Logger:
    log = logging.getLogger("SVR_DEN_DYS_PAPER")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    fh = logging.FileHandler(run_dir/"console.log", mode="w"); fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    for h in list(log.handlers): log.removeHandler(h)
    log.addHandler(ch); log.addHandler(fh)
    return log

# -------------------------- UTILIDADES BASE --------------------------
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

def stats13_array(arr: np.ndarray, prefix: str) -> dict:
    s = pd.Series(arr).dropna()
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

def stats13(series: pd.Series, prefix: str) -> dict:
    return stats13_array(to_num(series).dropna().values, prefix)

def lang_from_row(r) -> str:
    p = str(r.get("name_chunk_audio_path", ""))
    if "aphasiabank_en" in p or r.get("LLengWAB", 0) == 3: return "en"
    if "aphasiabank_es" in p or r.get("LLengWAB", 0) == 2: return "es"
    return "ca"

def safe_div(num: float, den: float) -> float:
    if den is None: return np.nan
    if not np.isfinite(den) or den <= 0: return np.nan
    return float(num / den)

def flatten_pause_durations(grp: pd.DataFrame) -> List[float]:
    vals: List[float] = []
    if "pause_durations" in grp.columns:
        for x in grp["pause_durations"]:
            try:
                if isinstance(x, str) and x.strip().startswith("["):
                    arr = json.loads(x)
                    vals.extend([float(v) for v in arr if v is not None])
            except Exception:
                continue
    if len(vals) == 0 and "seconds_per_pause" in grp.columns:
        vals = [float(v) for v in to_num(grp["seconds_per_pause"]).dropna().values]
    return vals

# --------------------- UTILIDADES PARA EMPAREJAR DYS-PAC -------------
def root_from_stem(stem: str) -> str:
    if not isinstance(stem, str): return ""
    return re.sub(r'(_\d+(?:\.\d+)?_\d+(?:\.\d+)?)$', '', stem)

def basename_noext_root(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        return ""
    stem = os.path.splitext(os.path.basename(path))[0]
    root = root_from_stem(stem)
    return root

def norm_key(s: str) -> str:
    if not isinstance(s, str): return ""
    return re.sub(r'[\s\-_/]+', '', s.strip().lower())

def normalize_dys_patient_columns(dysp: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "duration_sec":              "dys_patient_duration_sec",
        "duration_sec_patient_audio":"dys_patient_duration_sec",
        "pauses_per_min":            "dys_patient_pauses_per_min",
        "long_pauses_per_min":       "dys_patient_long_pauses_per_min",
        "short_pauses_per_min":      "dys_patient_short_pauses_per_min",
        "pauses_per_word":           "dys_patient_pauses_per_word",
        "long_pauses_per_word":      "dys_patient_long_pauses_per_word",
        "short_pauses_per_word":     "dys_patient_short_pauses_per_word",
        "pause_total_sec":           "dys_patient_pause_total_sec",
        "pause_mean_sec":            "dys_patient_pause_mean_sec",
        "dys_pause_total_sec":       "dys_patient_pause_total_sec",
        "dys_pause_mean_sec":        "dys_patient_pause_mean_sec",
        "num_pauses":                "dys_patient_num_pauses",
        "num_long_pauses":           "dys_patient_num_long_pauses",
        "num_short_pauses":          "dys_patient_num_short_pauses",
        "dys_pause_durations":       "dys_patient_pause_durations",
        "pause_durations":           "dys_patient_pause_durations",
    }
    out = dysp.copy()
    rename_dict = {c: colmap[c] for c in dysp.columns if c in colmap}
    out = out.rename(columns=rename_dict)
    # FIX: si el CSV ya trae nombres tipo 'dys_*' (no 'dys_patient_*'), los paso a 'dys_patient_*'
    add_prefix = {c: c.replace("dys_", "dys_patient_", 1)
                  for c in out.columns if c.startswith("dys_") and not c.startswith("dys_patient_")}
    if add_prefix:
        out = out.rename(columns=add_prefix)
    for c in set(list(colmap.values())):
        if c not in out.columns:
            out[c] = np.nan
    return out

# ------------------------- FEATURES A NIVEL PACIENTE -----------------
def build_DEN_for_patient(grp: pd.DataFrame) -> pd.Series:
    w_words = to_num(grp.get("num_palabras", pd.Series(index=grp.index)))
    w_min   = (to_num(grp["Duración"])/60.0) if "Duración" in grp.columns else None
    out = {}
    if "words_per_min" in grp.columns:
        out["den_words_per_min"] = wmean(grp["words_per_min"], w_min)
    if "phones_per_min" in grp.columns:
        out["den_phones_per_min"] = wmean(grp["phones_per_min"], w_min)
    if "W" in grp.columns:
        out["den_W"] = wmean(grp["W"], w_words)
    if "OCW" in grp.columns:
        out["den_OCW"] = wmean(grp["OCW"], w_words)
    if "words_per_utt" in grp.columns:
        out.update(stats13(grp["words_per_utt"], "den_words_utt"))
    if "phones_per_utt" in grp.columns:
        out.update(stats13(grp["phones_per_utt"], "den_phones_utt"))
    def add_ratio(col, key):
        if col in grp.columns:
            out[key] = wmean(grp[col], w_words)
    add_ratio("nouns","den_nouns")
    add_ratio("verbs","den_verbs")
    add_ratio("nouns_per_verb","den_nouns_per_verb")
    if "nouns" in grp.columns and "verbs" in grp.columns:
        num = wmean(grp["nouns"], w_words)
        den = wmean(grp["nouns"] + grp["verbs"], w_words)
        out["den_noun_ratio"] = float(num/den) if (den and np.isfinite(den) and den>0) else np.nan
    add_ratio("light_verbs","den_light_verbs")
    add_ratio("determiners","den_determiners")
    add_ratio("demonstratives","den_demonstratives")
    add_ratio("prepositions","den_prepositions")
    add_ratio("adjectives","den_adjectives")
    add_ratio("adverbs","den_adverbs")
    add_ratio("pronoun_ratio","den_pronoun_ratio")
    add_ratio("function_words","den_function_words")
    return pd.Series(out, dtype="float64")

def build_DYS_for_patient(grp: pd.DataFrame) -> pd.Series:
    out = {}
    tot_sec   = float(to_num(grp.get("Duración", pd.Series())).sum())   if "Duración" in grp.columns else np.nan
    tot_min   = safe_div(tot_sec, 60.0)
    tot_words = float(to_num(grp.get("num_palabras", pd.Series())).sum()) if "num_palabras" in grp.columns else np.nan
    tot_phone = float(to_num(grp.get("num_phones", pd.Series())).sum())   if "num_phones" in grp.columns else np.nan
    n_fillers = float(to_num(grp.get("num_fillers", pd.Series())).sum())  if "num_fillers" in grp.columns else np.nan
    n_pauses  = float(to_num(grp.get("num_pauses", pd.Series())).sum())   if "num_pauses"  in grp.columns else np.nan
    n_long    = float(to_num(grp.get("num_long_pauses", pd.Series())).sum()) if "num_long_pauses" in grp.columns else np.nan
    n_short   = float(to_num(grp.get("num_short_pauses", pd.Series())).sum()) if "num_short_pauses" in grp.columns else np.nan
    out["dys_fillers_per_min"]   = safe_div(n_fillers, tot_min)
    out["dys_fillers_per_word"]  = safe_div(n_fillers, tot_words)
    out["dys_fillers_per_phone"] = safe_div(n_fillers, tot_phone)
    out["dys_pauses_per_min"]        = safe_div(n_pauses, tot_min)
    out["dys_long_pauses_per_min"]   = safe_div(n_long,   tot_min)
    out["dys_short_pauses_per_min"]  = safe_div(n_short,  tot_min)
    out["dys_pauses_per_word"]       = safe_div(n_pauses, tot_words)
    out["dys_long_pauses_per_word"]  = safe_div(n_long,   tot_words)
    out["dys_short_pauses_per_word"] = safe_div(n_short,  tot_words)
    pause_vals = flatten_pause_durations(grp)
    out.update(stats13_array(np.array(pause_vals), "dys_pause_sec"))
    return pd.Series(out, dtype="float64")

def to_int(pred):
    return np.rint(np.asarray(pred)).clip(0, 100).astype(int)

def scatter(y_true, y_pred, title, out_png):
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    rho,_ = spearmanr(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.scatter(y_true, y_pred, alpha=.7)
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("QA real (0–100)")
    plt.ylabel("QA predicho")
    plt.title(f"{title}\nMAE {mae:.2f} | R² {r2:.2f} | ρ {rho:.2f}")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    return mae, r2, rho

# ------------------------------- MAIN --------------------------------
if __name__ == "__main__":

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path("results") / f"SVR_DEN_DYS_PAPER_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = set_logger(run_dir)
    log.info("------------------------------------------------------------")
    log.info("Inicio pipeline SVR con features DEN+DYS (protocolo paper)")

    try:
        shutil.copy2(__file__, run_dir / pathlib.Path(__file__).name)
    except Exception:
        pass

    # Cargar datos
    paths = []
    aph_path = CSV_APH_ES_DYS if os.path.isfile(CSV_APH_ES_DYS) else CSV_APH_ES
    cat_path = CSV_CAT_DYS    if os.path.isfile(CSV_CAT_DYS)    else CSV_CAT
    if os.path.isfile(aph_path): paths.append(aph_path)
    if os.path.isfile(cat_path): paths.append(cat_path)
    if not paths:
        log.error("No se encontraron CSV de entrada."); sys.exit(1)

    dfs = [pd.read_csv(p, encoding="utf-8") for p in paths]
    df = pd.concat(dfs, ignore_index=True).dropna(subset=["QA","CIP"])
    df["lang"] = df.apply(lang_from_row, axis=1)

    if "words_per_utt" not in df.columns and "promedio_palabras_por_frase" in df.columns:
        df["words_per_utt"] = to_num(df["promedio_palabras_por_frase"])

    log.info("Agregando features DEN a nivel paciente …")
    den_pat = df.groupby("CIP").apply(build_DEN_for_patient)
    log.info("Agregando features DYS a nivel paciente …")
    dys_pat = df.groupby("CIP").apply(build_DYS_for_patient)

    df_pat = pd.concat([den_pat, dys_pat], axis=1).reset_index()
    df_pat["QA"]   = df.groupby("CIP")["QA"].first().values
    df_pat["lang"] = df.groupby("CIP")["lang"].first().values

    # ------------------- Inyección de DYS por paciente -----------------
    if os.path.isfile(DYS_PAT_PATH):
        log.info(f"Cargando DYS por paciente desde {DYS_PAT_PATH}")
        dysp = pd.read_csv(DYS_PAT_PATH)
        dysp = normalize_dys_patient_columns(dysp)  # FIX: renombra y asegura prefijo patient

        first_paths = df.groupby("CIP")["name_chunk_audio_path"].first().astype(str)
        df_pat["patient_key"] = df_pat["CIP"].map(lambda c: basename_noext_root(first_paths.get(c, "")))
        df_pat["patient_key"] = df_pat["patient_key"].map(norm_key)

        if "patient_id" not in dysp.columns:
            cand = [c for c in ["file", "patient", "id"] if c in dysp.columns]
            if cand:
                dysp = dysp.rename(columns={cand[0]: "patient_id"})
            else:
                dysp["patient_id"] = np.nan
        dysp["patient_key"] = dysp["patient_id"].astype(str).map(norm_key)

        cols_to_add = [c for c in dysp.columns if c.startswith("dys_")]
        before = len(df_pat)
        df_pat = df_pat.merge(dysp[["patient_key"] + cols_to_add], on="patient_key", how="left")
        matched = df_pat["dys_patient_duration_sec"].notna().sum() if "dys_patient_duration_sec" in df_pat.columns else 0
        log.info(f"Emparejados con DYS-paciente: {matched} de {before}")
    else:
        log.warning(f"No existe DYS_PAT_PATH: {DYS_PAT_PATH}. Se omiten DYS por paciente.")

    # ------------------- listado de features finales -------------------
    den_cols = [c for c in df_pat.columns if c.startswith("den_")]
    dys_cols = [c for c in df_pat.columns if c.startswith("dys_")]
    feat_cols = den_cols + dys_cols

    # FIX: excluir cualquier columna NO numérica (p.ej., 'dys_patient_pause_durations' texto JSON)
    num_feat_cols = df_pat[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    if len(num_feat_cols) != len(feat_cols):
        log.info("Filtrando features no numéricas: %d -> %d", len(feat_cols), len(num_feat_cols))
        feat_cols = num_feat_cols

    log.info("Total features DEN: %d | DYS: %d | TOTAL: %d", len([c for c in feat_cols if c.startswith("den_")]),
             len([c for c in feat_cols if c.startswith("dys_")]), len(feat_cols))
    if not feat_cols:
        log.error("No se han construido features. Revisa columnas de entrada.")
        sys.exit(1)
    with open(run_dir/"features_used.txt","w",encoding="utf-8") as f:
        f.write(f"Features usadas ({len(feat_cols)}):\n")
        for c in feat_cols: f.write(f"- {c}\n")

    # Subconjuntos por idioma
    df_en = df_pat[df_pat.lang=="en"].reset_index(drop=True)
    df_es = df_pat[df_pat.lang=="es"].reset_index(drop=True)
    df_ca = df_pat[df_pat.lang=="ca"].reset_index(drop=True)
    n_en, n_es, n_ca = len(df_en), len(df_es), len(df_ca)
    log.info("Pacientes  en:%d  es:%d  ca:%d", n_en, n_es, n_ca)
    if n_en < 4:
        log.error("Se requieren ≥4 pacientes EN para GroupKFold=4."); sys.exit(1)

    X_en, y_en, g_en = df_en[feat_cols].values, df_en["QA"].values, df_en["CIP"].values

    # Pipeline + Grid para SVR (como en el paper)
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
    # FIX: n_jobs=1 y verbose=2 para ver progreso y evitar cuelgues por multiproceso
    gs_en = GridSearchCV(pipe, grid, cv=gkf, scoring="neg_mean_absolute_error",
                         n_jobs=1, verbose=2)
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

    # Calibración post-hoc (Isotonic)
    iso = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
    iso.fit(cv_pred_en, y_en)
    joblib.dump(iso, run_dir/"calibrator_en.pkl")

    # EN_in (diagnóstico in-sample)
    p_en_in_raw = best_en.predict(X_en).clip(0, 100)
    mae_in_raw, r2_in_raw, rho_in_raw = scatter(
        y_en, p_en_in_raw, "EN_IN_RAW", run_dir/"en_in_raw_scatter.png"
    )
    p_en_in_raw_int = to_int(p_en_in_raw)
    acc1 = np.mean(np.abs(p_en_in_raw_int - y_en) <= 1)
    acc5 = np.mean(np.abs(p_en_in_raw_int - y_en) <= 5)
    acc_exact = np.mean(p_en_in_raw_int == y_en)
    log.info("EN_IN (RAW integers) → Acc@1 %.2f%% | Acc@5 %.2f%% | Exact %.2f%%",
             100*acc1, 100*acc5, 100*acc_exact)
    pd.DataFrame({
        "CIP": df_en.CIP, "QA": y_en,
        "pred": p_en_in_raw, "pred_int": p_en_in_raw_int,
        "abs_err": np.abs(p_en_in_raw - y_en),
        "abs_err_int": np.abs(p_en_in_raw_int - y_en)
    }).to_csv(run_dir/"en_in_raw_preds.csv", index=False)
    log.info("EN_IN (RAW) → MAE %.2f | R² %.2f | ρ %.2f", mae_in_raw, r2_in_raw, rho_in_raw)

    p_en_in_cal = iso.predict(p_en_in_raw).clip(0, 100)
    mae_in_cal, r2_in_cal, rho_in_cal = scatter(
        y_en, p_en_in_cal, "EN_IN_CAL", run_dir/"en_in_cal_scatter.png"
    )
    p_en_in_cal_int = to_int(p_en_in_cal)
    acc1 = np.mean(np.abs(p_en_in_cal_int - y_en) <= 1)
    acc5 = np.mean(np.abs(p_en_in_cal_int - y_en) <= 5)
    acc_exact = np.mean(p_en_in_cal_int == y_en)
    log.info("EN_IN (CAL integers) → Acc@1 %.2f%% | Acc@5 %.2f%% | Exact %.2f%%",
             100*acc1, 100*acc5, 100*acc_exact)
    pd.DataFrame({
        "CIP": df_en.CIP, "QA": y_en,
        "pred_cal": p_en_in_cal, "pred_cal_int": p_en_in_cal_int,
        "abs_err_cal": np.abs(p_en_in_cal - y_en),
        "abs_err_cal_int": np.abs(p_en_in_cal_int - y_en)
    }).to_csv(run_dir/"en_in_cal_preds.csv", index=False)
    log.info("EN_IN (CAL) → MAE %.2f | R² %.2f | ρ %.2f", mae_in_cal, r2_in_cal, rho_in_cal)

    # Evaluación en INT (ES) y EXT (CA)
    def eval_and_save(tag, df_split):
        if len(df_split)==0:
            log.info("%s vacío", tag.upper()); return
        X = df_split[feat_cols].values; y = df_split["QA"].values

        p_raw = best_en.predict(X).clip(0,100)
        m1,r1,ro1 = scatter(y, p_raw, f"{tag.upper()}_RAW", run_dir/f"{tag}_raw_scatter.png")
        p_raw_int = to_int(p_raw)
        acc1 = np.mean(np.abs(p_raw_int - y) <= 1)
        acc5 = np.mean(np.abs(p_raw_int - y) <= 5)
        acc_exact = np.mean(p_raw_int == y)
        log.info("%-10s (RAW) → MAE %.2f | R² %.2f | ρ %.2f", tag.upper(), m1, r1, ro1)
        log.info("%-10s (RAW integers) → Acc@1 %.2f%% | Acc@5 %.2f%% | Exact %.2f%%",
                 tag.upper(), 100*acc1, 100*acc5, 100*acc_exact)
        pd.DataFrame({
            "CIP": df_split.CIP, "QA": y, "pred": p_raw, "pred_int": p_raw_int,
            "abs_err": np.abs(p_raw - y), "abs_err_int": np.abs(p_raw_int - y)
        }).to_csv(run_dir/f"{tag}_raw_preds.csv", index=False)

        p_cal = iso.predict(p_raw).clip(0,100)
        m2,r2,ro2 = scatter(y, p_cal, f"{tag.upper()}_CAL", run_dir/f"{tag}_cal_scatter.png")
        p_cal_int = to_int(p_cal)
        acc1 = np.mean(np.abs(p_cal_int - y) <= 1)
        acc5 = np.mean(np.abs(p_cal_int - y) <= 5)
        acc_exact = np.mean(p_cal_int == y)
        log.info("%-10s (CAL) → MAE %.2f | R² %.2f | ρ %.2f", tag.upper(), m2, r2, ro2)
        log.info("%-10s (CAL integers) → Acc@1 %.2f%% | Acc@5 %.2f%% | Exact %.2f%%",
                 tag.upper(), 100*acc1, 100*acc5, 100*acc_exact)
        pd.DataFrame({
            "CIP": df_split.CIP, "QA": y, "pred_cal": p_cal, "pred_cal_int": p_cal_int,
            "abs_err_cal": np.abs(p_cal - y), "abs_err_cal_int": np.abs(p_cal_int - y)
        }).to_csv(run_dir/f"{tag}_cal_preds.csv", index=False)

    eval_and_save("int", df_es)
    eval_and_save("ext", df_ca)

    log.info("Proceso completado — resultados en %s", run_dir)
