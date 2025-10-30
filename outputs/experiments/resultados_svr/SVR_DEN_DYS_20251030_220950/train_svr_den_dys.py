#!/usr/bin/env python3
# train_svr_den_dys.py
# -*- coding: utf-8 -*-
"""
SVR - Prediccion WAB-AQ con features DEN+DYS
Lee datos desde /lustre/ (archivos con EN, ES, CA)
Incluye severity classification y metricas completas
"""

import os
import sys
import re
import datetime
import pathlib
import shutil
import warnings
warnings.filterwarnings("ignore")

# FIX: stdout sin buffer + limitar hilos BLAS
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from typing import List

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, pearsonr

# ======================== CONFIGURACION PATHS ========================

# Datos en /lustre/
DATA_LUSTRE = "/lustre/ific.uv.es/ml/upc150/upc1503/data"

CSV_APH_DYS = os.path.join(DATA_LUSTRE, "df_aphbank_pos_metrics_with_dys.csv")
CSV_CAT_DYS = os.path.join(DATA_LUSTRE, "df_catalan_pos_metrics_with_dys.csv")
CSV_DYS_PAT = os.path.join(DATA_LUSTRE, "dys_pauses_by_patient.csv")

# Resultados en /lhome/
PROJECT_BASE = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025/outputs/experiments"
RESULTS_BASE = os.path.join(PROJECT_BASE, "resultados_svr")

os.makedirs(RESULTS_BASE, exist_ok=True)

# ======================== SEVERIDAD ========================
SEVERITY_BINS = [0, 25, 50, 75, 100]
SEVERITY_LABELS = ['Very Severe', 'Severe', 'Moderate', 'Mild']

def qa_to_severity(qa_scores):
    return pd.cut(
        qa_scores, 
        bins=SEVERITY_BINS, 
        labels=SEVERITY_LABELS,
        include_lowest=True
    )

# ======================== LOGGER ========================
def set_logger(run_dir: pathlib.Path) -> logging.Logger:
    log = logging.getLogger("SVR_DEN_DYS")
    log.setLevel(logging.INFO)
    
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    
    fh = logging.FileHandler(run_dir / "console.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    
    for h in list(log.handlers):
        log.removeHandler(h)
    
    log.addHandler(ch)
    log.addHandler(fh)
    
    return log

# ======================== UTILIDADES ========================
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def to_int(pred):
    return np.rint(np.asarray(pred)).clip(0, 100).astype(int)

def wmean(values: pd.Series, weights=None) -> float:
    v = to_num(values)
    if weights is None:
        return float(np.nanmean(v))
    w = to_num(weights)
    mask = ~(v.isna() | w.isna())
    if not mask.any():
        return float(np.nanmean(v))
    return float(np.average(v[mask].values, weights=w[mask].values))

def safe_div(num: float, den: float) -> float:
    if den is None:
        return np.nan
    if not np.isfinite(den) or den <= 0:
        return np.nan
    return float(num / den)

def stats13_array(arr: np.ndarray, prefix: str) -> dict:
    s = pd.Series(arr).dropna()
    if len(s) == 0:
        keys = ["q1","q2","q3","iqr12","iqr23","iqr13","p01","p99",
                "range01_99","mean","std","skew","kurt"]
        return {f"{prefix}_{k}": np.nan for k in keys}
    
    q1 = s.quantile(0.25)
    q2 = s.quantile(0.50)
    q3 = s.quantile(0.75)
    p01 = s.quantile(0.01)
    p99 = s.quantile(0.99)
    
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

# ======================== DETECCION DE IDIOMA ========================
def detect_language(row):
    """Detecta idioma desde name_chunk_audio_path"""
    path = str(row.get("name_chunk_audio_path", ""))
    if "aphasiabank_en" in path:
        return "en"
    elif "aphasiabank_es" in path:
        return "es"
    else:
        return "ca"

# ======================== FEATURES POR PACIENTE ========================
def build_DEN_for_patient(grp: pd.DataFrame) -> pd.Series:
    """Construye features DEN agregadas por paciente"""
    w_words = to_num(grp.get("num_palabras", pd.Series(index=grp.index)))
    w_min = (to_num(grp["Duración"])/60.0) if "Duración" in grp.columns else None
    
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
    
    add_ratio("nouns", "den_nouns")
    add_ratio("verbs", "den_verbs")
    add_ratio("nouns_per_verb", "den_nouns_per_verb")
    
    if "nouns" in grp.columns and "verbs" in grp.columns:
        num = wmean(grp["nouns"], w_words)
        den = wmean(grp["nouns"] + grp["verbs"], w_words)
        out["den_noun_ratio"] = float(num/den) if (den and np.isfinite(den) and den>0) else np.nan
    
    add_ratio("light_verbs", "den_light_verbs")
    add_ratio("determiners", "den_determiners")
    add_ratio("demonstratives", "den_demonstratives")
    add_ratio("prepositions", "den_prepositions")
    add_ratio("adjectives", "den_adjectives")
    add_ratio("adverbs", "den_adverbs")
    add_ratio("pronoun_ratio", "den_pronoun_ratio")
    add_ratio("function_words", "den_function_words")
    
    return pd.Series(out, dtype="float64")

def build_DYS_for_patient(grp: pd.DataFrame) -> pd.Series:
    """Construye features DYS agregadas por paciente"""
    out = {}
    
    tot_sec = float(to_num(grp.get("Duración", pd.Series())).sum()) if "Duración" in grp.columns else np.nan
    tot_min = safe_div(tot_sec, 60.0)
    tot_words = float(to_num(grp.get("num_palabras", pd.Series())).sum()) if "num_palabras" in grp.columns else np.nan
    tot_phone = float(to_num(grp.get("num_phones", pd.Series())).sum()) if "num_phones" in grp.columns else np.nan
    
    n_fillers = float(to_num(grp.get("num_fillers", pd.Series())).sum()) if "num_fillers" in grp.columns else np.nan
    n_pauses = float(to_num(grp.get("num_pauses", pd.Series())).sum()) if "num_pauses" in grp.columns else np.nan
    n_long = float(to_num(grp.get("num_long_pauses", pd.Series())).sum()) if "num_long_pauses" in grp.columns else np.nan
    n_short = float(to_num(grp.get("num_short_pauses", pd.Series())).sum()) if "num_short_pauses" in grp.columns else np.nan
    
    out["dys_fillers_per_min"] = safe_div(n_fillers, tot_min)
    out["dys_fillers_per_word"] = safe_div(n_fillers, tot_words)
    out["dys_fillers_per_phone"] = safe_div(n_fillers, tot_phone)
    
    out["dys_pauses_per_min"] = safe_div(n_pauses, tot_min)
    out["dys_long_pauses_per_min"] = safe_div(n_long, tot_min)
    out["dys_short_pauses_per_min"] = safe_div(n_short, tot_min)
    
    out["dys_pauses_per_word"] = safe_div(n_pauses, tot_words)
    out["dys_long_pauses_per_word"] = safe_div(n_long, tot_words)
    out["dys_short_pauses_per_word"] = safe_div(n_short, tot_words)
    
    pause_vals = flatten_pause_durations(grp)
    out.update(stats13_array(np.array(pause_vals), "dys_pause_sec"))
    
    return pd.Series(out, dtype="float64")

# ======================== METRICAS ========================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    try:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
    except:
        pearson_r, pearson_p = np.nan, np.nan
    
    try:
        spearman_rho, spearman_p = spearmanr(y_true, y_pred)
    except:
        spearman_rho, spearman_p = np.nan, np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_rho': spearman_rho,
        'Spearman_p': spearman_p
    }

def compute_accuracy_metrics(y_true, y_pred_int):
    y_true = np.asarray(y_true)
    y_pred_int = np.asarray(y_pred_int)
    
    errors = np.abs(y_pred_int - y_true)
    
    return {
        'Acc@1': float(np.mean(errors <= 1)),
        'Acc@5': float(np.mean(errors <= 5)),
        'Acc@10': float(np.mean(errors <= 10)),
        'Exact': float(np.mean(errors == 0))
    }

def compute_severity_accuracy(y_true, y_pred):
    sev_true = qa_to_severity(y_true)
    sev_pred = qa_to_severity(y_pred)
    
    acc = accuracy_score(sev_true, sev_pred)
    cm = confusion_matrix(sev_true, sev_pred, labels=SEVERITY_LABELS)
    report = classification_report(
        sev_true, sev_pred, 
        labels=SEVERITY_LABELS,
        output_dict=True,
        zero_division=0
    )
    
    return {
        'severity_accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': report
    }

# ======================== VISUALIZACION ========================
def plot_scatter(y_true, y_pred, title, out_png, metrics=None):
    if metrics is None:
        metrics = compute_metrics(y_true, y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Identidad")
    
    plt.xlabel("QA Real (0-100)", fontsize=12, fontweight='bold')
    plt.ylabel("QA Predicho (0-100)", fontsize=12, fontweight='bold')
    
    text = ("MAE:      {:.2f}\n"
            "RMSE:     {:.2f}\n"
            "R2:       {:.3f}\n"
            "Pearson:  {:.3f}\n"
            "Spearman: {:.3f}").format(
                metrics['MAE'],
                metrics['RMSE'],
                metrics['R2'],
                metrics['Pearson_r'],
                metrics['Spearman_rho']
            )
    
    plt.text(0.05, 0.95, text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
             fontsize=10,
             family='monospace')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, title, out_png):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=SEVERITY_LABELS,
           yticklabels=SEVERITY_LABELS,
           ylabel='Severidad Real',
           xlabel='Severidad Predicha')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, "{}\n({:.1f}%)".format(cm[i, j], cm_norm[i, j] * 100),
                   ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black",
                   fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

# ======================== EVALUACION ========================
def evaluate_split(y_true, y_pred, split_name, run_dir, log, df_ids=None):
    log.info("\n{}".format("="*70))
    log.info("EVALUACION: {}".format(split_name))
    log.info("="*70)
    
    # Metricas de regresion
    metrics_cont = compute_metrics(y_true, y_pred)
    log.info("Metricas de regresion:")
    log.info("  MAE:        {:.3f}".format(metrics_cont['MAE']))
    log.info("  RMSE:       {:.3f}".format(metrics_cont['RMSE']))
    log.info("  R2:         {:.3f}".format(metrics_cont['R2']))
    log.info("  Pearson r:  {:.3f} (p={:.4f})".format(
        metrics_cont['Pearson_r'], metrics_cont['Pearson_p']))
    log.info("  Spearman:   {:.3f} (p={:.4f})".format(
        metrics_cont['Spearman_rho'], metrics_cont['Spearman_p']))
    
    # Metricas de accuracy por distancia
    y_pred_int = to_int(y_pred)
    metrics_int = compute_accuracy_metrics(y_true, y_pred_int)
    log.info("\nMetricas de accuracy por distancia:")
    log.info("  Acc@1:  {:.2f}%".format(100*metrics_int['Acc@1']))
    log.info("  Acc@5:  {:.2f}%".format(100*metrics_int['Acc@5']))
    log.info("  Acc@10: {:.2f}%".format(100*metrics_int['Acc@10']))
    log.info("  Exact:  {:.2f}%".format(100*metrics_int['Exact']))
    
    # Metricas de clasificacion por severidad
    sev_metrics = compute_severity_accuracy(y_true, y_pred_int)
    log.info("\nMetricas de clasificacion por severidad:")
    log.info("  Accuracy: {:.2f}%".format(100*sev_metrics['severity_accuracy']))
    
    report = sev_metrics['classification_report']
    log.info("\n  Accuracy por clase:")
    for label in SEVERITY_LABELS:
        if label in report:
            log.info("    {}: Precision={:.2f}%, Recall={:.2f}%, F1={:.2f}%".format(
                label,
                100*report[label]['precision'],
                100*report[label]['recall'],
                100*report[label]['f1-score']
            ))
    
    # Guardar metricas
    all_metrics = {**metrics_cont, **metrics_int}
    all_metrics['severity_accuracy'] = sev_metrics['severity_accuracy']
    
    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(run_dir / "{}_metrics.csv".format(split_name), index=False)
    
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(run_dir / "{}_severity_report.csv".format(split_name))
    
    cm_df = pd.DataFrame(
        sev_metrics['confusion_matrix'],
        index=SEVERITY_LABELS,
        columns=SEVERITY_LABELS
    )
    cm_df.to_csv(run_dir / "{}_confusion_matrix.csv".format(split_name))
    
    # Guardar predicciones
    preds_df = pd.DataFrame({
        'QA_real': y_true,
        'QA_pred_cont': y_pred,
        'QA_pred_int': y_pred_int,
        'severity_real': qa_to_severity(y_true),
        'severity_pred': qa_to_severity(y_pred_int),
        'error_cont': y_pred - y_true,
        'error_int': y_pred_int - y_true,
        'abs_error_int': np.abs(y_pred_int - y_true)
    })
    
    if df_ids is not None:
        for col in df_ids.columns:
            preds_df[col] = df_ids[col].values
    
    preds_df.to_csv(run_dir / "{}_predictions.csv".format(split_name), index=False)
    
    # Generar graficos
    log.info("\nGenerando graficos...")
    
    plot_scatter(
        y_true, y_pred,
        "{} - Scatter Plot".format(split_name),
        run_dir / "{}_scatter.png".format(split_name),
        metrics_cont
    )
    
    plot_confusion_matrix(
        sev_metrics['confusion_matrix'],
        "{} - Matriz de Confusion (Severidad)".format(split_name),
        run_dir / "{}_confusion_matrix.png".format(split_name)
    )
    
    log.info("Archivos guardados: {}_*.csv/png".format(split_name))
    
    return metrics_cont, metrics_int, sev_metrics

# ======================== MAIN ========================
def main():
    # Setup
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path(RESULTS_BASE) / "SVR_DEN_DYS_{}".format(ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log = set_logger(run_dir)
    log.info("="*70)
    log.info("INICIO: SVR con datos desde /lustre/")
    log.info("="*70)
    log.info("Directorio de resultados: {}".format(run_dir))
    log.info("Timestamp: {}".format(ts))
    
    try:
        shutil.copy2(__file__, run_dir / pathlib.Path(__file__).name)
    except Exception as e:
        log.warning("No se pudo copiar script: {}".format(e))
    
    # ==================== CARGAR DATOS ====================
    log.info("\n" + "="*70)
    log.info("CARGANDO DATOS DESDE /lustre/")
    log.info("="*70)
    
    # Verificar archivos
    for path in [CSV_APH_DYS, CSV_CAT_DYS]:
        if not os.path.exists(path):
            log.error("No existe: {}".format(path))
            sys.exit(1)
    
    # Cargar CSVs
    log.info("Cargando AphasiaBank (EN+ES)...")
    df_aph = pd.read_csv(CSV_APH_DYS)
    log.info("  Chunks: {}".format(len(df_aph)))
    
    log.info("Cargando Catalan (CA)...")
    df_cat = pd.read_csv(CSV_CAT_DYS)
    log.info("  Chunks: {}".format(len(df_cat)))
    
    # Concatenar
    df = pd.concat([df_aph, df_cat], ignore_index=True)
    log.info("Total chunks: {}".format(len(df)))
    
    # Filtrar con QA y CIP
    df = df.dropna(subset=["QA", "CIP"])
    log.info("Chunks con QA y CIP: {}".format(len(df)))
    
    # Detectar idioma
    df['lang'] = df.apply(detect_language, axis=1)
    
    log.info("\nDistribucion de chunks por idioma:")
    log.info("{}".format(df['lang'].value_counts()))
    
    # ==================== AGREGACION POR PACIENTE ====================
    log.info("\n" + "="*70)
    log.info("AGREGANDO FEATURES POR PACIENTE")
    log.info("="*70)
    
    log.info("Construyendo features DEN...")
    den_pat = df.groupby("CIP").apply(build_DEN_for_patient)
    
    log.info("Construyendo features DYS...")
    dys_pat = df.groupby("CIP").apply(build_DYS_for_patient)
    
    # Combinar
    df_pat = pd.concat([den_pat, dys_pat], axis=1).reset_index()
    df_pat["QA"] = df.groupby("CIP")["QA"].first().values
    df_pat["lang"] = df.groupby("CIP")["lang"].first().values
    
    log.info("Total pacientes: {}".format(len(df_pat)))
    log.info("\nPacientes por idioma:")
    for lang in ['en', 'es', 'ca']:
        n = len(df_pat[df_pat['lang'] == lang])
        log.info("  {}: {}".format(lang.upper(), n))
    
    # ==================== PREPARAR FEATURES ====================
    log.info("\n" + "="*70)
    log.info("PREPARANDO FEATURES")
    log.info("="*70)
    
    den_cols = [c for c in df_pat.columns if c.startswith('den_')]
    dys_cols = [c for c in df_pat.columns if c.startswith('dys_')]
    feat_cols = den_cols + dys_cols
    
    # Filtrar solo numericas
    feat_cols = df_pat[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    log.info("Features DEN: {}".format(len([c for c in feat_cols if c.startswith('den_')])))
    log.info("Features DYS: {}".format(len([c for c in feat_cols if c.startswith('dys_')])))
    log.info("TOTAL: {}".format(len(feat_cols)))
    
    if len(feat_cols) == 0:
        log.error("No hay features numericas")
        sys.exit(1)
    
    # Guardar lista
    with open(run_dir / "features_used.txt", "w", encoding="utf-8") as f:
        f.write("FEATURES UTILIZADAS\n")
        f.write("{}\n".format("="*70))
        f.write("Total: {}\n\n".format(len(feat_cols)))
        
        f.write("DEN features ({}):\n".format(
            len([c for c in feat_cols if c.startswith('den_')])))
        for c in sorted([c for c in feat_cols if c.startswith('den_')]):
            f.write("  - {}\n".format(c))
        
        f.write("\nDYS features ({}):\n".format(
            len([c for c in feat_cols if c.startswith('dys_')])))
        for c in sorted([c for c in feat_cols if c.startswith('dys_')]):
            f.write("  - {}\n".format(c))
    
    # ==================== SPLITS POR IDIOMA ====================
    log.info("\n" + "="*70)
    log.info("PREPARANDO SPLITS")
    log.info("="*70)
    
    df_en = df_pat[df_pat['lang'] == 'en'].reset_index(drop=True)
    df_es = df_pat[df_pat['lang'] == 'es'].reset_index(drop=True)
    df_ca = df_pat[df_pat['lang'] == 'ca'].reset_index(drop=True)
    
    n_en, n_es, n_ca = len(df_en), len(df_es), len(df_ca)
    
    log.info("Distribucion por idioma:")
    log.info("  EN (train):               {} pacientes".format(n_en))
    log.info("  ES (internal validation): {} pacientes".format(n_es))
    log.info("  CA (external test):       {} pacientes".format(n_ca))
    
    if n_en < 4:
        log.error("Se requieren >=4 pacientes EN para GroupKFold (n_en={})".format(n_en))
        sys.exit(1)
    
    # Preparar datos EN
    X_en = df_en[feat_cols].values
    y_en = df_en['QA'].values
    g_en = df_en['CIP'].values
    
    log.info("\nDatos de entrenamiento (EN):")
    log.info("  X shape: {}".format(X_en.shape))
    log.info("  y shape: {}".format(y_en.shape))
    log.info("  Grupos unicos: {}".format(len(np.unique(g_en))))
    log.info("  QA range: [{:.1f}, {:.1f}]".format(y_en.min(), y_en.max()))
    log.info("  QA mean +/- std: {:.2f} +/- {:.2f}".format(y_en.mean(), y_en.std()))
    
    # ==================== MODELO ====================
    log.info("\n" + "="*70)
    log.info("CONFIGURACION Y ENTRENAMIENTO DEL MODELO")
    log.info("="*70)
    
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr", SVR())
    ])
    
    log.info("Pipeline:")
    log.info("  1. SimpleImputer (strategy='median')")
    log.info("  2. StandardScaler")
    log.info("  3. SVR")
    
    param_grid = {
        "svr__C": [1, 10, 100],
        "svr__epsilon": [0.1, 1],
        "svr__kernel": ["rbf", "linear"],
        "svr__shrinking": [True, False]
    }
    
    log.info("\nHiperparametros a explorar:")
    total_combinations = 1
    for param, values in param_grid.items():
        log.info("  {}: {}".format(param, values))
        total_combinations *= len(values)
    log.info("  Total combinaciones: {}".format(total_combinations))
    
    gkf = GroupKFold(n_splits=4)
    log.info("\nCross-validation: GroupKFold (n_splits=4)")
    log.info("  Objetivo: speaker-independent evaluation")
    log.info("  Metrica: negative MAE")
    
    log.info("\nIniciando GridSearchCV...")
    log.info("  (esto puede tomar varios minutos)")
    
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        verbose=2,
        return_train_score=True
    )
    
    gs.fit(X_en, y_en, groups=g_en)
    
    best_model = gs.best_estimator_
    
    log.info("\n{}".format("="*70))
    log.info("GridSearchCV COMPLETADO")
    log.info("="*70)
    log.info("Mejores hiperparametros:")
    for param, value in gs.best_params_.items():
        log.info("  {}: {}".format(param, value))
    log.info("Mejor MAE (CV): {:.3f}".format(-gs.best_score_))
    
    joblib.dump(best_model, run_dir / "model_best_cv.pkl")
    log.info("Modelo guardado: model_best_cv.pkl")
    
    cv_results = pd.DataFrame(gs.cv_results_)
    cv_results.to_csv(run_dir / "cv_results_full.csv", index=False)
    log.info("Resultados CV guardados: cv_results_full.csv")
    
    # ==================== PREDICCIONES CV ====================
    log.info("\n" + "="*70)
    log.info("PREDICCIONES CROSS-VALIDATION (OUT-OF-FOLD)")
    log.info("="*70)
    
    cv_preds = np.zeros_like(y_en, dtype=float)
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_en, y_en, groups=g_en), 1):
        log.info("  Fold {}/4: train={}, test={}".format(
            fold_idx, len(train_idx), len(test_idx)))
        
        model_fold = clone(best_model)
        model_fold.fit(X_en[train_idx], y_en[train_idx])
        cv_preds[test_idx] = model_fold.predict(X_en[test_idx])
    
    log.info("Predicciones out-of-fold completadas")
    
    evaluate_split(
        y_en, cv_preds, "CV_EN", run_dir, log,
        df_ids=df_en[['CIP']]
    )
    
    # ==================== CALIBRACION ====================
    log.info("\n" + "="*70)
    log.info("CALIBRACION POST-HOC (Isotonic Regression)")
    log.info("="*70)
    
    log.info("Entrenando calibrador en predicciones CV...")
    calibrator = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
    calibrator.fit(cv_preds, y_en)
    
    joblib.dump(calibrator, run_dir / "calibrator.pkl")
    log.info("Calibrador guardado: calibrator.pkl")
    
    cv_preds_cal = calibrator.predict(cv_preds)
    evaluate_split(
        y_en, cv_preds_cal, "CV_EN_CALIBRATED", run_dir, log,
        df_ids=df_en[['CIP']]
    )
    
    # ==================== MODELO FINAL ====================
    log.info("\n" + "="*70)
    log.info("ENTRENAMIENTO MODELO FINAL (TODO EN)")
    log.info("="*70)
    
    log.info("Reentrenando modelo en TODO el set EN...")
    best_model.fit(X_en, y_en)
    
    joblib.dump(best_model, run_dir / "model_final_all_EN.pkl")
    log.info("Modelo final guardado: model_final_all_EN.pkl")
    
    # ==================== EVALUACION EN_IN ====================
    log.info("\n" + "="*70)
    log.info("EVALUACION IN-SAMPLE (EN)")
    log.info("="*70)
    
    log.info("Predicciones in-sample (diagnostico)...")
    
    preds_en_in = best_model.predict(X_en)
    evaluate_split(
        y_en, preds_en_in, "EN_IN_RAW", run_dir, log,
        df_ids=df_en[['CIP']]
    )
    
    preds_en_in_cal = calibrator.predict(preds_en_in)
    evaluate_split(
        y_en, preds_en_in_cal, "EN_IN_CALIBRATED", run_dir, log,
        df_ids=df_en[['CIP']]
    )
    
    # ==================== EVALUACION ES ====================
    if n_es > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACION INTERNAL VALIDATION (ES)")
        log.info("="*70)
        
        X_es = df_es[feat_cols].values
        y_es = df_es['QA'].values
        
        log.info("Evaluando en {} pacientes ES...".format(n_es))
        
        preds_es = best_model.predict(X_es)
        evaluate_split(
            y_es, preds_es, "INT_ES_RAW", run_dir, log,
            df_ids=df_es[['CIP']]
        )
        
        preds_es_cal = calibrator.predict(preds_es)
        evaluate_split(
            y_es, preds_es_cal, "INT_ES_CALIBRATED", run_dir, log,
            df_ids=df_es[['CIP']]
        )
    else:
        log.warning("No hay pacientes ES para evaluacion")
    
    # ==================== EVALUACION CA ====================
    if n_ca > 0:
        log.info("\n" + "="*70)
        log.info("EVALUACION EXTERNAL TEST (CA)")
        log.info("="*70)
        
        X_ca = df_ca[feat_cols].values
        y_ca = df_ca['QA'].values
        
        log.info("Evaluando en {} pacientes CA...".format(n_ca))
        
        preds_ca = best_model.predict(X_ca)
        evaluate_split(
            y_ca, preds_ca, "EXT_CA_RAW", run_dir, log,
            df_ids=df_ca[['CIP']]
        )
        
        preds_ca_cal = calibrator.predict(preds_ca)
        evaluate_split(
            y_ca, preds_ca_cal, "EXT_CA_CALIBRATED", run_dir, log,
            df_ids=df_ca[['CIP']]
        )
    else:
        log.warning("No hay pacientes CA para evaluacion")
    
    # ==================== RESUMEN FINAL ====================
    log.info("\n" + "="*70)
    log.info("RESUMEN FINAL")
    log.info("="*70)
    
    log.info("\nDirectorio de resultados: {}".format(run_dir))
    
    log.info("\nPacientes procesados:")
    log.info("  EN (train):               {}".format(n_en))
    log.info("  ES (internal validation): {}".format(n_es))
    log.info("  CA (external test):       {}".format(n_ca))
    log.info("  TOTAL:                    {}".format(n_en + n_es + n_ca))
    
    log.info("\nFeatures utilizadas:")
    log.info("  DEN: {}".format(len([c for c in feat_cols if c.startswith('den_')])))
    log.info("  DYS: {}".format(len([c for c in feat_cols if c.startswith('dys_')])))
    log.info("  TOTAL: {}".format(len(feat_cols)))
    
    try:
        cv_metrics = pd.read_csv(run_dir / "CV_EN_CALIBRATED_metrics.csv").iloc[0]
        log.info("\nMetricas CV (calibrado):")
        log.info("  MAE:                {:.3f}".format(cv_metrics['MAE']))
        log.info("  R2:                 {:.3f}".format(cv_metrics['R2']))
        log.info("  Spearman:           {:.3f}".format(cv_metrics['Spearman_rho']))
        log.info("  Acc@5:              {:.1f}%".format(100*cv_metrics['Acc@5']))
        log.info("  Severity Accuracy:  {:.1f}%".format(100*cv_metrics['severity_accuracy']))
    except:
        pass
    
    log.info("\n" + "="*70)
    log.info("PROCESO COMPLETADO EXITOSAMENTE")
    log.info("="*70)
    log.info("Timestamp: {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log.info("Resultados en: {}".format(run_dir))
    log.info("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUMPIDO] Proceso cancelado por el usuario")
        sys.exit(1)
    except Exception as e:
        print("\n\n[ERROR FATAL] {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)