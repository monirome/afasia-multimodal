#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga Common Voice (español y catalán) desde Mozilla Data Collective (MDC)
y crea:
  1) Audios WAV en:
       /lustre/.../data/audios_completos/{Spanish,Catalan}/Control/
  2) Transcripciones .CHA en:
       /lustre/.../data/transcripciones/{Spanish,Catalan}/Control/
  3) Metadata CSV en:
       /lustre/.../afasia_cat/datos/01_metadata/

Instalación recomendada en el servidor:
  cd /lustre/ific.uv.es/ml/upc150/upc1503/afasia_cat/codigos_julio2025/
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install datacollective pandas numpy soundfile tqdm
"""

import os
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from datacollective import DataCollective


# ---------------------------------------------------------------------
# CREDENCIALES Y RUTAS
# ---------------------------------------------------------------------

# PON AQUÍ TU API KEY DE MOZILLA DATA COLLECTIVE ENTRE COMILLAS
MDC_API_KEY = "61e8ae89d3f7c10bc157c3bdc8991e72d9f2fa9b536ed050b52f85e39d11b87d"

# No hace falta cambiar esto normalmente
MDC_API_URL = "https://datacollective.mozillafoundation.org/api"

# Carpeta donde se descargarán los datasets de MDC en el servidor
MDC_DOWNLOAD_PATH = "/lustre/ific.uv.es/ml/upc150/upc1503/mdc_datasets"

# Exportamos las variables para que datacollective las use
os.environ["MDC_API_KEY"] = MDC_API_KEY
os.environ["MDC_API_URL"] = MDC_API_URL
os.environ["MDC_DOWNLOAD_PATH"] = MDC_DOWNLOAD_PATH

# Rutas internas del proyecto (tal y como lo tienes ahora)
BASE_DIR = Path("/lustre/ific.uv.es/ml/upc150/upc1503")

METADATA_DIR = BASE_DIR / "afasia_cat/datos/01_metadata"
AUDIO_BASE = BASE_DIR / "data/audios_completos"
TRANSCRIPT_BASE = BASE_DIR / "data/transcripciones"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_BASE.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_BASE.mkdir(parents=True, exist_ok=True)

# Subcarpeta de grupo (Control / PWA, etc.)
GROUP_SUBFOLDER = "Control"

# Dataset IDs de Common Voice Scripted Speech 23.0 en MDC
DATASETS = {
    "es": {
        "dataset_id": "cmflnuzw51ddgmwjkxpm9z1lw",   # Spanish
        "csv": "commonvoice23_es_control.csv",
        "audio_dir": "Spanish",
        "transcript_dir": "Spanish",
    },
    "ca": {
        "dataset_id": "cmflnuzw4kexmk1fr32acgnve",   # Catalan
        "csv": "commonvoice23_ca_control.csv",
        "audio_dir": "Catalan",
        "transcript_dir": "Catalan",
    },
}

# Parámetros de muestreo
MAX_SAMPLES_PER_LANG = 500
MIN_DURATION = 2.0   # segundos
MAX_DURATION = 30.0  # segundos
MIN_UPVOTES = 1      # votos mínimos


# ---------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------

def ensure_cv_corpus_extracted(download_root: Path, language_code: str) -> Path:
    """
    Devuelve el directorio del corpus para un idioma, tipo:
      download_root/cv-corpus-23.0-YYYY-MM-DD/es

    Estrategia:
      1) Buscar cv-corpus-23.0-*/<lang>/ ya existente.
      2) Si no existe, buscar un tar mcv-scripted-<lang>-v23.0.tar.gz
         en download_root, extraerlo allí, y volver a buscar.
    """
    # 1) Buscar cv-corpus-23.0-*/lang
    for p in download_root.iterdir():
        if p.is_dir() and p.name.startswith("cv-corpus-23.0"):
            lang_dir = p / language_code
            if lang_dir.is_dir():
                return lang_dir

    # 2) No encontrado; intentar extraer desde tar
    tar_pattern = f"mcv-scripted-{language_code}-v23.0.tar.gz"
    tar_candidates = list(download_root.glob(tar_pattern))
    if not tar_candidates:
        raise FileNotFoundError(
            f"No se encontró directorio cv-corpus-23.0-*/{language_code} "
            f"ni fichero {tar_pattern} en {download_root}."
        )

    tar_path = tar_candidates[0]
    print(f"\nNo se encontró cv-corpus-23.0-*/{language_code}, pero sí:")
    print(f"  {tar_path}")
    print("Extrayendo el contenido (puede tardar bastante)...")

    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(download_root)

    # Volver a buscar después de extraer
    for p in download_root.iterdir():
        if p.is_dir() and p.name.startswith("cv-corpus-23.0"):
            lang_dir = p / language_code
            if lang_dir.is_dir():
                print(f"Extracción completada. Usando {lang_dir}")
                return lang_dir

    raise FileNotFoundError(
        f"Tras extraer {tar_path}, no se encontró cv-corpus-23.0-*/{language_code}."
    )


def create_cha_file(participant_id, text, language, gender, age, output_path):
    """
    Crea archivo .CHA compatible con AphasiaBank.
    language: 'es' o 'ca'
    """
    age_map = {
        "teens": 18,
        "twenties": 25,
        "thirties": 35,
        "fourties": 45,
        "forties": 45,
        "fifties": 55,
        "sixties": 65,
        "seventies": 75,
        "eighties": 85,
        "nineties": 95,
    }

    gender_norm = ""
    if isinstance(gender, str):
        g = gender.lower()
        if "male" in g:
            gender_norm = "male"
        elif "female" in g:
            gender_norm = "female"

    age_numeric = ""
    if isinstance(age, str):
        age_numeric = age_map.get(age.lower(), "")
    elif isinstance(age, (int, float)):
        age_numeric = int(age)

    lang_code = "spa" if language == "es" else "cat"

    cha_content = (
        "@UTF8\n"
        "@Begin\n"
        f"@Languages:\t{lang_code}\n"
        "@Participants:\tPAR Participant\n"
        f"@ID:\t{lang_code}|CommonVoice23|PAR||{gender_norm}|{age_numeric}||control|||\n"
        f"@Media:\t{participant_id}, audio\n"
        f"*PAR:\t{text}\n"
        "@End\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cha_content)


def load_validated_tsv(cv_root: Path) -> pd.DataFrame:
    """
    Carga el TSV principal del dataset Common Voice.
    Intenta 'validated.tsv' y si no existiera, usa 'train.tsv'.
    """
    validated = cv_root / "validated.tsv"
    train = cv_root / "train.tsv"

    if validated.exists():
        tsv_path = validated
    elif train.exists():
        tsv_path = train
    else:
        raise FileNotFoundError(
            f"No se encontraron ni validated.tsv ni train.tsv en {cv_root}"
        )

    print(f"  Usando TSV: {tsv_path.name}")
    df = pd.read_csv(tsv_path, sep="\t")
    return df


def download_language_from_mdc(language_code: str,
                               dataset_conf: dict,
                               client: DataCollective,
                               download_root: Path) -> pd.DataFrame:
    """
    Descarga desde MDC el dataset Common Voice 23.0 para un idioma,
    filtra y crea WAV + CHA + CSV de metadata.
    """
    print("\n" + "=" * 80)
    print(f"PROCESANDO COMMON VOICE 23.0 - {language_code.upper()} (MDC)")
    print("=" * 80)

    dataset_id = dataset_conf["dataset_id"]

    # Directorios de salida locales: Language/Control
    audio_dir = AUDIO_BASE / dataset_conf["audio_dir"] / GROUP_SUBFOLDER
    transcript_dir = TRANSCRIPT_BASE / dataset_conf["transcript_dir"] / GROUP_SUBFOLDER
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    print("\nDirectorios de salida:")
    print(f"  Audios WAV      -> {audio_dir}")
    print(f"  Archivos .CHA   -> {transcript_dir}")
    print(f"  Metadata CSV    -> {METADATA_DIR / dataset_conf['csv']}")

    # 1) Descargar dataset desde MDC (si ya está, no vuelve a bajarlo)
    print("\nDescargando dataset desde Mozilla Data Collective (si es necesario)...")
    client.get_dataset(dataset_id)
    print("Descarga OK o ya existente.")

    # 2) Localizar raíz del corpus para el idioma (cv-corpus-23.0-*/lang)
    cv_root = ensure_cv_corpus_extracted(download_root, language_code)
    print(f"\nRaíz del dataset en disco: {cv_root}")

    # 3) Cargar TSV (validated o train)
    df = load_validated_tsv(cv_root)
    print(f"  Filas en TSV: {len(df)}")

    # Columna de texto
    if "text" in df.columns:
        text_col = "text"
    elif "sentence" in df.columns:
        text_col = "sentence"
    else:
        raise KeyError("No se encontró columna 'text' ni 'sentence' en el TSV.")

    # Columna de acento
    accent_col = None
    if "accent" in df.columns:
        accent_col = "accent"
    elif "accents" in df.columns:
        accent_col = "accents"

    # Asegurar columnas de votos
    if "up_votes" in df.columns:
        df["up_votes"] = pd.to_numeric(df["up_votes"], errors="coerce").fillna(0).astype(int)
    else:
        df["up_votes"] = 0

    if "down_votes" in df.columns:
        df["down_votes"] = pd.to_numeric(df["down_votes"], errors="coerce").fillna(0).astype(int)
    else:
        df["down_votes"] = 0

    df_votes = df[
        (df["up_votes"] >= MIN_UPVOTES) &
        (df["down_votes"] <= df["up_votes"])
    ].copy()

    print(f"  Filas tras filtro de votos: {len(df_votes)}")

    # 4) Iterar filas, cargar audio y filtrar por duracion
    metadata_list = []
    successful = 0
    skipped_duration = 0

    clips_dir = cv_root / "clips"
    pbar = tqdm(total=MAX_SAMPLES_PER_LANG, desc=language_code.upper())

    for idx, row in df_votes.iterrows():
        if successful >= MAX_SAMPLES_PER_LANG:
            break

        rel_path = row["path"]
        audio_path = clips_dir / rel_path

        if not audio_path.exists():
            continue

        try:
            audio_array, sr = sf.read(audio_path)
        except Exception:
            continue

        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        duration = len(audio_array) / float(sr)

        if duration < MIN_DURATION or duration > MAX_DURATION:
            skipped_duration += 1
            continue

        participant_id = f"CV23_{language_code.upper()}_{successful:05d}"

        out_wav_path = audio_dir / f"{participant_id}.wav"
        try:
            sf.write(out_wav_path, audio_array, sr)
        except Exception:
            continue

        text = str(row[text_col]) if not pd.isna(row[text_col]) else ""
        gender = row.get("gender", None)
        age = row.get("age", None)

        out_cha_path = transcript_dir / f"{participant_id}.cha"
        try:
            create_cha_file(participant_id, text, language_code, gender, age, out_cha_path)
        except Exception:
            if out_wav_path.exists():
                out_wav_path.unlink()
            continue

        accent_val = None
        if accent_col and accent_col in row and not pd.isna(row[accent_col]):
            accent_val = row[accent_col]

        metadata_entry = {
            "participant_id": participant_id,
            "audio_filename": str(out_wav_path),
            "cha_filename": str(out_cha_path),
            "text": text,
            "age": age if not pd.isna(age) else None,
            "gender": gender if not pd.isna(gender) else None,
            "accent": accent_val,
            "duration_sec": round(duration, 2),
            "language": language_code,
            "group": "Control",
            "QA": None,
            "aphasia_type": None,
            "severity": None,
            "group_original": "control",
            "dataset": "CommonVoice23_MDC",
            "cv_split_source": "validated_or_train",
            "up_votes": int(row["up_votes"]),
            "down_votes": int(row["down_votes"]),
            "client_id": row.get("client_id", None),
        }

        metadata_list.append(metadata_entry)
        successful += 1
        pbar.update(1)

    pbar.close()

    if not metadata_list:
        print("No se obtuvo ninguna muestra válida para", language_code)
        return None

    df_meta = pd.DataFrame(metadata_list)
    csv_path = METADATA_DIR / dataset_conf["csv"]
    df_meta.to_csv(csv_path, index=False)

    print("\nFiltrado final:")
    print(f"  Descargados y convertidos: {successful}")
    print(f"  Saltados por duración: {skipped_duration}")
    print(f"  Metadata CSV: {csv_path}")
    print(f"  Duración total: {df_meta['duration_sec'].sum() / 3600:.2f} horas")
    print(f"  Duración media: {df_meta['duration_sec'].mean():.2f} seg")

    return df_meta


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 80)
    print("DESCARGA COMMON VOICE 23.0 (ES Y CA) DESDE MOZILLA DATA COLLECTIVE")
    print("=" * 80)

    if MDC_API_KEY == "<AQUI_TU_API_KEY_MDC>":
        print("\nERROR: No has puesto tu API key. Edita el script y sustituye:")
        print('  MDC_API_KEY = "<AQUI_TU_API_KEY_MDC>"')
        print("por tu clave real de MDC.")
        return

    print("\nConfiguración:")
    print("  - Dataset: Common Voice Scripted Speech 23.0 (MDC)")
    print("  - Idiomas: español (es), catalán (ca)")
    print(f"  - Máx. por idioma: {MAX_SAMPLES_PER_LANG} muestras")
    print(f"  - Duración permitida: {MIN_DURATION}-{MAX_DURATION} segundos")
    print(f"  - Votos mínimos: {MIN_UPVOTES}")
    print(f"  - Subcarpeta de grupo: {GROUP_SUBFOLDER}")

    download_root = Path(MDC_DOWNLOAD_PATH)
    download_root.mkdir(parents=True, exist_ok=True)

    client = DataCollective()

    input("\nPresiona ENTER para comenzar la descarga y procesado...")

    results = {}

    for lang_code, conf in DATASETS.items():
        df_meta = download_language_from_mdc(lang_code, conf, client, download_root)
        results[lang_code] = df_meta

    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    total_samples = sum(len(df) for df in results.values() if df is not None)
    total_hours = sum(df["duration_sec"].sum() / 3600 for df in results.values() if df is not None)

    print(f"\nTotal muestras: {total_samples}")
    print(f"Total horas aproximadas: {total_hours:.2f} h")

    for lang_code, df_meta in results.items():
        if df_meta is None:
            continue
        conf = DATASETS[lang_code]
        print(f"\n  {lang_code.upper()}:")
        print(f"    Muestras: {len(df_meta)}")
        print(f"    Duración: {df_meta['duration_sec'].sum() / 3600:.2f} h")
        print(f"    Audios: {AUDIO_BASE / conf['audio_dir'] / GROUP_SUBFOLDER}")
        print(f"    Transcripciones: {TRANSCRIPT_BASE / conf['transcript_dir'] / GROUP_SUBFOLDER}")
        print(f"    Metadata: {METADATA_DIR / conf['csv']}")

    print("\n" + "=" * 80)
    print("PROXIMOS PASOS")
    print("=" * 80)
    print("\n1. Verificar archivos descargados:")
    print(f"   ls {AUDIO_BASE}/Spanish/{GROUP_SUBFOLDER}/*.wav | wc -l")
    print(f"   ls {AUDIO_BASE}/Catalan/{GROUP_SUBFOLDER}/*.wav | wc -l")
    print(f"   ls {TRANSCRIPT_BASE}/Spanish/{GROUP_SUBFOLDER}/*.cha | wc -l")
    print(f"   ls {TRANSCRIPT_BASE}/Catalan/{GROUP_SUBFOLDER}/*.cha | wc -l")
    print("\n2. Ejecutar tus scripts de alineamiento y extracción de features como siempre.")


if __name__ == "__main__":
    main()
