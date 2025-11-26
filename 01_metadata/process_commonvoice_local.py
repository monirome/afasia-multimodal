#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

BASE = Path("/lustre/ific.uv.es/ml/upc150/upc1503")

CV_CA_ROOT = BASE / "mdc_datasets/cv-corpus-23.0-2025-09-05/ca"

TSV_PATH = CV_CA_ROOT / "validated.tsv"
CLIPS_DIR = CV_CA_ROOT / "clips"

OUT_AUDIO = BASE / "data/audios_completos/Catalan/Control"
OUT_CHA = BASE / "data/transcripciones/Catalan/Control"
OUT_AUDIO.mkdir(parents=True, exist_ok=True)
OUT_CHA.mkdir(parents=True, exist_ok=True)

OUT_CSV = BASE / "afasia_cat/datos/01_metadata/commonvoice23_ca_control.csv"

MAX_SAMPLES = 500
MIN_DUR = 2.0
MAX_DUR = 30.0

def create_cha(pid, text, gender, age, out_path):
    lang_code = "cat"
    gender_norm = ""
    if isinstance(gender, str):
        g = gender.lower()
        if "male" in g:
            gender_norm = "male"
        elif "female" in g:
            gender_norm = "female"

    age_norm = ""
    if isinstance(age, str):
        amap = {
            "teens": 18, "twenties": 25, "thirties": 35,
            "fourties": 45, "forties": 45, "fifties": 55,
            "sixties": 65, "seventies": 75, "eighties": 85,
            "nineties": 95
        }
        age_norm = amap.get(age.lower(), "")
    elif isinstance(age, (int, float)):
        age_norm = int(age)

    content = (
        "@UTF8\n"
        "@Begin\n"
        f"@Languages:\t{lang_code}\n"
        "@Participants:\tPAR Participant\n"
        f"@ID:\t{lang_code}|CommonVoice23|PAR||{gender_norm}|{age_norm}||control|||\n"
        f"@Media:\t{pid}, audio\n"
        f"*PAR:\t{text}\n"
        "@End\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    print("\nCargando TSV catalán...")
    df = pd.read_csv(TSV_PATH, sep="\t")
    print(f"Total filas TSV: {len(df)}")

    # Asegurar columnas de votos
    df["up_votes"] = pd.to_numeric(df.get("up_votes", 0), errors="coerce").fillna(0).astype(int)
    df["down_votes"] = pd.to_numeric(df.get("down_votes", 0), errors="coerce").fillna(0).astype(int)

    df_valid = df[
        (df["up_votes"] >= 1) &
        (df["down_votes"] <= df["up_votes"])
    ].copy()

    print(f"Filas tras voto mínimo: {len(df_valid)}")

    results = []
    count = 0

    print("Procesando audios catalanes...")

    for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid)):

        if count >= MAX_SAMPLES:
            break

        rel = row["path"]
        mp3_path = CLIPS_DIR / rel

        if not mp3_path.exists():
            continue

        # Intentar leer el audio (ignoramos errores)
        try:
            audio, sr = sf.read(mp3_path)
        except Exception:
            continue

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        dur = len(audio) / float(sr)
        if dur < MIN_DUR or dur > MAX_DUR:
            continue

        pid = f"CV23_CA_{count:05d}"
        wav_path = OUT_AUDIO / f"{pid}.wav"

        # Intentar guardar WAV
        try:
            sf.write(wav_path, audio, sr)
        except Exception:
            continue

        # Texto
        text = row["sentence"] if "sentence" in row else (row["text"] if "text" in row else "")

        # CHA
        cha_path = OUT_CHA / f"{pid}.cha"
        try:
            create_cha(pid, text, row.get("gender"), row.get("age"), cha_path)
        except Exception:
            if wav_path.exists():
                wav_path.unlink()
            continue

        # Añadir metadata
        results.append({
            "participant_id": pid,
            "audio_filename": str(wav_path),
            "cha_filename": str(cha_path),
            "text": text,
            "age": row.get("age"),
            "gender": row.get("gender"),
            "accent": row.get("accent") if "accent" in row else None,
            "duration_sec": round(dur, 2),
            "language": "ca",
            "group": "Control",
            "QA": None,
            "aphasia_type": None,
            "severity": None,
            "dataset": "CommonVoice23_MDC"
        })

        count += 1

    # Guardar CSV final
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)

    print("\n======================================")
    print("CATALÁN COMPLETADO")
    print("======================================")
    print(f"Muestras finales: {count}")
    print(f"CSV guardado en: {OUT_CSV}")
    print(f"WAV guardados en: {OUT_AUDIO}")
    print(f"CHA guardados en: {OUT_CHA}")


if __name__ == "__main__":
    main()
