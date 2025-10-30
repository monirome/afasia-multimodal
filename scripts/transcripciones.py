#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrae los intervalos (start/end) de habla del paciente (PAR) desde archivos .cha
en inglés y español, y genera un CSV unificado.

Estructura esperada:
data/transcripciones/
 ├── English/*.cha
 └── Spanish/*.cha

Salida:
data/df_chunks_patient.csv
"""

import os
import pandas as pd
import pylangacq as pla

# ----------------------------- RUTAS ---------------------------------
BASE_DIR = "/lhome/ext/upc150/upc1503/afasia_cat/codigos_julio2025/data"
CHA_DIRS = [
    os.path.join(BASE_DIR, "transcripciones/English"),
    os.path.join(BASE_DIR, "transcripciones/Spanish"),
]
OUT_CSV = os.path.join(BASE_DIR, "df_chunks_patient.csv")

# -------------------------- FUNCIÓN PRINCIPAL ------------------------
def read_chat_files(chat_dirs):
    all_rows = []
    total_files = 0

    for chat_folder in chat_dirs:
        if not os.path.isdir(chat_folder):
            print(f"[WARN] Carpeta no encontrada: {chat_folder}")
            continue

        files = [os.path.join(chat_folder, f) for f in os.listdir(chat_folder) if f.endswith(".cha")]
        print(f"📂 Leyendo {len(files)} archivos en {chat_folder}")
        total_files += len(files)

        for fpath in files:
            try:
                ds = pla.read_chat(fpath)
            except Exception as e:
                print(f"[ERROR] No se pudo leer {fpath}: {e}")
                continue

            utts = ds.utterances(participants="PAR")
            if not utts:
                continue

            for u in utts:
                marks = u.time_marks
                if not marks or marks[0] is None or marks[1] is None:
                    continue

                row = {
                    "file": os.path.basename(fpath),
                    "mark_start": marks[0],
                    "mark_end": marks[1],
                    "start_sec": marks[0] / 1000.0,
                    "end_sec": marks[1] / 1000.0,
                    "duration_sec": (marks[1] - marks[0]) / 1000.0,
                    "transcription": u.tiers.get("PAR", ""),
                    "lang": "en" if "English" in chat_folder else "es",
                }

                # metadatos del encabezado (si existen)
                try:
                    header = ds.headers()[0]["Participants"]["PAR"]
                    row["sex"] = header.get("sex", "")
                    row["age"] = header.get("age", "")
                    row["WAB_AQ"] = header.get("custom", "")
                    row["aphasia_type"] = header.get("group", "")
                except Exception:
                    row["sex"] = row["age"] = row["WAB_AQ"] = row["aphasia_type"] = ""

                # ID de paciente = nombre del archivo sin extensión
                row["patient_id"] = os.path.splitext(os.path.basename(fpath))[0]

                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No se extrajo ninguna utterance de paciente.")
        return df

    df["age"] = df["age"].astype(str).str.extract(r"(\d+)")
    df = df[df["duration_sec"] > 0]
    df.reset_index(drop=True, inplace=True)

    print(f"Total archivos procesados: {total_files}")
    print(f"Total utterances extraídas: {len(df)}")
    return df

# ----------------------------- MAIN ---------------------------------
def main():
    df = read_chat_files(CHA_DIRS)
    if df.empty:
        print("No se genero ningún dato.")
        return

    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Guardado CSV → {OUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
