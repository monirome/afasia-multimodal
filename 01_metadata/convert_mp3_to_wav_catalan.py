#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path
from tqdm import tqdm
import os

BASE_DIR = Path("/lustre/ific.uv.es/ml/upc150/upc1503")
MP3_DIR = BASE_DIR / "data/audios_completos/Catalan/Control"

TARGET_SR = 16000
TARGET_CHANNELS = 1

# Pon esto a False si algún día quieres conservar los mp3
DELETE_ORIGINAL_MP3 = True

def main():
    if not MP3_DIR.is_dir():
        print(f"La carpeta {MP3_DIR} no existe.")
        return

    mp3_files = sorted(MP3_DIR.glob("*.mp3"))
    if not mp3_files:
        print(f"No se encontraron archivos .mp3 en {MP3_DIR}")
        return

    print(f"Se van a procesar {len(mp3_files)} archivos MP3 en:")
    print(f"  {MP3_DIR}\n")

    converted = 0
    deleted = 0
    skipped = 0

    for mp3_path in tqdm(mp3_files, desc="Procesando MP3"):
        wav_path = mp3_path.with_suffix(".wav")

        # Si ya existe un WAV válido, solo borrar el MP3 (si toca) y seguir
        if wav_path.exists() and os.path.getsize(wav_path) > 0:
            skipped += 1
            if DELETE_ORIGINAL_MP3:
                try:
                    mp3_path.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"No se pudo borrar {mp3_path}: {e}")
            continue

        # Si no existe el WAV, convertir con ffmpeg
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-y",
            "-i", str(mp3_path),
            "-ac", str(TARGET_CHANNELS),
            "-ar", str(TARGET_SR),
            str(wav_path),
        ]

        result = subprocess.run(cmd)

        # Si ffmpeg fue bien y el wav existe y tiene tamaño > 0, borrar el mp3 si procede
        if result.returncode == 0 and wav_path.exists() and os.path.getsize(wav_path) > 0:
            converted += 1
            if DELETE_ORIGINAL_MP3:
                try:
                    mp3_path.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"No se pudo borrar {mp3_path}: {e}")
        else:
            print(f"Error convirtiendo {mp3_path}, se mantiene el MP3.")

    print("\nConversión terminada.")
    print(f"  MP3 con WAV previo (saltados en conversión): {skipped}")
    print(f"  WAV creados en esta ejecución: {converted}")
    print(f"  MP3 borrados: {deleted}")
    print(f"Revisa los WAV en: {MP3_DIR}")

if __name__ == "__main__":
    main()
