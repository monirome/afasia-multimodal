#!/usr/bin/env python3
# generate_whisperx_alignments.py
# -*- coding: utf-8 -*-
"""
GENERACION DE WORD ALIGNMENTS CON WHISPERX - FORCED ALIGNMENT MODE
Replica metodología de Le et al. (2018) que usa P2FA
"""

import os
import glob
import argparse
import warnings
import re
warnings.filterwarnings("ignore")

import pandas as pd
import torch

try:
    import whisperx
    HAS_WHISPERX = True
except ImportError:
    HAS_WHISPERX = False
    print("WhisperX no esta instalado")
    print("Instalalo con: pip install whisperx")

# ======================== CONFIGURACION ========================
DEFAULT_MODEL = "large-v2"
DEFAULT_BATCH_SIZE = 16
AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".flac"}
DEFAULT_OUTPUT = "../data/word_alignments_ALL.csv"

# ======================== LIMPIEZA Y TROCEO ====================
def normalize_chat_for_alignment(s: str) -> str:
    """
    Normaliza AGRESIVAMENTE una línea *PAR:* de CHAT para alineación robusta
    """
    t = s.lower()  # Todo a minúsculas

    # 1) Eliminar timecodes tipo 18646_19427
    t = re.sub(r'\b\d+_\d+\b', ' ', t)

    # 2) Eliminar marcas CHAT problemáticas
    t = re.sub(r'\+<', ' ', t)
    t = re.sub(r'\+(?:/|//|")', ' ', t)
    t = re.sub(r'&\+\S+', ' ', t)
    t = re.sub(r'&\S+', ' ', t)
    t = re.sub(r'\[[^\]]*\]', ' ', t)
    t = re.sub(r'<[^>]*>', ' ', t)
    t = re.sub(r':\w+', ' ', t)
    t = re.sub(r'\(.*?\)', ' ', t)

    # 3) Normalizar guiones bajos y guiones
    t = t.replace('_', ' ')
    t = t.replace('-', ' ')

    # 4) Quitar puntuación
    t = re.sub(r'[""„‡†,.!?;:]', ' ', t)

    # 5) Conservar solo letras/números/apóstrofos/espacios
    t = re.sub(r"[^0-9a-zA-ZÀ-ÿ'\s]", ' ', t)

    # 6) Espacios múltiples
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def chunk_by_tokens(text: str, max_tokens: int = 30):
    """
    Divide un texto en trozos de hasta max_tokens tokens (aprox. palabras).
    """
    toks = re.findall(r"\b[\wÀ-ÿ']+\b", text)
    for i in range(0, len(toks), max_tokens):
        yield ' '.join(toks[i:i + max_tokens])


# ======================== UTILIDADES ==========================
def find_audio_files(audio_base):
    """Encuentra todos los archivos de audio recursivamente y elige el mayor si hay duplicados."""
    audio_files = {}

    print("Buscando audios en: {}".format(audio_base))

    for ext in AUDIO_EXTENSIONS:
        pattern = os.path.join(audio_base, "**/*{}".format(ext))
        for filepath in glob.glob(pattern, recursive=True):
            patient_id = os.path.splitext(os.path.basename(filepath))[0]

            if patient_id in audio_files:
                existing_size = os.path.getsize(audio_files[patient_id])
                new_size = os.path.getsize(filepath)
                if new_size > existing_size:
                    audio_files[patient_id] = filepath
            else:
                audio_files[patient_id] = filepath

    return audio_files


def parse_only_ids_arg(only_ids_value: str):
    """
    Soporta:
      --only_ids "id1,id2,id3"
      --only_ids /ruta/a/ids.csv (una columna, sin cabecera)
    """
    if not only_ids_value:
        return set()
    path = only_ids_value.strip()
    if os.path.exists(path) and os.path.isfile(path):
        try:
            df = pd.read_csv(path, header=None)
            return set(df.iloc[:, 0].astype(str).str.strip().tolist())
        except Exception:
            pass
    # lista separada por comas
    return set([x.strip() for x in path.split(",") if x.strip()])


# ← NUEVA FUNCIÓN: Extraer y preparar segmentos desde el .CHA
def extract_patient_segments_from_cha(patient_id, cha_dir, chunk_tokens: int):
    """
    Extrae SOLO *PAR:* del .CHA, limpia y devuelve una lista de segmentos cortos
    para el forced alignment. Cada segmento está ya normalizado y troceado.
    """
    cha_pattern = os.path.join(cha_dir, "**/{}.cha".format(patient_id))
    cha_files = glob.glob(cha_pattern, recursive=True)

    if not cha_files:
        return None

    raw_utts = []
    try:
        with open(cha_files[0], 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.startswith('*PAR:'):
                    continue
                text = line.replace('*PAR:', '').strip()
                # Limpieza ligera antes de la normalización fuerte
                text = re.sub(r'\(.*?\)', ' ', text)
                text = re.sub(r'@\w+', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    raw_utts.append(text)
    except Exception as e:
        print("      Error leyendo .CHA: {}".format(e))
        return None

    if not raw_utts:
        return None

    # Normalizar y trocear
    segments = []
    for u in raw_utts:
        u2 = normalize_chat_for_alignment(u)
        if not u2:
            continue
        for ch in chunk_by_tokens(u2, max_tokens=chunk_tokens):
            if ch:
                segments.append(ch)

    return segments if segments else None


# ======================== WHISPERX FORCED ALIGNMENT ========================
def process_audio_whisperx_forced(audio_path, segment_texts, align_model,
                                  metadata, device, language):
    """
    FORCED ALIGNMENT con WhisperX - VERSIÓN ROBUSTA
    """
    try:
        # Cargar audio
        audio = whisperx.load_audio(audio_path)
        sr = 16000.0
        total_dur = len(audio) / sr
        
        if total_dur <= 0:
            print("    Error: audio con duración 0")
            return []

        # Preparar segmentos con tiempos proporcionales
        token_counts = [len(re.findall(r"\b[\wÀ-ÿ']+\b", s)) for s in segment_texts]
        token_counts = [max(1, n) for n in token_counts]
        total_tokens = float(sum(token_counts))

        segments = []
        cursor = 0.0
        for txt, n_tok in zip(segment_texts, token_counts):
            dur = total_dur * (n_tok / total_tokens)
            start = cursor
            end = min(total_dur, cursor + dur)
            cursor = end
            segments.append({"text": txt, "start": start, "end": end})

        print("    Alineando {} segmentos...".format(len(segments)))
        
        # Alinear con configuración robusta
        try:
            aligned_result = whisperx.align(
                segments,
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False,
                interpolate_method="nearest",
                return_timestamps="word"
            )
        except TypeError:
            # Versión antigua de WhisperX (sin parámetros extra)
            aligned_result = whisperx.align(
                segments,
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )

        # Extraer palabras
        word_alignments = []
        for segment in aligned_result.get("segments", []):
            for word_info in segment.get("words", []):
                if not word_info:
                    continue
                    
                word = (word_info.get("word") or "").strip()
                start = word_info.get("start")
                end = word_info.get("end")
                
                if not word or start is None or end is None:
                    continue
                
                # Validar tiempos razonables
                if start < 0 or end < start or end > total_dur:
                    continue
                
                word_alignments.append({
                    "word": word,
                    "start_sec": float(start),
                    "end_sec": float(end)
                })

        print("    {} palabras alineadas".format(len(word_alignments)))
        return word_alignments

    except Exception as e:
        print("    Error: {}".format(e))
        import traceback
        traceback.print_exc()
        return []


# ======================== MAIN ========================
def main():
    """Pipeline principal"""

    parser = argparse.ArgumentParser(
        description="Genera word alignments con WhisperX FORCED ALIGNMENT",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--audio_base",
        required=True,
        help="Carpeta raiz con audios (busqueda recursiva)"
    )

    parser.add_argument(
        "--cha_dir",
        required=True,
        help="Carpeta con archivos .CHA (transcripciones manuales)"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Archivo CSV de salida (default: {})".format(DEFAULT_OUTPUT)
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Dispositivo (default: cuda si esta disponible)"
    )

    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "es", "ca"],
        help="Idioma (default: en)"
    )

    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Procesar solo N pacientes (para testing rapido)"
    )

    parser.add_argument(
        "--only_ids",
        default=None,
        help="CSV con ids (1 columna) o lista separada por comas, para filtrar pacientes"
    )

    parser.add_argument(
        "--chunk_tokens",
        type=int,
        default=30,
        help="Maximo de tokens por segmento para forced alignment (default: 30)"
    )

    args = parser.parse_args()

    # ==================== VERIFICACIONES ====================
    print("=" * 70)
    print("GENERACION DE WORD ALIGNMENTS - FORCED ALIGNMENT MODE")
    print("Replica metodología P2FA de Le et al. (2018)")
    print("=" * 70)

    if not HAS_WHISPERX:
        print("\nWhisperX no esta disponible")
        print("Instalalo con: pip install whisperx")
        return

    if not os.path.exists(args.audio_base):
        print("\nERROR: No existe directorio de audios: {}".format(args.audio_base))
        return

    if not os.path.exists(args.cha_dir):
        print("\nERROR: No existe directorio de .CHA: {}".format(args.cha_dir))
        return

    print("\nConfiguracion:")
    print("  Device: {}".format(args.device))
    print("  Idioma: {}".format(args.language))
    print("  Audio base: {}".format(args.audio_base))
    print("  CHA dir: {}".format(args.cha_dir))
    print("  Output: {}".format(args.output))
    print("  chunk_tokens: {}".format(args.chunk_tokens))
    if args.only_ids:
        print("  only_ids: {}".format(args.only_ids))
    print("\n  MODO: FORCED ALIGNMENT (solo palabras del paciente)")

    # Verificar CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\n  CUDA no disponible. Usando CPU (sera MAS LENTO)")
        args.device = "cpu"

    if args.device == "cuda":
        print("\nGPU detectada: {}".format(torch.cuda.get_device_name(0)))

    # ==================== BUSCAR AUDIOS ====================
    print("\n" + "=" * 70)
    print("BUSCANDO AUDIOS")
    print("=" * 70)

    audio_files = find_audio_files(args.audio_base)

    if len(audio_files) == 0:
        print("No se encontraron archivos de audio")
        return

    # Filtrar por only_ids si procede
    only = set()
    if args.only_ids:
        only = parse_only_ids_arg(args.only_ids)
        if only:
            audio_files = {k: v for k, v in audio_files.items() if k in only}
            print("Filtrado por only_ids: {} audios".format(len(audio_files)))

    print("Encontrados: {} audios".format(len(audio_files)))

    # Limitar para testing
    if args.max_patients:
        audio_files = dict(list(audio_files.items())[:args.max_patients])
        print("\n  Limitado a {} pacientes (testing)".format(args.max_patients))

    # ==================== CARGAR MODELO DE ALINEACION ====================
    print("\n" + "=" * 70)
    print("CARGANDO MODELO WHISPERX")
    print("=" * 70)

    print("\nCargando modelo de alineacion para {}...".format(args.language))

    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=args.language,
            device=args.device
        )
        print("Modelo de alineacion cargado")
    except Exception as e:
        print("Error cargando modelo de alineacion: {}".format(e))
        return

    # ==================== PROCESAR AUDIOS ====================
    print("\n" + "=" * 70)
    print("PROCESANDO {} AUDIOS - FORCED ALIGNMENT".format(len(audio_files)))
    print("=" * 70)

    all_alignments = []
    successful = 0
    failed = 0
    skipped_no_cha = 0

    for i, (patient_id, audio_path) in enumerate(audio_files.items(), 1):
        print("\n[{0:4d}/{1}] {2}".format(i, len(audio_files), patient_id))
        print("  {}".format(os.path.basename(audio_path)))

        # PASO 1: Extraer segmentos desde el .CHA (solo *PAR:)
        segment_texts = extract_patient_segments_from_cha(
            patient_id,
            args.cha_dir,
            args.chunk_tokens
        )

        if not segment_texts:
            print("  No se encontró .CHA o sin utterances *PAR:, saltando...")
            skipped_no_cha += 1
            continue

        total_palabras = sum(len(s.split()) for s in segment_texts)
        print("  Transcripción .CHA: {} palabras en {} segmentos".format(
            total_palabras, len(segment_texts))
        )

        # PASO 2: Forced alignment con WhisperX en segmentos
        word_aligns = process_audio_whisperx_forced(
            audio_path,
            segment_texts,
            align_model,
            metadata,
            args.device,
            args.language
        )

        if word_aligns:
            for wa in word_aligns:
                wa["patient_id"] = patient_id
                wa["group"] = "Unknown"
            all_alignments.extend(word_aligns)
            successful += 1
        else:
            failed += 1

    # ==================== GUARDAR RESULTADOS ====================
    if all_alignments:
        print("\n" + "=" * 70)
        print("GUARDANDO RESULTADOS")
        print("=" * 70)

        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Creado directorio: {}".format(output_dir))

        df_out = pd.DataFrame(all_alignments)
        df_out = df_out[["patient_id", "group", "word", "start_sec", "end_sec"]]
        df_out = df_out.sort_values(["patient_id", "start_sec"])

        df_out.to_csv(args.output, index=False)

        print("\nProcesados exitosamente: {}/{}".format(successful, len(audio_files)))
        if failed > 0:
            print("Fallidos: {}/{}".format(failed, len(audio_files)))
        if skipped_no_cha > 0:
            print("Sin .CHA: {}/{}".format(skipped_no_cha, len(audio_files)))

        print("\nEstadisticas:")
        print("  Total palabras: {:,}".format(len(df_out)))
        if successful > 0:
            print("  Palabras/paciente (promedio): {:.1f}".format(len(df_out) / successful))
        print("  Pacientes: {}".format(df_out['patient_id'].nunique()))

        print("\nIMPORTANTE: Estas son SOLO palabras del PACIENTE (*PAR:)")
        print("   No incluye speech del investigador")

        print("\nGuardado en: {}".format(args.output))

        print("\n" + "=" * 70)
        print("PROCESO COMPLETADO")
        print("=" * 70)

    else:
        print("\nNo se generaron alineaciones")


if __name__ == "__main__":
    main()