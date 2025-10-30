#!/usr/bin/env python3
# generate_whisperx_alignments.py
# -*- coding: utf-8 -*-
"""
GENERACION DE WORD ALIGNMENTS CON WHISPERX
Para replicar Le et al. (2018) con tecnologia moderna
"""

import os
import glob
import argparse
import warnings
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

# ======================== UTILIDADES ========================
def find_audio_files(audio_base):
    """
    Encuentra todos los archivos de audio recursivamente
    """
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

def load_manual_transcriptions(csv_path):
    """
    Carga transcripciones manuales si existen
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        
        if "patient_id" not in df.columns or "transcription" not in df.columns:
            print("  CSV debe tener columnas: patient_id, transcription")
            return {}
        
        return dict(zip(df["patient_id"], df["transcription"]))
    
    except Exception as e:
        print("  Error cargando transcripciones: {}".format(e))
        return {}

# ======================== WHISPERX PIPELINE ========================
def process_audio_whisperx(audio_path, model, align_model, metadata, 
                           device, batch_size, manual_transcript=None):
    """
    Procesa un audio con WhisperX
    """
    
    try:
        # Cargar audio
        audio = whisperx.load_audio(audio_path)
        
        # 1. ASR
        if manual_transcript:
            print("    Usando transcripcion manual")
            result = {
                "segments": [{
                    "text": manual_transcript,
                    "start": 0.0,
                    "end": len(audio) / 16000.0
                }],
                "language": "en"
            }
        else:
            print("    Transcribiendo con Whisper...")
            result = model.transcribe(audio, batch_size=batch_size)
            print("    Transcripcion completada")
        
        # 2. Forced Alignment
        print("    Alineando palabras...")
        aligned_result = whisperx.align(
            result["segments"], 
            align_model, 
            metadata,
            audio, 
            device,
            return_char_alignments=False
        )
        
        # 3. Extraer word-level timestamps
        word_alignments = []
        for segment in aligned_result.get("segments", []):
            for word_info in segment.get("words", []):
                word_alignments.append({
                    "word": word_info["word"].strip(),
                    "start_sec": float(word_info["start"]),
                    "end_sec": float(word_info["end"])
                })
        
        print("    {} palabras alineadas".format(len(word_alignments)))
        
        return word_alignments
    
    except Exception as e:
        print("    Error: {}".format(e))
        return []

# ======================== MAIN ========================
def main():
    """Pipeline principal"""
    
    parser = argparse.ArgumentParser(
        description="Genera word alignments con WhisperX para Le et al. (2018)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--audio_base", 
        required=True,
        help="Carpeta raiz con audios (busqueda recursiva)"
    )
    
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Archivo CSV de salida (default: {})".format(DEFAULT_OUTPUT)
    )
    
    parser.add_argument(
        "--transcriptions_csv",
        default=None,
        help="CSV opcional con transcripciones manuales (patient_id, transcription)"
    )
    
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Modelo de Whisper (default: large-v2)"
    )
    
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Dispositivo (default: cuda si esta disponible)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size para inferencia (default: 16)"
    )
    
    parser.add_argument(
        "--language",
        default=None,
        help="Idioma (en, es, ca). Si None, se detecta automaticamente"
    )
    
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Procesar solo N pacientes (para testing rapido)"
    )
    
    args = parser.parse_args()
    
    # ==================== VERIFICACIONES ====================
    print("="*70)
    print("GENERACION DE WORD ALIGNMENTS CON WHISPERX")
    print("="*70)
    
    if not HAS_WHISPERX:
        print("\nWhisperX no esta disponible")
        print("Instalalo con:")
        print("  pip install whisperx")
        print("  # Para GPU:")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return
    
    if not os.path.exists(args.audio_base):
        print("\nERROR: No existe directorio de audios: {}".format(args.audio_base))
        return
    
    print("\nConfiguracion:")
    print("  Modelo Whisper: {}".format(args.model))
    print("  Device: {}".format(args.device))
    print("  Batch size: {}".format(args.batch_size))
    print("  Idioma: {}".format(args.language if args.language else 'auto-detect'))
    print("  Audio base: {}".format(args.audio_base))
    print("  Output: {}".format(args.output))
    
    # Verificar CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\n  CUDA no disponible. Usando CPU (sera MAS LENTO)")
        args.device = "cpu"
    
    if args.device == "cuda":
        print("\nGPU detectada: {}".format(torch.cuda.get_device_name(0)))
        compute_type = "float16"
    else:
        print("\n  Usando CPU (puede ser lento para muchos audios)")
        compute_type = "int8"
    
    # ==================== BUSCAR AUDIOS ====================
    print("\n" + "="*70)
    print("BUSCANDO AUDIOS")
    print("="*70)
    
    audio_files = find_audio_files(args.audio_base)
    
    if len(audio_files) == 0:
        print("No se encontraron archivos de audio")
        print("Extensiones buscadas: {}".format(AUDIO_EXTENSIONS))
        return
    
    print("Encontrados: {} audios".format(len(audio_files)))
    
    # Mostrar primeros 5
    print("\nPrimeros 5 audios:")
    for i, (pid, path) in enumerate(list(audio_files.items())[:5], 1):
        print("  {}. {}: {}".format(i, pid, os.path.basename(path)))
    
    if len(audio_files) > 5:
        print("  ... y {} mas".format(len(audio_files) - 5))
    
    # Cargar transcripciones manuales
    manual_transcripts = load_manual_transcriptions(args.transcriptions_csv)
    if manual_transcripts:
        print("\nTranscripciones manuales: {}".format(len(manual_transcripts)))
    
    # Limitar para testing
    if args.max_patients:
        audio_files = dict(list(audio_files.items())[:args.max_patients])
        print("\n  Limitado a {} pacientes (testing)".format(args.max_patients))
    
    # ==================== CARGAR MODELOS ====================
    print("\n" + "="*70)
    print("CARGANDO MODELOS WHISPERX")
    print("="*70)
    
    print("\nCargando modelo Whisper ({})...".format(args.model))
    print("(esto puede tomar algunos minutos la primera vez)")
    
    try:
        model = whisperx.load_model(
            args.model, 
            device=args.device,
            compute_type=compute_type,
            language=args.language
        )
        print("Modelo ASR cargado")
    except Exception as e:
        print("Error cargando modelo ASR: {}".format(e))
        return
    
    print("\nCargando modelo de alineacion...")
    
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=args.language if args.language else "en",
            device=args.device
        )
        print("Modelo de alineacion cargado")
    except Exception as e:
        print("Error cargando modelo de alineacion: {}".format(e))
        return
    
    # ==================== PROCESAR AUDIOS ====================
    print("\n" + "="*70)
    print("PROCESANDO {} AUDIOS".format(len(audio_files)))
    print("="*70)
    
    all_alignments = []
    successful = 0
    failed = 0
    
    for i, (patient_id, audio_path) in enumerate(audio_files.items(), 1):
        print("\n[{:4d}/{}] {}".format(i, len(audio_files), patient_id))
        print("  {}".format(os.path.basename(audio_path)))
        
        # Duracion del audio
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            print("  Duracion: {:.1f}s".format(duration))
        except:
            pass
        
        # Obtener transcripcion manual si existe
        manual_trans = manual_transcripts.get(patient_id)
        
        # Procesar con WhisperX
        word_aligns = process_audio_whisperx(
            audio_path, 
            model, 
            align_model,
            metadata,
            args.device,
            args.batch_size,
            manual_transcript=manual_trans
        )
        
        if word_aligns:
            for wa in word_aligns:
                wa["patient_id"] = patient_id
            
            all_alignments.extend(word_aligns)
            successful += 1
        else:
            failed += 1
    
    # ==================== GUARDAR RESULTADOS ====================
    if all_alignments:
        print("\n" + "="*70)
        print("GUARDANDO RESULTADOS")
        print("="*70)
        
        # Crear directorio de salida si no existe
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Creado directorio: {}".format(output_dir))
        
        # Crear DataFrame
        df_out = pd.DataFrame(all_alignments)
        df_out = df_out[["patient_id", "word", "start_sec", "end_sec"]]
        df_out = df_out.sort_values(["patient_id", "start_sec"])
        
        # Guardar CSV
        df_out.to_csv(args.output, index=False)
        
        print("\nProcesados exitosamente: {}/{}".format(successful, len(audio_files)))
        if failed > 0:
            print("Fallidos: {}/{}".format(failed, len(audio_files)))
        
        print("\nEstadisticas:")
        print("  Total palabras: {:,}".format(len(df_out)))
        print("  Palabras/paciente (promedio): {:.1f}".format(len(df_out)/successful))
        print("  Pacientes: {}".format(df_out['patient_id'].nunique()))
        
        # Distribucion de duraciones
        df_out['duration'] = df_out['end_sec'] - df_out['start_sec']
        print("\n  Duracion palabras (segundos):")
        print("    Min:    {:.3f}".format(df_out['duration'].min()))
        print("    Media:  {:.3f}".format(df_out['duration'].mean()))
        print("    Max:    {:.3f}".format(df_out['duration'].max()))
        
        print("\nGuardado en: {}".format(args.output))
        
        print("\nPrimeras 10 palabras:")
        print(df_out.head(10).to_string(index=False))
        
        print("\n" + "="*70)
        print("PROCESO COMPLETADO")
        print("="*70)
        
    else:
        print("\nNo se generaron alineaciones")
        print("Verifica que:")
        print("  - Los audios sean validos")
        print("  - WhisperX este correctamente instalado")
        print("  - Haya suficiente memoria (GPU o RAM)")

if __name__ == "__main__":
    main()