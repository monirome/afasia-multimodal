#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
GENERACIÓN DE WORD ALIGNMENTS CON WHISPERX
Para replicar Le et al. (2018) con tecnología moderna
==============================================================================

WhisperX combina:
- Whisper (OpenAI): ASR estado del arte
- Forced alignment: timestamps palabra por palabra

VENTAJAS vs Kaldi:
- Instalación simple (pip install)
- Multilingüe (99 idiomas)
- No requiere entrenamiento
- GPU-friendly

ENTRADA:
- Carpeta con audios (.wav, .mp3, .mp4, etc.)
- Opcionalmente: transcripciones manuales

SALIDA:
- word_alignments.csv con columnas:
  * patient_id
  * word
  * start_sec
  * end_sec

REQUISITOS:
pip install whisperx torch torchaudio
# GPU (recomendado):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
==============================================================================
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
    print("WhisperX no está instalado")
    print("Instálalo con: pip install whisperx")

# ======================== CONFIGURACIÓN ========================
DEFAULT_MODEL = "large-v2"  # Opciones: tiny, base, small, medium, large-v2, large-v3
DEFAULT_BATCH_SIZE = 16
AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".m4a", ".flac"}

# ======================== UTILIDADES ========================
def find_audio_files(audio_base):
    """
    Encuentra todos los archivos de audio recursivamente
    
    Args:
        audio_base: directorio raíz
    
    Returns:
        dict: {patient_id: ruta_audio}
    """
    audio_files = {}
    
    print(f"Buscando audios en: {audio_base}")
    
    for ext in AUDIO_EXTENSIONS:
        pattern = os.path.join(audio_base, f"**/*{ext}")
        for filepath in glob.glob(pattern, recursive=True):
            # Patient ID = nombre del archivo sin extensión
            patient_id = os.path.splitext(os.path.basename(filepath))[0]
            
            # Si hay duplicados, quedarse con el más grande
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
    
    Args:
        csv_path: ruta al CSV
    
    Returns:
        dict: {patient_id: transcription}
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
        print(f"  Error cargando transcripciones: {e}")
        return {}

# ======================== WHISPERX PIPELINE ========================
def process_audio_whisperx(audio_path, model, align_model, metadata, 
                           device, batch_size, manual_transcript=None):
    """
    Procesa un audio con WhisperX:
    1. ASR (si no hay transcripción manual)
    2. Forced alignment palabra por palabra
    
    Args:
        audio_path: ruta al audio
        model: modelo WhisperX cargado
        align_model: modelo de alineación
        metadata: dict con info del modelo
        device: 'cuda' o 'cpu'
        batch_size: batch size para inferencia
        manual_transcript: transcripción manual opcional
    
    Returns:
        list de dicts: [{word, start_sec, end_sec}, ...]
    """
    
    try:
        # Cargar audio
        audio = whisperx.load_audio(audio_path)
        
        # 1. ASR (transcripción automática)
        if manual_transcript:
            print(f"    ✓ Usando transcripción manual")
            # Crear formato compatible con WhisperX
            result = {
                "segments": [{
                    "text": manual_transcript,
                    "start": 0.0,
                    "end": len(audio) / 16000.0
                }],
                "language": "en"  # Se detectará en alignment
            }
        else:
            print(f"    ⏳ Transcribiendo con Whisper...")
            result = model.transcribe(audio, batch_size=batch_size)
            print(f"    ✓ Transcripción completada")
        
        # 2. Forced Alignment
        print(f"    ⏳ Alineando palabras...")
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
        
        print(f"    ✓ {len(word_alignments)} palabras alineadas")
        
        return word_alignments
    
    except Exception as e:
        print(f"    Error: {e}")
        return []

# ======================== MAIN ========================
def main():
    """Pipeline principal"""
    
    parser = argparse.ArgumentParser(
        description="Genera word alignments con WhisperX para Le et al. (2018)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Argumentos principales
    parser.add_argument(
        "--audio_base", 
        required=True,
        help="Carpeta raíz con audios (búsqueda recursiva)"
    )
    
    parser.add_argument(
        "--output",
        default="word_alignments.csv",
        help="Archivo CSV de salida (default: word_alignments.csv)"
    )
    
    # Transcripciones manuales (opcional)
    parser.add_argument(
        "--transcriptions_csv",
        default=None,
        help="CSV opcional con transcripciones manuales (patient_id, transcription)"
    )
    
    # Configuración del modelo
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
        help="Dispositivo (default: cuda si está disponible)"
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
        help="Idioma (en, es, ca). Si None, se detecta automáticamente"
    )
    
    # Testing
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Procesar solo N pacientes (para testing rápido)"
    )
    
    args = parser.parse_args()
    
    # ==================== VERIFICACIONES ====================
    print("="*70)
    print("GENERACIÓN DE WORD ALIGNMENTS CON WHISPERX")
    print("="*70)
    
    if not HAS_WHISPERX:
        print("\nWhisperX no está disponible")
        print("Instálalo con:")
        print("  pip install whisperx")
        print("  # Para GPU:")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return
    
    print(f"\nConfiguración:")
    print(f"  Modelo Whisper: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Idioma: {args.language if args.language else 'auto-detect'}")
    print(f"  Audio base: {args.audio_base}")
    print(f"  Output: {args.output}")
    
    # Verificar CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\n  CUDA no disponible. Usando CPU (será MÁS LENTO)")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"\n✓ GPU detectada: {torch.cuda.get_device_name(0)}")
        compute_type = "float16"
    else:
        print(f"\n  Usando CPU (puede ser lento para muchos audios)")
        compute_type = "int8"
    
    # ==================== BUSCAR AUDIOS ====================
    print("\n" + "="*70)
    print("BUSCANDO AUDIOS")
    print("="*70)
    
    audio_files = find_audio_files(args.audio_base)
    
    if len(audio_files) == 0:
        print("No se encontraron archivos de audio")
        print(f"Extensiones buscadas: {AUDIO_EXTENSIONS}")
        return
    
    print(f"✓ Encontrados: {len(audio_files)} audios")
    
    # Mostrar primeros 5
    print("\nPrimeros 5 audios:")
    for i, (pid, path) in enumerate(list(audio_files.items())[:5], 1):
        print(f"  {i}. {pid}: {os.path.basename(path)}")
    
    if len(audio_files) > 5:
        print(f"  ... y {len(audio_files) - 5} más")
    
    # Cargar transcripciones manuales (opcional)
    manual_transcripts = load_manual_transcriptions(args.transcriptions_csv)
    if manual_transcripts:
        print(f"\n✓ Transcripciones manuales: {len(manual_transcripts)}")
    
    # Limitar para testing
    if args.max_patients:
        audio_files = dict(list(audio_files.items())[:args.max_patients])
        print(f"\n  Limitado a {args.max_patients} pacientes (testing)")
    
    # ==================== CARGAR MODELOS ====================
    print("\n" + "="*70)
    print("CARGANDO MODELOS WHISPERX")
    print("="*70)
    
    # Modelo ASR
    print(f"\nCargando modelo Whisper ({args.model})...")
    print("(esto puede tomar algunos minutos la primera vez)")
    
    try:
        model = whisperx.load_model(
            args.model, 
            device=args.device,
            compute_type=compute_type,
            language=args.language
        )
        print("✓ Modelo ASR cargado")
    except Exception as e:
        print(f"Error cargando modelo ASR: {e}")
        return
    
    # Modelo de alineación
    print("\nCargando modelo de alineación...")
    
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=args.language if args.language else "en",
            device=args.device
        )
        print("✓ Modelo de alineación cargado")
    except Exception as e:
        print(f"Error cargando modelo de alineación: {e}")
        return
    
    # ==================== PROCESAR AUDIOS ====================
    print("\n" + "="*70)
    print(f"PROCESANDO {len(audio_files)} AUDIOS")
    print("="*70)
    
    all_alignments = []
    successful = 0
    failed = 0
    
    for i, (patient_id, audio_path) in enumerate(audio_files.items(), 1):
        print(f"\n[{i:4d}/{len(audio_files)}] {patient_id}")
        print(f"  📁 {os.path.basename(audio_path)}")
        
        # Duración del audio
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            print(f"  ⏱️  Duración: {duration:.1f}s")
        except:
            pass
        
        # Obtener transcripción manual si existe
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
            # Agregar patient_id
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
        
        # Crear DataFrame
        df_out = pd.DataFrame(all_alignments)
        df_out = df_out[["patient_id", "word", "start_sec", "end_sec"]]
        df_out = df_out.sort_values(["patient_id", "start_sec"])
        
        # Guardar CSV
        df_out.to_csv(args.output, index=False)
        
        # Resumen
        print(f"\n✓ Procesados exitosamente: {successful}/{len(audio_files)}")
        if failed > 0:
            print(f"✗ Fallidos: {failed}/{len(audio_files)}")
        
        print(f"\nEstadísticas:")
        print(f"  Total palabras: {len(df_out):,}")
        print(f"  Palabras/paciente (promedio): {len(df_out)/successful:.1f}")
        print(f"  Pacientes: {df_out['patient_id'].nunique()}")
        
        # Distribución de duraciones
        df_out['duration'] = df_out['end_sec'] - df_out['start_sec']
        print(f"\n  Duración palabras (segundos):")
        print(f"    Min:    {df_out['duration'].min():.3f}")
        print(f"    Media:  {df_out['duration'].mean():.3f}")
        print(f"    Max:    {df_out['duration'].max():.3f}")
        
        print(f"\n✓ Guardado en: {args.output}")
        
        # Mostrar primeras filas
        print("\nPrimeras 10 palabras:")
        print(df_out.head(10).to_string(index=False))
        
        print("\n" + "="*70)
        print("PROCESO COMPLETADO")
        print("="*70)
        
    else:
        print("\nNo se generaron alineaciones")
        print("Verifica que:")
        print("  - Los audios sean válidos")
        print("  - WhisperX esté correctamente instalado")
        print("  - Haya suficiente memoria (GPU o RAM)")

if __name__ == "__main__":
    main()