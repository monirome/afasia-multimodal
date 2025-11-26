#!/usr/bin/env python3
# 03_features/build_dataset_base.py
# -*- coding: utf-8 -*-
"""
Construye data/dataset_base.csv DIRECTAMENTE desde los .CHA.

- Recorre las carpetas con .CHA (DEFAULT_CHA_DIRS o las que pases con --cha-roots).
- Extrae metadata WAB (QA, grupo, idioma, sexo, edad, tipo de afasia) desde los headers con pylangacq.
- Extrae la transcripción PAR limpia usando la misma lógica que aphasia_paper_replication_FINAL.py.
- Genera un CSV con columnas:
    patient_id
    group             (control / pwa / None)
    QA
    language
    sex
    age
    aphasia_type
    group_original
    transcript
"""

import os
import sys
import glob
import argparse
import re
import pandas as pd
import pylangacq as pla

# ======================== RUTAS POR DEFECTO ========================
# Las mismas que usas en aphasia_paper_replication_FINAL.py
DEFAULT_CHA_DIRS = [
    "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones/English",
    "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones/Spanish",
]

# ==================== UTILIDADES TEXTO (copiado del otro script) ====================
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ\u00f1\u00d1']+", re.UNICODE)

def tokenize_words(text: str):
    if not text:
        return []
    text = text.replace("_", " ")
    return _WORD_RE.findall(text)

def clean_chat_text(text):
    t = text
    t = re.sub(r'\b\d+_\d+\b', '', t)
    t = re.sub(r'\[[\^:]\]', '', t)
    t = re.sub(r'<[^>]*>', '', t)
    t = re.sub(r'\[[^\]]*\]', '', t)
    t = re.sub(r'xxx|www', '', t)
    t = re.sub(r'&=\w+', '', t)
    t = re.sub(r'&\w+', '', t)  # &uh, &mm
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'\(\.\)', ' ', t)
    t = re.sub(r'\([\d.]+\)', ' ', t)
    t = re.sub(r'\([^)]*\)', ' ', t)
    t = re.sub(r'[+/]', ' ', t)
    t = re.sub(r'[:;]', ' ', t)
    t = re.sub(r"[^\w\s']", ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def extract_par_utterances_from_cha(cha_path):
    """
    Extrae:
    - transcript: todas las palabras PAR limpiadas y concatenadas en minúsculas
    - utterances: lista de enunciados limpios
    """
    if not cha_path or not os.path.exists(cha_path):
        return None, []
    utterances = []
    full_transcript = []
    try:
        with open(cha_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        current = ""
        is_par = False
        for line in lines:
            line = line.rstrip('\n')
            if line.startswith('*PAR:'):
                if current and is_par:
                    clean = clean_chat_text(current)
                    if clean:
                        utterances.append(clean)
                        full_transcript.extend(tokenize_words(clean))
                current = line[5:]
                is_par = True
            elif line.startswith('*'):
                if current and is_par:
                    clean = clean_chat_text(current)
                    if clean:
                        utterances.append(clean)
                        full_transcript.extend(tokenize_words(clean))
                current = ""
                is_par = False
            elif line.startswith('\t') and is_par:
                current += " " + line[1:]
        if current and is_par:
            clean = clean_chat_text(current)
            if clean:
                utterances.append(clean)
                full_transcript.extend(tokenize_words(clean))
    except Exception:
        return None, []
    if not full_transcript:
        return None, []
    return ' '.join(full_transcript).lower(), utterances

# ======================== METADATA DESDE .CHA ========================

def extract_metadata_df_from_cha(cha_roots):
    """
    Extrae metadata desde TODOS los .CHA en cha_roots (lista de carpetas).

    Devuelve un DataFrame con columnas:
        patient_id
        group_original
        QA
        sex
        age
        aphasia_type
        language
        group   (simplificado: control / pwa / None)
    """
    print("=" * 70)
    print("EXTRACCION METADATA DESDE .CHA PARA DATASET BASE")
    print("=" * 70)

    if not cha_roots:
        raise ValueError("Lista de cha_roots vacía; pasa --cha-roots o define DEFAULT_CHA_DIRS.")

    cha_files = []
    for root in cha_roots:
        if not root or not os.path.exists(root):
            continue
        pattern = os.path.join(root, "**", "*.cha")
        cha_files.extend(glob.glob(pattern, recursive=True))

    cha_files = sorted(set(cha_files))
    print("\nArchivos .CHA encontrados: {}".format(len(cha_files)))

    if len(cha_files) == 0:
        raise FileNotFoundError(
            "No se encontraron archivos .CHA en las carpetas:\n  " +
            "\n  ".join(cha_roots)
        )

    results = []
    errors = []

    for i, filepath in enumerate(cha_files, 1):
        if i % 100 == 0:
            print("  Procesando {}/{}...".format(i, len(cha_files)))

        patient_id = os.path.splitext(os.path.basename(filepath))[0]

        try:
            ds = pla.read_chat(filepath)
            headers = ds.headers()

            if len(headers) == 0:
                errors.append({'patient_id': patient_id, 'error': 'Sin headers'})
                continue

            header = headers[0]

            # Idioma
            lang = 'en'
            if 'Languages' in header:
                langs = header['Languages']
                lang_raw = (langs[0] if isinstance(langs, list) else langs).lower()
                if 'spa' in lang_raw or 'spanish' in lang_raw:
                    lang = 'es'
                elif 'cat' in lang_raw or 'catalan' in lang_raw:
                    lang = 'ca'

            # Participant PAR
            if 'Participants' not in header or 'PAR' not in header['Participants']:
                errors.append({'patient_id': patient_id, 'error': 'Sin PAR'})
                continue

            par_info = header['Participants']['PAR']

            group_raw = par_info.get('group', '').strip().lower() or None
            sex = par_info.get('sex', '').strip() or None

            age_raw = par_info.get('age', '').strip()
            age = None
            if age_raw:
                try:
                    age = int(age_raw.split(';')[0].split('.')[0])
                except Exception:
                    age = None

            custom = par_info.get('custom', '').strip()
            wab_aq = None
            aphasia_type = None

            if custom and custom != '':
                parts = str(custom).split('|')
                for part in parts:
                    part = part.strip()
                    try:
                        score = float(part)
                        if 0 <= score <= 100:
                            wab_aq = score
                    except Exception:
                        if part and not part.replace('.', '').isdigit():
                            aphasia_type = part

            results.append({
                'patient_id': patient_id,
                'group_original': group_raw,
                'QA': wab_aq,
                'sex': sex,
                'age': age,
                'aphasia_type': aphasia_type,
                'language': lang
            })

        except Exception as e:
            errors.append({'patient_id': patient_id, 'error': str(e)})

    df = pd.DataFrame(results)
    df_errors = pd.DataFrame(errors)

    print("\n" + "=" * 70)
    print("SIMPLIFICANDO GRUPOS: Control vs PWA")
    print("=" * 70)

    def simplify_group(row):
        g = row['group_original']
        if g == 'notaphasicbywab':
            return 'control'
        elif g in ['anomic', 'broca', 'conduction', 'wernicke', 'global',
                   'transmotor', 'transsensory', 'isolation', 'pwa']:
            return 'pwa'
        elif g == 'control':
            # Los "control" con QA bajo los tratamos como pwa (como en tu script)
            return 'pwa'
        else:
            return None

    df['group'] = df.apply(simplify_group, axis=1)

    print("\nDistribución ORIGINAL (group_original):")
    if 'group_original' in df.columns:
        print(df['group_original'].value_counts(dropna=False).head(20))
    else:
        print("No hay columna group_original")

    print("\nDistribución SIMPLIFICADA (group):")
    print(df['group'].value_counts(dropna=False))

    df_with_qa = df[df['QA'].notna()]
    if len(df_with_qa) > 0:
        print("\n" + "=" * 70)
        print("ESTADISTICAS WAB-AQ POR GRUPO")
        print("=" * 70)
        for grp in ['control', 'pwa']:
            df_grp = df_with_qa[df_with_qa['group'] == grp]
            if len(df_grp) > 0:
                print("\n{}:".format(grp.upper()))
                print("  N: {}".format(len(df_grp)))
                print("  QA media: {:.1f} ± {:.1f}".format(df_grp['QA'].mean(), df_grp['QA'].std()))
                print("  Rango: [{:.1f}, {:.1f}]".format(df_grp['QA'].min(), df_grp['QA'].max()))

    if len(df_errors) > 0:
        print("\nArchivos con errores de metadata: {}".format(len(df_errors)))

    return df

# ======================== DATASET BASE ========================

def build_dataset_base(cha_roots, output_csv):
    """
    Construye dataset_base.csv:

    - Usa extract_metadata_df_from_cha(cha_roots) para obtener QA, grupo, etc.
    - Recorre los mismos .CHA para extraer transcript PAR con
      extract_par_utterances_from_cha.
    """
    print("\n" + "=" * 70)
    print("CONSTRUYENDO DATASET BASE DESDE .CHA")
    print("=" * 70)
    print("CHA_ROOTS:")
    for r in cha_roots:
        print("  - {}".format(r))
    print("OUTPUT: {}".format(output_csv))

    df_meta = extract_metadata_df_from_cha(cha_roots)

    cha_files = []
    for root in cha_roots:
        if not root or not os.path.exists(root):
            continue
        pattern = os.path.join(root, "**", "*.cha")
        cha_files.extend(glob.glob(pattern, recursive=True))
    cha_files = sorted(set(cha_files))

    pid_to_path = {}
    for path in cha_files:
        pid = os.path.splitext(os.path.basename(path))[0]
        if pid not in pid_to_path:
            pid_to_path[pid] = path

    print("\nPacientes con metadata: {}".format(len(df_meta)))
    print("Pacientes con .CHA indexado: {}".format(len(pid_to_path)))

    rows = []
    missing_cha = 0
    missing_transcript = 0

    for _, row in df_meta.iterrows():
        pid = row['patient_id']
        cha_path = pid_to_path.get(pid)

        if not cha_path or not os.path.exists(cha_path):
            missing_cha += 1
            continue

        transcript, utterances = extract_par_utterances_from_cha(cha_path)
        if not transcript:
            missing_transcript += 1
            continue

        rows.append({
            'patient_id': pid,
            'group': row['group'],
            'QA': row['QA'],
            'language': row['language'],
            'sex': row['sex'],
            'age': row['age'],
            'aphasia_type': row['aphasia_type'],
            'group_original': row['group_original'],
            'transcript': transcript
        })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        raise RuntimeError("No se pudo construir ningún registro con transcript + metadata.")

    df_out = df_out.sort_values('patient_id')

    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    print("RESUMEN DATASET BASE")
    print("=" * 70)
    print("Filas finales: {}".format(len(df_out)))
    print("Pacientes sin .CHA: {}".format(missing_cha))
    print("Pacientes sin transcript PAR: {}".format(missing_transcript))
    print("Guardado en: {}".format(output_csv))

    print("\nColumnas del dataset_base:")
    print(df_out.columns.tolist())

    return df_out

# ======================== MAIN CLI ========================

def main():
    parser = argparse.ArgumentParser(
        description="Construye data/dataset_base.csv desde los .CHA"
    )
    parser.add_argument(
        "--cha-roots",
        nargs="+",
        default=None,
        help="Carpetas con .CHA. Si no se indica, se usan DEFAULT_CHA_DIRS."
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "dataset_base.csv"),
        help="Ruta de salida para dataset_base.csv (por defecto data/dataset_base.csv)"
    )
    args = parser.parse_args()

    if args.cha_roots is not None and len(args.cha_roots) > 0:
        cha_roots = args.cha_roots
    else:
        if not DEFAULT_CHA_DIRS:
            raise ValueError(
                "DEFAULT_CHA_DIRS está vacío y no se han pasado --cha-roots.\n"
                "Indica las carpetas de .CHA con el parámetro --cha-roots."
            )
        cha_roots = DEFAULT_CHA_DIRS

    build_dataset_base(cha_roots, args.output)

if __name__ == "__main__":
    main()
