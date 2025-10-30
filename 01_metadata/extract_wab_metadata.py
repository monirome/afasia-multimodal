#!/usr/bin/env python3
# extract_wab_metadata.py
# -*- coding: utf-8 -*-
"""
Extrae WAB-AQ + IDIOMA desde .CHA usando pylangacq

Metodo reproducible desde archivos .CHA raw (no requiere CSV procesado)
"""

import os
import glob
import sys
import pandas as pd
import pylangacq as pla

# ======================== CONFIGURACION ========================
CHA_DIR = "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones"
OUTPUT_VALID = "../data/patient_metadata_WAB.csv"
OUTPUT_ERRORS = "../data/patient_metadata_errors.csv"

def extract_metadata_from_cha(cha_dir, output_valid, output_errors):
    """
    Extrae WAB-AQ + idioma desde archivos .CHA
    """
    print("="*70)
    print("EXTRACCION METADATA COMPLETA DESDE .CHA")
    print("="*70)
    
    # Buscar archivos
    print("\nBuscando archivos .CHA en: {}".format(cha_dir))
    cha_files = glob.glob(os.path.join(cha_dir, '**/*.cha'), recursive=True)
    print("Archivos encontrados: {}".format(len(cha_files)))
    
    if len(cha_files) == 0:
        print("\nERROR: No se encontraron archivos .CHA")
        sys.exit(1)
    
    results = []
    errors = []
    
    for i, filepath in enumerate(cha_files, 1):
        if i % 100 == 0:
            print("  Procesando {}/{}...".format(i, len(cha_files)))
        
        try:
            # Leer archivo
            ds = pla.read_chat(filepath)
            headers = ds.headers()
            
            if len(headers) == 0:
                errors.append({
                    'patient_id': os.path.splitext(os.path.basename(filepath))[0],
                    'error': 'Sin headers'
                })
                continue
            
            header = headers[0]
            
            # Extraer idioma desde @Languages
            lang = 'en'  # default
            if 'Languages' in header:
                langs = header['Languages']
                if isinstance(langs, list) and len(langs) > 0:
                    lang_raw = langs[0].lower()
                elif isinstance(langs, str):
                    lang_raw = langs.lower()
                else:
                    lang_raw = 'eng'
                
                # Normalizar
                if lang_raw in ['eng', 'english']:
                    lang = 'en'
                elif lang_raw in ['spa', 'spanish', 'espanol', 'esp']:
                    lang = 'es'
                elif lang_raw in ['cat', 'catalan', 'catala']:
                    lang = 'ca'
                else:
                    lang = 'en'
            
            # Extraer participant info
            if 'Participants' not in header:
                errors.append({
                    'patient_id': os.path.splitext(os.path.basename(filepath))[0],
                    'error': 'Sin Participants'
                })
                continue
            
            participants = header['Participants']
            
            if 'PAR' not in participants:
                errors.append({
                    'patient_id': os.path.splitext(os.path.basename(filepath))[0],
                    'error': 'Sin PAR'
                })
                continue
            
            par_info = participants['PAR']
            
            # Extraer patient_id
            patient_id = os.path.splitext(os.path.basename(filepath))[0]
            
            # Extraer sex
            sex = par_info.get('sex', '')
            if not sex or sex == '':
                sex = None
            
            # Extraer age
            age = par_info.get('age', '')
            if age and age != '':
                try:
                    age_str = str(age).split(';')[0].split('.')[0]
                    age = int(age_str) if age_str.isdigit() else None
                except:
                    age = None
            else:
                age = None
            
            # Extraer custom field (contiene tipo afasia y WAB-AQ)
            custom = par_info.get('custom', '')
            
            # Parsear custom para extraer aphasia_type y QA
            aphasia_type = None
            wab_aq = None
            
            if custom and custom != '':
                # Formato tipico: "Broca|65.5|" o "Anomic|82.3"
                parts = str(custom).split('|')
                
                if len(parts) >= 1 and parts[0]:
                    aphasia_type = parts[0].strip()
                
                if len(parts) >= 2 and parts[1]:
                    try:
                        wab_aq = float(parts[1])
                        # Validar rango
                        if not (0 <= wab_aq <= 100):
                            wab_aq = None
                    except:
                        wab_aq = None
            
            # Guardar resultado
            results.append({
                'patient_id': patient_id,
                'QA': wab_aq,
                'sex': sex,
                'age': age,
                'aphasia_type': aphasia_type,
                'language': lang
            })
            
        except Exception as e:
            errors.append({
                'patient_id': os.path.splitext(os.path.basename(filepath))[0],
                'error': str(e)
            })
    
    # Crear DataFrames
    df_results = pd.DataFrame(results)
    df_errors = pd.DataFrame(errors)
    
    # Filtrar pacientes con QA valido
    df_valid = df_results[df_results['QA'].notna()].copy()
    
    print("\n" + "="*70)
    print("RESULTADOS")
    print("="*70)
    print("Pacientes con metadata: {}".format(len(df_results)))
    print("Pacientes con QA valido: {}".format(len(df_valid)))
    print("Pacientes con errores: {}".format(len(df_errors)))
    
    if len(df_valid) > 0:
        print("\nDistribucion por idioma:")
        print(df_valid['language'].value_counts())
        
        print("\nEstadisticas WAB-AQ:")
        print("  Media: {:.1f}".format(df_valid['QA'].mean()))
        print("  Std:   {:.1f}".format(df_valid['QA'].std()))
        print("  Min:   {:.1f}".format(df_valid['QA'].min()))
        print("  Max:   {:.1f}".format(df_valid['QA'].max()))
        
        print("\nDistribucion por idioma + tipo de afasia:")
        for lang in df_valid['language'].unique():
            df_lang = df_valid[df_valid['language'] == lang]
            print("\n  {}:".format(lang.upper()))
            print("    N: {}".format(len(df_lang)))
            print("    QA: {:.1f} +/- {:.1f}".format(df_lang['QA'].mean(), df_lang['QA'].std()))
            if 'aphasia_type' in df_lang.columns:
                print("    Tipos mas comunes:")
                print(df_lang['aphasia_type'].value_counts().head(5).to_string().replace('\n', '\n      '))
        
        # Crear directorio de salida si no existe
        output_dir = os.path.dirname(output_valid)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("\nCreado directorio: {}".format(output_dir))
        
        # Guardar
        df_valid.to_csv(output_valid, index=False)
        print("\nGuardado en: {}".format(output_valid))
        
        # Mostrar primeras filas
        print("\nPrimeras 10 filas:")
        print(df_valid[['patient_id', 'QA', 'language', 'sex', 'age', 'aphasia_type']].head(10).to_string(index=False))
    else:
        print("\nNo se encontraron pacientes con QA valido")
    
    if len(df_errors) > 0:
        df_errors.to_csv(output_errors, index=False)
        print("\nErrores guardados en: {}".format(output_errors))
        print("\nEjemplos de errores (primeros 5):")
        for _, err in df_errors.head(5).iterrows():
            print("  {}: {}".format(err['patient_id'], err['error']))
    
    print("\n" + "="*70)
    return df_valid, df_errors

if __name__ == "__main__":
    extract_metadata_from_cha(CHA_DIR, OUTPUT_VALID, OUTPUT_ERRORS)