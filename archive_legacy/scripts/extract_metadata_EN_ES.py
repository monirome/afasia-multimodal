#!/usr/bin/env python3
# extract_metadata_EN_ES.py
# -*- coding: utf-8 -*-

"""
Extrae metadata (QA + idioma) desde .CHA para EN y ES
"""
import os
import glob
import pandas as pd
import pylangacq as pla

def extract_en_es():
    print("="*70)
    print("EXTRACCIÓN METADATA EN + ES DESDE .CHA")
    print("="*70)
    
    # Directorios
    dirs = [
        "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones",  # EN
        "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones/Spanish"  # ES
    ]
    
    cha_files = []
    for d in dirs:
        if os.path.exists(d):
            cha_files.extend(glob.glob(os.path.join(d, '**/*.cha'), recursive=True))
    
    print(f"\nArchivos .CHA encontrados: {len(cha_files)}")
    
    results = []
    
    for i, filepath in enumerate(cha_files, 1):
        if i % 100 == 0:
            print(f"  Procesando {i}/{len(cha_files)}...")
        
        try:
            ds = pla.read_chat(filepath)
            headers = ds.headers()
            
            if len(headers) == 0 or 'Participants' not in headers[0]:
                continue
            
            header = headers[0]
            
            # Detectar idioma
            lang = 'en'
            if 'Languages' in header:
                langs = header['Languages']
                lang_raw = langs[0].lower() if isinstance(langs, list) else str(langs).lower()
                
                if lang_raw in ['spa', 'spanish', 'español', 'esp']:
                    lang = 'es'
                else:
                    lang = 'en'
            
            # Detectar idioma por path si no está en header
            if 'Spanish' in filepath or 'spanish' in filepath:
                lang = 'es'
            
            participants = header['Participants']
            if 'PAR' not in participants:
                continue
            
            par_info = participants['PAR']
            patient_id = os.path.splitext(os.path.basename(filepath))[0]
            
            # Extraer datos
            sex = par_info.get('sex', None)
            age_raw = par_info.get('age', '')
            age = None
            if age_raw:
                try:
                    age = int(str(age_raw).split(';')[0])
                except:
                    pass
            
            custom = par_info.get('custom', '')
            aphasia_type = None
            wab_aq = None
            
            if custom:
                parts = str(custom).split('|')
                if len(parts) >= 1 and parts[0]:
                    aphasia_type = parts[0].strip()
                if len(parts) >= 2 and parts[1]:
                    try:
                        wab_aq = float(parts[1])
                        if not (0 <= wab_aq <= 100):
                            wab_aq = None
                    except:
                        pass
            
            results.append({
                'patient_id': patient_id,
                'QA': wab_aq,
                'sex': sex,
                'age': age,
                'aphasia_type': aphasia_type,
                'language': lang
            })
            
        except Exception as e:
            continue
    
    df = pd.DataFrame(results)
    df = df[df['QA'].notna()]
    
    print(f"\n{'='*70}")
    print("RESULTADOS")
    print("="*70)
    print(f"Total pacientes con QA: {len(df)}")
    print(f"\nDistribución por idioma:")
    print(df['language'].value_counts())
    
    for lang in df['language'].unique():
        df_lang = df[df['language'] == lang]
        print(f"\n{lang.upper()}:")
        print(f"  N: {len(df_lang)}")
        print(f"  QA: {df_lang['QA'].mean():.1f} ± {df_lang['QA'].std():.1f}")
    
    df.to_csv('metadata_EN_ES_from_CHA.csv', index=False)
    print(f"\n✓ Guardado: metadata_EN_ES_from_CHA.csv")
    
    return df

if __name__ == "__main__":
    extract_en_es()
