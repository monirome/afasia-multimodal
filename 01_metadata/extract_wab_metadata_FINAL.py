#!/usr/bin/env python3
# extract_wab_metadata_FINAL.py
# -*- coding: utf-8 -*-
"""
Extrae metadata desde .CHA con LOGICA CORREGIDA:
- Control = notaphasicbywab (QA >= 93.8)
- PWA = todos los tipos de afasia
"""

import os
import glob
import sys
import pandas as pd
import pylangacq as pla

# ======================== CONFIGURACION ========================
CHA_DIR = "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones"
OUTPUT_CSV = "../data/patient_metadata_WAB.csv"
OUTPUT_ERRORS = "../data/patient_metadata_errors.csv"

def extract_metadata_from_cha(cha_dir, output_csv, output_errors):
    """Extrae metadata TAL CUAL viene en el .CHA"""
    
    print("="*70)
    print("EXTRACCION METADATA DESDE .CHA - VERSION FINAL")
    print("="*70)
    
    cha_files = glob.glob(os.path.join(cha_dir, '**/*.cha'), recursive=True)
    print("\nArchivos .CHA encontrados: {}".format(len(cha_files)))
    
    if len(cha_files) == 0:
        print("\nERROR: No se encontraron archivos .CHA")
        sys.exit(1)
    
    results = []
    errors = []
    
    for i, filepath in enumerate(cha_files, 1):
        if i % 100 == 0:
            print("  Procesando {}/{}...".format(i, len(cha_files)))
        
        try:
            ds = pla.read_chat(filepath)
            headers = ds.headers()
            
            if len(headers) == 0:
                errors.append({'patient_id': os.path.splitext(os.path.basename(filepath))[0],
                              'error': 'Sin headers'})
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
            
            # Participant
            if 'Participants' not in header or 'PAR' not in header['Participants']:
                errors.append({'patient_id': os.path.splitext(os.path.basename(filepath))[0],
                              'error': 'Sin PAR'})
                continue
            
            par_info = header['Participants']['PAR']
            patient_id = os.path.splitext(os.path.basename(filepath))[0]
            
            # GRUPO (raw)
            group_raw = par_info.get('group', '').strip().lower()
            
            # Otros campos
            sex = par_info.get('sex', '').strip() or None
            age_raw = par_info.get('age', '').strip()
            age = None
            if age_raw:
                try:
                    age = int(age_raw.split(';')[0].split('.')[0])
                except:
                    pass
            
            # WAB-AQ desde custom
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
                    except:
                        if part and not part.replace('.', '').isdigit():
                            aphasia_type = part
            
            results.append({
                'patient_id': patient_id,
                'group_original': group_raw if group_raw else None,
                'QA': wab_aq,
                'sex': sex,
                'age': age,
                'aphasia_type': aphasia_type,
                'language': lang
            })
            
        except Exception as e:
            errors.append({'patient_id': os.path.splitext(os.path.basename(filepath))[0],
                          'error': str(e)})
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    df_errors = pd.DataFrame(errors)
    
    # ==================== SIMPLIFICAR GRUPOS ====================
    print("\n" + "="*70)
    print("SIMPLIFICANDO GRUPOS: Control vs PWA")
    print("="*70)
    
    def simplify_group(row):
        """
        Control = notaphasicbywab (los únicos con QA ~95-100)
        PWA = todos los tipos de afasia
        """
        g = row['group_original']
        
        # Control verdadero
        if g == 'notaphasicbywab':
            return 'control'
        
        # PWA = tipos de afasia + pwa genérico
        elif g in ['anomic', 'broca', 'conduction', 'wernicke', 'global',
                   'transmotor', 'transsensory', 'isolation', 'pwa']:
            return 'pwa'
        
        # "control" con QA bajo → probablemente PWA mal etiquetado
        elif g == 'control':
            return 'pwa'  # Según análisis, estos tienen QA ~22
        
        # Sin grupo
        else:
            return None
    
    df['group'] = df.apply(simplify_group, axis=1)
    
    print("\nDistribución ORIGINAL:")
    print(df['group_original'].value_counts(dropna=False).head(15))
    
    print("\nDistribución SIMPLIFICADA:")
    print(df['group'].value_counts(dropna=False))
    
    # Estadísticas por grupo
    df_with_qa = df[df['QA'].notna()]
    
    if len(df_with_qa) > 0:
        print("\n" + "="*70)
        print("ESTADISTICAS WAB-AQ POR GRUPO")
        print("="*70)
        
        for grp in ['control', 'pwa']:
            df_grp = df_with_qa[df_with_qa['group'] == grp]
            if len(df_grp) > 0:
                print("\n{}:".format(grp.upper()))
                print("  N: {}".format(len(df_grp)))
                print("  QA: {:.1f} ± {:.1f}".format(df_grp['QA'].mean(), df_grp['QA'].std()))
                print("  Range: [{:.1f}, {:.1f}]".format(df_grp['QA'].min(), df_grp['QA'].max()))
                
                # Validación
                qa_mean = df_grp['QA'].mean()
                if grp == 'control':
                    if qa_mean >= 93.8:
                        print("  ✓ CORRECTO: QA >= 93.8 (cutoff clínico)")
                    else:
                        print("  ⚠ WARNING: QA < 93.8 (esperado >= 93.8)")
                else:  # pwa
                    if qa_mean < 93.8:
                        print("  ✓ CORRECTO: QA < 93.8 (indica afasia)")
                    else:
                        print("  ⚠ WARNING: QA >= 93.8 (muy alto para PWA)")
    
    # Guardar
    print("\n" + "="*70)
    print("GUARDANDO RESULTADOS")
    print("="*70)
    
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Reordenar columnas
    cols = ['patient_id', 'group', 'QA', 'language', 'sex', 'age', 'aphasia_type', 'group_original']
    df = df[cols]
    
    df.to_csv(output_csv, index=False)
    print("\nMetadata guardada: {}".format(output_csv))
    print("  Total: {}".format(len(df)))
    print("  Con grupo: {}".format(df['group'].notna().sum()))
    print("  Con WAB-AQ: {}".format(df['QA'].notna().sum()))
    
    if len(df_errors) > 0:
        df_errors.to_csv(output_errors, index=False)
    
    print("\n" + "="*70)
    print("EJEMPLOS (primeras 20)")
    print("="*70)
    print(df[['patient_id', 'group', 'QA', 'language', 'group_original']].head(20).to_string(index=False))
    
    print("\n" + "="*70)
    print("COMPLETADO")
    print("="*70)
    
    return df, df_errors

if __name__ == "__main__":
    extract_metadata_from_cha(CHA_DIR, OUTPUT_CSV, OUTPUT_ERRORS)
