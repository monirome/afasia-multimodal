#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrae WAB-AQ desde .CHA usando pylangacq
"""

import os
import glob
import pandas as pd
import pylangacq as pla

def extract_wab_from_cha_correct(cha_dir):
    """
    Extrae WAB-AQ desde archivos .CHA usando pylangacq
    """
    print("="*70)
    print("EXTRACCIÓN WAB-AQ DESDE .CHA (usando pylangacq)")
    print("="*70)
    
    # Buscar archivos
    print(f"\nBuscando archivos .CHA en: {cha_dir}")
    cha_files = glob.glob(os.path.join(cha_dir, '**/*.cha'), recursive=True)
    print(f"Archivos encontrados: {len(cha_files)}")
    
    results = []
    errors = []
    
    for i, filepath in enumerate(cha_files, 1):
        if i % 100 == 0:
            print(f"  Procesando {i}/{len(cha_files)}...")
        
        try:
            # Leer archivo
            ds = pla.read_chat(filepath)
            headers = ds.headers()
            
            if len(headers) > 0 and 'Participants' in headers[0]:
                participants = headers[0]['Participants']
                
                if 'PAR' in participants:
                    par_info = participants['PAR']
                    
                    # Extraer información
                    patient_id = os.path.splitext(os.path.basename(filepath))[0]
                    wab_aq = par_info.get('custom', None)
                    sex = par_info.get('sex', None)
                    age = par_info.get('age', None)
                    aphasia_type = par_info.get('group', None)
                    
                    # Limpiar WAB-AQ (puede ser string)
                    if wab_aq is not None:
                        try:
                            wab_aq = float(wab_aq)
                            # Validar rango
                            if 0 <= wab_aq <= 100:
                                results.append({
                                    'patient_id': patient_id,
                                    'QA': wab_aq,
                                    'sex': sex,
                                    'age': age if age else None,
                                    'aphasia_type': aphasia_type
                                })
                            else:
                                errors.append({'patient_id': patient_id, 'error': f'WAB-AQ fuera de rango: {wab_aq}'})
                        except (ValueError, TypeError):
                            errors.append({'patient_id': patient_id, 'error': f'WAB-AQ no numérico: {wab_aq}'})
                    else:
                        errors.append({'patient_id': patient_id, 'error': 'Campo custom vacío'})
        
        except Exception as e:
            patient_id = os.path.splitext(os.path.basename(filepath))[0]
            errors.append({'patient_id': patient_id, 'error': str(e)})
    
    return results, errors

def main():
    cha_dir = '/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones'
    
    results, errors = extract_wab_from_cha_correct(cha_dir)
    
    print(f"\n{'='*70}")
    print("RESULTADOS")
    print("="*70)
    print(f"Pacientes con WAB-AQ válido: {len(results)}")
    print(f"Pacientes con errores: {len(errors)}")
    
    if len(results) > 0:
        # Crear DataFrame
        df = pd.DataFrame(results)
        
        # Limpiar edad (quitar meses)
        if 'age' in df.columns:
            df['age'] = df['age'].astype(str).str[:2]
        
        # Ordenar
        df = df.sort_values('patient_id')
        
        # Guardar
        df.to_csv('patient_metadata_WAB.csv', index=False)
        print(f"\n✓ Guardado en: patient_metadata_WAB.csv")
        
        # Estadísticas
        print(f"\nEstadísticas WAB-AQ:")
        print(f"  Media: {df['QA'].mean():.1f}")
        print(f"  Std:   {df['QA'].std():.1f}")
        print(f"  Min:   {df['QA'].min():.1f}")
        print(f"  Max:   {df['QA'].max():.1f}")
        
        print(f"\nDistribución por tipo de afasia:")
        print(df['aphasia_type'].value_counts())
        
        print(f"\nPrimeros 10 pacientes:")
        print(df[['patient_id', 'QA', 'sex', 'age', 'aphasia_type']].head(10).to_string(index=False))
    else:
        print("\n No se encontraron pacientes con WAB-AQ válido")
    
    # Guardar errores para inspección
    if len(errors) > 0:
        pd.DataFrame(errors).to_csv('patients_wab_errors.csv', index=False)
        print(f"\n Errores guardados en: patients_wab_errors.csv")
        print(f"\n Ejemplos de errores (primeros 5):")
        for err in errors[:5]:
            print(f"  {err['patient_id']}: {err['error']}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
