#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrae scores WAB-AQ desde archivos .CHA
"""

import os
import re
import glob
import pandas as pd

def extract_wab_from_cha(filepath):
    """
    Busca WAB-AQ en un archivo .CHA
    
    Patrones comunes:
    - WAB AQ: 85.2
    - AQ: 72.5
    - Aphasia Quotient: 45.8
    - WAB-AQ: 91.3
    """
    patient_id = os.path.splitext(os.path.basename(filepath))[0]
    wab_score = None
    
    patterns = [
        r'WAB[-\s]*AQ\s*[:=]\s*([0-9]+\.?[0-9]*)',
        r'AQ\s*[:=]\s*([0-9]+\.?[0-9]*)',
        r'Aphasia\s+Quotient\s*[:=]\s*([0-9]+\.?[0-9]*)',
        r'WAB\s*[:=]\s*([0-9]+\.?[0-9]*)',
        r'QA\s*[:=]\s*([0-9]+\.?[0-9]*)',
    ]
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Buscar con cada patrón
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    # Validar rango (WAB-AQ: 0-100)
                    if 0 <= score <= 100:
                        wab_score = score
                        break
        
        return patient_id, wab_score, content[:500] if wab_score is None else None
    
    except Exception as e:
        return patient_id, None, str(e)

def main():
    cha_dir = '/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones'
    
    print("="*70)
    print("EXTRACCIÓN WAB-AQ DESDE .CHA")
    print("="*70)
    
    # Buscar archivos
    print(f"\nBuscando archivos .CHA en: {cha_dir}")
    cha_files = glob.glob(os.path.join(cha_dir, '**/*.cha'), recursive=True)
    print(f"Archivos encontrados: {len(cha_files)}")
    
    # Extraer scores
    results = []
    no_score = []
    
    for i, filepath in enumerate(cha_files, 1):
        if i % 100 == 0:
            print(f"  Procesando {i}/{len(cha_files)}...")
        
        patient_id, score, preview = extract_wab_from_cha(filepath)
        
        if score is not None:
            results.append({'patient_id': patient_id, 'QA': score})
        else:
            no_score.append({'patient_id': patient_id, 'preview': preview})
    
    # Resultados
    print(f"\n{'='*70}")
    print("RESULTADOS")
    print("="*70)
    print(f"Pacientes con WAB-AQ: {len(results)}")
    print(f"Pacientes sin WAB-AQ: {len(no_score)}")
    
    if len(results) > 0:
        df = pd.DataFrame(results).sort_values('patient_id')
        df.to_csv('patient_metadata_WAB.csv', index=False)
        print(f"\n✓ Guardado en: patient_metadata_WAB.csv")
        
        # Estadísticas
        print(f"\nEstadísticas WAB-AQ:")
        print(f"  Media: {df['QA'].mean():.1f}")
        print(f"  Std:   {df['QA'].std():.1f}")
        print(f"  Min:   {df['QA'].min():.1f}")
        print(f"  Max:   {df['QA'].max():.1f}")
        
        print(f"\nPrimeros 10 pacientes:")
        print(df.head(10).to_string(index=False))
    else:
        print("\n No se encontraron scores WAB-AQ")
    
    # Guardar pacientes sin score para inspección
    if len(no_score) > 0:
        pd.DataFrame(no_score).to_csv('patients_no_wab.csv', index=False)
        print(f"\nPacientes sin score guardados en: patients_no_wab.csv")
        print(f"\nEjemplos de archivos sin score (primeros 5):")
        for item in no_score[:5]:
            print(f"\n  {item['patient_id']}:")
            if item['preview'] and len(item['preview']) > 0:
                print(f"    Preview: {item['preview'][:200]}...")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
