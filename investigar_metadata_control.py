#!/usr/bin/env python3
# investigar_metadata_control.py
"""
Investiga cómo están codificados los participantes Control en AphasiaBank
extrayendo directamente desde los .CHA files.
"""

import os
import glob
import pandas as pd
import pylangacq as pla

# ======================== CONFIGURACION ========================
CHA_DIR = "/lustre/ific.uv.es/ml/upc150/upc1503/data/transcripciones"

def extract_metadata_from_cha_files(cha_dir):
    """Extrae metadata de TODOS los .CHA para investigar grupos"""
    
    print("=" * 70)
    print("INVESTIGANDO METADATA CONTROL EN APHASIABANK")
    print("=" * 70)
    
    cha_files = glob.glob(os.path.join(cha_dir, '**/*.cha'), recursive=True)
    print(f"\nArchivos .CHA encontrados: {len(cha_files)}")
    
    if len(cha_files) == 0:
        print("ERROR: No se encontraron archivos .CHA")
        return None
    
    results = []
    
    for i, filepath in enumerate(cha_files, 1):
        if i % 100 == 0:
            print(f"  Procesando {i}/{len(cha_files)}...")
        
        try:
            ds = pla.read_chat(filepath)
            headers = ds.headers()
            
            if len(headers) == 0:
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
            continue
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("ANÁLISIS DE GRUPOS")
    print("=" * 70)
    
    print(f"\nTotal registros extraídos: {len(df)}")
    
    print("\n📊 Distribución por group_original:")
    print(df['group_original'].value_counts(dropna=False).head(20))
    
    print("\n📊 Distribución por idioma:")
    print(df['language'].value_counts())
    
    # Analizar Control específicamente
    print("\n" + "=" * 70)
    print("ANÁLISIS DETALLADO: CONTROL")
    print("=" * 70)
    
    # Buscar diferentes variantes de "control"
    control_variants = ['notaphasicbywab', 'control', 'typical', 'healthy', 'nonaphasic']
    
    for variant in control_variants:
        df_variant = df[df['group_original'] == variant]
        if len(df_variant) > 0:
            print(f"\n🟢 Grupo: '{variant}' (n={len(df_variant)})")
            
            # WAB-AQ stats
            if 'QA' in df_variant.columns:
                qa_values = df_variant['QA'].dropna()
                if len(qa_values) > 0:
                    print(f"  WAB-AQ:")
                    print(f"    Con valores: {len(qa_values)}/{len(df_variant)}")
                    print(f"    Media: {qa_values.mean():.2f}")
                    print(f"    Std: {qa_values.std():.2f}")
                    print(f"    Min: {qa_values.min():.2f}")
                    print(f"    Max: {qa_values.max():.2f}")
                    print(f"    Mediana: {qa_values.median():.2f}")
                else:
                    print(f"  WAB-AQ: Todos NaN")
            
            # Mostrar ejemplos
            print(f"\n  Ejemplos (primeros 5):")
            sample_cols = ['patient_id', 'QA', 'aphasia_type', 'language', 'age', 'sex']
            available_cols = [col for col in sample_cols if col in df_variant.columns]
            print(df_variant[available_cols].head(5).to_string(index=False))
    
    # Analizar PWA
    print("\n" + "=" * 70)
    print("ANÁLISIS DETALLADO: PWA")
    print("=" * 70)
    
    pwa_variants = ['anomic', 'broca', 'conduction', 'wernicke', 'global', 
                    'transmotor', 'transsensory', 'isolation', 'pwa']
    
    df_pwa = df[df['group_original'].isin(pwa_variants)]
    
    if len(df_pwa) > 0:
        print(f"\n🔴 Total PWA: {len(df_pwa)}")
        
        print("\n  Distribución por tipo de afasia:")
        print(df_pwa['group_original'].value_counts())
        
        qa_values = df_pwa['QA'].dropna()
        if len(qa_values) > 0:
            print(f"\n  WAB-AQ en PWA:")
            print(f"    Con valores: {len(qa_values)}/{len(df_pwa)}")
            print(f"    Media: {qa_values.mean():.2f}")
            print(f"    Std: {qa_values.std():.2f}")
            print(f"    Min: {qa_values.min():.2f}")
            print(f"    Max: {qa_values.max():.2f}")
            print(f"    Mediana: {qa_values.median():.2f}")
    
    # Recomendación final
    print("\n" + "=" * 70)
    print("RECOMENDACIÓN PARA COMMON VOICE")
    print("=" * 70)
    
    control_qa = df[df['group_original'] == 'notaphasicbywab']['QA'].dropna()
    
    if len(control_qa) > 0:
        print(f"\n✅ Los Control en AphasiaBank (notaphasicbywab) tienen:")
        print(f"   WAB-AQ media: {control_qa.mean():.2f}")
        print(f"   WAB-AQ rango: [{control_qa.min():.2f}, {control_qa.max():.2f}]")
        print(f"\n📝 Para Common Voice usar:")
        print(f"   wab_aq = None  (o dejar vacío)")
        print(f"   Razón: Los controles sanos NO necesitan evaluación WAB")
        print(f"   (El WAB-AQ solo se aplica a pacientes con sospecha de afasia)")
    else:
        print("\n⚠️  No se encontraron valores WAB-AQ en Control")
        print("\n📝 Para Common Voice usar:")
        print("   wab_aq = None")
        print("   aphasia_type = None")
        print("   severity = None")
    
    print("\n   Campos obligatorios para Control:")
    print("   - group = 'control' (o 'Control')")
    print("   - wab_aq = None")
    print("   - aphasia_type = None")
    print("   - severity = None")
    
    return df


if __name__ == "__main__":
    df = extract_metadata_from_cha_files(CHA_DIR)
    
    if df is not None:
        # Guardar para referencia
        output_csv = "investigacion_metadata_control.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Datos guardados en: {output_csv}")