#!/usr/bin/env python3
# utils/detect_group_utils.py
# -*- coding: utf-8 -*-
"""
Funciones auxiliares para detectar Control vs PWA
Minimalista - solo lo esencial
"""

import pylangacq as pla

def detect_group_from_cha(filepath):
    """
    Detecta si es Control o PWA desde archivo .CHA
    Returns: 'Control', 'PWA', o 'Unknown'
    """
    try:
        ds = pla.read_chat(filepath)
        headers = ds.headers()
        
        if not headers:
            return 'PWA'  # Default
        
        header = headers[0]
        
        if 'Participants' in header and 'PAR' in header['Participants']:
            par_info = header['Participants']['PAR']
            custom = par_info.get('custom', '')
            
            if custom and custom != '':
                # Formato: "tipo_afasia|score|..."
                aphasia_type = str(custom).split('|')[0].strip().lower()
                
                # Es control si no hay tipo de afasia
                if aphasia_type in ['control', 'none', '', 'healthy', 'normal']:
                    return 'Control'
                # Si hay cualquier tipo de afasia, es PWA
                elif aphasia_type:
                    return 'PWA'
        
        # Si no hay info de afasia, probablemente es control
        return 'Control' if not custom else 'PWA'
    
    except:
        return 'PWA'  # Default en caso de error