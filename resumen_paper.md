# Metodología del paper (Le et al., 2018) 

## Objetivo

Predecir WAB-AQ (Western Aphasia Battery Aphasia Quotient) usando features lingüísticas extraídas de transcripciones automáticas.

---

## 1. Datos (paper)

**Dataset:** AphasiaBank  
**Idioma:** solo inglés  
**Total de participantes:** ~600

### Distribución por grupos

| Grupo | Nº aprox. de hablantes | Protocolo | Rango WAB-AQ | Sub-datasets |
|-------|------------------------|-----------|--------------|--------------|
| PWA (Afasia) | ~530 | Cinderella | 0–100 | Aphasia, English, Fridriksson, Kurland, Wright |
| Control | ~70 | Cinderella | 93.8–100 | – |

---

## 2. Transcripciones (paper)

**Método:** forced alignment con Kaldi ASR

- **Input:** audio + transcripciones CHAT (`.cha`)
- **Output:** marcas temporales a nivel de palabra (word-level timestamps)
- No utilizan WhisperX 

---

## 3. Features (130 en total, paper)

### Grupos de features

| Grupo   | Nº features | Descripción breve |
|---------|------------:|-------------------|
| DEN (Density)   | 42 | Palabras/minuto, ratios POS, longitud de enunciados (utterances) |
| DYS (Dysfluency) | 22 | Fillers, pausas, duración de pausas |
| LEX (Lexical)    | 66 | TTR, frecuencia, imageability, Age of Acquisition (AoA), familiarity |
| POS-LM          | 26 | Cross-entropy de modelos POS bigrama/trigrama |
| PVE             | 52 | Variabilidad acústica en repeticiones |
| DTW             | 39 | Distancia acústica de posteriorgrams con DTW |

### Estadísticas por métrica

Se aplican 13 estadísticas a 17 métricas base del tipo `{X}`:

```text
{X} → min, p10, q1, median, q3, p90, max, mean, std, skew, kurt, iqr, mad
```

---

## 4. Feature selection (paper)

**Método:** Sequential Forward Selection (SFS)

1. Iniciar con un conjunto vacío de features.
2. Probar cada feature individualmente.
3. Añadir la feature que más mejora el rendimiento en CV (MAE).
4. Repetir mientras añadir features siga mejorando el rendimiento.
5. Resultado final: ~42 features seleccionadas de las 130.

- Implementación propia (no detallan librería).
- Reducción: 130 → ~42 features “óptimas”.

---

## 5. Modelo (paper)

- **Algoritmo:** Support Vector Regression (SVR)
- **Kernel:** RBF (Radial Basis Function)

### Hiperparámetros explorados

(Esquema aproximado reportado en el paper):

```text
svr__C:        [0.1, 1, 10, 100, 1000]
svr__epsilon:  [0.01, 0.1, 1]
svr__gamma:    ['scale', 'auto']
```

**Optimización:** Grid Search con validación cruzada 5-fold.

---

## 6. Metodología de evaluación (paper)

### Split de datos

```text
TRAINING SET:
  - Control: 100% (siempre en training)
  - PWA: 75% (seleccionados por CV)

TEST SET:
  - PWA: 25% (withhold por CV)
```

### Cross-validation

- **Método:** 4-fold Stratified GroupKFold
- **Grupos:** `patient_id` (el mismo paciente nunca está en train y test a la vez).
- **Estratificación:** por sub-dataset (Aphasia, English, Fridriksson, Kurland, Wright).

Proceso por fold:

```text
Fold 1:
  - Train: PWA (75%) + Control (100%) → entrenar SVR
  - Test:  PWA (25%) → predecir
  - Métrica: MAE

Repetir para 4 folds.

CV final: media de los 4 MAEs.
```

---

## 7. Métricas de evaluación (paper)

### Regresión

- MAE (Mean Absolute Error) – métrica principal.
- RMSE (Root Mean Squared Error).
- Pearson r (correlación lineal).

### Clasificación por severidad

Bins de severidad:

```text
Very Severe:  0–25
Severe:      25–50
Moderate:    50–75
Mild:        75–100
```

Métricas de clasificación:

- Accuracy.
- Precision, Recall, F1 por clase.

---

## 8. Resultados del paper

```text
MAE:       8–10 (aprox.; reportan sobre todo RMSE)
RMSE:      ~12–14
Pearson r: 0.75–0.85
Features:  ~42 (tras feature selection)
```

Nota: el paper no reporta MAE explícitamente; se infiere a partir de RMSE y correlaciones.

---

# NUESTRO ESTUDIO 

## Objetivo

Mismo objetivo que el paper: predecir WAB-AQ a partir de features cuantitativas extraídas de transcripciones automáticas.

---

## 1. Datos (NUESTRO)

**Dataset:** AphasiaBank (multilingüe: inglés, español, catalán)  
**Total de participantes:** 506

### Distribución por grupos

| Grupo  | Nº hablantes | Protocolos principales        | Rango WAB-AQ | Idiomas                    |
|--------|-------------:|------------------------------|--------------|----------------------------|
| PWA    | 421          | Cinderella, Window, otros    | 0.2–93.6     | English, Spanish, Catalan  |
| Control| 70           | –                            | 93.8–100     | English, Spanish, Catalan  |

### Distribución por severidad (solo PWA)

| Severidad    | Rango WAB-AQ | Nº PWA | % aproximado |
|--------------|--------------|-------:|-------------:|
| Very Severe  | 0–25         | 44     | 10%          |
| Severe       | 25–50        | 55     | 13%          |
| Moderate     | 50–75        | 156    | 37%          |
| Mild         | 75–100       | 166    | 39%          |

Diferencias frente al paper:

- Multilingüe (EN/ES/CA) en lugar de solo inglés.
- Menos muestras PWA (421 frente a ~530).

---

## 2. Transcripciones 

**Método:** forced alignment con WhisperX.

- Input: audio + transcripciones CHAT (`.cha`).
- Output: marcas temporales a nivel de palabra.
- Modelo base: `large-v2` de Whisper.
- Modelos de alineación: EN y ES (CA haciendo fallback a ES).

Features acústicas extraídas (para tenerlas disponibles, aunque no se usan en las features lingüísticas finales):

- MFCC (39 dimensiones): 12 coeficientes + energía + primeras y segundas derivadas (Δ, ΔΔ).
- MFB (40 dimensiones): log mel-filterbanks.

Diferencias frente al paper:

- WhisperX es más moderno y robusto que Kaldi.
- Pipeline multilingüe (EN/ES/CA).

---

## 3. Features 

Has implementado 65 features en total, centradas en DEN, DYS y una parte mínima de LEX.

### Cobertura frente al paper

| Grupo   | # features implementadas | Cobertura frente al paper |
|---------|-------------------------:|----------------------------|
| DEN     | 42                      | 100% (42/42)               |
| DYS     | 22                      | 100% (22/22)               |
| LEX     | 1                       | 1.5% (1/66)                |
| POS-LM  | 0                       | 0% (0/26)                  |
| PVE     | 0                       | 0% (0/52)                  |
| DTW     | 0                       | 0% (0/39)                  |
| **Total** | **65**                 | **50% (65/130)**           |

### Detalle de grupos implementados

**DEN (42 features) – implementado al 100 %**

Ejemplos de variables:

- Ratios globales:
  - `den_words_per_min`
  - `den_phones_per_min`
  - `den_W` (proporción de palabras)
  - `den_OCW` (open-class words)
- Estadísticas de longitud de enunciado:
  - `den_words_utt_{13 stats}` (13 estadísticas sobre nº de palabras por enunciado).
  - `den_phones_utt_{13 stats}` (13 estadísticas sobre nº de fonemas por enunciado).
- Ratios POS:
  - `den_nouns`
  - `den_verbs`
  - `den_prepositions`
  - `den_determiners`
  - `den_light_verbs`
  - `den_function_words`
  - etc. (hasta cubrir ~12 ratios POS).

**DYS (22 features) – implementado al 100 %**

Ejemplos de variables:

- Ratios de disfluencias:
  - `dys_fillers_per_min`
  - `dys_fillers_per_word`
  - `dys_fillers_per_phone`
  - `dys_pauses_per_min`
  - otros ratios derivados (hasta 9 ratios).
- Duración de pausas:
  - `dys_pause_sec_{13 stats}` (13 estadísticas sobre duración de pausas).

**LEX (1 feature) – implementación mínima**

- `lex_ttr` (Type-Token Ratio).

Quedan pendientes (no implementadas todavía):

- `{lex_freq_{13 stats}}` → requiere frecuencias léxicas (SUBTLEX u otros).
- `{lex_img_{13 stats}}` → imageability (MRC Database u otros).
- `{lex_aoa_{13 stats}}` → Age of Acquisition (Brysbaert).
- `{lex_fam_{13 stats}}` → familiarity (MRC Database u otros).
- `{lex_phones_{13 stats}}` → longitud fonémica (CMUdict, etc.).

### Scripts principales

- `build_den_dys.py` → genera DEN + DYS.
- `build_lex_SIMPLE.py` → genera el TTR de LEX.

Diferencias frente al paper:

- DEN y DYS replicados por completo (exactamente igual que en el paper)
- LEX solo incluye TTR.
- POS-LM, PVE y DTW no implementados (por complejidad y/o dependencia de recursos externos).

---

## 4. Feature selection y experimentos 

Lo que he hecho hasta ahora:

- No has aplicado feature selection automática (no hay SFS implementado en la pipeline actual).
- Has probado dos configuraciones de features:

### Experimento 1: 29 features “simples”

Selección manual (sin las 13 estadísticas completas), centrada en ratios y medias:

- 18 features DEN.
- 10 features DYS.
- 1 feature LEX (`lex_ttr`).

Resultados:

```text
MAE:   12.29
RMSE:  16.70
R²:    0.505
r (Pearson): 0.710
Acc@5: 26.6%
Accuracy severidad: 56.8%
```

### Experimento 2: 65 features completas

Uso de todas las features disponibles de DEN, DYS y LEX:

- 42 DEN.
- 22 DYS.
- 1 LEX.

Resultados:

```text
MAE:   14.55
RMSE:  18.85
R²:    0.369
r (Pearson): 0.607
Acc@5: 21.4%
Accuracy severidad: 51.8%
```

Comentario: al añadir más features sin selección previa se observa sobreajuste y empeoramiento de métricas.

Diferencia clave frente al paper:

- El paper sí usa SFS para seleccionar ~42 features óptimas.
- La pipeline actual usa todas las features disponibles sin selección automática.

---

## 5. Modelo 

- **Algoritmo:** Support Vector Regression (SVR).
- **Kernel:** RBF, con comparación puntual con kernel lineal.

### Pipeline

En términos de scikit-learn:

```text
[SimpleImputer(strategy='median')]
→ [StandardScaler]
→ [SVR]
```

### Hiperparámetros explorados

```text
svr__C:        [1, 10, 100]
svr__epsilon:  [0.1, 1]
svr__kernel:   ['rbf', 'linear']
svr__shrinking:[True, False]
```

Total de combinaciones: 24.

Optimización: Grid Search con validación cruzada (CV=5) anidada dentro de los folds de GroupKFold.

Diferencias frente al paper:

- Mismo modelo base (SVR con kernel RBF).
- Menos valores de `C` explorados que en el paper original.

---

## 6. Metodología de evaluación 

### Split de datos

Has replicado la lógica del paper:

```text
TRAINING SET (por fold):
  - Control: 70 (100% siempre en training)
  - PWA:     ~315 (75% del total PWA)

TEST SET (por fold):
  - PWA: ~105 (25% del total PWA)
```

- Los sujetos control nunca aparecen en el conjunto de test.
- Solo PWA en test y en las métricas finales (como en el paper).

### Cross-validation

- **Método:** 4-fold GroupKFold.
- **Grupos:** `patient_id` (asegura que cada paciente solo aparece en train o test, pero no en ambos).

Proceso por fold:

```text
Fold 1:
  - Train: PWA (315) + Control (70) = 385
  - Test:  PWA (105)
  - Dentro del fold: Grid Search con CV=5 (solo en training)
  - Métrica principal de selección: MAE

Repetido para 4 folds.

CV final: promedio de los 4 MAEs.
```

Diferencias frente al paper:

- Mismo número de folds (4).
- Misma lógica de grupos por paciente.
- No se ha estratificado por sub-dataset.

---

## 7. Post-procesamiento 

### Calibración

He aplicado calibración posterior sobre las predicciones de WAB-AQ:

- Método: Isotonic Regression.
- Entrenamiento: sobre predicciones obtenidas vía CV.
- Ajuste final: se recortan las predicciones al rango [0, 100].

Efecto:

- Ligera mejora en algunas métricas de error.
- El paper no menciona ninguna técnica de calibración explícita.

---

## 8. Métricas de evaluación 

### Regresión

Has utilizado las mismas métricas que el paper, y algunas adicionales:

- MAE.
- RMSE.
- Pearson r.
- Spearman ρ (añadida por ti).

### Clasificación por severidad

Mismas bandas que el paper:

```text
Very Severe:  0–25
Severe:      25–50
Moderate:    50–75
Mild:        75–100
```

Métricas calculadas:

- Accuracy global.
- Precision, Recall y F1 por clase.
- Acc@1, Acc@5, Acc@10 (magnitud del error en puntos WAB).

### Outputs adicionales

- Matrices de confusión.
- Diagramas de dispersión predicho vs real.
- Permutation importance.
- Valores SHAP para interpretabilidad.

---

## 9. Resultados 

### Resumen de resultados por experimento

| Experimento | Features usadas                            | MAE   | RMSE  | R²    | Pearson r | Acc@5 | Accuracy severidad |
|------------:|---------------------------------------------|------:|------:|------:|----------:|------:|-------------------:|
| 1           | 29 (DEN + DYS simples + `lex_ttr`)          | 12.29 | 16.70 | 0.505 | 0.710     | 26.6% | 56.8%              |
| 2           | 65 (todas las DEN/DYS + `lex_ttr`)          | 14.55 | 18.85 | 0.369 | 0.607     | 21.4% | 51.8%              |

Comparado con el paper:

- El paper reporta MAE aproximado 8–10, RMSE ~12–14 y Pearson r entre 0.75 y 0.85.
- Nuestros resultados actuales están claramente por debajo de esos valores, aunque la correlación es razonable.

### Importancia de variables (top 5)

| Ranking | Feature              | Interpretación                          |
|--------:|----------------------|-----------------------------------------|
| 1       | `lex_ttr`           | Diversidad léxica (Type-Token Ratio)    |
| 2       | `den_determiners`   | Proporción de determinantes             |
| 3       | `den_function_words`| Proporción de palabras funcionales      |
| 4       | `den_prepositions`  | Proporción de preposiciones             |
| 5       | `den_light_verbs`   | Proporción de light verbs               |

---

# Comparación lado a lado

| Aspecto              | Paper (Le et al., 2018)                  | Nuestra implementación                               |
|----------------------|------------------------------------------|------------------------------------------------|
| Dataset total        | ~600 speakers (solo EN)                  | 506 speakers (EN/ES/CA)                         |
| PWA                  | ~530                                     | 421                                            |
| Control              | ~70                                      | 70                                             |
| Transcripción        | Kaldi ASR                                | WhisperX                                       |
| Features totales     | 130                                      | 65 (50%)                                       |
| DEN                  | 42                                       | 42                                             |
| DYS                  | 22                                       | 22                                             |
| LEX                  | 66                                       | 1                                              |
| POS-LM               | 26                                       | 0                                              |
| PVE                  | 52                                       | 0                                              |
| DTW                  | 39                                       | 0                                              |
| Feature selection    | SFS (Sequential Forward Selection)       | No aplicado (todas las features)               |
| Modelo               | SVR + RBF                                | SVR + RBF                                      |
| CV                   | 4-fold Stratified GroupKFold (subsets)   | 4-fold GroupKFold (por paciente, sin subsets)  |
| Control en train     | Siempre                                  | Siempre                                        |
| MAE                  | ~8–10                                    | 12.29 (29 feat) / 14.55 (65 feat)              |
| R²                   | ~0.75                                    | 0.505 / 0.369                                  |
| Pearson r            | 0.75–0.85                                | 0.710 / 0.607                                  |

---

# Conclusiones y próximos pasos

## Fortalezas de nuestra implementación

- Features DEN y DYS replicadas al 100 % respecto al paper.
- Metodología de split muy cercana al diseño original (Control siempre en train; test solo con PWA).
- Uso de WhisperX, que ofrece transcripciones y alineaciones más robustas que Kaldi.
- Configuración multilingüe (EN/ES/CA), lo que amplía el alcance del estudio.
- Código reproducible y bien documentado (scripts separados para cada bloque de features).
- Añadido de herramientas modernas de interpretabilidad (SHAP, permutation importance).

## Debilidades frente al paper

- Solo se utiliza el 50 % de las features totales (65/130).
- LEX está casi sin desarrollar (1/66 features).
- No se ha aplicado feature selection automática (SFS u otra variante).
- Menor número de muestras PWA (421 vs ~530).
- Mayor heterogeneidad por mezclar idiomas (EN/ES/CA) en un mismo modelo.

## Causas probables de resultados peores

- Falta de features clave: LEX y POS-LM son muy informativas en el paper.
- Ausencia de selección de features: usar 65 features con ~400 PWA favorece el sobreajuste.
- Dataset algo más pequeño y más heterogéneo (multilingüe).
- Ligera reducción del espacio de hiperparámetros respecto al paper.

## Próximos pasos 

### Prioridad alta 

1. Implementar POS-LM (26 features):
   - Entrenar modelos de POS bigrama/trigrama por idioma.
   - Extraer cross-entropy sobre las secuencias de POS generadas por nuestras transcripciones.
   - Añadir estas 26 features al conjunto.

2. Añadir feature selection tipo SFS:
   - Replicar la lógica del paper:
     - Empezar con conjunto vacío.
     - Añadir iterativamente la feature que más mejore el MAE en CV.
   - Fijar un número máximo de features (por ejemplo, 30–40).
   - Comparar rendimiento con:
     - 29 features “simples”.
     - 65 features completas.
     - Subconjunto seleccionado por SFS.

3. Basar el paper/tesis en un modelo con selección de features:
   - Argumentar que la reducción de dimensionalidad es clave cuando el nº de muestras es limitado.
   - Reportar el conjunto de features seleccionadas y su interpretación clínica.

### Prioridad media 

1. Completar LEX (para inglés inicialmente):
   - Integrar bases de datos:
     - Frecuencias: SUBTLEX (o similar).
     - Imageability y familiarity: MRC Database.
     - Age of Acquisition: Brysbaert.
     - Longitud fonémica: CMUdict.
   - Calcular las 13 estadísticas para cada métrica léxica, al menos para los datos en inglés.
   - En ES/CA, decidir si:
     - Se omiten esas features.
     - Se usan solo algunas aproximaciones (frecuencia, longitud de palabra, etc.).

2. Refinar el espacio de hiperparámetros de SVR:
   - Añadir valores de `C` más pequeños y más grandes.
   - Ajustar mejor `epsilon` según la escala real de WAB-AQ.

### Prioridad baja 

1. Implementar PVE y DTW:
   - Requiere:
     - Cálculo de posteriorgrams.
     - Emparejamiento de repeticiones de palabras/frases.
     - Aplicar DTW para medir distancias acústicas.
   - Aporta fidelidad al paper, pero es costoso en tiempo de desarrollo y cómputo.
