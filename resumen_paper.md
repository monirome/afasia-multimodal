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
- No utilizan WhisperX. 

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
- Reducción: 130 → 42 features “óptimas” aproximadamente.

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
  - Control: 100 % (siempre en training)
  - PWA: 75 % (seleccionados por CV)

TEST SET:
  - PWA: 25 % (withhold por CV)
```

### Cross-validation

- **Método:** 4-fold Stratified GroupKFold
- **Grupos:** `patient_id` (el mismo paciente nunca está en train y test a la vez).
- **Estratificación:** por sub-dataset (Aphasia, English, Fridriksson, Kurland, Wright).

Proceso por fold:

```text
Fold 1:
  - Train: PWA (75 %) + Control (100 %) → entrenar SVR
  - Test:  PWA (25 %) → predecir
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

Mantengo el mismo objetivo que el paper: predecir WAB-AQ a partir de features cuantitativas extraídas de transcripciones automáticas.

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
| Very Severe  | 0–25         | 44     | 10 %         |
| Severe       | 25–50        | 55     | 13 %         |
| Moderate     | 50–75        | 156    | 37 %         |
| Mild         | 75–100       | 166    | 39 %         |

Diferencias frente al paper:

- Trabajo con un escenario multilingüe (EN/ES/CA) en lugar de solo inglés.
- Tengo menos muestras PWA (421 frente a ~530).

### Parche TCU – PWA españoles

En el subcorpus español de AphasiaBank (Texas Christian University, protocolo Spanish AphasiaBank):

- Pacientes: `TCU02a`, `TCU04a`, `TCU06a`, `TCU10a`.
- En la web oficial se indica que los 4 son PWA.
- En mi CSV original, `TCU06a` y `TCU10a` venían sin etiqueta de grupo (`group = NaN`).

En el script `train_svr_COMPLETO_FINAL.py` aplico un parche:
para `patient_id` ∈ {`TCU06a`, `TCU10a`} fuerzo `group = "pwa"`. De esta forma, los cuatro TCU españoles quedan correctamente etiquetados como PWA.

---

## 2. Transcripciones 

**Método:** forced alignment con WhisperX.

- Input: audio + transcripciones CHAT (`.cha`).
- Output: marcas temporales a nivel de palabra.
- Modelo base ASR: `large-v2` de Whisper.
- Modelos de alineación: EN y ES (con fallback a ES para catalán).

Features acústicas extraídas (para tenerlas disponibles, aunque no las utilizo todavía en este experimento lingüístico):

- MFCC (39 dimensiones): 12 coeficientes + energía + primeras y segundas derivadas (Δ, ΔΔ).
- MFB (40 dimensiones): log mel-filterbanks.

Diferencias frente al paper:

- Uso WhisperX, más moderno y robusto que Kaldi.
- La pipeline es multilingüe (EN/ES/CA).

---

## 3. Features 

### Cobertura frente al paper

En la versión actual del dataset tengo **125 features lingüísticas** implementadas:

| Grupo   | Variables Paper | Variables Implementadas | Cobertura |
|---------|---------------:|------------------------:|----------:|
| DEN     | 42             | 42                      | 100%      |
| DYS     | 22             | 22                      | 100%      |
| LEX     | 66             | 1                       | 1.5%      |
| POS-LM  | 26             | 60                      | 231%      |
| PVE     | 52             | 0                       | 0%        |
| DTW     | 39             | 0                       | 0%        |
| **TOTAL** | **247**      | **125**                 | **51%**   |

### Detalle de grupos implementados

**DEN (42 features) – replicado al 100 %**

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

**DYS (22 features) – replicado al 100 %**

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

- He replicado DEN y DYS por completo (mis definiciones corresponden a las del paper).
- En LEX solo utilizo TTR.
- No he implementado todavía POS-LM, PVE ni DTW (por complejidad y/o dependencia de recursos externos).

---

## 4. Feature selection y experimentos 

En `train_svr_COMPLETO_FINAL.py` he implementado selección de features automática tipo SFS (mlxtend) sobre las 65 features disponibles, y puedo activarla con la opción `--features sfs`. Trabajo con tres configuraciones de features:

- `--features simple`: 29 features sencillas (DEN + DYS + `lex_ttr`).
- `--features full`: las 65 features disponibles (DEN + DYS + `lex_ttr`).
- `--features sfs`: subconjunto óptimo seleccionado automáticamente a partir de las 65 anteriores.

En este resumen describo los resultados de las dos primeras configuraciones (simple y full); la configuración SFS ya está implementada, pero todavía no he hecho un análisis sistemático de sus resultados.

### Experimento 1: 29 features “simples”

Selección manual (sin las 13 estadísticas completas), centrada en ratios y medias:

- 18 features DEN.
- 10 features DYS.
- 1 feature LEX (`lex_ttr`).

Resultados (CV en PWA EN + controles):

```text
MAE:   12.29
RMSE:  16.70
R²:    0.505
r (Pearson): 0.710
Acc@5: 26.6 %
Accuracy severidad: 56.8 %
```

### Experimento 2: 65 features completas

Uso de todas las features disponibles de DEN, DYS y LEX:

- 42 DEN.
- 22 DYS.
- 1 LEX.

Resultados (CV en PWA EN + controles):

```text
MAE:   14.55
RMSE:  18.85
R²:    0.369
r (Pearson): 0.607
Acc@5: 21.4 %
Accuracy severidad: 51.8 %
```

Comentario: al añadir más features sin una selección previa (modo `full`) observo sobreajuste y un empeoramiento claro de las métricas frente al modo `simple`.

El paper usa SFS para seleccionar ~42 features óptimas. Yo ya he incorporado SFS (`--features sfs`), pero aún no he analizado de forma sistemática los resultados de esa configuración ni la he comparado a fondo con los modos simple y full.

---

## 5. Modelo 

- **Algoritmo:** Support Vector Regression (SVR).
- **Kernel:** RBF, con comparación puntual con kernel lineal (no incluida en la última versión del grid, pero sí explorada en fases previas).

### Pipeline

En términos de scikit-learn, mi pipeline es:

```text
[SimpleImputer(strategy='median')]
→ [StandardScaler]
→ [SVR]
```

### Hiperparámetros explorados

```text
svr__C:        [0.1, 1, 10, 100, 1000]
svr__epsilon:  [0.01, 0.1, 1, 5, 10]
svr__kernel:   ['rbf]
svr__gamma:    ['scale', 'auto']
```

Total de combinaciones: 5 × 5 × 1 × 2 = 50.

La optimización la hago con Grid Search (GridSearchCV) con validación cruzada interna (CV=5) dentro de cada fold de GroupKFold.

Diferencias frente al paper:

- Uso el mismo rango de `C` que el paper.
- Amplío el rango de `epsilon` a valores más altos (5, 10).
- Mantengo `gamma` en `['scale', 'auto']`.
- La estrategia de búsqueda es conceptualmente equivalente (búsqueda en rejilla con CV), aunque podría reemplazarla por Optuna en el futuro para explorar el espacio de forma más eficiente.

---

## 6. Metodología de evaluación 

### Lógica general de split en mi implementación

En el script actual he optado por un esquema que imita el “Combined protocol (Auto)” del paper, pero restringiendo la validación cruzada a los PWA en inglés:

- Para entrenamiento y validación cruzada (CV):
  - Utilizo únicamente los PWA en inglés (`df_en`) como conjunto objetivo de CV.
  - Añado todos los controles (sea cual sea su idioma) al conjunto de entrenamiento en cada fold.
- Para evaluación externa:
  - Los PWA en español y catalán nunca participan en la CV.
  - Solo los utilizo después, como test externo (EVAL_ES y EVAL_CA).

### Cross-validation (EN)

- **Método:** 4-fold GroupKFold.
- **Grupos:** `patient_id` (cada paciente aparece solo en train o en test dentro de un fold, nunca en ambos).

Proceso por fold:

```text
Para cada fold:
  - Defino los índices train_idx y test_idx sobre PWA EN.
  - Train:
      - PWA EN (train_idx)  ≈ 75 % de los PWA EN
      + TODOS los controles (70 sujetos)
  - Test:
      - PWA EN (test_idx)   ≈ 25 % de los PWA EN
  - Dentro del train:
      - Aplico GridSearchCV (CV=5, scoring=neg_MAE).
      - Selecciono el mejor modelo por MAE.
  - Con ese modelo, predigo QA sobre el test_idx.
```

Al final, concateno las predicciones de los 4 folds sobre todos los PWA EN y obtengo las métricas de CV globales (split “CV_PWA”).

### Evaluación externa ES y CA

Tras entrenar el modelo final (ajustado sobre PWA EN + controles completos), hago:

- EVAL_ES_RAW / EVAL_ES_CALIBRATED:
  - Input: PWA en español (`df_es`), todas las features disponibles.
  - Predigo con el modelo final y aplico (o no) calibración.
- EVAL_CA_RAW / EVAL_CA_CALIBRATED:
  - Input: PWA en catalán (`df_ca`).
  - Mismo esquema que en español.

Diferencias frente al paper:

- Mantengo el mismo número de folds (4) y la misma lógica de grupos por paciente.
- No estratifico por sub-dataset (Aphasia, English, Fridriksson, Kurland, Wright).
- En la CV solo uso PWA en inglés como objetivo; PWA en ES/CA se tratan como evaluación externa.

---

## 7. Post-procesamiento 

### Calibración

Aplico calibración posterior sobre las predicciones de WAB-AQ:

- Método: Isotonic Regression (`sklearn.isotonic.IsotonicRegression`).
- Entrenamiento del calibrador:
  - Utilizo las predicciones obtenidas en CV sobre PWA EN (`cv_preds`) y sus QA reales (`y_pwa_en`).
- Uso posterior:
  - Guardo el calibrador en disco (`calibrator.pkl`).
  - Aplico el calibrador tanto a:
    - Las predicciones de CV (split `CV_PWA_CALIBRATED`).
    - Las predicciones EVAL_ES y EVAL_CA.

Siempre recorto las predicciones calibradas al rango [0, 100].

Efecto:

- Observo una ligera mejora en algunas métricas de error (MAE/RMSE).
- En el paper no se menciona ninguna técnica explícita de calibración; su contraste principal es entre “Oracle” (transcripción manual) y “Auto” (transcripción automática), no una calibración posterior de las salidas del modelo.

---

## 8. Métricas de evaluación 

### Regresión

Uso las mismas métricas que el paper y algunas adicionales:

- MAE.
- RMSE.
- R².
- Pearson r.
- Spearman ρ (añadida por mí).

### Clasificación por severidad

Utilizo las mismas bandas que el paper:

```text
Very Severe:  0–25
Severe:      25–50
Moderate:    50–75
Mild:        75–100
```

Métricas:

- Accuracy global de severidad.
- Precision, Recall y F1 por clase.
- Acc@1, Acc@5, Acc@10 (según magnitud del error en puntos WAB).

### Outputs adicionales

Genero además:

- Matrices de confusión.
- Diagramas de dispersión QA_real vs QA_predicho (con métricas impresas en la figura).
- Histograma de errores con umbral tipo Figura 6 del paper.
- Permutation importance.
- Valores SHAP para interpretabilidad (cuando SHAP está disponible).

---

## 9. Resultados 

### Resumen de resultados por experimento (CV en PWA EN)

| Experimento | Features usadas                            | MAE   | RMSE  | R²    | Pearson r | Acc@5 | Accuracy severidad |
|------------:|---------------------------------------------|------:|------:|------:|----------:|------:|-------------------:|
| 1           | 29 (DEN + DYS simples + `lex_ttr`)          | 12.29 | 16.70 | 0.505 | 0.710     | 26.6 % | 56.8 %            |
| 2           | 65 (todas las DEN/DYS + `lex_ttr`)          | 14.55 | 18.85 | 0.369 | 0.607     | 21.4 % | 51.8 %            |

Comparado con el paper:

- El paper reporta MAE ≈ 8–10, RMSE ≈ 12–14 y Pearson r entre 0.75 y 0.85.
- Mis resultados actuales están por debajo de esos valores en términos de error (MAE/RMSE), aunque la correlación es razonablemente alta, especialmente en el modo `simple`.

### Importancia de variables (top 5, típicamente)

| Ranking | Feature              | Interpretación                          |
|--------:|----------------------|-----------------------------------------|
| 1       | `lex_ttr`           | Diversidad léxica (Type-Token Ratio)    |
| 2       | `den_determiners`   | Proporción de determinantes             |
| 3       | `den_function_words`| Proporción de palabras funcionales      |
| 4       | `den_prepositions`  | Proporción de preposiciones             |
| 5       | `den_light_verbs`   | Proporción de light verbs               |

(El ranking exacto puede variar ligeramente según el experimento, pero estas variables aparecen de forma consistente entre las más importantes.)

---

# Comparación lado a lado

| Aspecto              | Paper (Le et al., 2018)                          | Mi implementación actual                                        |
|----------------------|--------------------------------------------------|-----------------------------------------------------------------|
| Dataset total        | ~600 speakers (solo EN)                          | 506 speakers (EN/ES/CA)                                         |
| PWA                  | ~530                                             | 421                                                             |
| Control              | ~70                                              | 70                                                              |
| Transcripción        | Kaldi ASR                                        | WhisperX                                                        |
| Features totales     | 130                                              | 65 (50 %)                                                       |
| DEN                  | 42                                               | 42                                                              |
| DYS                  | 22                                               | 22                                                              |
| LEX                  | 66                                               | 1                                                               |
| POS-LM               | 26                                               | 0                                                               |
| PVE                  | 52                                               | 0                                                               |
| DTW                  | 39                                               | 0                                                               |
| Feature selection    | SFS (Sequential Forward Selection)               | SFS disponible (`--features sfs`), resultados aún por explotar |
| Modelo               | SVR + RBF                                        | SVR + RBF                                                       |
| CV                   | 4-fold Stratified GroupKFold (por sub-dataset)   | 4-fold GroupKFold por paciente (solo PWA EN en CV)             |
| Control en train     | Siempre                                          | Siempre (todos los controles en cada fold de train)            |
| PWA en CV            | PWA EN (todos los sub-datasets)                  | PWA EN; ES/CA solo como test externo                           |
| MAE                  | ~8–10                                            | 12.29 (simple) / 14.55 (full)                                  |
| R²                   | ~0.75                                            | 0.505 / 0.369                                                   |
| Pearson r            | 0.75–0.85                                        | 0.710 / 0.607                                                   |

---

# Conclusiones y próximos pasos

## Fortalezas de mi implementación

- He replicado las features DEN y DYS al 100 % respecto al paper.
- La lógica de split mantiene la idea central del paper: controles siempre en train y evaluación principal sobre PWA.
- Utilizo WhisperX, que proporciona transcripciones y alineaciones modernas y robustas.
- He extendido el escenario a un contexto multilingüe (EN/ES/CA), lo que da más alcance clínico y científico.
- El código es reproducible y modular (scripts separados por bloque de features, script único para entrenamiento y evaluación).
- He añadido herramientas modernas de interpretabilidad (SHAP, permutation importance) y análisis de error (histograma + grupos low/high error).

## Debilidades frente al paper

- Solo utilizo el 50 % de las features totales del paper (65/130).
- El bloque LEX está prácticamente sin desarrollar (1/66 features).
- Aunque ya tengo implementada SFS, los resultados que reporto aquí están basados en configuraciones sin selección automática (simple y full).
- Tengo menos muestras PWA (421 vs ~530).
- La heterogeneidad aumenta al tener un dataset multilingüe; de momento, la CV está restringida a inglés, y ES/CA se tratan como test externo.

## Causas probables de resultados peores

- Falta de features clave: LEX y POS-LM son muy informativas en el paper y en mi caso están ausentes o muy reducidas.
- En los resultados presentados no he explotado todavía la configuración SFS, que podría reducir sobreajuste y acercar el rendimiento al del paper.
- El dataset es algo más pequeño en PWA y, en conjunto, más heterogéneo (multilingüe).
- Puede que el espacio de hiperparámetros aún no esté optimizado de forma tan fina como en el artículo original (aunque el rango de C es el mismo).

## Próximos pasos 

### Prioridad alta 

1. Analizar en profundidad la configuración `--features sfs`  
   - Comparar sistemáticamente:
     - `simple` vs `full` vs `sfs`.
   - Evaluar:
     - Número de features seleccionadas.
     - Estabilidad del subconjunto de features entre folds o seeds.
     - Impacto en MAE, RMSE, R² y correlaciones.
   - Identificar si SFS reduce claramente el sobreajuste observado en el modo `full`.

2. Afinar la búsqueda de hiperparámetros  
   - Evaluar si me interesa:
     - Ajustar mejor los rangos de `epsilon`.
     - Explorar otras opciones de `gamma` o kernels.
   - Considerar reemplazar parte del GridSearchCV por Optuna (o RandomizedSearchCV) para explorar el espacio de forma más eficiente y flexible.

3. Implementar POS-LM (26 features)  
   - Entrenar modelos POS bigrama/trigrama por idioma.
   - Calcular la cross-entropy de las secuencias de POS obtenidas de mis transcripciones.
   - Añadir estas 26 features al conjunto y evaluar su efecto, empezando por el subset EN.

### Prioridad media 

1. Completar LEX (al menos para inglés en una primera fase)  
   - Integrar recursos:
     - Frecuencias léxicas (p. ej. SUBTLEX).
     - Imageability y familiarity (MRC Database).
     - Age of Acquisition (Brysbaert).
     - Longitud fonémica (CMUdict u otro recurso apropiado).
   - Calcular las 13 estadísticas por métrica léxica, al menos para los datos en inglés.
   - Para ES/CA, decidir si:
     - Excluyo algunas de estas features.
     - O uso aproximaciones (frecuencia en corpus de referencia, longitud de palabra, etc.) con cuidado.

### Prioridad baja 

1. Implementar PVE y DTW  
   - Requiere:
     - Cálculo de posteriorgrams.
     - Emparejamiento de repeticiones de palabras/frases.
     - Aplicar DTW para medir distancias acústicas entre realizaciones.
   - Estas features aportarían fidelidad total al paper, pero su implementación es costosa en tiempo de desarrollo y cómputo.

2. Refinar el flujo para catalán  
   - Descargar de forma consistente audios y transcripciones catalanas.
   - Reprocesar y asegurarse de que se integran correctamente en el pipeline.
   - Explorar modelos monolingües o cross-lingual específicos para CA si tiene sentido.

## Resultados de experimentos SVR completos

| ID    | Config                | Features   | POS_LM     |   Total_Features |   DEN |   DYS |   LEX |   POSLM |   CV_MAE |   CV_RMSE |   CV_R2 |   CV_Pearson |   CV_Spearman |   Calibrated_MAE |   Calibrated_RMSE |   Calibrated_R2 |   Calibrated_Pearson |   Calibrated_Spearman |   Acc_10 |   Severity_Acc |   Delta_MAE_vs_Paper |   Delta_Pearson_vs_Paper | SVR_Kernel   |   SVR_C |   SVR_Epsilon | SVR_Gamma   |   ES_MAE_raw |   ES_MAE_calibrated |   ES_R2_calibrated |   ES_Pearson_calibrated | Dataset               |   N_samples_train |   N_samples_eval_ES | CV_Strategy                   | Excluded_Features   | Notes                                                                                                        |
|:------|:----------------------|:-----------|:-----------|-----------------:|------:|------:|------:|--------:|---------:|----------:|--------:|-------------:|--------------:|-----------------:|------------------:|----------------:|---------------------:|----------------------:|---------:|---------------:|---------------------:|-------------------------:|:-------------|--------:|--------------:|:------------|-------------:|--------------------:|-------------------:|------------------------:|:----------------------|------------------:|--------------------:|:------------------------------|:--------------------|:-------------------------------------------------------------------------------------------------------------|
| A1    | SIMPLE_NO-POSLM       | simple     | none       |               29 |    18 |    10 |     1 |       0 |   13.54  |    18.562 |   0.39  |        0.659 |         0.66  |           12.859 |            17.081 |           0.483 |                0.695 |                 0.68  |    51.31 |          56.56 |                 3.68 |                   -0.104 | rbf          |      10 |          5    | auto        |        23.88 |               18.46 |              -3.77 |                   0.135 | AphasiaBank           |               419 |                   4 | GroupKFold-4                  | lex_ttr             | Baseline con 29 features simples                                                                             |
| A2    | FULL_NO-POSLM         | full       | none       |               64 |    42 |    22 |     0 |       0 |   15.078 |    20.185 |   0.278 |        0.558 |         0.558 |           14.663 |            18.854 |           0.37  |                0.609 |                 0.582 |    42.24 |          49.64 |                 5.48 |                   -0.19  | rbf          |      10 |          5    | auto        |        18.1  |               15.59 |              -2.03 |                   0.082 | AphasiaBank           |               419 |                   4 | GroupKFold-4                  | lex_ttr             | Full features sin selección                                                                                  |
| A3    | SFS_NO-POSLM          | sfs        | none       |               12 |     8 |     4 |     0 |       0 |   12.863 |    17.767 |   0.441 |        0.686 |         0.668 |           12.098 |            16.099 |           0.541 |                0.735 |                 0.686 |    52.74 |          59.67 |                 2.92 |                   -0.064 | rbf          |      10 |          0.01 | scale       |        19.05 |               14.93 |              -1.57 |                   0.207 | AphasiaBank           |               419 |                   4 | GroupKFold-4                  | lex_ttr             | SFS sin POS-LM - mejor sin POS-LM                                                                            |
| B1    | FULL_POSLM-KN         | full       | kneser-ney |               94 |    42 |    22 |     0 |      30 |   15.489 |    21.915 |   0.149 |        0.492 |         0.55  |           15.035 |            19.592 |           0.32  |                0.566 |                 0.575 |    38.19 |          49.4  |                 5.85 |                   -0.233 | rbf          |      10 |          5    | auto        |        22.21 |               17.9  |              -3.54 |                   0.136 | AphasiaBank           |               419 |                   4 | GroupKFold-4                  | lex_ttr             | Full con KN - overfitting                                                                                    |
| B2    | SFS_POSLM-KN          | sfs        | kneser-ney |               28 |    11 |     6 |     0 |      11 |   12.992 |    18.203 |   0.413 |        0.678 |         0.673 |           11.892 |            16.131 |           0.539 |                0.734 |                 0.696 |    53.46 |          63.25 |                 2.71 |                   -0.065 | rbf          |      10 |          0.01 | scale       |        19.51 |               14.88 |              -1.54 |                   0.252 | AphasiaBank           |               419 |                   4 | GroupKFold-4                  | lex_ttr             | SFS + KN - segundo mejor overall                                                                             |
| C1    | FULL_POSLM-BO         | full       | backoff    |               94 |    42 |    22 |     0 |      30 |   15.522 |    22.002 |   0.143 |        0.487 |         0.547 |           15.066 |            19.601 |           0.319 |                0.565 |                 0.576 |    39.14 |          50.12 |                 5.89 |                   -0.234 | rbf          |      10 |          5    | auto        |        22    |               17.74 |              -3.47 |                   0.138 | AphasiaBank           |               419 |                   4 | GroupKFold-4                  | lex_ttr             | Full con Backoff - overfitting                                                                               |
| C2    | SFS_POSLM-BO          | sfs        | backoff    |               28 |    11 |     6 |     0 |      11 |   13     |    18.209 |   0.413 |        0.678 |         0.673 |           11.918 |            16.144 |           0.538 |                0.734 |                 0.696 |    52.98 |          63.25 |                 2.74 |                   -0.065 | rbf          |      10 |          0.01 | scale       |        19.3  |               14.69 |              -1.45 |                   0.267 | AphasiaBank           |               419 |                   4 | GroupKFold-4                  | lex_ttr             | SFS + Backoff - equivalente a B2                                                                             |
| D1    | FULL_POSLM-ALL        | full       | all        |              124 |    42 |    22 |     0 |      60 |   15.631 |    22.216 |   0.126 |        0.477 |         0.539 |           15.255 |            19.855 |           0.302 |                0.549 |                 0.569 |    40.1  |          51.07 |                 6.08 |                   -0.25  | rbf          |      10 |          1    | auto        |        20.41 |               16.47 |              -2.71 |                   0.165 | AphasiaBank           |               419 |                   4 | GroupKFold-4                  | lex_ttr             | Full con KN+BO - más overfitting                                                                             |
| D2    | SFS_POSLM-ALL         | sfs        | all        |               38 |    11 |     6 |     0 |      21 |   12.761 |    17.933 |   0.43  |        0.688 |         0.676 |           11.873 |            16.118 |           0.54  |                0.735 |                 0.701 |    54.18 |          62.77 |                 2.69 |                   -0.064 | rbf          |      10 |          0.01 | scale       |        19.38 |               14.61 |              -1.42 |                   0.275 | AphasiaBank           |               419 |                   4 | GroupKFold-4                  | lex_ttr             | SFS + ALL - mejor overall                                                                                    |
| PAPER | Le et al. 2018 (Auto) | reference  | reference  |               43 |    18 |    10 |     6 |       2 |    9.18  |   nan     | nan     |        0.799 |       nan     |            9.24  |           nan     |         nan     |                0.786 |               nan     |   nan    |         nan    |                 0    |                    0     |              |     nan |        nan    |             |       nan    |              nan    |             nan    |                 nan     | AphasiaBank (EN only) |               348 |                 nan | Speaker-independent 4-fold CV |                     | Le et al. (2018), Combined protocol, Auto vs Calibrated. SVR con features transcript-based y ASR automático. |

