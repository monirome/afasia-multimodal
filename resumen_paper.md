# Metodología del paper (Le et al., 2018)

## Objetivo

Predecir WAB-AQ (Western Aphasia Battery Aphasia Quotient) usando features lingüísticas extraídas de transcripciones automáticas.

---

## 1. Datos (paper)

* **Dataset:** AphasiaBank
* **Idioma:** solo inglés
* **Total de participantes:** ~600

### Distribución por grupos

| Grupo | Nº aprox. de hablantes | Protocolo | Rango WAB-AQ | Sub-datasets |
| :--- | :--- | :--- | :--- | :--- |
| **PWA (Afasia)** | ~530 | Cinderella | 0–100 | Aphasia, English, Fridriksson, Kurland, Wright |
| **Control** | ~70 | Cinderella | 93.8–100 | – |

---

## 2. Transcripciones (paper)

**Método:** forced alignment con **Kaldi ASR**

* **Input:** audio + transcripciones CHAT (`.cha`).
* **Output:** marcas temporales a nivel de palabra (word-level timestamps).
* No utilizan WhisperX.

---

## 3. Features (130 en total, paper)

### Grupos de features

| Grupo | Nº features | Descripción breve |
| :--- | :--- | :--- |
| **DEN** (Density) | 42 | Palabras/minuto, ratios POS, longitud de enunciados (utterances). |
| **DYS** (Dysfluency) | 22 | Fillers, pausas, duración de pausas. |
| **LEX** (Lexical) | 66 | TTR, frecuencia, imageability, Age of Acquisition (AoA), familiarity. |
| **POS-LM** | 26 | Cross-entropy de modelos POS bigrama/trigrama. |
| **PVE** | 52 | Variabilidad acústica en repeticiones. |
| **DTW** | 39 | Distancia acústica de posteriorgrams con DTW. |

### Estadísticas por métrica

Se aplican 13 estadísticas a 17 métricas base del tipo `{X}`:
`{X} → min, p10, q1, median, q3, p90, max, mean, std, skew, kurt, iqr, mad`.

---

## 4. Feature selection (paper)

**Método:** Sequential Forward Selection (SFS)

1.  Iniciar con un conjunto vacío de features.
2.  Probar cada feature individualmente.
3.  Añadir la feature que más mejora el rendimiento en CV (MAE).
4.  Repetir mientras añadir features siga mejorando el rendimiento.
5.  **Resultado final:** ~42 features seleccionadas de las 130.

---

## 5. Modelo (paper)

* **Algoritmo:** Support Vector Regression (SVR).
* **Kernel:** RBF (Radial Basis Function).

### Hiperparámetros explorados

* `svr__C`: [0.1, 1, 10, 100, 1000]
* `svr__epsilon`: [0.01, 0.1, 1]
* `svr__gamma`: ["scale", "auto"]

**Optimización:** Grid Search con validación cruzada 5-fold.

---

## 6. Metodología de evaluación (paper)

### Split de datos

* **TRAINING SET:**
    * Control: 100 % (siempre en training).
    * PWA: 75 % (seleccionados por CV).
* **TEST SET:**
    * PWA: 25 % (withhold por CV).

### Cross-validation

* **Método:** 4-fold Stratified GroupKFold.
* **Grupos:** `patient_id` (el mismo paciente nunca está en train y test a la vez).
* **Estratificación:** por sub-dataset (Aphasia, English, Fridriksson, Kurland, Wright).

**Proceso por fold:**
1.  Train: PWA (75 %) + Control (100 %) → entrenar SVR.
2.  Test: PWA (25 %) → predecir.
3.  Métrica: MAE.
4.  **CV final:** media de los 4 MAEs.

---

## 7. Métricas de evaluación (paper)

### Regresión
* MAE (Mean Absolute Error) – métrica principal.
* RMSE (Root Mean Squared Error).
* Pearson r (correlación lineal).

### Clasificación por severidad
**Bins de severidad:**
* Very Severe: 0–25
* Severe: 25–50
* Moderate: 50–75
* Mild: 75–100

---

## 8. Resultados del paper

* **MAE:** 8–10 (aprox.; reportan sobre todo RMSE).
* **RMSE:** ~12–14.
* **Pearson r:** 0.75–0.85.
* **Features:** ~42 (tras feature selection).

***

# NUESTRO ESTUDIO (Actualizado)

## Objetivo

Predecir el WAB-AQ utilizando características lingüísticas extraídas de transcripciones automáticas generadas por **WhisperX**, superando las limitaciones del SVR clásico mediante **Ensemble Learning Heterogéneo** y **Modelos de Lenguaje Sintáctico (POS-LM)** avanzados.

---

## 1. Datos (NUESTRO)

* **Dataset:** AphasiaBank (multilingüe: inglés, español, catalán).
* **Total de participantes:** 506.

### Distribución por grupos

| Grupo | Nº hablantes | Protocolos | Rango WAB-AQ | Idiomas |
| :--- | :--- | :--- | :--- | :--- |
| **PWA** | 421 | Cinderella, Window, otros | 0.2–93.6 | EN, ES, CA |
| **Control** | 70 | – | 93.8–100 | EN, ES, CA |

### Distribución por severidad (PWA)

* **Muy Severa (0-25):** 10 %
* **Severa (25-50):** 13 %
* **Moderada (50-75):** 37 %
* **Leve (75-100):** 39 %

*Diferencias frente al paper:* Trabajo con un escenario multilingüe y un desbalance natural hacia casos leves.

### Parche TCU – PWA españoles

En el subcorpus español de AphasiaBank (Texas Christian University, protocolo Spanish AphasiaBank):
* Pacientes: `TCU02a`, `TCU04a`, `TCU06a`, `TCU10a`.
* En la web oficial se indica que los 4 son PWA.
* En el CSV original, `TCU06a` y `TCU10a` venían sin etiqueta de grupo (`group = NaN`).

En el script de entrenamiento se aplica un parche: para `patient_id` ∈ {`TCU06a`, `TCU10a`} fuerzo `group = "pwa"`. De esta forma, los cuatro TCU españoles quedan correctamente etiquetados como PWA y se incluyen en el conjunto de test.

---

## 2. Transcripciones

**Método:** forced alignment con **WhisperX**.

* **Input:** audio + transcripciones CHAT (`.cha`).
* **Output:** marcas temporales a nivel de palabra y fonema.
* **Modelo base ASR:** `large-v2` de Whisper.
* **Modelos de alineación:** EN y ES (con fallback a ES para catalán).

Features acústicas extraídas (disponibles en el dataset pero no utilizadas en el Ensemble final por priorizar explicabilidad lingüística):
* **MFCC (39 dimensiones):** 12 coeficientes + energía + primeras y segundas derivadas (Δ, ΔΔ).
* **MFB (40 dimensiones):** log mel-filterbanks.

*Diferencias frente al paper:* Uso WhisperX, más moderno y robusto al ruido que Kaldi. La pipeline es multilingüe.

---

## 3. Features (125 Implementadas)

### Cobertura frente al paper

En la versión actual del dataset tengo **125 features lingüísticas** implementadas, expandiendo significativamente la sección POS-LM.

| Grupo | Variables Paper | Implementadas | Cobertura | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| **DEN** | 42 | 42 | 100% | Ratios de palabras, velocidad, clases gramaticales. |
| **DYS** | 22 | 22 | 100% | Pausas, fillers, silencios. |
| **LEX** | 66 | 1 | Parcial | Solo TTR. Se detectó `lex_phones_std` como clave. |
| **POS-LM** | 26 | **60** | **231%** | Expandido con Backoff y Kneser-Ney. |
| **PVE/DTW** | 91 | 0 | 0% | No implementado. |

### Detalle de grupos implementados

**DEN (42 features) – replicado al 100 %**
* Ratios globales: `den_words_per_min`, `den_phones_per_min`, `den_OCW` (open-class words).
* Estadísticas de longitud de enunciado: `den_words_utt_{13 stats}`.
* Ratios POS: `den_nouns`, `den_verbs`, `den_prepositions`, `den_determiners`, etc.

**DYS (22 features) – replicado al 100 %**
* Ratios de disfluencias: `dys_fillers_per_min`, `dys_pauses_per_min`.
* Duración de pausas: `dys_pause_sec_{13 stats}`.

**POS-LM (60 features) – Innovación:**
A diferencia del paper, se han implementado dos métodos de suavizado para calcular la "sorpresa sintáctica" (Perplejidad):
1.  **Backoff (Laplace):** 30 features (Bigram + Trigram CE/PPL). Método robusto para secuencias no vistas.
2.  **Kneser-Ney smoothing:** 30 features. Estado del arte clásico en n-gramas.

**LEX (1 feature + derivada):**
* Implementación mínima: `lex_ttr` (Type-Token Ratio).
* **Hallazgo Clave:** Se ha calculado una nueva variable `lex_phones_std` (desviación estándar de la longitud de palabras en fonemas).
    * *Interpretación:* Pacientes graves tienen desviación cercana a 0 (solo monosílabos). Pacientes leves tienen desviación alta. Esta variable ha resultado ser el predictor más potente.

---

## 4. Modelado: Del SVR al Ensemble Optimizado

Se ha sustituido el SVR único por una arquitectura de **Ensemble Heterogéneo** optimizado matemáticamente.

### A. Modelos Individuales (Componentes)

1.  **TabPFN (Transformer Tabular):**
    * Transformer pre-entrenado que funciona como un "Large Language Model" para datos tabulares.
    * **Rol:** Es el modelo más preciso individualmente.
2.  **CatBoost (Gradient Boosting):**
    * Árboles de decisión robustos.
    * **Rol:** Aporta estabilidad y manejo de ruido.
3.  **EBM (Explainable Boosting Machine):**
    * Modelo "Glassbox" (Generalized Additive Model).
    * **Rol:** Aporta explicabilidad clínica visual.

### B. Optimización de Pesos (Nelder-Mead / SLSQP)

Se utiliza un algoritmo de optimización para encontrar la combinación lineal perfecta de los tres modelos:

$$\text{Predicción} = w_1 \cdot \text{TabPFN} + w_2 \cdot \text{CatBoost} + w_3 \cdot \text{EBM}$$

* **Pesos resultantes:** TabPFN (86.5%), CatBoost (12.1%), EBM (1.4%).

---

## 5. Metodología de evaluación (NUESTRA)

### Cross-validation
* **Método:** 5-fold Stratified GroupKFold (asegurando balance de severidad en cada fold).
* **Target:** PWA en inglés (entrenamiento y validación).
* **Test Externo:** Pacientes en Español y Catalán.

### Post-procesamiento
1.  **Calibración Isotónica:** Ajuste de las probabilidades/puntuaciones para corregir sesgos en los extremos (0 y 100).
2.  **Clinical Override (Propuesta):** Regla lógica para corregir fallos en audios muy cortos (*"Si word_count < 10, limitar predicción máxima"*).

---

## 6. Resultados Experimentales Actualizados

Esta tabla compara el rendimiento final de nuestro sistema frente al baseline del paper y los modelos previos.

| ID | Modelo / Config | Features | N_Feat | CV_MAE (Raw) | **Calibrated MAE** | **Pearson (r)** | Notas Técnicas |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E_FINAL** | **Ensemble Optimizado** | **Híbrido** | **-** | **-** | **9.69** | **0.842** | **Nuevo Estándar.** Pesos optimizados (SLSQP). |
| **E1** | TabPFN (Individual) | KBest-40 | 40 | 11.06 | 10.27 | 0.803 | Mejor modelo individual. Supera correlación del estado del arte. |
| **E2** | EBM (Glassbox) | Full | 108 | 12.02 | 11.21 | 0.779 | Modelo totalmente interpretable. |
| **E3** | CatBoost | Full | 108 | 11.95 | 11.44 | 0.751 | Base robusta basada en árboles. |
| **B2** | SVR (Baseline Mejorado) | SFS | 28 | 12.99 | 11.89 | 0.734 | Mejor configuración de SVR encontrada. |
| **A1** | SVR (Baseline Simple) | Manual | 29 | 13.54 | 12.86 | 0.695 | Punto de partida inicial. |
| **REF** | **Paper (Le et al. 2018)** | **SVR** | **43** | **9.18** | **-** | **0.799** | **Estado del Arte (Referencia).** |

### Conclusiones Principales

1.  **Superación en Correlación:** Nuestro **Ensemble Optimizado (r=0.842)** supera claramente la correlación reportada en el paper original (r=0.799), lo que indica una mejor capacidad para ordenar a los pacientes por gravedad.
2.  **Acercamiento al Estado del Arte en MAE:** Con un **MAE de 9.69**, nos hemos situado a solo 0.5 puntos del resultado del paper (9.18), pero utilizando transcripciones totalmente automáticas (WhisperX) y en un entorno más realista y ruidoso.
3.  **Superioridad de los Nuevos Modelos:** La transición de SVR (MAE ~11.89) a Transformers Tabulares (MAE ~10.27) ha supuesto una mejora directa de más de 1.5 puntos, validando la modernización del stack tecnológico.
4.  **Importancia del Diagnóstico de Error:** Se ha identificado que los errores residuales (>20 puntos) se deben casi exclusivamente a pacientes severos con audios extremadamente cortos, lo que abre la puerta a una corrección mediante reglas clínicas simples.

---

# Comparación Lado a Lado

| Aspecto | Paper (Le et al., 2018) | Mi Implementación Actual |
| :--- | :--- | :--- |
| **Dataset** | ~600 speakers (Solo EN) | 506 speakers (EN/ES/CA) |
| **Transcripción** | Kaldi ASR | **WhisperX** (Más robusto) |
| **Features** | 130 (Manuales + Acústicas) | 125 (DEN/DYS 100% + **POS-LM Avanzado**) |
| **Feature Selection** | SFS (Sequential Forward Selection) | **K-Best (Mutual Info)** + Importancia intrínseca |
| **Modelo Principal** | SVR + RBF | **Ensemble (TabPFN + CatBoost + EBM)** |
| **Validación** | 4-Fold por sub-dataset | **5-Fold Stratified** por severidad |
| **Optimización** | Grid Search | **Nelder-Mead / SLSQP** (Pesos) |
| **MAE Final** | ~9.18 | **9.69** (Calibrado y Optimizado) |
| **Correlación (r)** | 0.799 | **0.842** (Superior al paper) |

---

# Conclusiones y Próximos Pasos

## Fortalezas de mi implementación
1.  **Tecnología de Vanguardia:** El uso de TabPFN (Transformers) y Ensemble Learning ha demostrado ser superior a los métodos clásicos (SVR), especialmente para mejorar la correlación (0.84 vs 0.79).
2.  **Robustez Multilingüe:** El sistema está diseñado para trabajar con inglés, español y catalán, ampliando el alcance del estudio original.
3.  **Innovación en Features:** La implementación de POS-LM con Backoff/Kneser-Ney ha resultado crítica para capturar la sintaxis rota, algo que las features acústicas puras no ven.
4.  **Diagnóstico de Error:** Se ha identificado que el error residual se concentra en casos "borde": pacientes graves con muy poca producción oral.

## Debilidades (A abordar)
1.  **Brecha de MAE:** A pesar de la mejora, el MAE (9.69) sigue estando 0.5 puntos por encima del paper (9.18). Esto se atribuye a la falta de features acústicas complejas (DTW/PVE) y al ruido inherente de un dataset más heterogéneo.
2.  **Dependencia del Audio:** El modelo falla cuando el audio es demasiado corto (<5s) porque no hay suficiente "señal" lingüística.

## Próximos Pasos (Hoja de Ruta Final)

### Prioridad Alta
1.  **Regla de Seguridad Clínica (Post-Processing):**
    * Implementar una regla lógica híbrida: *"Si el recuento de palabras es < 10 y la duración < 5s, limitar la predicción máxima a 40 puntos"*.
    * Se espera que esto elimine los *outliers* graves y baje el MAE global hacia el objetivo de 9.2.

2.  **Validación Externa Pura:**
    * Evaluar el Ensemble final estrictamente sobre los subconjuntos de **Español y Catalán** para confirmar la capacidad multilingüe del sistema sin reentrenamiento.

### Prioridad Media
1.  **Consolidación de `lex_phones_std`:**
    * Formalizar el uso de la variabilidad de longitud de palabra como un biomarcador digital clave. Realizar un estudio estadístico aislado de esta variable.

### Prioridad Baja
1.  **Features Acústicas (PVE/DTW):**
    * Aunque completarían la replicación del paper, el coste computacional es alto y la mejora marginal estimada es baja dado el éxito del Ensemble actual.