# Automatic Assessment of Aphasia Severity using DEN+DYS Features

Automatic prediction of WAB-AQ (Western Aphasia Battery - Aphasia Quotient) scores using acoustic-prosodic features from spontaneous speech, replicating the methodology from **Le et al. (2018)**.


## Overview

This repository implements an **automatic speech analysis pipeline** for predicting aphasia severity using:

- **28 acoustic-prosodic features** (DEN + DYS) extracted from speech
- **Support Vector Regression (SVR)** with RBF/Linear kernels
- **Cross-validation** with GroupKFold (speaker-independent)
- **Isotonic calibration** for improved predictions
- **Severity classification** (Very Severe, Severe, Moderate, Mild)

### Reference Paper

> **Le, D., et al. (2018)**. "Automatic Assessment of Speech Intelligibility for Individuals with Aphasia."  
> *Speech Communication*, 100, 1-11.


## Features

### Information Density (DEN) - 18 features
- Words/min, Phones/min
- Word and phone production per utterance
- Part-of-speech ratios (nouns, verbs, adjectives, etc.)
- Function word ratios
- Light verb usage

### Dysfluency (DYS) - 10 features
- Fillers per minute/word/phone
- Pause counts (total, long, short)
- Pause durations (mean)

---

## Pipeline Architecture
```
INPUT DATA
  - Audio files (.wav, .mp3, .mp4)
  - Transcriptions (.cha format - CHAT)
            |
            v
+---------------------------------------------------------+
| STEP 1: Metadata Extraction                            |
| Script: 01_metadata/extract_metadata_from_csv.py       |
|         (or extract_wab_metadata.py for .CHA only)     |
| Output: data/patient_metadata_WAB.csv                  |
|         (patient_id, QA, sex, age, aphasia_type, lang) |
+---------------------------------------------------------+
            |
            v
+---------------------------------------------------------+
| STEP 2: Word-level Alignments (WhisperX)              |
| Script: 02_alignments/generate_whisperx_alignments.py |
| Output: data/word_alignments_ALL.csv                  |
|         (patient_id, word, start_sec, end_sec)         |
+---------------------------------------------------------+
            |
            v
+---------------------------------------------------------+
| STEP 3: Extract DEN+DYS Features                      |
| Script: 03_features/build_den_dys.py                  |
| Output: data/features_den_dys_COMPLETO.csv            |
|         (patient_id, language, 28 features)            |
+---------------------------------------------------------+
            |
            v
+---------------------------------------------------------+
| STEP 4: Merge Features + Metadata                     |
| Script: 04_merge/merge_features_metadata.py           |
| Output: data/dataset_FINAL_EN_ES.csv                  |
+---------------------------------------------------------+
            |
            v
+---------------------------------------------------------+
| STEP 5: Train & Evaluate SVR                          |
| Script: 05_models/train_svr_den_dys.py                |
| Output: resultados_svr/SVR_DEN_DYS_YYYYMMDD_HHMMSS/   |
|         - Models (pkl)                                 |
|         - Metrics (csv)                                |
|         - Plots (png)                                  |
|         - Confusion matrices                           |
|         - Logs (console.log)                           |
+---------------------------------------------------------+
```

## Dataset Structure
```
project/
├── data/
│   ├── audios_completos/          # Audio files
│   │   ├── patient001.wav
│   │   ├── patient002.wav
│   │   └── ...
│   │
│   └── transcripciones/           # CHAT transcriptions
│       ├── patient001.cha
│       ├── patient002.cha
│       └── ...
```

### .CHA File Format (CHAT)

Transcription files must include:
- `@ID` header with participant metadata
- `*PAR:` lines with patient utterances
- WAB-AQ score in participant's `custom` field (format: `AphasiaType|Score`)

Example:
```
@UTF8
@Begin
@Languages: eng
@Participants: PAR Participant
@ID: eng|AphasiaBank|PAR|55;00.|male|aphasia||Participant|Broca|65.5|
*PAR: the boy is kicking the ball .
*PAR: and the dog is running .
@End
```

---

## Usage

### Complete Pipeline

Execute the pipeline step-by-step using HTCondor:
```bash
# Navigate to jobs directory
cd jobs

# Step 1: Extract metadata
condor_submit metadata_extraction.sub

# Step 2: Generate word alignments (choose CPU or GPU)
condor_submit whisperx_cpu_ALL.sub    # CPU (slower but always available)
# OR
condor_submit whisperx_gpu.sub        # GPU (10-20x faster)

# Step 3: Extract features
condor_submit extract_features.sub

# Step 4: Merge data
condor_submit merge_features_metadata.sub

# Step 5: Train SVR
condor_submit svr_pipeline.sub

# Monitor jobs
condor_q
watch -n 1 condor_q

# View logs
tail -f ../logs/svr_*.out
```

---

## Step-by-Step Guide

### Step 1: Metadata Extraction

Extract WAB-AQ scores and demographic information.

**Two methods available:**

#### Method A: From CSV AphasiaBank (Faster)
```bash
cd 01_metadata
python3 extract_metadata_from_csv.py
```

**Requirements**: Pre-processed CSV file `df_aphbank_pos_metrics.csv`

#### Method B: From .CHA Files (Reproducible)
```bash
cd 01_metadata
python3 extract_wab_metadata.py
```

**Requirements**: Only .CHA transcription files

**Output**: `data/patient_metadata_WAB.csv`
```csv
patient_id,QA,sex,age,aphasia_type,language
patient001,65.5,male,55,Broca,en
patient002,82.3,female,48,Anomic,en
patient003,45.2,male,62,Wernicke,es
...
```

**Time**: 
- Method A: ~1 minute
- Method B: ~5-10 minutes

---

### Step 2: Word-level Alignments (WhisperX)

Extract word-level timestamps using WhisperX.

#### Local Execution
```bash
cd 02_alignments

# CPU version (slower)
python3 generate_whisperx_alignments.py \
  --audio_base /path/to/audios_completos \
  --output ../data/word_alignments_ALL.csv \
  --model base \
  --device cpu

# GPU version (faster)
python3 generate_whisperx_alignments.py \
  --audio_base /path/to/audios_completos \
  --output ../data/word_alignments_ALL.csv \
  --model large-v2 \
  --device cuda
```

#### HTCondor Execution
```bash
cd jobs

# CPU
condor_submit whisperx_cpu_ALL.sub

# GPU
condor_submit whisperx_gpu.sub
```

**Output**: `data/word_alignments_ALL.csv`
```csv
patient_id,word,start_sec,end_sec
patient001,the,0.150,0.300
patient001,boy,0.350,0.550
patient001,is,0.600,0.700
...
```

**Time**: 
- CPU: ~10-20 seconds per audio
- GPU: ~1-2 seconds per audio

---

### Step 3: Extract DEN+DYS Features

Extract 28 acoustic-prosodic features from transcriptions + alignments.

#### Local Execution
```bash
cd 03_features

python3 build_den_dys.py \
  --cha_dir /path/to/transcripciones \
  --word_align_csv ../data/word_alignments_ALL.csv \
  --output ../data/features_den_dys_COMPLETO.csv
```

#### HTCondor Execution
```bash
cd jobs
condor_submit extract_features.sub
```

**Output**: `data/features_den_dys_COMPLETO.csv`
```csv
patient_id,language,den_words_per_min,den_phones_per_min,...,dys_pause_sec_mean
patient001,en,120.5,450.2,0.92,0.75,...,0.35
patient002,en,95.3,380.1,0.88,0.68,...,0.42
...
```

**Time**: ~2-3 seconds per patient

---

### Step 4: Merge Features + Metadata

Combine features with metadata.

#### Local Execution
```bash
cd 04_merge
python3 merge_features_metadata.py
```

#### HTCondor Execution
```bash
cd jobs
condor_submit merge_features_metadata.sub
```

**Output**: `data/dataset_FINAL_EN_ES.csv`
```csv
patient_id,language,QA,sex,age,aphasia_type,den_words_per_min,...,dys_pause_sec_mean
patient001,en,65.5,male,55,Broca,120.5,...,0.35
patient002,en,82.3,female,48,Anomic,95.3,...,0.42
...
```

**Time**: < 1 minute

---

### Step 5: Train SVR Model

Train and evaluate SVR with cross-validation.

#### Configuration

Update paths in `05_models/train_svr_den_dys.py` (lines 50-55):
```python
PROJECT_BASE = "/path/to/your/project"
DATA_BASE = os.path.join(PROJECT_BASE, "data")
CSV_FINAL = os.path.join(DATA_BASE, "dataset_FINAL_EN_ES.csv")
RESULTS_BASE = os.path.join(PROJECT_BASE, "resultados_svr")
```

#### Local Execution
```bash
cd 05_models
python3 train_svr_den_dys.py
```

#### HTCondor Execution
```bash
cd jobs
condor_submit svr_pipeline.sub
```

**Time**: ~20-40 minutes (depends on dataset size and grid search)

---

## Output Files

### SVR Results Structure
```
resultados_svr/
└── SVR_DEN_DYS_20251030_103000/
    ├── console.log                          # Full execution log
    ├── features_used.txt                    # List of 28 features
    ├── train_svr_den_dys.py                 # Copy of training script
    │
    ├── model_best_cv.pkl                    # Best CV model
    ├── model_final_all_EN.pkl               # Final retrained model
    ├── calibrator.pkl                       # Isotonic calibrator
    │
    ├── cv_results_full.csv                  # GridSearchCV results
    │
    ├── CV_EN_metrics.csv                    # CV regression metrics
    ├── CV_EN_predictions.csv                # CV predictions with errors
    ├── CV_EN_scatter.png                    # Scatter plot (pred vs real)
    ├── CV_EN_residuals.png                  # Residual analysis
    ├── CV_EN_errors.png                     # Error distribution
    ├── CV_EN_severity_report.csv            # Severity classification report
    ├── CV_EN_confusion_matrix.csv           # Confusion matrix (numbers)
    ├── CV_EN_confusion_matrix.png           # Confusion matrix (visual)
    │
    ├── CV_EN_CALIBRATED_*.csv/png           # Calibrated CV results
    ├── EN_IN_RAW_*.csv/png                  # In-sample evaluation
    ├── EN_IN_CALIBRATED_*.csv/png           # Calibrated in-sample
    │
    ├── INT_ES_RAW_*.csv/png                 # Internal validation (Spanish)
    ├── INT_ES_CALIBRATED_*.csv/png          # Calibrated internal validation
    │
    ├── EXT_CA_RAW_*.csv/png                 # External test (Catalan)
    └── EXT_CA_CALIBRATED_*.csv/png          # Calibrated external test
```

### Key Metrics Files

#### `CV_EN_CALIBRATED_metrics.csv`
```csv
MAE,RMSE,R2,Pearson_r,Spearman_rho,Acc@1,Acc@5,Acc@10,Exact,severity_accuracy
8.5,11.2,0.72,0.85,0.84,0.15,0.68,0.89,0.05,0.75
```

**Metrics Explained:**
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **R2**: R-squared (higher is better, max 1.0)
- **Pearson_r**: Pearson correlation (higher is better, max 1.0)
- **Spearman_rho**: Spearman correlation (higher is better, max 1.0)
- **Acc@N**: % predictions within N points of true value
- **Exact**: % predictions exactly correct (after rounding)
- **severity_accuracy**: % correct severity classification

#### `CV_EN_CALIBRATED_predictions.csv`
```csv
patient_id,QA_real,QA_pred_cont,QA_pred_int,severity_real,severity_pred,error_cont,error_int,abs_error_int
patient001,65.5,68.2,68,Moderate,Moderate,-2.7,-3,3
patient002,82.3,79.8,80,Mild,Mild,2.5,2,2
patient003,45.2,38.9,39,Severe,Severe,6.3,6,6
...
```

#### `CV_EN_severity_report.csv`
```csv
,precision,recall,f1-score,support
Very Severe,0.80,0.73,0.76,22
Severe,0.68,0.72,0.70,36
Moderate,0.76,0.79,0.78,132
Mild,0.81,0.76,0.79,160
accuracy,,,0.75,350
macro avg,0.76,0.75,0.76,350
weighted avg,0.76,0.75,0.76,350
```

---

## Methodology

### Feature Extraction

**DEN Features (18)**:
1. Words/min
2. Phones/min  
3. W ratio (words / words+interjections)
4. Open-class word ratio
5. Words/utterance (mean)
6. Phones/utterance (mean)
7-18. POS ratios (nouns, verbs, nouns/verb ratio, noun ratio, light verbs, determiners, demonstratives, prepositions, adjectives, adverbs, pronoun ratio, function words)

**DYS Features (10)**:
19. Fillers/min
20. Fillers/word
21. Fillers/phone
22. Pauses/min (total)
23. Long pauses/min (>400ms)
24. Short pauses/min (<=400ms)
25. Pauses/word (total)
26. Long pauses/word
27. Short pauses/word
28. Pause duration (mean seconds)

### Model Training

- **Algorithm**: Support Vector Regression (SVR)
- **Kernels tested**: RBF, Linear
- **Validation**: 4-fold GroupKFold (speaker-independent)
- **Hyperparameters Grid Search**: 
  - C in {1, 10, 100}
  - epsilon in {0.1, 1}
  - kernel in {rbf, linear}
  - shrinking in {True, False}
  - Total: 24 combinations
- **Preprocessing**: 
  - SimpleImputer (median strategy)
  - StandardScaler (mean=0, std=1)
- **Calibration**: Isotonic Regression (post-hoc, clipped to [0,100])

### Severity Classification

WAB-AQ scores are binned into severity categories:
- **Very Severe**: 0-25
- **Severe**: 25-50
- **Moderate**: 50-75
- **Mild**: 75-100

The model predicts continuous QA scores, which are then classified into severity categories for additional evaluation.

---

## Project Structure
```
aphasia-severity-prediction/
├── 01_metadata/
│   ├── extract_metadata_from_csv.py    # Method A: from CSV
│   └── extract_wab_metadata.py         # Method B: from .CHA
│
├── 02_alignments/
│   └── generate_whisperx_alignments.py # WhisperX word-level alignment
│
├── 03_features/
│   └── build_den_dys.py                # DEN+DYS feature extraction
│
├── 04_merge/
│   └── merge_features_metadata.py      # Combine features + metadata
│
├── 05_models/
│   └── train_svr_den_dys.py            # SVR training pipeline
│
├── jobs/                               # HTCondor submit files
│   ├── metadata_extraction.sub
│   ├── whisperx_cpu_ALL.sub
│   ├── whisperx_gpu.sub
│   ├── extract_features.sub
│   ├── merge_features_metadata.sub
│   └── svr_pipeline.sub
│
├── data/                               # Intermediate data files
│   ├── patient_metadata_WAB.csv
│   ├── word_alignments_ALL.csv
│   ├── features_den_dys_COMPLETO.csv
│   └── dataset_FINAL_EN_ES.csv
│
├── logs/                               # HTCondor logs
│   ├── metadata_*.out/err/log
│   ├── whisperx_*.out/err/log
│   ├── extract_*.out/err/log
│   ├── merge_*.out/err/log
│   └── svr_*.out/err/log
│
├── resultados_svr/                     # SVR training results
│   └── SVR_DEN_DYS_YYYYMMDD_HHMMSS/
│
├── archive_legacy/                     # Old scripts and outputs
│   ├── scripts/
│   └── outputs/
│
├── notebooks/                          # Jupyter notebooks for analysis
│
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

- Le et al. (2018) for the original methodology
- spaCy for NLP processing
- scikit-learn for machine learning tools
