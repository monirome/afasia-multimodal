# Automatic Assessment of Aphasia Severity using DEN+DYS Features

Automatic prediction of WAB-AQ (Western Aphasia Battery - Aphasia Quotient) scores using acoustic-prosodic features from spontaneous speech, replicating the methodology from **Le et al. (2018)**.

---

## Overview

This repository implements an **automatic speech analysis pipeline** for predicting aphasia severity using:

- **28 acoustic-prosodic features** (DEN + DYS) extracted from speech
- **Support Vector Regression (SVR)** with RBF/Linear kernels
- **Cross-validation** with GroupKFold (speaker-independent)
- **Isotonic calibration** for improved predictions

### Reference Paper

> **Le, D., et al. (2018)**. "Automatic Assessment of Speech Intelligibility for Individuals with Aphasia."  
> *Speech Communication*, 100, 1-11.

---

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
- Pause durations (mean, distribution)

---

## Pipeline Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT DATA                                   │
│  • Audio files (.wav)                                            │
│  • Transcriptions (.cha format - CHAT)                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Word-level Alignments (WhisperX)                       │
│  Script: generate_word_alignments_whisperx.py                   │
│  Output: word_alignments_ALL.csv                                │
│          (patient_id, word, start_sec, end_sec)                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Extract WAB-AQ Scores                                  │
│  Script: extract_wab_pylangacq.py                               │
│  Output: patient_metadata_WAB.csv                               │
│          (patient_id, QA, sex, age, aphasia_type)               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Extract DEN+DYS Features                               │
│  Script: extract_den_dys_complete.py                            │
│  Output: features_den_dys_COMPLETO.csv                          │
│          (patient_id, language, 28 features)                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Merge Features + Metadata                              │
│  Script: merge_data.py (inline in notebooks)                    │
│  Output: dataset_FINAL_den_dys_wab.csv                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Train & Evaluate SVR                                   │
│  Script: afasia_train_SVR_DEN_DYS_pipeline.py                   │
│  Output: resultados_svr/SVR_DEN_DYS_YYYYMMDD_HHMMSS/            │
│          • Models (pkl)                                          │
│          • Metrics (csv)                                         │
│          • Plots (png)                                           │
│          • Logs (console.log)                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Requirements

### System Requirements
- **OS**: Linux (tested on CentOS)
- **Python**: 3.9+
- **Memory**: 16GB+ RAM recommended
- **GPU**: Optional (for faster WhisperX processing)

### Python Dependencies
```txt
# Core ML
numpy>=1.26.4,<2.0
pandas>=2.2.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Speech Processing
whisperx==3.7.4
pylangacq>=0.19.0
spacy>=3.7.0

# NLP Models
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Utils
joblib>=1.3.0
```

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/aphasia-severity-prediction.git
cd aphasia-severity-prediction
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install core packages
pip install -r requirements.txt

# Install spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm  # if Spanish data
python -m spacy download ca_core_news_sm  # if Catalan data

# Install FFmpeg (required for WhisperX)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# CentOS/RHEL:
sudo yum install ffmpeg

# Or download static binary:
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz
cp ffmpeg-*/ffmpeg venv/bin/
```

---

## Dataset Structure
```
data/
├── audios_completos/          # Audio files
│   ├── patient001.wav
│   ├── patient002.wav
│   └── ...
│
└── transcripciones/           # CHAT transcriptions
    ├── patient001.cha
    ├── patient002.cha
    └── ...
```

### .CHA File Format (CHAT)

Transcription files must include:
- `@ID` header with participant metadata
- `*PAR:` lines with patient utterances
- WAB-AQ score in participant's `custom` field

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

### Step 1: Word Alignments (WhisperX)

Extract word-level timestamps using WhisperX.

#### Script: `generate_word_alignments_whisperx.py`
```bash
# Local execution
python3 generate_word_alignments_whisperx.py \
  --audio_base /path/to/audios \
  --output word_alignments_ALL.csv \
  --model base \
  --device cpu

# HTCondor submission
condor_submit whisperx_cpu_ALL.sub
```

#### HTCondor Submit File: `whisperx_cpu_ALL.sub`
```bash
executable = /path/to/venv/bin/python3
arguments  = generate_word_alignments_whisperx.py --audio_base /path/to/audios --output word_alignments_ALL.csv --model base --device cpu

output = logs/whisperx_$(Cluster).$(Process).out
error  = logs/whisperx_$(Cluster).$(Process).err
log    = logs/whisperx_$(Cluster).$(Process).log

request_cpus   = 8
request_memory = 32000

environment = "PYTHONUNBUFFERED=1 PATH=/path/to/venv/bin:$PATH"
should_transfer_files = NO

queue
```

**Output**: `word_alignments_ALL.csv`
```csv
patient_id,word,start_sec,end_sec
patient001,the,0.150,0.300
patient001,boy,0.350,0.550
...
```

**Time**: ~10 seconds per audio (CPU), ~1-2 seconds (GPU)

---

### Step 2: Extract WAB-AQ Scores

Extract WAB-AQ scores from .CHA transcription headers.

#### Script: `extract_wab_pylangacq.py`
```bash
python3 extract_wab_pylangacq.py
```

**Output**: `patient_metadata_WAB.csv`
```csv
patient_id,QA,sex,age,aphasia_type
patient001,65.5,male,55,Broca
patient002,82.3,female,48,Anomic
...
```

---

### Step 3: Extract DEN+DYS Features

Extract 28 acoustic-prosodic features from transcriptions + alignments.

#### Script: `extract_den_dys_complete.py`
```bash
# Local execution
python3 extract_den_dys_complete.py \
  --cha_dir /path/to/transcripciones \
  --word_align_csv word_alignments_ALL.csv \
  --output features_den_dys_COMPLETO.csv

# HTCondor submission
condor_submit extract_features.sub
```

#### HTCondor Submit File: `extract_features.sub`
```bash
executable = /path/to/venv/bin/python3
arguments  = extract_den_dys_complete.py --cha_dir /path/to/transcripciones --word_align_csv word_alignments_ALL.csv --output features_den_dys_COMPLETO.csv

output = logs/extract_$(Cluster).$(Process).out
error  = logs/extract_$(Cluster).$(Process).err
log    = logs/extract_$(Cluster).$(Process).log

request_cpus   = 8
request_memory = 32000

environment = "PYTHONUNBUFFERED=1"
should_transfer_files = NO

queue
```

**Output**: `features_den_dys_COMPLETO.csv`
```csv
patient_id,language,den_words_per_min,den_phones_per_min,...,dys_pause_sec_mean
patient001,en,120.5,450.2,...,0.35
patient002,en,95.3,380.1,...,0.42
...
```

**Time**: ~2-3 seconds per patient

---

### Step 4: Merge Data

Combine features with metadata.
```python
import pandas as pd

# Load data
df_features = pd.read_csv('features_den_dys_COMPLETO.csv')
df_metadata = pd.read_csv('patient_metadata_WAB.csv')

# Merge
df_final = pd.merge(
    df_features,
    df_metadata[['patient_id', 'QA', 'sex', 'age', 'aphasia_type']],
    on='patient_id',
    how='inner'
)

# Clean NaN
df_final = df_final.dropna(subset=['den_light_verbs', 'dys_pause_sec_mean'])

# Save
df_final.to_csv('dataset_FINAL_den_dys_wab.csv', index=False)

print(f"Final dataset: {len(df_final)} patients")
```

**Output**: `dataset_FINAL_den_dys_wab.csv`

---

### Step 5: Train SVR Model

Train and evaluate SVR with cross-validation.

#### Script: `afasia_train_SVR_DEN_DYS_pipeline.py`

**Prerequisites**: Update paths in script
```python
# Line ~35-40
CSV_FINAL = "/path/to/dataset_FINAL_den_dys_wab.csv"
RESULTS_BASE = "/path/to/resultados_svr"
```

**Execution**:
```bash
# Local execution
python3 afasia_train_SVR_DEN_DYS_pipeline.py

# HTCondor submission
condor_submit svr_pipeline.sub
```

#### HTCondor Submit File: `svr_pipeline.sub`
```bash
executable = /path/to/venv/bin/python3
arguments  = afasia_train_SVR_DEN_DYS_pipeline.py

output = logs/svr_$(Cluster).$(Process).out
error  = logs/svr_$(Cluster).$(Process).err
log    = logs/svr_$(Cluster).$(Process).log

request_cpus   = 8
request_memory = 32000

environment = "PYTHONUNBUFFERED=1"
should_transfer_files = NO

queue
```

**Time**: ~20-40 minutes (depends on dataset size and grid search)

---

## Output Files

### SVR Results Structure
```
resultados_svr/
└── SVR_DEN_DYS_20251030_103000/
    ├── console.log                    # Full execution log
    ├── features_used.txt              # List of 28 features
    │
    ├── model_best_cv.pkl              # Best CV model
    ├── model_final_all_EN.pkl         # Final retrained model
    ├── calibrator.pkl                 # Isotonic calibrator
    │
    ├── cv_results_full.csv            # GridSearchCV results
    │
    ├── CV_EN_metrics.csv              # Cross-validation metrics
    ├── CV_EN_predictions.csv          # CV predictions
    ├── CV_EN_scatter.png              # Scatter plot
    ├── CV_EN_residuals.png            # Residual analysis
    ├── CV_EN_errors.png               # Error distribution
    │
    ├── CV_EN_CALIBRATED_*.csv/png     # Calibrated CV results
    ├── EN_IN_RAW_*.csv/png            # In-sample evaluation
    ├── EN_IN_CALIBRATED_*.csv/png     # Calibrated in-sample
    │
    ├── INT_ES_*.csv/png               # Internal validation (Spanish)
    └── EXT_CA_*.csv/png               # External test (Catalan)
```

### Key Metrics Files

#### `CV_EN_CALIBRATED_metrics.csv`
```csv
MAE,RMSE,R2,Pearson_r,Spearman_rho,Acc@1,Acc@5,Acc@10,Exact
8.5,11.2,0.72,0.85,0.84,0.15,0.68,0.89,0.05
```

#### `CV_EN_CALIBRATED_predictions.csv`
```csv
patient_id,QA_real,QA_pred_cont,QA_pred_int,error_cont,error_int,abs_error_int
patient001,65.5,68.2,68,-2.7,-3,3
patient002,82.3,79.8,80,2.5,2,2
...
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
7-18. POS ratios (nouns, verbs, adjectives, etc.)

**DYS Features (10)**:
19. Fillers/min
20. Fillers/word
21. Fillers/phone
22-24. Pauses/min (total, long >400ms, short ≤400ms)
25-27. Pauses/word (total, long, short)
28. Pause duration (mean)

### Model Training

- **Algorithm**: Support Vector Regression (SVR)
- **Kernels tested**: RBF, Linear
- **Validation**: 4-fold GroupKFold (speaker-independent)
- **Hyperparameters**: 
  - C ∈ {1, 10, 100}
  - ε ∈ {0.1, 1}
  - Shrinking ∈ {True, False}
- **Calibration**: Isotonic Regression (post-hoc)

---

## Troubleshooting

### Common Issues

#### 1. **WhisperX Error: FFmpeg not found**
```bash
# Install FFmpeg in venv
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz
cp ffmpeg-*/ffmpeg venv/bin/
chmod +x venv/bin/ffmpeg

# Update .sub file
environment = "PYTHONUNBUFFERED=1 PATH=/path/to/venv/bin:$PATH"
```

#### 2. **spaCy Model Not Found**
```bash
python -m spacy download en_core_web_sm
```

#### 3. **NumPy Version Conflict**
```bash
pip install "numpy<2.0"
```

#### 4. **Memory Errors**

Increase memory in .sub files:
```bash
request_memory = 64000  # 64GB
```
