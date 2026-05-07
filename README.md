# MACHI · HIMARI · HIKARI
### Complaint Driver Classification Pipeline

> Thousands of customers contact their water utility every year. They're angry about their bills, confused by outages, struggling to get through on the phone, or dealing with something far more serious — financial hardship, domestic violence, a meter dispute that's become a crisis. The complaints are real. The signal is there. But reading every verbatim, tagging every topic, routing every case? That doesn't scale.
>
> This pipeline reads the raw text so analysts don't have to.

---

## The Names

The three components are named after characters from *Interviews with Monster Girls* (*Demi-chan wa Kataritai*) — an anime about a high school teacher who genuinely wants to understand the demi-humans in his class. It felt right for a project built around *listening* to people at their most frustrated. The names also happen to backronym perfectly into their actual technical roles, which I consider a minor victory.

| Name | Character | Role in pipeline | Expanded acronym |
|------|-----------|-----------------|-----------------|
| **MACHI** | Kyoko Machi — the dullahan, a being defined by *separation* | Separates PII from the rest of the text | **M**asking **A**lgorithm for **C**ontextual & **H**euristic **I**dentification |
| **HIMARI** | Himari Takanashi — the ordinary human twin, the *translator* between worlds | Translates raw verbatims into something the model can understand | **H**euristic **I**nput **M**apping & **A**rray **R**eduction **I**nterface |
| **HIKARI** | Hikari Takanashi — the vampire, *perceptive* and direct | Perceives the complaint driver signal in preprocessed text | **H**igh-dimensional **I**nference & **K**ey-driver **A**nalysis **R**outing **I**ntelligence |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Data & Label Origins](#3-data--label-origins)
4. [Component Reference](#4-component-reference)
   - [MACHI — PII Scrubber](#machi--pii-scrubber-machi_v2pkl)
   - [HIMARI — Preprocessing Orchestrator](#himari--preprocessing-orchestrator)
   - [HIKARI — Complaint Driver Classifier](#hikari--complaint-driver-classifier-hikari_v2keras)
5. [Complaint Driver Labels](#5-complaint-driver-labels)
6. [Repository Structure](#6-repository-structure)
7. [Environment & Dependencies](#7-environment--dependencies)
8. [Quickstart — Running Inference](#8-quickstart--running-inference)
9. [Deployment Guide](#9-deployment-guide)
10. [Known Limitations & Future Work](#10-known-limitations--future-work)

---

## 1. Project Overview

Customer complaints submitted via email, web forms, and phone logs contain rich qualitative signal that is difficult to analyse at scale. This project builds a three-stage ML pipeline that automates the extraction of complaint drivers from free-text verbatims.

The problem it solves is real: in a CX analytics context, an analyst team might receive thousands of complaints per month. Manual TextIQ classification in Qualtrics covers historical data, but it doesn't scale to new incoming cases without repeated human review. This pipeline changes that.

The three stages are:

1. **PII Redaction (MACHI)** — named-entity recognition strips personal information before any text is stored or processed downstream. No analyst should ever see a customer's address or phone number in a model output.
2. **Text Preprocessing (HIMARI)** — channel-aware cleaning handles the structural differences between an email thread, a web form submission, and a phone transcript. Then normalises, tokenises, and prepares the sequence for the classifier.
3. **Multi-Label Classification (HIKARI)** — a Bidirectional LSTM with frozen GloVe embeddings assigns one or more *complaint driver* labels to each verbatim, enabling structured reporting on customer pain points at scale.

The pipeline was developed in a **Databricks / Azure environment** with PySpark for data ingestion, and produces a labelled output DataFrame suitable for BI dashboarding.

---

## 2. Pipeline Architecture

```
Raw Customer Verbatim
        │
        ▼
┌───────────────────────────────────────────────┐
│  HIMARI  (Preprocessing Orchestrator)         │
│                                               │
│  1. Channel extraction                        │
│     • Email  → strip headers & reply threads  │
│     • Web    → extract "Case Description"     │
│     • Other  → pass-through                   │
│                                               │
│  2. Normalisation                             │
│     • Lowercase  •  Strip whitespace          │
│     • Remove non-ASCII characters             │
│                                               │
│  3. PII Redaction  ──────────────► MACHI      │
│     • Regex pre-pass (emails, phones,         │
│       addresses, AU postcodes, IDs)           │
│     • CRF sequential tagging                  │
│       (BIO-tag NER → [redacted])              │
│                                               │
│  4. Stopword Removal                          │
│     • NLTK English + domain custom words      │
│                                               │
│  5. Sequence wrapping                         │
│     • "<start> … <end>"                       │
│     • Truncate to MAX_SEQUENCE_LENGTH - 2     │
└───────────────────────────────────────────────┘
        │
        ▼  cleaned text
┌───────────────────────────────────────────────┐
│  HIKARI  (Complaint Driver Classifier)        │
│                                               │
│  Tokenizer (tokenizer.pkl)                    │
│    → integer sequence → pad to 196 tokens     │
│                                               │
│  BiLSTM model (hikari_v2.keras)               │
│    Embedding (GloVe 100d, frozen)             │
│    → BiLSTM(64) → GlobalMaxPool1D             │
│    → BatchNorm → Dropout(0.3)                 │
│    → Dense(128, relu) → Dropout(0.3)          │
│    → Dense(64, relu)  → Dropout(0.3)          │
│    → Dense(10, sigmoid)                       │
│                                               │
│  Threshold @ 0.5 → multi-hot label vector     │
└───────────────────────────────────────────────┘
        │
        ▼
Predicted Drivers + Confidence Scores
e.g. [("Value for Money", 0.87), ("Trust", 0.61)]
```

---

## 3. Data & Label Origins

### Training Data

| Component | Dataset |
|-----------|---------|
| MACHI (NER/PII) | [`ai4privacy/pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) — English subset (`english_pii_43k.jsonl`) from HuggingFace |
| HIKARI (Classification) | Internal customer verbatims labelled with complaint driver topics |

### Label Generation — Qualtrics TextIQ

The ground-truth labels used to train HIKARI were derived using **Qualtrics TextIQ**, a built-in NLP tool within the Qualtrics CX platform. TextIQ performs unsupervised topic discovery and sentiment analysis on open-text survey responses. Each verbatim was associated with one or more complaint driver *topics*, and the resulting topic-verbatim associations were exported and used as the multi-label training targets.

This means the model learns to replicate the topic associations that TextIQ and human reviewers established — making it suitable for scaling classification to new verbatims without repeated manual review.

### Label Encoding

Topics were one-hot encoded into a binary multi-hot label matrix `y` of shape `(n_samples, 10)`. A single verbatim may carry multiple labels — a complaint about a billing error from a vulnerable customer might be labelled both *Value for Money* and *Vulnerability & FDV*, which is exactly the kind of nuance the pipeline needs to preserve.

---

## 4. Component Reference

---

### MACHI — PII Scrubber (`machi_v2.pkl`)

**Notebook:** `PII_Scrubber_LSTM_Model_-_MACHI_v1.ipynb`

#### Purpose
Prevent downstream storage or analysis of personally identifiable customer information. MACHI runs before any text leaves the preprocessing stage. The dullahan reference isn't just aesthetic — like Machi herself, the whole point is to keep certain things *separated*.

#### Architecture (v1 — Training Reference)
The training notebook implements a **Bidirectional LSTM** sequence tagger:

```
Input (token IDs, padded to MAX_LEN)
  └─ Embedding (128-dim, learned from scratch)
       └─ BiLSTM (64 units, return_sequences=True)
            └─ TimeDistributed Dense (n_tags, softmax)
```

Trained with `sparse_categorical_crossentropy` for 3 epochs on the `ai4privacy` BIO-tagged dataset.

#### Deployed Artefact (v2 — Inference)
In production (HIMARI), the deployed PII model is `machi_v2.pkl` — a **CRF (Conditional Random Field)** tagger loaded via `joblib` / `sklearn_crfsuite`. The CRF was trained with hand-engineered lexical features per token:

| Feature | Description |
|---------|-------------|
| `word.lower` | Lowercase form of the token |
| `word[-3:]` | Suffix trigram |
| `word.isupper` | All-caps flag |
| `word.istitle` | Title-case flag |
| `word.isdigit` | Numeric flag |
| `-1:word.*` | Same features for the preceding token |
| `+1:word.*` | Same features for the following token |
| `BOS` / `EOS` | Sentence boundary markers |

#### Redaction Strategy
MACHI uses a **two-pass hybrid** approach for robustness:

1. **Regex pre-pass** — catches patterns that tokenisers break apart (emails, AU mobile numbers `XXXX XXX XXX`, street addresses matched against the full AU Post street-type list, state/postcode patterns, 7/9-digit numeric IDs, Medicare/concession formats).
2. **CRF sequential pass** — BIO-tagged NER on the regex-cleaned text. Any `B-` entity span and its trailing `I-` tokens are replaced with `[redacted]`.

#### I/O

| | |
|---|---|
| **Input** | Raw text string |
| **Output** | Text string with PII replaced by `[redacted]` |

---

### HIMARI — Preprocessing Orchestrator

**Notebook:** `Complaint_Driver_Preprocessing_Script_-_HIMARI_v2.ipynb`

#### Purpose
HIMARI is the orchestration layer that sits between raw CRM data and the HIKARI classifier. It is channel-aware — it knows that an email complaint looks structurally different from a web form submission — and applies appropriate extraction before normalising and feeding into HIKARI. Think of Himari as the human in the room: she doesn't have supernatural abilities, but she understands the context everyone else is operating in and makes the pipeline actually work.

#### Processing Steps

```python
def himari_preprocess(text, case_origin, stop_words):
```

| Step | Detail |
|------|--------|
| **1. Channel extraction** | `Email` → strips `From/Sent/To/Subject` headers, cuts at inline reply markers (`On … wrote:`), drops forwarded message blocks. `Web` → extracts content between `Case Description:` and `First Name:` regex anchors. `Other` → pass-through. |
| **2. Normalise** | `.lower().strip()` |
| **3. PII redaction** | Calls `redact_crf(text)` — the MACHI hybrid pass described above |
| **4. Stopword removal** | NLTK English stopwords + domain custom word (`'water'`, since the domain is a water utility) |
| **5. Truncation** | Filtered tokens truncated to `MAX_SEQUENCE_LENGTH - 2` to ensure the final sequence fits HIKARI's input shape after adding `<start>` and `<end>` |
| **6. Wrap** | `"<start> " + body + " <end>"` |

#### Batch Processing (`process_cases`)
HIMARI v2 includes a `process_cases()` function that ingests a Spark-loaded complaints DataFrame with columns `['Case Number', 'Description', 'Case Origin']` and returns a DataFrame with `['Case Number', 'clean_comment', 'drivers']` — enabling bulk scoring of new cases.

#### I/O

| | |
|---|---|
| **Input** | Raw verbatim string + case origin string (`"Email"` / `"Web"` / `"Other"`) |
| **Output** | Cleaned, tokenised, redacted string with `<start>` / `<end>` markers |

---

### HIKARI — Complaint Driver Classifier (`hikari_v2.keras`)

**Notebook:** `Complaint_Driver_LSTM_RNN_Model_-_HIKARI_v2.ipynb`

#### Purpose
Multi-label classification of preprocessed customer verbatims into complaint driver categories. Designed to operate at scale on CRM exports, replacing manual TextIQ classification for new incoming complaints. The name fits: HIKARI is perceptive, fast, and goes straight to the point.

#### Model Architecture

```
Input: integer token sequence, shape (None, 196)
  │
  ├─ Embedding (vocab_size × 100, GloVe 6B 100d, trainable=False)
  │
  ├─ Bidirectional LSTM (64 units, return_sequences=True)
  │     • Processes token sequence in both directions
  │     • return_sequences=True preserves temporal resolution for pooling
  │
  ├─ GlobalMaxPooling1D
  │     • Collapses sequence dimension, retaining most salient feature per dim
  │
  ├─ BatchNormalization
  │
  ├─ Dropout (0.3)
  │
  ├─ Dense (128, ReLU)
  │
  ├─ Dropout (0.3)
  │
  ├─ Dense (64, ReLU)
  │
  ├─ Dropout (0.3)
  │
  └─ Dense (10, Sigmoid)   ← one output node per complaint driver
```

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Embedding | GloVe 6B 100-dimensional, frozen |
| Optimiser | Adam (default learning rate) |
| Loss | Binary cross-entropy (multi-label) |
| Metric | Accuracy |
| Epochs | 75 (best val_accuracy checkpoint saved) |
| Batch size | 32 |
| Data split | 70% train / 15% validation / 15% test |
| Random seed | `20250417` |
| Sequence max length | 196 tokens (post-padding / truncation) |
| Inference threshold | 0.5 (configurable) |

#### Evaluation

The model is evaluated with:
- **Per-class classification report** (precision, recall, F1-score)
- **ROC-AUC per driver** (multi-label one-vs-rest)
- **Multi-label confusion matrix** (one 2×2 matrix per driver)
- **F1 bar chart** visualisation across all 10 drivers

#### I/O

| | |
|---|---|
| **Input** | Integer-padded token sequence of shape `(1, 196)` |
| **Output** | Probability vector of shape `(10,)` — one sigmoid score per driver |
| **Post-threshold** | List of `(driver_name, confidence)` tuples where `confidence ≥ 0.5` |

---

## 5. Complaint Driver Labels

The model outputs a probability score for each of the following 10 complaint drivers. Labels were defined and initially tagged in Qualtrics TextIQ.

| Index | Driver | Description |
|-------|--------|-------------|
| 0 | **Customer Service** | Staff behaviour, responsiveness, call handling |
| 1 | **Digital** | App, website, or digital tool issues |
| 2 | **Online Experience** | Self-service portal, account management online |
| 3 | **Outages and Faults** | Service interruptions, infrastructure failures |
| 4 | **Process** | Internal workflows, wait times, escalation paths |
| 5 | **Reputation** | Brand trust, media, social perception concerns |
| 6 | **Sustainability** | Environmental concerns, green credentials |
| 7 | **Trust** | Transparency, honesty, data handling concerns |
| 8 | **Value for Money** | Billing, pricing, charges, concessions |
| 9 | **Vulnerability & FDV** | Customers in hardship, family/domestic violence context |

---

## 6. Repository Structure

```
📁 hikari-himari-machi/
│
├── 📓 PII_Scrubber_LSTM_Model_-_MACHI_v1.ipynb
│       Training notebook for the MACHI PII tagger (BiLSTM reference implementation)
│
├── 📓 Complaint_Driver_Preprocessing_Script_-_HIMARI_v2.ipynb
│       Preprocessing orchestrator — channel-aware cleaning, PII redaction, tokenisation
│
├── 📓 Complaint_Driver_LSTM_RNN_Model_-_HIKARI_v2.ipynb
│       HIKARI model training, evaluation, and inference notebook
│
├── 🌐 app.py
│       Streamlit inference app for Hugging Face Spaces deployment
│
├── 📋 requirements.txt
│       Python dependencies for HF Spaces / local deployment
│
├── 📋 README_SPACE.md
│       Hugging Face Space card (rename to README.md when creating the HF Space)
│
├── 📁 keras-pkl/
│   ├── 🤖 hikari_v2.keras
│   │       Saved Keras v3 model — full model including weights and architecture
│   │
│   ├── 📦 machi_v2.pkl
│   │       Trained CRF PII tagger — loaded via joblib
│   │
│   └── 📦 tokenizer.pkl
│           Fitted Keras Tokenizer — maps words to integer indices
│           (must match the vocabulary HIKARI was trained on)
│
└── 📄 README.md
```

---

## 7. Environment & Dependencies

The pipeline was developed and tested in a **Databricks Runtime** environment (Azure). For local or Hugging Face Spaces deployment:

```bash
pip install tensorflow scikit-learn sklearn_crfsuite joblib nltk streamlit numpy
```

| Library | Role |
|---------|------|
| `tensorflow` / `keras` | Loading and running `hikari_v2.keras` |
| `sklearn_crfsuite` | Running the CRF tagger in `machi_v2.pkl` |
| `joblib` | Deserialising `machi_v2.pkl` |
| `nltk` | English stopwords (`nltk.download('stopwords')`) |
| `numpy` | Array operations |
| `streamlit` | Web UI for HF Spaces deployment |
| `pyspark` | Required only for batch ingestion from Azure Data Lake; not needed for single-comment inference |

> **Note on GloVe:** GloVe embeddings (`glove.6B.100d`) were used during training with `trainable=False`. The embedding weights are frozen inside `hikari_v2.keras` — **GloVe does not need to be downloaded for inference**.

---

## 8. Quickstart — Running Inference

```python
import joblib
import pickle
import re
import nltk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import int_shape

nltk.download('stopwords')
from nltk.corpus import stopwords

# ── Load artefacts ─────────────────────────────────────────────────────────
hikari_model = load_model('keras-pkl/hikari_v2.keras', compile=False)
crf_model    = joblib.load('keras-pkl/machi_v2.pkl')

with open('keras-pkl/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_SEQUENCE_LENGTH = int_shape(hikari_model.input)[1]  # auto-detects 196

stop_words = set(stopwords.words('english')).union({'water'})

TOPICS = [
    'Customer Service', 'Digital', 'Online Experience', 'Outages and Faults',
    'Process', 'Reputation', 'Sustainability', 'Trust',
    'Value for Money', 'Vulnerability & FDV'
]

# ── Paste in the helper functions from HIMARI notebook ────────────────────
# (word2features, sent2features, tokenize_with_offsets,
#  STREET_PATTERN, redact_crf, himari_preprocess)
# Or copy directly from app.py — all helpers are inlined there.

# ── Run inference ──────────────────────────────────────────────────────────
def predict(raw_comment: str, case_origin: str = "Other", threshold: float = 0.5):
    cleaned  = himari_preprocess(raw_comment, case_origin, stop_words)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH,
                             padding='post', truncating='post')
    probs    = hikari_model.predict(padded, verbose=0)[0]

    results = [(TOPICS[i], round(float(p), 3))
               for i, p in enumerate(probs) if p >= threshold]
    return sorted(results, key=lambda x: x[1], reverse=True), cleaned

# Example
drivers, cleaned = predict(
    "My bill went up $200 with no explanation and I can't log into the portal.",
    case_origin="Email"
)
print("Cleaned:", cleaned)
print("Drivers:", drivers)
```

---

## 9. Deployment Guide

### Hugging Face Spaces (Recommended)

All three artefact files (`hikari_v2.keras`, `machi_v2.pkl`, `tokenizer.pkl`) total approximately **9.4 MB** — well within HF Spaces limits. The full inference pipeline, including all HIMARI helper functions, is inlined in `app.py`.

**Steps to deploy:**

```bash
# 1. Create a new HF Space (Streamlit SDK) via the HF website

# 2. Clone the Space repository
git clone https://huggingface.co/spaces/<your-username>/hikari-complaint-classifier

# 3. Copy project files into the Space root
cp app.py requirements.txt <space-dir>/
cp -r keras-pkl/ <space-dir>/

# 4. Use README_SPACE.md as the Space README
cp README_SPACE.md <space-dir>/README.md

# 5. Push
cd <space-dir>
git add .
git commit -m "feat: deploy HIKARI complaint driver classifier"
git push
```

### Can the three artefact files be operationalised without the notebooks?

**Yes — `hikari_v2.keras`, `machi_v2.pkl`, and `tokenizer.pkl` are sufficient for inference.** All HIMARI helper functions are inlined in `app.py`. No training data, GloVe files, or Spark environment is needed at inference time.

### Alternative: FastAPI Backend

For production API deployment (e.g., behind an internal CRM integration):

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load artefacts at startup (see Quickstart above)

class Request(BaseModel):
    verbatim: str
    case_origin: str = "Other"
    threshold: float = 0.5

@app.post("/predict")
def predict_drivers(req: Request):
    drivers, cleaned = predict(req.verbatim, req.case_origin, req.threshold)
    return {"cleaned_text": cleaned, "drivers": drivers}
```

---

## 10. Known Limitations & Future Work

### Limitations

| Area | Detail |
|------|--------|
| **Fixed vocabulary** | The tokenizer was fitted on training verbatims only. Domain-specific new words (e.g., new product names) will be mapped to `<oov>` and may reduce accuracy. |
| **Label imbalance** | Some drivers (e.g., *Sustainability*, *Reputation*) likely appeared far less frequently in training data than others (e.g., *Value for Money*, *Customer Service*). F1 scores vary substantially across classes. |
| **Threshold sensitivity** | A global threshold of 0.5 is applied across all drivers. Per-driver tuned thresholds (derived from ROC curves) would improve precision/recall trade-offs for low-frequency classes. |
| **AU-specific PII patterns** | The MACHI regex pre-pass targets Australian phone/address/postcode formats. Adaptation required for other markets. |
| **Static GloVe embeddings** | GloVe 6B 100d was trained on general web text. Domain-specific embeddings trained on utility industry text may improve classification of sector-specific language. |

### Future Work

- **Threshold optimisation per class** — select per-driver cutoffs using validation-set ROC curves.
- **Fine-tuned transformer** — replace the BiLSTM + GloVe stack with a domain-adapted BERT or DistilBERT.
- **Active learning loop** — route low-confidence predictions (`max_prob < 0.6`) back to a human review queue for new labelled data.
- **Confidence calibration** — sigmoid outputs are not calibrated probabilities; Platt scaling or isotonic regression could improve reliability.
- **MACHI v2 CRF retraining notebook** — a dedicated training notebook for the deployed CRF (`machi_v2.pkl`) would complete the documentation trail.
