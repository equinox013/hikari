---
title: HIKARI Complaint Driver Classifier
emoji: ✦
colorFrom: yellow
colorTo: gray
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
license: mit
---

# HIKARI · HIMARI · MACHI
### Complaint Driver Classification Pipeline

An end-to-end NLP pipeline that reads raw customer verbatims, strips PII, and multi-label classifies each complaint into one or more of **10 complaint driver categories** — built for utility sector CX analytics.

## How to use

1. Paste a raw customer complaint (email, web form, or phone log transcript) into the text area.
2. Select the **Case Origin** so HIMARI applies the correct channel-extraction rules.
3. Adjust the **confidence threshold** if needed (default 0.5).
4. Click **Classify**.

The app runs the full three-stage pipeline:
- **MACHI** redacts PII before any text is analysed
- **HIMARI** normalises and tokenises the cleaned text
- **HIKARI** predicts complaint driver labels and confidence scores

## Model Details

| Component | Architecture | Training Data |
|-----------|-------------|---------------|
| MACHI (PII redaction) | CRF tagger (sklearn-crfsuite) with hand-engineered lexical features | [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) |
| HIKARI (classifier) | Bidirectional LSTM + GloVe 6B 100d (frozen) + GlobalMaxPool1D | Internal utility sector verbatims labelled via Qualtrics TextIQ |

## Complaint Driver Labels

`Customer Service` · `Digital` · `Online Experience` · `Outages and Faults` · `Process` · `Reputation` · `Sustainability` · `Trust` · `Value for Money` · `Vulnerability & FDV`

## Notes

- PII patterns are AU-specific (mobile formats, postcodes, street types).
- GloVe weights are frozen inside `hikari_v2.keras` — no external embedding download required.
- This demo is for portfolio/research purposes. Do not submit real customer PII.
