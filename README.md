# IEMOCAP Emotion Recognition: Data Science Project

3rd year data science project, Bar-Ilan University (Engineering Faculty).

**Team:** Chen Shmila, Alma Hova, Ohad Lavie

---

## Overview

The goal of this project is to build a multimodal emotion recognition system from speech data. We group the original 10 IEMOCAP emotion labels into 3 classes — **Negative**, **Positive**, and **Neutral** — and compare text-only, audio-only, and multimodal fusion approaches.

---

## Dataset

**IEMOCAP**: Interactive Emotional Dyadic Motion Capture corpus.

- 10,039 utterances from 10 speakers (5 female, 5 male) across 5 dyadic sessions
- Each session contains both improvised and scripted scenarios
- Features include soft-label emotion scores, dimensional emotion ratings (activation, valence, dominance), and acoustic features (pitch, speaking rate, RMS energy)
- 10 emotion classes: `neutral`, `frustrated`, `angry`, `sad`, `happy`, `excited`, `surprise`, `fear`, `disgust`, `other`
- Loaded via HuggingFace: [`AbstractTTS/IEMOCAP`](https://huggingface.co/datasets/AbstractTTS/IEMOCAP)

---

## Project Structure

```
notebooks/
├── iemocap-analysis.ipynb     # EDA notebook
├── text_classification.ipynb  # RoBERTa text-only classifier
├── audio_classification.ipynb # Wav2Vec2 audio-only classifier
└── late_fusion.ipynb          # RoBERTa + Wav2Vec2 late (decision-level) fusion
docs/
├── report.md                  # Full project report
└── Projects_3rd_year_course_booklet.pdf
```

All modelling notebooks share the same evaluation setup:
- 3-class emotion grouping: **Negative · Positive · Neutral**
- Train on Sessions 1–4, test on Session 5 (fully speaker-independent)
- Class imbalance handled with inverse-frequency class weights in the loss
- Early stopping (patience = 3 epochs on val Macro F1)
- Reported metrics: accuracy, Macro F1-Score, confusion matrix

---

## Results Summary

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| RoBERTa (text-only) | 0.6793 | 0.6519 | 0.6900 |
| Wav2Vec2 (audio-only) | 0.6779 | 0.6324 | 0.6799 |
| Late Fusion (best α) | 0.8392 | 0.8173 | 0.8431 |

All models evaluated on Session 5 (n=2,170 utterances, fully speaker-independent).

---

## Progress

### Exploratory Data Analysis ✓
- Dataset overview: shape, types, missing values
- Emotion class distribution and class imbalance analysis
- Outlier detection across all numerical features
- Emotion distribution by gender
- Valence–Activation 2D emotional space
- Acoustic profiles (pitch, RMS, speaking rate) per emotion class
- Correlation analysis of numerical features
- Dimensional emotion profiles (activation, valence, dominance per class)
- Session type analysis (improvised vs. scripted)

### Text Classification (RoBERTa) ✓
- `roberta-base` fine-tuned on the `transcription` column
- Train on Sessions 1–4, held-out test on Session 5 (speaker-independent)
- Early stopping on validation Macro F1 (patience = 3), best at epoch 4 of 7
- **Test results — Accuracy: 0.6793 | Macro F1: 0.6519**

### Audio Classification (Wav2Vec2) ✓
- `facebook/wav2vec2-base` fine-tuned on raw 16kHz waveforms
- Convolutional feature encoder frozen; transformer layers fine-tuned
- Train on Sessions 1–4, held-out test on Session 5 (speaker-independent)
- **Test results — Accuracy: 0.6779 | Macro F1: 0.6324**

### Late Fusion ✓
- Weighted combination of text and audio output logits: α\_text = 0.65, α\_audio = 0.35
- No additional training — uses checkpoints from the two unimodal models
- **Test results — Accuracy: 0.8392 | Macro F1: 0.8173** (+0.17 over text-only)

### Up Next
- Re-run audio and fusion models with corrected 8.5s max duration
- Final results comparison across all three models
- Export report to PDF
