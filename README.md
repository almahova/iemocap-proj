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
├── intermediate_fusion.ipynb  # RoBERTa + Wav2Vec2 intermediate (feature-level) fusion
└── late_fusion.ipynb          # RoBERTa + Wav2Vec2 late (decision-level) fusion
docs/
└── Projects_3rd_year_course_booklet.pdf
```

All modelling notebooks share the same evaluation setup:
- 3-class emotion grouping: **Negative · Positive · Neutral**
- Train on Sessions 1–4, test on Session 5 (fully speaker-independent)
- Class imbalance handled with inverse-frequency class weights in the loss
- Early stopping (patience = 3 epochs on val Macro F1)
- Reported metrics: accuracy, Macro F1-Score, confusion matrix

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
- Acoustic feature comparison by gender

### Text Classification (RoBERTa) ✓
- `roberta-base` fine-tuned on the `transcription` column
- Train on Sessions 1–4, held-out test on Session 5
- Early stopping on validation Macro F1 (patience = 3)

### Audio Classification (Wav2Vec2) ✓
- `facebook/wav2vec2-base` fine-tuned end-to-end on raw audio
- Train on Sessions 1–4, held-out test on Session 5 (speaker-independent)
- Feature encoder frozen; linear warmup + decay scheduler
- **Test results (Ses05): Macro F1 0.6324**

### Late Fusion ✓
- Weighted sum of text and audio logits (α\_text = 0.65, α\_audio = 0.35)
- Evaluated on Session 5 test set
- **Test results (Ses05): Macro F1 0.8173**

### In Progress: Intermediate Fusion
- Concatenates RoBERTa and Wav2Vec2 hidden-state embeddings before a shared MLP head
- Differential learning rates: 1e-5 for pretrained encoders, 1e-4 for fusion head
- Training pending

### Up Next
- Run intermediate fusion training and evaluation
- Compare all four approaches (text-only, audio-only, late fusion, intermediate fusion)
- Report writing
