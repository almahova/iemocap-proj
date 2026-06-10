# IEMOCAP Emotion Recognition: Data Science Project

3rd year data science project, Bar-Ilan University (Engineering Faculty).

**Team:** Chen Shmila, Alma Hova, Ohad Lavie

---

## Overview

The goal of this project is to build a multiclass emotion recognition model from speech data. Given an audio utterance, the model should predict the speaker's emotional state from one of 10 emotion categories.

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
iemocap_modelling.ipynb        # Original modelling notebook (Wav2Vec2, LOSO-CV)
notebooks/
├── iemocap-analysis.ipynb     # EDA notebook
├── text_classification.ipynb  # RoBERTa text-only classifier
├── audio_classification.ipynb # Wav2Vec2 audio-only classifier (LOSO-CV)
├── intermediate_fusion.ipynb  # RoBERTa + Wav2Vec2 intermediate (feature-level) fusion
└── late_fusion.ipynb          # RoBERTa + Wav2Vec2 late (decision-level) fusion
docs/
└── Projects_3rd_year_course_booklet.pdf
```

All modelling notebooks share the same setup:
- 3-class emotion grouping: **Negative · Positive · Neutral**
- Class imbalance handled with inverse-frequency class weights in the loss
- Early stopping (patience = 3 epochs on val Macro F1)
- Reported metrics: accuracy, Macro F1-Score, confusion matrix

---

## Progress

### Exploratory Data Analysis
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

### Text Classification (RoBERTa)
- `roberta-base` fine-tuned on the `transcription` column
- 80/10/10 stratified train/val/test split
- Best val Macro F1 reached at epoch 4, early stopped at epoch 7
- **Test results:** Accuracy **0.7052**, Macro F1 **0.6648**

### In Progress: Audio Classification (Wav2Vec2)
- `facebook/wav2vec2-base` fine-tuned end-to-end on raw audio
- Leave-One-Session-Out cross-validation (5 folds, fully speaker-independent)
- Evaluation: Macro F1 per fold + aggregated confusion matrix
- Currently running fold-by-fold training

### In Progress: Multimodal Fusion
- **Intermediate fusion**: combine RoBERTa text embeddings with Wav2Vec2 audio embeddings before classification: data loading/splitting set up, training pending
- **Late fusion**: combine the independent text and audio model predictions at the decision level: notebook scaffolded, not yet implemented

### Up Next
- Finish audio classification LOSO-CV run
- Train and evaluate intermediate and late fusion models
- Compare all approaches (text-only, audio-only, intermediate fusion, late fusion)
- Report writing
