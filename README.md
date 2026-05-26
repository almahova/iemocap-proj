# IEMOCAP Emotion Recognition — Data Science Project

3rd year data science project, Bar-Ilan University (Engineering Faculty).

**Team:** Chen Shmila, Alma Hova, Ohad Lavie

---

## Overview

The goal of this project is to build a multiclass emotion recognition model from speech data. Given an audio utterance, the model should predict the speaker's emotional state from one of 10 emotion categories.

---

## Dataset

**IEMOCAP** — Interactive Emotional Dyadic Motion Capture corpus.

- 10,039 utterances from 10 speakers (5 female, 5 male) across 5 dyadic sessions
- Each session contains both improvised and scripted scenarios
- Features include soft-label emotion scores, dimensional emotion ratings (activation, valence, dominance), and acoustic features (pitch, speaking rate, RMS energy)
- 10 emotion classes: `neutral`, `frustrated`, `angry`, `sad`, `happy`, `excited`, `surprise`, `fear`, `disgust`, `other`
- Loaded via HuggingFace: [`AbstractTTS/IEMOCAP`](https://huggingface.co/datasets/AbstractTTS/IEMOCAP)

---

## Project Structure

```
notebooks/
└── iemocap-analysis.ipynb   # Main analysis notebook
docs/
└── Projects_3rd_year_course_booklet.pdf
```

---

## Progress

### ✅ Exploratory Data Analysis
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

### 🔜 Up Next
- Feature engineering
- Model training and evaluation
- Speaker-independent cross-validation (GroupKFold)
