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

## Results Summary

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| RoBERTa (text-only) | 0.6793 | 0.6519 | 0.6948 |
| Wav2Vec2 (audio-only) | 0.6880 | 0.6403 | 0.6874 |
| Late Fusion (best α=0.60) | — | 0.7104 | — |

All models evaluated on Session 5 (n=2,170 utterances, fully speaker-independent).

---

## Repository Structure

The project is developed across multiple branches, each representing a phase:

| Branch | Contents |
|--------|----------|
| `main` | Project overview |
| `eda-analysis` | EDA notebook — dataset exploration and analysis |
| `data-modelling` | All modelling notebooks, report, and final deliverables |

All modelling work — including the final report and presentation — is on the **`data-modelling`** branch.
