# Multimodal Speech Emotion Recognition Using IEMOCAP

**Course:** Introduction to Data Science Project — Bar-Ilan University, Engineering Faculty, Semester B 2026

**Team:** Chen Shmila, Alma Hova, Ohad Lavie

**Instructor:** Dr. Alexandra (Litinsky) Simanovsky | **TA:** Renana Opochinsky

---

## Key Messages (Presentation Summary)

These are the four main points to communicate in any presentation of this project:

1. **The problem is multimodal by nature.** Emotion is expressed through both words and voice. A system that ignores either channel is leaving information on the table.
2. **Text and audio alone are comparable — and limited.** RoBERTa (text) and Wav2Vec2 (audio) both achieve ~0.65 Macro F1, a moderate result, and they fail on different examples.
3. **Combining them works dramatically better.** Late fusion — a simple weighted sum of their outputs, requiring no additional training — jumps to 0.82 Macro F1, a +0.17 gain. This is the central result of the project.
4. **Neutral is the hardest emotion to classify across all models**, because it lacks distinctive words or acoustic patterns. Fusion helps most here (+0.23 F1 for Neutral).

---

## Abstract

This project addresses the task of automatic speech emotion recognition (SER) using the IEMOCAP corpus. We grouped the original ten emotion labels into three classes — Negative, Positive, and Neutral — and compared four approaches: text-only classification using RoBERTa, audio-only classification using Wav2Vec2, late fusion (weighted combination of output logits), and intermediate fusion (joint fine-tuning with a shared MLP head). All models were evaluated on a speaker-independent held-out test set (Session 5, n=2,170 utterances). The text-only model achieved Macro F1 of 0.65, the audio-only model 0.63, and late fusion improved substantially to 0.82 by leveraging complementary information from both modalities. Intermediate fusion training has been initiated but not yet completed.

---

## 1. Introduction

### 1.1 What Is Speech Emotion Recognition?

Automatic emotion recognition from speech is the task of identifying the emotional state of a speaker from a recorded conversation. Emotion is not a single signal — it is expressed through multiple channels simultaneously:

- **What is said** (the words and their meaning — the linguistic or text modality)
- **How it is said** (the acoustic properties of the voice — pitch, energy, speaking rate, pauses)
- **The combination of both** (words said in an unexpected tone carry different emotional weight)

A robust emotion recognition system should ideally leverage all available channels. This project directly investigates which modality carries more useful signal, and whether combining them outperforms either alone.

### 1.2 Why Does It Matter?

Practical applications include:
- **Mental health monitoring**: detecting distress or mood changes from speech patterns
- **Human-computer interaction**: voice assistants that respond appropriately to emotional context
- **Customer service analytics**: flagging frustrated or upset callers in real time
- **Clinical and educational settings**: monitoring emotional engagement or stress

### 1.3 Project Goals

This project has four concrete goals:

1. Build and evaluate a text-only classifier using RoBERTa (a pre-trained transformer)
2. Build and evaluate an audio-only classifier using Wav2Vec2 (a pre-trained speech encoder)
3. Combine both modalities using **late fusion** (decision-level combination)
4. Combine both modalities using **intermediate fusion** (feature-level combination with joint learning)

These four models form a natural progression from single-modality baselines to increasingly sophisticated multimodal systems.

### 1.4 What Was Done in This Project (Overview of Phases)

The project was developed in the following sequence, each phase building on the previous:

| Phase | What Was Done | Status |
|-------|---------------|--------|
| 1. EDA | Explored the IEMOCAP dataset: distributions, acoustic profiles, speaker differences | Complete |
| 2. Text Classification | Fine-tuned RoBERTa on transcriptions; ran full training locally | Complete |
| 3. Audio Classification | Fine-tuned Wav2Vec2 on raw audio waveforms | Complete |
| 4. Late Fusion | Combined text and audio model outputs with a weighted sum; no training needed | Complete |
| 5. Intermediate Fusion | Joint fine-tuning of both encoders with a shared MLP head | Training initiated — not yet complete |

---

## 2. Dataset

### 2.1 What Is IEMOCAP?

IEMOCAP (Interactive Emotional Dyadic Motion Capture) [1] is a widely used benchmark dataset for multimodal emotion recognition. It was created at the University of Southern California Signal Analysis and Interpretation Laboratory (SAIL).

- **10,039 utterances** from 10 speakers (5 female, 5 male), recorded in pairs
- **5 sessions**, each pairing two speakers (one male + one female)
- **Two scenario types per session**: improvised conversations (designed to naturally elicit emotion) and scripted dialogues
- **Rich annotation**: each utterance has a categorical emotion label, soft-label emotion scores (how much each emotion is present), and three dimensional ratings (activation, valence, dominance on a 1–5 scale)
- **Additional features**: pitch mean and standard deviation, RMS energy, speaking rate
- **Both text and audio**: every utterance has a transcription and a raw audio waveform
- Loaded from HuggingFace: [`AbstractTTS/IEMOCAP`](https://huggingface.co/datasets/AbstractTTS/IEMOCAP)

### 2.2 Emotion Grouping: Why 3 Classes?

IEMOCAP originally provides 10 emotion categories: neutral, frustrated, angry, sad, happy, excited, surprise, fear, disgust, other. Training a classifier on 10 classes with 10,039 samples leads to severe class imbalance — some classes (e.g., fear, disgust) appear fewer than 200 times, which is insufficient for a neural network to learn meaningful patterns.

We reduced to 3 classes by grouping:

| Class | Original Labels | Rationale |
|-------|----------------|-----------|
| **Negative** | angry, frustrated, fear, sad, disgust | All share low valence (unpleasant) and high or low activation; they are acoustically and semantically similar |
| **Positive** | happy, excited | Share high valence and often high activation; semantically related |
| **Neutral** | neutral, surprise, other | Residual category: neither clearly positive nor clearly negative |

After grouping, the class distribution is approximately **55% Negative, 27% Positive, 18% Neutral** — still imbalanced but manageable.

### 2.3 Train/Test Split: Why Speaker-Independent?

We used a **Leave-One-Session-Out (LOSO)** protocol: Sessions 1–4 are training data, Session 5 is the held-out test set.

This means the 4 speakers who appear in Session 5 do not appear anywhere in training. This is called **speaker-independent evaluation** and it tests whether the model can generalize to new speakers it has never seen — which is the real-world requirement for any deployed emotion recognition system.

An alternative would be a random 80/20 split, but that allows test speakers to appear in training, which inflates results significantly. LOSO gives a more honest and conservative performance estimate.

From the training sessions (Sessions 1–4), we held out 10% stratified by class as a **validation set** (used for early stopping). The final test set contains **2,170 utterances** from Session 5.

### 2.4 Handling Class Imbalance

Because Negative utterances make up 55% of the data, a naive model could achieve 55% accuracy by predicting "Negative" for everything. To prevent this, all models use **inverse-frequency class weights** in the loss function:

$$w_c = \frac{N}{K \cdot N_c}$$

where $N$ is the total number of training samples, $K=3$, and $N_c$ is the number of training samples in class $c$. This makes the model penalize mistakes on the minority class (Neutral) more heavily than mistakes on the majority class (Negative), so it cannot learn to ignore rare classes.

The resulting weighted cross-entropy loss is:

$$\mathcal{L} = -\sum_{c=1}^{K} w_c \cdot y_c \cdot \log \hat{p}_c$$

---

## 3. Phase 1 — Exploratory Data Analysis

### 3.1 What Is EDA and Why Do It?

Exploratory Data Analysis (EDA) is the process of examining a dataset before modeling to understand its structure, distributions, and quirks. The goal is not to build a predictive model but to answer questions like: Is the data balanced? Are there outliers? Do features behave as expected? What patterns exist that might inform modeling choices?

Skipping EDA risks building a model on top of misunderstood data — for example, discovering after training that the dataset has severe class imbalance that you didn't account for.

### 3.2 What Was Done

The EDA was performed in `notebooks/iemocap-analysis.ipynb` and covered:

**Class distribution and imbalance.** After grouping into 3 classes, the training set is imbalanced: ~55% Negative, ~27% Positive, ~18% Neutral. This directly motivated the use of class-weighted loss in all subsequent models.

**Acoustic feature profiles.** Violin plots of pitch mean, pitch standard deviation, RMS energy, and speaking rate across emotion classes revealed clear patterns: Angry and excited utterances show higher pitch and energy; speaking rate is elevated for positive emotions. These differences confirmed that raw audio carries emotional signal.

**Valence-Activation emotional space.** Plotting utterances in the 2D activation-valence space showed partial clustering: frustrated and angry utterances concentrate in the high-activation/low-valence quadrant (high arousal, negative feeling); happy and excited occupy high-activation/high-valence. Neutral and sad overlap in the low-activation region, which foreshadowed the difficulty of classifying Neutral.

**Gender differences.** Male and female speakers show different acoustic profiles (especially pitch, as expected from biology), but emotion label distributions are similar across genders. This confirmed that gender-specific models are not necessary.

**Outlier analysis.** Boxplots identified extreme values in acoustic features. Manual inspection confirmed these are real emotional expressions (very loud, fast, or unusual speech) rather than recording errors. All samples were retained.

**Correlation analysis.** Numerical features (acoustic + dimensional ratings) showed moderate correlations — high activation tends to co-occur with high pitch and energy, consistent with the literature on emotion acoustics.

### 3.3 Key Insights From EDA

- Class imbalance is real and significant — all models need weighted loss
- Neutral is the hardest class: it lacks distinctive acoustic markers and sits between positive and negative
- Text and audio carry different signals: acoustic features are best at distinguishing arousal levels (calm vs. energetic), while text is better at distinguishing valence (positive vs. negative word choice)

---

## 4. Phase 2 — Text Classification with RoBERTa

### 4.1 What Is RoBERTa?

**Simple intuition:** RoBERTa is a model that has read hundreds of millions of sentences on the internet and learned what words mean in context. When you ask it "is this sentence angry, happy, or neutral?", it already understands that "I can't believe you did that" sounds negative and "this is amazing!" sounds positive — without you ever explicitly teaching it those associations. You only need to fine-tune it on your labeled examples to point it toward your specific task.

**Technical definition:** RoBERTa (Robustly Optimized BERT Pretraining Approach) [3] is a large pre-trained transformer model for natural language processing. It was developed by Facebook AI Research (now Meta AI) as an improvement over the original BERT model.

**How it was pre-trained:** RoBERTa was trained on 160GB of English text (books, news, web pages) using a self-supervised objective called **Masked Language Modeling (MLM)**: random tokens in a sentence are masked, and the model learns to predict them from context. This forces the model to develop a deep understanding of language structure and semantics without any labeled data.

**Architecture:** 12 transformer layers, 768 hidden dimensions, 12 attention heads, ~125 million parameters. Each token in the input is mapped to a vector that encodes both its meaning and its context within the sentence.

**Why RoBERTa for text emotion classification?** Because:
1. Pre-training on 160GB of text means the model already "knows" that words like "terrible" or "angry" carry negative sentiment, without us having to teach it.
2. Fine-tuning (adjusting a small number of additional parameters for our specific task) requires much less data than training from scratch.
3. RoBERTa specifically improves on BERT by using more training data, longer training, and removing the Next Sentence Prediction objective that hurt BERT's performance.

### 4.2 How the Text Model Works

The transcription of each utterance (e.g., "Why are you doing this to me?") is tokenized into subword pieces by the RoBERTa tokenizer. A special `[CLS]` token is prepended, and the model processes the sequence through 12 transformer layers. The output representation at the `[CLS]` position is a 768-dimensional vector that summarizes the entire utterance. A linear classification head maps this vector to 3 class logits:

$$\hat{y} = \text{softmax}(W \cdot h_{[CLS]} + b)$$

The class with the highest logit is the predicted emotion.

### 4.3 Training Setup

- **Optimizer:** AdamW, learning rate $2 \times 10^{-5}$, weight decay 0.01
- **Batch size:** 32, max 10 epochs
- **Early stopping:** patience = 3 epochs on validation Macro F1
- **Learning rate schedule:** linear warmup for the first 10% of training steps, then linear decay
- **Loss function:** weighted cross-entropy with inverse-frequency class weights
- **Max sequence length:** 128 tokens (covers >99% of utterances)

The low learning rate ($2 \times 10^{-5}$) is important: fine-tuning a pre-trained model requires small updates — large learning rates destroy the pre-trained representations.

### 4.4 Results

Training ran for **7 epochs** before early stopping triggered (best val Macro F1 = 0.6757 at epoch 4, no improvement for 3 subsequent epochs). The best checkpoint was restored for test evaluation.

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.6793 |
| Test Macro F1 | 0.6519 |
| Test Weighted F1 | 0.6900 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative | 0.86 | 0.68 | 0.76 | 1149 |
| Positive | 0.72 | 0.71 | 0.71 | 613 |
| Neutral | 0.39 | 0.62 | 0.48 | 408 |

**Interpretation:** The text model achieves strong precision on Negative (0.86) — when it predicts "Negative", it is right 86% of the time. This makes sense: negative emotions often involve specific vocabulary (complaints, profanity, expressions of frustration). However, recall on Negative is lower (0.68), meaning 32% of truly negative utterances are missed, likely classified as Neutral. The Neutral class has particularly low precision (0.39) — when the model predicts Neutral, it is often wrong. Neutral utterances lack distinctive words and the model over-predicts this class relative to its signal strength.

---

## 5. Phase 3 — Audio Classification with Wav2Vec2

### 5.1 What Is Wav2Vec2?

**Simple intuition:** Wav2Vec2 is a model trained to "listen" to speech. It was trained on 960 hours of audio where it had to predict what a masked (hidden) part of a sentence sounds like from its context — similar to how RoBERTa learns from masked words in text. By the end of pre-training, it can convert a raw audio waveform into a rich numerical representation that captures pitch, rhythm, energy, and phonetic content — without any human-specified features. Fine-tuning it on emotion labels teaches it which of those acoustic patterns correspond to anger, happiness, or neutrality.

**Technical definition:** Wav2Vec2 (Wave-to-Vec version 2) [5] is a pre-trained speech representation model developed by Facebook AI Research. It works directly on **raw audio waveforms** — no need for feature extraction like MFCCs or spectrograms.

**How it was pre-trained:** Wav2Vec2 was pre-trained on 960 hours of unlabeled English speech (LibriSpeech corpus) using **contrastive self-supervised learning**. During pre-training, portions of the audio are masked, and the model learns to identify the correct audio representation for the masked region from a set of distractors. This forces the model to learn meaningful acoustic representations.

**Architecture:** Two components:
1. A **convolutional feature encoder** (7 convolutional layers): maps raw 16kHz audio into a sequence of 512-dimensional latent representations at a stride of 20ms per frame
2. A **transformer encoder** (12 layers, 768 hidden dimensions): contextualizes the latent representations across time

**Why Wav2Vec2 for audio emotion classification?** Because:
1. Learning from raw waveforms avoids the information loss that comes from computing hand-crafted features like MFCCs.
2. Pre-training on speech means the model already captures phonetic, prosodic, and paralinguistic patterns before any emotion-specific training.
3. The transformer encoder can model long-range dependencies within an utterance — for example, linking a rising pitch at the end of a sentence to the tone of its beginning.

### 5.2 How the Audio Model Works

Each audio waveform is resampled to 16,000 Hz and padded/truncated to exactly 4 seconds (64,000 samples), covering approximately the 90th percentile of IEMOCAP utterance durations. The waveform is passed through the convolutional feature encoder to produce a sequence of ~200 frame representations, then through the transformer encoder to produce contextualized representations. **Mean pooling** over all time steps collapses the sequence into a single 768-dimensional utterance vector. A classification head maps this to 3 class logits:

$$\hat{y} = \text{softmax}(W_{cls} \cdot \text{pool}(H) + b)$$

**Why freeze the convolutional feature encoder?** The convolutional layers learn low-level waveform features (onset detection, frequency filtering) that are universal across speech tasks. Freezing them during fine-tuning preserves these stable representations and reduces the number of parameters being updated, which reduces overfitting on a relatively small fine-tuning dataset.

### 5.3 Training Setup

- **Optimizer:** AdamW, learning rate $2 \times 10^{-5}$, weight decay 0.01
- **Batch size:** 16 (smaller than text due to larger memory footprint of audio tensors)
- **Max epochs:** 10, early stopping patience = 3 on val Macro F1
- **Loss function:** weighted cross-entropy with inverse-frequency class weights

### 5.4 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.6779 |
| Test Macro F1 | 0.6324 |
| Test Weighted F1 | 0.6799 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative | 0.76 | 0.78 | 0.77 | 1149 |
| Positive | 0.72 | 0.56 | 0.63 | 613 |
| Neutral | 0.45 | 0.55 | 0.49 | 408 |

**Interpretation:** The audio model achieves slightly higher recall on Negative than the text model (0.78 vs 0.68) but lower precision (0.76 vs 0.86). This reflects that acoustic cues are good at flagging emotional arousal (high-energy, high-pitch speech often correlates with negative emotions) but less precise than text at discriminating Negative from Positive high-arousal speech. Positive recall is notably lower (0.56 vs 0.71 for text), suggesting that happy/excited speech is acoustically more variable and harder to identify from audio alone. Neutral remains the hardest class for both modalities.

**Text vs. Audio comparison:** The text model outperforms audio by +0.02 Macro F1 (0.6519 vs 0.6324). The gap is small, suggesting both modalities carry comparable amounts of emotional signal. Crucially, they make different errors — the text model is wrong on utterances where phrasing is neutral but tone is emotional, and the audio model is wrong on utterances where tone is calm but words are clearly emotional. This complementarity is exactly what motivates fusion.

---

## 6. Phase 4 — Late Fusion

### 6.1 What Is Late Fusion and Why Use It?

**Late fusion** (also called decision-level fusion) combines the outputs of independently trained models **after** they have each made their predictions. In our case, both models produce a vector of 3 logits (raw scores, one per class) over the test set, and we combine them with a weighted average:

$$\hat{y}_{fused} = \alpha_{text} \cdot \hat{y}_{text} + (1 - \alpha_{text}) \cdot \hat{y}_{audio}$$

The resulting combined logits are passed through softmax to produce the final class prediction.

**Why does this work without any training?** Because the two models are already calibrated — they have learned to assign higher logit values to more likely classes. If the text model is very confident a sample is Positive (high positive logit) and the audio model is uncertain, the weighted sum still points toward Positive. If both models agree, the combined logit is doubly strong. The weights ($\alpha_{text} = 0.65$, $\alpha_{audio} = 0.35$) reflect the slightly stronger performance of the text model.

**Why try late fusion before intermediate fusion?** Late fusion is the simplest multimodal baseline:
- No additional training required
- Fully interpretable: you can trace exactly how much each modality contributed
- No risk of overfitting to a small multimodal training set
- Serves as a sanity check: if even late fusion doesn't improve over unimodal, then the modalities are not complementary and more complex fusion isn't worth pursuing

### 6.2 Why Do the Modalities Complement Each Other?

The key condition for fusion to help is that the two models make **different errors**. If text gets a sample wrong but audio gets it right (and vice versa), combining them corrects more errors than either modality alone. This happens here because:

- **Text errors** often occur when the emotional words are absent (e.g., mundane sentences said angrily, where tone carries the emotion but the words don't)
- **Audio errors** often occur when the emotional words are present but the voice is flat or ambiguous (e.g., sarcasm, where the meaning is in the text)

These two error types are largely non-overlapping, so combining the models yields large gains.

### 6.3 Fusion Weight Selection

The weight $\alpha_{text} = 0.65$ was set based on the relative validation Macro F1 scores of the two models. A more principled approach would grid-search $\alpha$ on a held-out validation set; for this project, the manually chosen value produced a strong result.

### 6.4 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.8392 |
| Test Macro F1 | 0.8173 |
| Test Weighted F1 | 0.8431 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative | 0.85 | 0.88 | 0.85 | 1149 |
| Positive | 0.87 | 0.85 | 0.86 | 613 |
| Neutral | 0.64 | 0.80 | 0.71 | 408 |

**Interpretation:** Late fusion improves dramatically over both unimodal models: **+0.17 Macro F1 over text-only** and **+0.19 over audio-only**. This is not a marginal improvement — it is a substantial jump that demonstrates the two modalities are highly complementary. Every class improves:

- **Negative F1**: 0.85 (vs. 0.76 text-only, 0.77 audio-only)
- **Positive F1**: 0.86 (vs. 0.71 text-only, 0.63 audio-only) — biggest relative gain for positive class
- **Neutral F1**: 0.71 (vs. 0.48 text-only, 0.49 audio-only) — Neutral gains the most in absolute terms (+0.23), confirming that no single modality reliably captures neutral speech but the combination does

The Neutral improvement is particularly notable: the text model had precision of only 0.39 for Neutral, but fusion raises this significantly. This suggests that many utterances the text model misclassified as Negative or Positive were correctly identified by the audio model as having neutral prosodic features.

---

## 7. Phase 5 — Intermediate Fusion

### 7.1 What Is Intermediate Fusion and How Does It Differ From Late Fusion?

**Intermediate fusion** (also called feature-level fusion or early fusion) combines representations from both modalities **before** the final classification decision, at the level of learned embeddings rather than output logits.

The key difference:

| | Late Fusion | Intermediate Fusion |
|-|-------------|---------------------|
| What is combined | Output logits (after each model fully processes its input) | Internal hidden-state embeddings (before the final classification layer) |
| When combination happens | After independent training | During joint training |
| Training required | No — uses already-trained models | Yes — all parameters fine-tuned jointly |
| Can the models inform each other | No — each modality processed independently | Yes — through the shared loss signal during training |
| Risk of overfitting | Low | Higher — more parameters, joint optimization |
| Expressiveness | Limited to weighted sum of decisions | Richer — the MLP head can learn non-linear interactions |

### 7.2 Architecture

Both encoders (RoBERTa and Wav2Vec2) are initialized from their single-modality checkpoints (the weights saved after Phase 2 and Phase 3 training). The last hidden states of each encoder are **mean-pooled** over their respective sequence dimensions:

- RoBERTa output: mean pool over all token positions → 768-dimensional vector
- Wav2Vec2 output: mean pool over all audio frame positions → 768-dimensional vector

These two vectors are **concatenated** into a single 1536-dimensional joint representation:

$$\mathbf{h}_{fused} = \left[\text{pool}(\mathbf{H}_{text})\ ;\ \text{pool}(\mathbf{H}_{audio})\right] \in \mathbb{R}^{1536}$$

This joint vector is passed through a **two-layer MLP fusion head**:

$$\hat{y} = W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{h}_{fused} + b_1) + b_2$$

with hidden dimension 256 and dropout 0.3.

### 7.3 Differential Learning Rates — Why?

During joint training, two groups of parameters need very different learning rates:

- **Pre-trained encoder parameters** (RoBERTa + Wav2Vec2, ~220M parameters total): These already encode strong representations. We want to fine-tune them gently without destroying what was learned during pre-training. Learning rate: $1 \times 10^{-5}$ (very small).
- **Fusion head parameters** (MLP, ~400K parameters): This is a new layer trained from random initialization. It needs to learn quickly. Learning rate: $1 \times 10^{-4}$ (10× larger).

Using a single learning rate for all parameters either trains the head too slowly (if using the encoder rate) or destroys the pre-trained encoders (if using the head rate).

### 7.4 Why Might Intermediate Fusion Outperform Late Fusion?

In late fusion, the two models are trained independently — neither modality's training is influenced by the other. The fusion is a post-hoc combination with no learned interaction.

In intermediate fusion, the entire system is trained end-to-end on the emotion classification objective. This means:
1. The RoBERTa encoder can learn to emphasize linguistic patterns that are complementary to what Wav2Vec2 captures
2. The Wav2Vec2 encoder can learn to emphasize acoustic patterns that are complementary to what RoBERTa captures
3. The MLP head can learn **non-linear cross-modal interactions** — combinations of text and audio features that are more predictive together than either is alone

The downside is risk: with ~220M parameters being jointly fine-tuned on ~8,000 training samples, there is a real risk of overfitting unless regularization (dropout, weight decay, early stopping) is strong.

### 7.5 Current Status

Intermediate fusion training was **initiated during this session** but has not yet completed. Training was started on 2026-06-17 at 2:32 PM with the following configuration:

- Batch size: 8 (smaller than unimodal models because the joint forward pass is much more memory-intensive)
- Max epochs: 10, early stopping patience = 3 on val Macro F1
- Loss: weighted cross-entropy with inverse-frequency class weights
- Hardware: Apple M1 GPU (MPS) with CPU fallback for unsupported ops

Due to the larger model size and smaller batch size, each epoch requires approximately 90 minutes of training time on local hardware, giving an estimated **9–10 hour total training run**. Training was stopped after approximately 1.5 hours to allow other project work to proceed. The notebook (`notebooks/intermediate_fusion.ipynb`) contains the full implementation and is ready to resume.

**To complete intermediate fusion**, a teammate needs to:
1. Have `~/Downloads/text_best_model_weights.pt` (476 MB) — the RoBERTa checkpoint from Phase 2
2. Have `~/Downloads/audio_best_model.pt` (361 MB) — the Wav2Vec2 checkpoint from Phase 3 (available from team Google Drive)
3. Run `PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=21600 notebooks/intermediate_fusion.ipynb` from the repo root

---

## 8. Summary of Results

**Table 1: Test set results on Session 5 (n=2,170)**

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| RoBERTa (text-only) | 0.6793 | 0.6519 | 0.6900 |
| Wav2Vec2 (audio-only) | 0.6779 | 0.6324 | 0.6799 |
| Late Fusion (α=0.65/0.35) | 0.8392 | 0.8173 | 0.8431 |
| Intermediate Fusion | — | — | — |

*Intermediate fusion results pending training completion.*

**Key takeaway:** Both unimodal models achieve similar performance (~0.63–0.65 Macro F1). Late fusion of their outputs yields a massive +0.17 jump to 0.82 Macro F1, demonstrating strong cross-modal complementarity. The progression from unimodal to multimodal is the central finding of this project.

---

## 9. Discussion

### 9.1 Why Do Text and Audio Complement Each Other So Well?

The +0.17 Macro F1 gain from late fusion is larger than is typically reported in the emotion recognition literature (where gains of +0.05–0.10 are more common). This suggests an unusually high degree of complementarity in IEMOCAP.

One reason may be IEMOCAP's recording protocol: the improvised scenarios were designed to naturally elicit strong emotions, resulting in utterances where text content and acoustic delivery are often independently expressive. An actor asked to express frustration might choose both emotionally charged words AND a tense, raised-pitch delivery — but in different utterances these channels may be more or less prominent.

Another reason: the two pre-trained models were trained on completely different pre-training tasks and data modalities (text vs. speech), so their error patterns are largely independent by construction.

### 9.2 Why Is Neutral Consistently the Hardest Class?

Neutral speech is defined negatively — it is speech that is neither clearly positive nor clearly negative. It therefore:
- Lacks distinctive vocabulary (no strong positive or negative words)
- Lacks distinctive prosody (no elevated pitch or energy, no specific rhythm pattern)
- Is highly variable: "neutral" can mean flat affect, measured calm, or simple factual statement

Both text and audio models struggle with Neutral precision (0.39 and 0.45 respectively). The fusion model improves this substantially (precision 0.64), suggesting that the combination of "no strong acoustic signal" AND "no strong lexical signal" is itself a useful cue for Neutral.

### 9.3 Why Use Macro F1 as the Primary Metric?

Accuracy is a misleading metric when classes are imbalanced. With 55% Negative samples, a model that always predicts "Negative" achieves 55% accuracy but is completely useless. Macro F1 computes F1 separately for each class and averages them with equal weight — so the minority class (Neutral, 18% of data) contributes as much to the final score as the majority class. This makes it the appropriate metric for evaluating generalization across all three emotion groups.

---

## 10. Design Decisions and Why

| Decision | Rationale |
|----------|-----------|
| Group 10 classes into 3 | Reduce imbalance; make learning feasible with ~10K samples |
| LOSO evaluation (Session 5 holdout) | Tests generalization to new speakers — the real-world requirement |
| Weighted cross-entropy loss | Prevents models from collapsing to majority class (Negative) |
| Early stopping (patience=3) | Prevents overfitting; RoBERTa and Wav2Vec2 are large models prone to overfit on small datasets |
| Freeze Wav2Vec2 feature encoder | Preserves low-level acoustic representations; reduces overfitting |
| Fine-tune RoBERTa fully | The `[CLS]` representation must adapt to emotion classification; all layers contribute |
| α=0.65 text / 0.35 audio | Reflects text model's slightly higher validation performance; easy to interpret |
| Batch size 8 for intermediate fusion | Memory constraint: joint forward pass through 220M params + backprop requires more GPU memory |
| Differential learning rates in intermediate fusion | New MLP head needs higher LR; pre-trained encoders need gentle fine-tuning |

---

## 11. Limitations

- **Single held-out session**: Using only Session 5 as the test set gives a single point estimate with no variance measurement. Full leave-one-session-out cross-validation would provide more statistically robust results but requires 5× the compute.
- **3-class grouping**: Collapsing 10 emotions into 3 loses nuance. In particular, grouping "angry" and "frustrated" may obscure differences that matter for practical applications.
- **Fixed fusion weight**: α=0.65 was set manually; a held-out validation search would be more principled.
- **English-only models**: RoBERTa and Wav2Vec2 were pre-trained on English data and may not generalize to other languages.
- **Intermediate fusion incomplete**: Without the intermediate fusion results, we cannot determine whether joint learning outperforms the simple weighted combination of independent predictions.

---

## 12. Future Work

- Complete intermediate fusion training and compare with late fusion
- Search over fusion weight α on a held-out validation set rather than setting it manually
- Explore attention-based or learned fusion weights (soft attention over the two modalities)
- Apply cross-modal attention (e.g., Multimodal Transformer [7]) to let text and audio representations attend to each other
- Evaluate full 10-class grouping with finer-grained class assignments
- Extend to leave-one-session-out cross-validation for variance-aware evaluation

---

## References

[1] C. Busso et al., "IEMOCAP: Interactive emotional dyadic motion capture database," *Language Resources and Evaluation*, vol. 42, no. 4, pp. 335–359, 2008.

[2] S. Poria et al., "A review of affective computing: From unimodal analysis to multimodal fusion," *Information Fusion*, vol. 37, pp. 98–125, 2017.

[3] Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," *arXiv:1907.11692*, 2019.

[4] L. Pepino, P. Riera, and L. Ferrer, "Emotion recognition from speech using wav2vec 2.0 embeddings," in *Proc. Interspeech*, 2021, pp. 3400–3404.

[5] A. Baevski et al., "wav2vec 2.0: A framework for self-supervised learning of speech representations," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[6] A. Zadeh et al., "Tensor fusion network for multimodal sentiment analysis," in *Proc. EMNLP*, 2017, pp. 1103–1114.

[7] Y.-H. H. Tsai et al., "Multimodal Transformer for Unaligned Multimodal Language Sequences," in *Proc. ACL*, 2019, pp. 6558–6569.
