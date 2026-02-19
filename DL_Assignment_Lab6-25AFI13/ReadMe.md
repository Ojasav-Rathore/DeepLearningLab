# Neural Machine Translation (English to Spanish) Using Seq2Seq and Attention

This repository contains the code for building, training, and evaluating Deep Learning models for Machine Translation. Specifically, it translates English sentences into Spanish using Sequence-to-Sequence (Seq2Seq) architectures implemented in PyTorch.

The project compares a standard Vanilla Seq2Seq model against two variants equipped with different Attention mechanisms (Bahdanau and Luong) to demonstrate how attention improves translation quality and handles longer sequences.

---

## Project Overview

The notebook (`DL-Lab6Assignment_25AFI13.ipynb`) covers the complete deep learning pipeline:

1. **Data Preprocessing:** Text normalization, tokenization, and dynamic vocabulary building.
2. **Data Loading:** Custom PyTorch `Dataset` and `DataLoader` with sequence padding.
3. **Modeling:** Implementation of Encoder-Decoder networks with and without Attention.
4. **Training:** Iterative training loop calculating Cross-Entropy Loss over 60 epochs.
5. **Evaluation:** Performance comparison using Bilingual Evaluation Understudy (BLEU) scores.
6. **Visualization:** Plotting training loss convergence and BLEU score comparisons.

---

## Dataset

* **Source:** The dataset consists of English-Spanish sentence pairs (`spa.txt`).
* **Size:** The code utilizes a subset of `10,000` sample pairs to optimize training time.
* **Splits:** * Training: 80%
* Validation: 10%
* Testing: 10%


* **Tokens:** Special tokens `<pad>` (0), `<sos>` (1), and `<eos>` (2) are used for sequence bounding and padding.

---

## Model Architectures

The project evaluates three distinct model architectures.

### 1. Vanilla Seq2Seq

A standard baseline model using Recurrent Neural Networks (LSTMs). The Encoder processes the entire source sentence into a single fixed-size context vector (the final hidden state), which is then passed to the Decoder to generate the target sentence step-by-step.

### 2. Bahdanau (Additive) Attention

Instead of relying on a single context vector, the Bahdanau mechanism allows the Decoder to "attend" to different parts of the source sentence at each decoding step. It calculates attention scores by passing the Decoder's hidden state and the Encoder's outputs through a feed-forward neural network (additive calculation).

### 3. Luong (Multiplicative) Attention

Similar to Bahdanau, but calculates attention scores using a dot-product (multiplicative) approach between the Decoder's hidden state and the Encoder's outputs. This is generally more computationally efficient than additive attention.

### Hyperparameters

* **Embedding Size:** 256
* **Hidden Size:** 512
* **Optimizer:** Adam (Learning Rate: 0.001)
* **Loss Function:** CrossEntropyLoss (ignoring `<pad>` indices)
* **Batch Size:** 64
* **Epochs:** 60

---

## Results & Evaluation

The models are evaluated using the **BLEU** score metric (via `nltk.translate.bleu_score`), using a smoothing function to account for shorter sequences.

Based on the final execution, the attention models outperform the vanilla baseline:

* **Vanilla Seq2Seq BLEU:** ~0.192
* **Luong Attention BLEU:** ~0.197
* **Bahdanau Attention BLEU:** ~0.201

The code generates two visual outputs:

1. **Training Loss Plot:** A line chart overlaying the loss descent of all three models over 60 epochs.
2. **BLEU Score Bar Chart:** A side-by-side comparison of the final evaluation scores.

---

## Requirements & Execution

**Dependencies:**

* Python 3.x
* PyTorch (with CUDA support recommended)
* NumPy
* NLTK (Natural Language Toolkit)
* Scikit-learn
* Matplotlib & Seaborn
