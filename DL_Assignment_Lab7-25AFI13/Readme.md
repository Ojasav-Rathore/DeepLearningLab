# Experiment 7: Sequence-to-Sequence Learning with Transformers

This repository contains a PyTorch implementation of a complete Transformer-based Encoder-Decoder model for English-to-Spanish machine translation. It replaces traditional LSTM-based architectures entirely with self-attention mechanisms and parallel computation.



## Objective
The goal of this project is to build a sequence-to-sequence (seq2seq) machine translation model from scratch to understand the inner workings of the Transformer architecture. 

## Features Implemented
This implementation includes all core components of a standard Transformer:
* **Embedding Layer:** Learns word embeddings for both source (English) and target (Spanish) vocabularies.
* **Positional Encoding:** Utilizes sinusoidal positional encoding added to the input embeddings to retain sequence order information.
* **Multi-Head Self-Attention:** Includes scaled dot-product attention extended to multiple heads, supporting masking for padded tokens and look-ahead masking for the decoder.
* **Transformer Encoder:** Stacked layers containing multi-head self-attention, position-wise Feed Forward Networks, residual connections, and layer normalization.
* **Transformer Decoder:** Stacked layers featuring masked multi-head self-attention, encoder-decoder (cross) attention, Feed Forward Networks, residual connections, and layer normalization.
* **Training Pipeline:** Implements Teacher Forcing, Cross-Entropy loss (with a padding mask), and the Adam optimizer.
* **Evaluation:** Evaluates translation performance using the BLEU score metric.

## Dataset
The model trains on an English-Spanish sentence pair dataset (`spa.txt`). 
* **Sampling:** To train within a reasonable timeframe, the script samples a subset of 10,000 sentence pairs.
* **Splits:** The data is split into 80% training, 10% validation, and 10% testing sets.
* **Preprocessing:** Handles tokenization, lowercasing, punctuation normalization, and sequence padding (`<sos>`, `<eos>`, `<pad>`, `<unk>`).

## Prerequisites
Ensure you have Python installed along with the following libraries:

```bash
pip install torch torchvision torchaudio
pip install nltk
