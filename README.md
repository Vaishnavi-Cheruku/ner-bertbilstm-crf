# BERT-BiLSTM-CRF NER (CoNLL-2003)
This project implements a Named Entity Recognition (NER) model using a hybrid architecture: BERT for contextual embeddings, BiLSTM for sequential modeling, and CRF for structured prediction.

Dataset: CoNLL-2003 (English)
Model: bert-base-cased + BiLSTM + CRF
Framework: PyTorch
Metric: Macro F1 (using seqeval)
Best Test F1: 0.8930

Key Features
Subword-aware tag alignment for BERT
Viterbi decoding via CRF
Training with AdamW, StepLR, gradient clipping
Checkpoint saving based on validation F1
