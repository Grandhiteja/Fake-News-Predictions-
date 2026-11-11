# TinyBERT Fake News Detector – Project Report

**Author:** Grandhi Sai Mani Teja
**Type:** NLP / deep learning project with a Streamlit web app
**Stack:** Python, PyTorch, Hugging Face Transformers, PEFT (LoRA), scikit-learn, Streamlit

---

## 1. Summary

This project classifies a news article as **REAL** or **FAKE** from its headline and body text. A lightweight transformer encoder (TinyBERT, 4 layers, 312 hidden size) with LoRA adapters encodes both texts. A cross-attention layer lets the headline attend to the body, a bidirectional GRU reads the combined sequence, and a linear + sigmoid head gives the probability that the article is fake.

The model was trained on the **WELFake** dataset in Google Colab (Tesla T4 GPU). On the held-out test split of **14,308 articles**, the published checkpoint scores **99.36% accuracy** and **0.9938 F1 (fake class)**. See Section 6.

The repository contains:

| File | Purpose |
|------|---------|
| `train.ipynb` | Full training notebook: data loading, split, model, two-stage training, evaluation, saving, and an evaluation of the published checkpoint |
| `app.py` | Streamlit app (204 lines) that loads the trained model and predicts on a pasted headline and body |
| `tinybert_fake_news_model/full_model.pth` | Trained checkpoint (~60 MB): full model state dict plus training config |
| `tinybert_fake_news_model/adapter_model.safetensors`, `adapter_config.json` | LoRA adapter weights (~80 KB) and settings (r = 4, alpha = 32, dropout 0.1, target modules query and value) |
| `tinybert_fake_news_model/tokenizer.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json` | WordPiece tokenizer (30,522-token vocabulary, lower-cased) |

---

## 2. Problem

Fake articles often pair a sensational headline with a body that does not support it, or use a recognisable writing style. The model reads the headline and the body separately and then explicitly compares them through cross-attention, instead of treating the article as one block of text.

---

## 3. Dataset

- **WELFake** (title, text, label; label 1 = fake, 0 = real). 72,134 rows; after dropping rows with a missing title, text or label, **71,537 articles** remain (36,509 fake, 35,028 real).
- **Split:** 80% train / 20% test, stratified by label, `random_state = 42`.
  - Train: 57,229 articles (29,207 fake, 28,022 real)
  - Test: 14,308 articles (7,302 fake, 7,006 real)
- **Tokenisation:** headline truncated or padded to 64 tokens, body to 448 tokens (64 + 448 = 512, BERT's maximum length).

---

## 4. Model architecture

```
Headline (64 tokens) ──► TinyBERT + LoRA ──► H  (64 × 312)
Body     (448 tokens) ──► TinyBERT + LoRA ──► B  (448 × 312)   (same shared encoder)

Cross-attention (8 heads): query = H, key = value = B, body padding masked
        ──► C (64 × 312): each headline token summarised by the body content it matches

concat [H ; C] along the sequence ──► (128 × 312)
Bi-GRU (64 hidden units per direction) ──► (128 × 128)
Global max-pool over the sequence ──► (128)
Dropout 0.2 ──► Linear 128 → 1 ──► Sigmoid ──► P(fake)
```

| Component | Setting |
|-----------|---------|
| Encoder | `huawei-noah/TinyBERT_General_4L_312D` (4 layers, hidden 312, 12 heads, FFN 1200) |
| LoRA | r = 4, alpha = 32, dropout 0.1, on the self-attention query and value projections |
| Cross-attention | `nn.MultiheadAttention`, 312 dims, 8 heads, dropout 0.1 |
| Bi-GRU | 1 layer, 64 hidden units per direction |
| Head | dropout 0.2, linear 128 → 1, sigmoid |

**Parameter counts:**

| Part | Parameters |
|------|-----------|
| TinyBERT encoder (original weights) | about 14.4 million |
| LoRA adapters | 19,968 |
| Cross-attention | 390,624 |
| Bi-GRU | 145,152 |
| Classifier | 129 |
| New layers added on top of TinyBERT | 555,873 |

---

## 5. Training

All training code is in `train.ipynb`.

| Setting | Value |
|---------|-------|
| Hardware | Google Colab, NVIDIA Tesla T4 |
| Loss | Binary cross-entropy (`nn.BCELoss`) on the sigmoid output |
| Optimiser | AdamW, weight decay 0.01 |
| Batch size | 16 |
| Epochs | 5 |
| Gradient clipping | max norm 1.0 |
| Decision threshold | 0.5 |

**Two-stage schedule:**

1. **Stage 1, epochs 1–2, learning rate 2e-4:** the whole TinyBERT backbone is frozen, including its LoRA layers. Only the cross-attention, Bi-GRU and classifier are trained, so the new layers learn on top of stable pre-trained features.
2. **Stage 2, epochs 3–5, learning rate 1e-5:** the backbone is unfrozen (original TinyBERT weights and LoRA) and the whole network is fine-tuned with a low learning rate, so the encoder adapts to news text without forgetting what it learned in pre-training.

The checkpoint `full_model.pth` stores the state dict and the config `{model_name, max_headline_len: 64, max_body_len: 448, batch_size: 16, epochs: 5}`.

---

## 6. Results

Evaluation of the published `full_model.pth` on the full test split (14,308 articles), threshold 0.5. This is reproduced in section 9 of `train.ipynb`.

| Metric | Value |
|--------|-------|
| Accuracy | **99.36%** |
| F1 (fake) | **0.9938** |
| Precision / recall (fake) | 0.9940 / 0.9936 |
| Precision / recall (real) | 0.9933 / 0.9937 |
| Errors | 91 of 14,308 |

**Confusion matrix:**

|  | Predicted REAL | Predicted FAKE |
|--|---------------|----------------|
| **Actual REAL** | 6,962 | 44 |
| **Actual FAKE** | 47 | 7,255 |

Errors are balanced: 44 real articles flagged as fake and 47 fake articles missed.

---

## 7. Web application (`app.py`)

1. `load_model()` (cached with `@st.cache_resource`) picks GPU if available, else CPU. It loads the tokenizer from `tinybert_fake_news_model/`, builds the same `TinyBertCrossAttnBiGRU` class, loads `full_model.pth` and switches to `eval()` mode. If `full_model.pth` is missing, it falls back to loading only the LoRA adapter. In that case the cross-attention, GRU and classifier stay untrained, so predictions are not meaningful.
2. The user pastes a headline and an article body and clicks **Analyze News**. Empty fields show a warning.
3. Both texts are tokenised (64 and 448 tokens) and passed through the model under `torch.no_grad()`.
4. The output `prob` is P(fake). The label is **FAKE** if `prob > 0.5`, else **REAL**. The displayed confidence is `prob` for FAKE and `1 - prob` for REAL. The raw probability is also shown.

**Run locally:**

```bash
pip install streamlit torch transformers peft
streamlit run app.py
```

The first run downloads the TinyBERT base model from Hugging Face, so it needs internet once.

---

## 8. Limitations

- **One dataset:** WELFake merges several public news datasets, so high accuracy can partly reflect source-specific style. Performance on news from other outlets, time periods or languages has not been tested.
- **No separate validation set:** the test split was also used to monitor each epoch. A train / validation / test split would give a cleaner final estimate.
- **Truncation:** only the first 448 body tokens are used.
- **Headline padding is not masked** before the Bi-GRU and max-pool.
- **GRU dropout** (`dropout=0.1` with `num_layers=1`) has no effect in PyTorch.
- **Confidence is not calibrated:** the sigmoid output is not guaranteed to match real-world probability, and the 0.5 threshold is fixed.
- The model detects patterns in text. It does **not** fact-check claims against external sources.

## 9. Possible improvements

- Evaluate on a different fake-news dataset (cross-dataset testing) to measure generalisation.
- Add a validation split with early stopping, and calibrate the probabilities (temperature scaling).
- Mask headline padding in the GRU / pooling step and handle long articles with chunking.
- Show cross-attention weights in the app to explain which body passages drove the decision.

---

## References

1. Jiao et al., *TinyBERT: Distilling BERT for Natural Language Understanding*, 2019. https://arxiv.org/abs/1909.10341
2. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. https://arxiv.org/abs/2106.09685
3. Vaswani et al., *Attention Is All You Need*, 2017. https://arxiv.org/abs/1706.03762
4. Verma et al., *WELFake: Word Embedding over Linguistic Features for Fake News Detection*, IEEE Transactions on Computational Social Systems, 2021.
