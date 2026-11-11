# Fake News Detection – TinyBERT + LoRA + Cross-Attention + Bi-GRU

Classifies a news article as **REAL** or **FAKE** from its headline and body. The headline attends to the body through cross-attention, so the model can pick up headline-body mismatches.

**Results on the WELFake test split (14,308 articles):** 99.36% accuracy, 0.9938 F1 (fake class), 91 errors. See `train.ipynb`, section 9.

## Architecture
Headline (64 tokens) and body (448 tokens) → shared TinyBERT-4L-312D encoder with LoRA (r=4, alpha=32, query/value) → cross-attention (headline = query, body = key/value, 8 heads) → Bi-GRU (64 per direction) → max-pool → dropout 0.2 → linear → sigmoid = P(fake).

## Training
WELFake dataset, 80/20 stratified split (random_state 42). Google Colab, Tesla T4, batch 16, BCE loss, AdamW.
- Epochs 1–2: backbone frozen, lr 2e-4 (trains cross-attention, Bi-GRU, classifier)
- Epochs 3–5: full fine-tuning, lr 1e-5

## Repository
| File | What it is |
|------|------------|
| `train.ipynb` | Training and evaluation notebook |
| `app.py` | Streamlit web app |
| `PROJECT_REPORT.md` | Full project report: method, results, limitations |
| `tinybert_fake_news_model/` | Trained checkpoint, LoRA adapter, tokenizer |

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```
The first run downloads the TinyBERT base model from Hugging Face.

## Limitations
Trained and tested on one dataset (WELFake). The model detects textual patterns and does not fact-check claims. See `PROJECT_REPORT.md`.
