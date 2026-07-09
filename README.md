# SLM-125M: a legal/financial language model, pretrained from scratch

A complete, reproducible pipeline that turns three public HuggingFace datasets into
a **125-million-parameter Llama-style language model**, pretrained **from scratch**
(fresh weights, fresh tokenizer) on [Modal](https://modal.com), then deployed for
inference behind a web frontend.

The whole build, data to deployed model, cost about **$57** and reaches a held-out
**validation perplexity of 8.50**.

- **Live demo:** https://slm-125m.vercel.app
- **Model weights:** https://huggingface.co/thesreedath/slm-125m-base
- **Inference API:** `POST https://thesreedath--slm-125m-inference-web.modal.run/generate`

> It is a **base completer**, not a chatbot: give it the start of a sentence and it
> continues in the legal register. The honest metric is perplexity; at 125M it
> speaks the register fluently but does not know facts (that would need RAG).

---

## What it does, in six phases

| Phase | What | Where |
|---|---|---|
| 0 | Setup + smoke test + measure real dataset sizes | `modal_app.py::smoke_test` / `::measure` |
| 1 | Stream + clean 3 datasets (deterministic rule chain) | `cleaning.py`, `modal_app.py::clean` |
| 2 | Near-dedup (MinHash-LSH) + exact-dedup + decontaminate | `dedup.py`, `modal_app.py::dedup` |
| 3 | Train a fresh 16K byte-level BPE tokenizer | `modal_app.py::tokenizer` |
| 4 | Tokenize + pack into uint16 1024-token windows, split 99/1 | `modal_app.py::tokenize` |
| 5 | Pretrain the 125M model (8xH100 DDP, 5 epochs) | `train.py`, `modal_app.py::pretrain` |
| 6 | Push to HuggingFace + serve + Vercel frontend | `modal_app.py` (Inference), `slm-125m-site/` |

Each phase has a readable write-up in [`docs/`](docs/).

## The data (not 70/20/10)

The two legal sources only hold about 2B tokens combined, so the mix is
**legal-first**: take all case law and all SEC filings, add a small web slice.
Realized mix by real tokens is roughly **40% case law, 40% SEC, 20% web**, about
2.19B training tokens (99/1 train/val split).

| Source | HuggingFace id | Role |
|---|---|---|
| case law | `HFforLegal/case-law` (split `us`) | US court opinions |
| SEC | `PleIAs/SEC` | financial filings (10-K etc.) |
| web | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | general-English fluency |

## The model

Vanilla Llama geometry (maps 1:1 to `transformers.LlamaConfig`): 12 layers, hidden
768, 12 heads, SwiGLU (inner 3072), RMSNorm, RoPE, tied embeddings, context 1024,
vocab 16,384. ~125.8M parameters.

## Results

- Validation perplexity: **8.50** (from 16.47 at step 1,000; clean monotonic curve)
- 5 epochs, 20,885 steps, 10.95B tokens seen, ~1.8h on 8xH100, **$56.58**
- Full data pipeline (Phases 0 to 4): CPU only, ~$0.18

---

## Reproduce it

Everything you need, from creating the Modal and HuggingFace accounts to the exact
commands, is in **[`docs/REPLICATION_GUIDE.md`](docs/REPLICATION_GUIDE.md)** (an
agent-ready brief). In short:

```bash
pip install modal
modal token new                       # or: modal token set --token-id ... --token-secret ...
cp .env.local.example .env.local      # fill in your tokens
source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
modal volume create slm-125m

modal run modal_app.py                 # Phase 0: smoke test
modal run modal_app.py::measure        # Phase 0: measure dataset sizes
modal run modal_app.py::clean --fineweb-shards 5   # Phase 1
modal run modal_app.py::dedup          # Phase 2
modal run modal_app.py::tokenizer      # Phase 3
modal run modal_app.py::tokenize       # Phase 4
modal run modal_app.py::smoke_pretrain # Phase 5 smoke (1x H100, ~$0.12)
modal run modal_app.py::pretrain       # Phase 5 full (8x H100, 5 epochs)
modal deploy modal_app.py              # Phase 6: inference endpoint
```

`config.py` is the single source of truth (model geometry, data mix, budgets,
paths, thresholds). Everything else imports from it.

## Repo layout

```
config.py          single source of truth
cleaning.py        deterministic 6-step cleaning chain (pure functions)
dedup.py           dedup + decontamination helpers (pure functions)
train.py           125M model + mmap data loader + DDP training loop + eval
modal_app.py       the Modal app: one function per phase + inference endpoint
slm-125m-site/     the Vercel frontend (single-file index.html)
docs/              per-phase write-ups + REPLICATION_GUIDE + SUMMARY
INITIAL.md         the original project brief
```

## Notes

- No secrets are committed. Put your tokens in `.env.local` (git-ignored).
- Data licensing: case law and SEC filings are public/government-domain;
  fineweb-edu is ODC-By. This repo contains code only, not the datasets or weights.
- Built by Vizuara AI Labs as an end-to-end small-language-model reference.
