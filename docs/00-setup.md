# Phase 0, Setup + smoke test

**Goal:** prove the plumbing works (Modal auth, the Volume, HF streaming, field
extraction, the cleaner) before we spend a cent on real data or GPU. Cost ≈ $0.

---

## What we are building (one line)

A 125M-parameter, Llama-style, decoder-only LM, pretrained **from scratch**
(fresh weights, fresh 16K byte-level BPE tokenizer) on a legal/financial-heavy
corpus, on Modal, then deployed behind a website for inference.

`config.py` is the single source of truth. It already sanity-checks:

```
model: 125,847,552 params (~125.8M) | vocab 16384 | 12L/768d/12h kv=12
target tokens: 10.0B (~79 tok/param)
pretrain: 8xH100 @ $3.95/hr each | cap $40.0
stages: setup -> clean -> dedup -> tokenizer -> tokenize -> pretrain -> deploy
```

---

## The compute layout

- **One Modal App** (`slm-125m`) + **one persistent Volume** (`slm-125m`) mounted
  at `/data`. Everything durable lives on the Volume.
- CPU containers for cleaning / tokenizer / tokenization (cheap). Data is streamed
  and cleaned before any GPU boots.
- 8xH100 single-node DDP for the one GPU phase (pretraining). Same total cost as
  1xH100, faster wall-clock.

On-Volume directory of record:

```
/data/clean/            cleaned .txt shards, per source          (Phase 1-2)
/data/tokenizer/        the trained 16K byte-level BPE           (Phase 3)
/data/tokens/train/     99% packed uint16 1024-token windows     (Phase 4)
/data/tokens/val/        1% packed windows                       (Phase 4)
/data/checkpoints/base/ final model (HF safetensors)             (Phase 5)
/data/checkpoints/ckpt.pt  resumable optimizer state             (Phase 5)
```

---

## Credentials

`.env.local` (git-ignored, already written) holds the Modal token. The token is
verified and Modal is authed on profile `thesreedath`:

```
Token verified successfully!
```

`HUGGINGFACE_TOKEN` is **not needed yet**. All three datasets are ungated, so the
HF token is only required to *push the finished model* in Phase 6. We will add it
to `.env.local` and create a Modal secret before deploy.

---

## The data mix (from `config.py`)

| Source | HF id | Split / config | Weight | Text field |
|---|---|---|---|---|
| case-law | `HFforLegal/case-law` | split `us` | 0.70 | `document` |
| sec | `PleIAs/SEC` | `train` | 0.20 | `text` |
| fineweb-edu | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | 0.10 | `text` |

Target: ~10B cleaned tokens (~7B case-law, ~2B SEC, ~1B web) ≈ ~79 tok/param.

---

## What Phase 0 will actually run (needs your thumbs-up)

Three steps, all ≈ $0:

1. **Create the Modal Volume `slm-125m`** (idempotent; no-op if it exists).
2. **Author `modal_app.py`** with the App, the CPU base image (`datasets`,
   `langdetect`, `tokenizers`, `transformers`), the Volume mount, and a
   `smoke_test` function.
3. **Run the 10-doc smoke test** on a Modal CPU container: stream **10 documents
   from each source**, run each through the cleaning chain, and print
   before/after (raw length, kept/dropped, drop reason, a cleaned excerpt). This
   confirms in one shot: network reachability, correct field extraction per
   source (`document` vs `text`), and that the cleaner behaves.

The cleaner used in the smoke test is the same fixed, deterministic rule chain
Phase 1 will use at scale:

1. `filter_lines`: drop lines <40 chars or >30% non-alphanumeric; collapse whitespace.
2. `strip_boilerplate`: delete lines matching known regexes (FORM 10-K, `Page N of M`,
   SEC headers, `/s/` signatures, Table of Contents, "All rights reserved").
3. length gate: drop the doc if <600 chars survive (`too_short`).
4. `is_repetitive`: drop if top-10 4-grams cover >50% of all 4-grams (`repetitive`).
5. `is_english`: `langdetect` on the first 5k chars; ASCII-ratio fallback (`non_english`).
6. optional strict OCR pass: drop docs where >3% of words look like OCR errors.

**No data is stored in Phase 0.** We stream 30 docs total, print, and discard.

---

## Deliverable

This doc + a passing 10-doc smoke test (before/after printout per source).

## After Phase 0

Stop. Review the smoke output together. Then write `docs/01-data.md` and, on your
approval, run the real streaming clean to ~10B tokens (Phase 1, CPU, low cost).
