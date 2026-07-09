# SLM From Scratch, Fresh Working Directory Guide (paste this in)

**Purpose:** copy this file into a brand-new, empty working directory, drop your
Modal token next to it, open Claude Code there, and start a clean 125M-parameter
SLM build from zero. This file is the brief for the coding agent: it defines the
scope, the phases, the exact data format at every step, and how the finished model
is deployed for inference.

Give the agent this file and say: *"Read SLM_BUILD_GUIDE.md and let's start Phase 0."*

---

## What we are building

A **125-million-parameter, Llama-style, decoder-only language model**, pretrained
**completely from scratch** (fresh weights, fresh tokenizer) on a domain-specific
corpus, on a **Modal** account, then deployed to a **website for inference**.

Scope for this build, in order:

1. Download the training data (stream from HuggingFace)
2. Clean it (rule-based pipeline)
3. Train the tokenizer (fresh 16K byte-level BPE)
4. Tokenize the data (packed token ids, 99/1 train/val split)
5. Pretrain the 125M model
6. Deploy the pretrained model to a website for inference

Nothing else (no SFT / alignment / RAG) in this build. Those are later.

---

## Credentials you must drop into this directory

Create a file `.env.local` in this directory (never commit it):

```bash
MODAL_TOKEN_ID=...            # from your new Modal account
MODAL_TOKEN_SECRET=...        # from your new Modal account
HUGGINGFACE_TOKEN=hf_...      # a HF token with write access (for pushing the model)
```

The agent will `source .env.local` and `modal token set` / create a Modal secret
so the training containers can read HuggingFace. All three datasets are **ungated**,
so the HF token is really only needed to *push the finished model*, not to read data.

---

## How we work (rules for the agent)

These are firm. They exist because the last build got stuck in a "black box."

- **Lean and transparent, phase by phase.** Do one phase, show the result, stop.
  No multi-hour fire-and-forget runs.
- **A short markdown doc before each phase.** Before executing a phase, write a
  readable `docs/NN-<phase>.md` explaining what will run and why. Get a thumbs-up
  before spending money.
- **Cost visibility + hard caps.** Only pretraining (Phase 5) uses a GPU. Check
  spend with `modal billing report --start <date> --json`. Put a `--max-usd` cap
  on the pretrain run.
- **`config.py` is the single source of truth.** Model geometry, tokenizer vocab,
  data mix, paths, all live there. Every other module imports from it.
- **Immutable, small, well-named code.** Pure functions for the data filters, one
  file per concern, type annotations, no in-place mutation.

---

## The compute layout (Modal)

- One Modal **App** for the build, one persistent **Volume** (call it `slm-125m`)
  mounted at `/data`. Everything durable lives on the Volume.
- Cleaning + tokenizer + tokenization = **CPU containers** (cheap). Data is streamed
  and cleaned before any GPU boots.
- Pretraining = **1× or 8× H100** (H100 is the best $/token at this scale). A single
  H100 pretrains a 125M model for roughly \$15 to 25; 8× H100 is faster for the same
  total cost.
- Directory of record on the Volume:

```
/data/clean/            cleaned .txt shards, per source          (Phase 1 to 2)
/data/tokenizer/        the trained 16K byte-level BPE            (Phase 3)
/data/tokens/train/     99% packed token windows (.bin + index)  (Phase 4)
/data/tokens/val/        1% packed token windows                 (Phase 4)
/data/checkpoints/base/ final model (HF safetensors)             (Phase 5)
/data/checkpoints/ckpt.pt  resumable optimizer state             (Phase 5)
```

---

## Phase 0, Setup + smoke test  (cost ≈ \$0)

1. `source .env.local`, `modal token set`, create the Modal Volume and an HF secret.
2. Write `config.py` with the model + tokenizer + data-mix constants (see below).
3. Smoke test: stream **10 documents** from each source, run them through the
   cleaner, print before/after. Confirms the network, the secret, and the field
   extraction all work before any real run.

**Deliverable:** `docs/00-setup.md` + a passing 10-doc smoke test.

---

## Phase 1, Download + clean  (CPU, low cost)

**Download strategy: stream, don't hoard.** Use HuggingFace `datasets` with
`streaming=True`. Pull documents one at a time, clean each on the fly, and **stop
each source at its token budget.** We never materialize the multi-TB raw datasets.

The data mix (put in `config.py`):

| Source | HF id | Weight | Text field |
|---|---|---|---|
| case-law | `HFforLegal/case-law` (split `us`) | 0.70 | `document` |
| sec | `PleIAs/SEC` | 0.20 | `text` |
| fineweb-edu | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | 0.10 | `text` |

**How much to pull:** build a cleaned corpus of about **10B tokens** (≈7B case-law,
2B SEC, 1B web). At ~4 chars/token that is ~40GB of clean text, a few percent of
the sources, and ~80 tokens/parameter for a 125M model (well past Chinchilla's ~20).

**The cleaning pipeline (fixed, rule-based, deterministic).** Per streamed document,
cheapest-check-first (a drop ends the chain, and every drop is counted by reason):

1. `filter_lines`, drop lines <40 chars or >30% non-alphanumeric; collapse whitespace.
2. `strip_boilerplate`, delete whole lines matching known regexes (FORM 10-K,
   `Page N of M`, SEC headers, `/s/` signatures, Table of Contents, All rights reserved).
3. length gate, drop the doc if <600 chars survive (`too_short`).
4. `is_repetitive`, drop if the top-10 4-grams cover >50% of all 4-grams (`repetitive`).
5. `is_english`, `langdetect` on the first 5k chars; ASCII-ratio fallback (`non_english`).
6. optional strict OCR pass, drop docs where >3% of words look like OCR errors.

Kept docs are written one-per-line to `/data/clean/<source>/*.txt`.

**Reusability note (important):** this cleaned corpus is the durable asset. Future
bigger models **reuse** it; we only ever stream *additional* data for a model that
needs more than 10B tokens, appending new cleaned shards. We never re-download or
re-clean what's already here.

**Deliverable:** `docs/01-data.md` + drop-count report per source.

---

## Phase 2, Dedup + contamination strip  (CPU, low cost)

1. **Near-dedup** the dominant source (case-law) with **MinHash-LSH** (5-word
   shingles → 64-num signature → LSH buckets, Jaccard 0.8). Drop near-duplicates;
   small sources pass through. Exact blake2b hashing is the fallback.
2. **Strip eval contamination**, drop docs resembling the benchmark sets (LexGLUE /
   CaseHOLD) so held-out evaluation stays honest.

**Deliverable:** `docs/02-dedup.md` + kept/dropped counts.

---

## Phase 3, Train the tokenizer  (CPU, ≈ \$0)

Train a **fresh 16,384-token byte-level BPE** tokenizer on the cleaned corpus.

- Byte-level → **no out-of-vocabulary token, ever**; robust to OCR garble/unicode.
- 16K is deliberately small: at 125M the embedding table is ~10% of the model, so a
  small vocab leaves more budget for the transformer layers.
- Reserve special tokens at train time: `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`
  (and chat tokens `<|user|>`, `<|assistant|>`, `<|system|>` for later alignment).
- Save as a HuggingFace `PreTrainedTokenizerFast` at `/data/tokenizer/`.

**Gotcha (write it down now):** `llama.cpp`'s `convert_hf_to_gguf.py` rejects a fresh
byte-level BPE (`BPE pre-tokenizer was not recognized`) and the hash changes every
retrain. If we later export GGUF, patch the converter to map our unknown byte-level
BPE → the `gpt-2` pre-tokenizer. Keep that patch idempotent.

**Deliverable:** `docs/03-tokenizer.md` + a round-trip encode/decode sanity check.

---

## Phase 4, Tokenize + split 99/1  (CPU, low cost)

**Data format of the training data (this is the concrete answer to "is there a
specific data format?"):**

- Each cleaned document is batch-encoded to token ids.
- An **`<|eos|>` id is appended after every document** as a separator.
- The flat id stream is **packed into fixed 1024-token windows** (the model's
  context length). No padding inside a window; the EOS marks document boundaries.
- Windows are stored as **`uint16` binary shards** (`*.bin`) plus an `index.json`
  (shard names, token counts, window counts, dtype, seq_len). `uint16` because a
  16K vocab fits in 16 bits, half the disk of `int32`.
- Tokenization is **parallel**: one worker per input shard, then merge + re-index.
  It reads only the local cleaned corpus, never HuggingFace.

**The split:** deterministic **99% train / 1% val**, every 100th packed window goes
to `/data/tokens/val/`, the rest to `/data/tokens/train/`. By-window and reproducible,
so no validation window can leak into training. Result: ~9.9B train + ~100M val tokens.

**Deliverable:** `docs/04-tokenize.md` + final token/window counts for train and val.

---

## Phase 5, Pretrain the 125M model  (GPU, the only real spend)

**Model architecture (put in `config.py`, maps 1:1 to `transformers.LlamaConfig`):**

| Field | Value |
|---|---|
| parameters | ~125M |
| layers | 12 |
| hidden size | 768 |
| attention heads | 12 (head dim 64), MHA (kv heads = heads) |
| MLP | SwiGLU, inner 3072 |
| normalization | RMSNorm, pre-norm |
| position embeddings | RoPE (rotary, 0 params) |
| context length | 1024 |
| vocab | 16,384 |
| embeddings | tied (input = output projection) |

**Training recipe:** next-token cross-entropy, AdamW (β 0.9/0.95, wd 0.1), cosine
LR with warmup, grad-clip 1.0, ~0.5M-token global batch, bf16, gradient checkpointing
off (small model). Single-node DDP if using 8× H100. Read `/data/tokens/train/`,
evaluate **perplexity** on `/data/tokens/val/` every N steps. Keep a resumable
`ckpt.pt` (optimizer + step) AND write the model as HF safetensors to
`/data/checkpoints/base/`.

**Honest framing:** at 125M the headline metric is **held-out validation perplexity**,
not MMLU (near-random at this size). The base model is a **completer**, not a chat
model, give it a passage prefix and it continues it.

**Cost:** ~\$15 to 25 for a solid single-epoch/10B-token run on H100; put a `--max-usd`
cap on it. Report loss + perplexity at real milestones only.

**Deliverable:** `docs/05-pretrain.md` + a loss/perplexity curve + real sample
generations from a legal/financial prefix.

---

## Phase 6, Deploy the pretrained model for inference

Here is exactly what the finished artifact is, where it runs, and how.

### What the pretrained model looks like (the artifact)

A standard HuggingFace model directory, the same shape as any Llama checkpoint:

```
checkpoints/base/
├── config.json              # LlamaConfig: 12L/768d/12h, vocab 16384, ctx 1024, RoPE
├── model.safetensors        # the weights, bf16, ~250MB
├── generation_config.json   # eos/bos/pad ids, default sampling
├── tokenizer.json           # the fast tokenizer (merges + vocab)
├── tokenizer_config.json
└── special_tokens_map.json  # <|bos|> <|eos|> <|pad|> <|unk|>
```

- **Format: `safetensors`** (safe, zero-copy, the HF-native format) + JSON config +
  the tokenizer files. This is directly loadable with
  `AutoModelForCausalLM.from_pretrained(...)` / `AutoTokenizer.from_pretrained(...)`.
- It is a **base completion model**: prompt = `<|bos|>` + your text, and it continues.
- Optional: a **quantized GGUF** export (Q4/Q8) via `llama.cpp` for laptop/local
  inference (needs the tokenizer patch from Phase 3). Not required for the website.

### Are we deploying it on HuggingFace?

**Two parts, and yes to the first:**

1. **The model weights → a HuggingFace repo** (e.g. `your-user/slm-125m-base`),
   pushed with the write token. This is the canonical home of the artifact.
2. **The inference API → a HuggingFace Space** (Docker SDK, FastAPI, CPU). The Space
   bundles the weights + tokenizer and exposes `POST /generate {prompt, max_new_tokens,
   temperature, top_p, top_k}` → `{generated}`. Open CORS so a static site can call it.
   - Free CPU Spaces sleep when idle → the first call is a ~20 to 40s cold start; the
     frontend should show a "waking the model" state and auto-retry.
   - Space gotcha: pin **Python 3.12** in the Dockerfile (gradio/others break on 3.13).
   - **Alternative:** serve from a **Modal endpoint** (`@modal.asgi_app`) that mounts
     the Volume, keeps everything on Modal, warmer, but costs while running. Pick one;
     HF Space is the zero-cost default.

### The website (frontend)

A small **static site deployed to Vercel** that calls the Space/Modal endpoint:

- A prompt box + sampling controls (temperature, max tokens, top-p) and the streamed
  completion.
- A short "what this is" panel: it speaks the domain register, it is a base completer,
  the honest metric is perplexity.
- **Gotcha to bake in:** a base model under greedy decoding can emit EOS after ~1
  token → enforce a `min_new_tokens`, and suppress rare non-ASCII vocab tokens +
  `top_k` to avoid unicode garbage.

**Deliverable:** `docs/06-deploy.md` + the live HF repo URL + Space URL + Vercel URL.

---

## Quick reference: what format is the data at each step?

| Step | On-disk format |
|---|---|
| Raw (HF) | streamed records (parquet rows), never fully stored |
| Cleaned corpus | UTF-8 `.txt`, **one document per line**, per-source folders |
| Tokenizer | HF `PreTrainedTokenizerFast` (`tokenizer.json` + configs) |
| Tokenized data | **`uint16` `.bin` shards**, EOS-separated, packed into 1024-token windows, `index.json` |
| Train/val split | same `.bin` format under `/tokens/train` (99%) and `/tokens/val` (1%) |
| Pretrained model | HF **`safetensors`** + `config.json` (LlamaConfig) + tokenizer files |
| Optional deploy | quantized **GGUF** for local/llama.cpp inference |

---

## One-line summary for the agent

Stream three HF datasets → clean with a fixed rule chain → dedup + decontaminate →
train a 16K byte-level BPE → pack into `uint16` 1024-token windows split 99/1 →
pretrain a 125M Llama-style model (RoPE/SwiGLU/RMSNorm, tied embeddings) on Modal →
push `safetensors` to HuggingFace → serve from a HF Space (or Modal endpoint) behind
a Vercel frontend. Lean, per-phase docs, approval + cost cap before GPU spend.
