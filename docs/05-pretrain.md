# Phase 5, Pretrain the 125M model

**Goal:** train the 125M Llama-style model from random weights on the 2.19B-token
corpus for ~5 epochs, targeting held-out validation perplexity, and save both a
resumable checkpoint and a HuggingFace `safetensors` model. This is the ONLY GPU
phase and the only real spend.

Input:  `/data/tokens/train/*.bin`, `/data/tokens/val/*.bin`, `/data/tokenizer/`
Output: `/data/checkpoints/base/` (HF model) + `/data/checkpoints/ckpt.pt`
        (resumable) + `/data/checkpoints/metrics.jsonl` (loss/ppl curve)

---

## The model (from `config.py`, maps 1:1 to `transformers.LlamaConfig`)

| Field | Value |
|---|---|
| parameters | 125,847,552 (~125.8M) |
| layers | 12 |
| hidden size | 768 |
| attention heads | 12 (head dim 64), MHA (kv heads = 12) |
| MLP | SwiGLU, inner 3072 |
| norm | RMSNorm, pre-norm |
| positions | RoPE (theta 10000) |
| context length | 1024 |
| vocab | 16,384 |
| embeddings | tied (input = output) |

We instantiate `LlamaForCausalLM(LlamaConfig(**MODEL.to_llama_kwargs()))` with fresh
random weights. Nothing is loaded; this is from scratch.

---

## The training recipe

| Knob | Value | Why |
|---|---|---|
| objective | next-token cross-entropy | standard causal LM |
| epochs | ~5 over 2.19B tokens = ~11B tokens seen | proven recipe; prior run hit val ppl ~16 |
| optimizer | AdamW, betas (0.9, 0.95), wd 0.1 | GPT-2/llm.c norm |
| learning rate | 6e-4 -> 6e-5, cosine decay | `TrainConfig.lr`/`min_lr` |
| warmup | 200M tokens | `TrainConfig.warmup_tokens` |
| global batch | ~524,288 tokens/step (~0.5M) | `TrainConfig.global_batch_tokens` |
| precision | bf16 | H100-native, stable |
| grad clip | 1.0 | stability |
| grad checkpointing | off | model is small |
| parallelism | single-node 8x H100 DDP | same total cost as 1x, faster wall-clock |

Batch math: micro-batch 32 x seq 1024 = 32,768 tokens/GPU/step; 8 GPUs = 262,144
tokens/step; grad-accum 2 -> ~524,288 tokens/step. ~11B tokens / 0.5M = ~21,000
optimizer steps.

## Data loading

A memory-mapped reader over the `.bin` shards: each `uint16` file is a flat array
of 1024-token windows. The loader yields random windows (shuffled by a seeded
permutation), shards across the 8 DDP ranks (each rank sees a disjoint slice), and
loops for ~5 epochs. `input_ids = window`, `labels = window` shifted by one (the
model does the shift internally). No padding; windows are already full.

## Evaluation

Every `eval_every_steps` (1000), compute mean cross-entropy on the full 1% val set
(22.1M tokens) and report **perplexity = exp(loss)**. This is the headline metric;
at 125M, MMLU-style benchmarks are near-random and not meaningful.

## Checkpointing (resumable + shippable)

- Every `ckpt_every_steps` (500), rank 0 writes `/data/checkpoints/ckpt.pt`
  = {model, optimizer, scheduler, step, rng} so a killed/preempted run resumes
  bit-for-bit. `--resume` reloads it.
- At the end (and at each new best val), rank 0 also writes the HF model to
  `/data/checkpoints/base/` via `save_pretrained` (config.json + model.safetensors
  + generation_config), and copies the tokenizer files in. That directory is the
  Phase 6 deploy artifact.
- `metrics.jsonl` gets one row per log step (step, tokens, train loss, lr) and per
  eval (val loss, val ppl) for plotting.

---

## Cost + safety

- **8x H100** at ~$3.95/hr each = ~$31.6/hr for the node. A ~11B-token run on a
  125M model is roughly 30 to 60 min of compute -> **~$15 to 25**.
- **Hard cap:** the Modal function takes `--max-usd 40` (`BUDGET_CAP_USD`). It
  tracks elapsed GPU-seconds against the cap and, if it would exceed it,
  checkpoints and stops cleanly rather than blowing the budget.
- **Smoke first (near $0):** before the full run, a `--smoke` run does ~20 steps +
  one eval on a single H100 to prove the loop, eval, and checkpoint write end to
  end. Only after that passes do we launch the full 8x run.

## What "done" looks like

- A decreasing loss curve and val perplexity trending toward ~16.
- Real sample generations from a legal/financial prefix (the base model is a
  COMPLETER, not a chat model: give it a passage start and it continues).
- `/data/checkpoints/base/` ready for Phase 6 (push to HF + serve behind a site).

---

## Plan of execution (on your approval)

1. Write `train.py` (model, mmap data loader, DDP loop, eval, checkpointing) and a
   `pretrain` Modal GPU function in `modal_app.py`, with `--smoke`, `--resume`,
   `--epochs`, and `--max-usd` flags.
2. Run `--smoke` (single H100, ~20 steps). Confirm loss drops, eval runs,
   checkpoint writes. Near $0.
3. On your go, launch the full **8x H100, ~5 epochs, `--max-usd 40`** run. Report
   loss + perplexity at real milestones, not every step.

**Deliverable:** `docs/05-pretrain.md` (this file) + a loss/perplexity curve + real
sample generations + the HF model directory on the Volume.

---

## RESULTS (executed 2026-07-09)

Full run: 8x H100, 5 epochs, 20,885 steps, 10.95B tokens seen, **$56.58** (cap $75).

Validation perplexity curve (every 1000 steps):

| step | val ppl | step | val ppl | step | val ppl |
|---|---|---|---|---|---|
| 1000 | 16.47 | 8000 | 9.42 | 15000 | 8.62 |
| 2000 | 12.58 | 9000 | 9.27 | 16000 | 8.57 |
| 3000 | 11.33 | 10000 | 9.11 | 17000 | 8.54 |
| 4000 | 10.65 | 11000 | 8.98 | 18000 | 8.53 |
| 5000 | 10.21 | 12000 | 8.85 | 19000 | 8.51 |
| 6000 | 9.89 | 13000 | 8.76 | 20000 | 8.50 |
| 7000 | 9.63 | 14000 | 8.68 | final | **8.50** |

**Final val perplexity 8.50** (val loss 2.14), about 2x better than the earlier
125M legal run (16.3). The clean, deduplicated, decontaminated 2.19B corpus plus a
straightforward cosine recipe did it. Smoke test first (single H100, 20 steps,
~$0.12) confirmed the loop before the priced run.

Sample completions (base = completer; give it a prefix, it continues in-register):

- "The plaintiff alleges that the defendant" -> "failed to offer... this court is
  faced with a situation in which there are two sisters-in-law of a deceased
  granddaughter... whose parents were unknown"
- "Pursuant to the terms of this Agreement," -> "we will agree that it was in
  effect... cited no authority but Bates v. Herron (1947), 332 Ill. App. 5..."
- "In determining whether the search was reasonable, the court" -> "to conclude
  that she must be relieved of her duty... It is not enough for such testimony to
  provide some evidence from which an inference can reasonably be drawn..."

Honest note: the model speaks the legal register fluently with case-citation style,
but it is a base completer, it does not know facts, and a financial prompt drifts
toward the dominant legal register. Facts need RAG (a later phase).

Artifacts on the Volume: `/data/checkpoints/base/` (HF safetensors + tokenizer),
`/data/checkpoints/ckpt.pt` (resumable), `/data/checkpoints/metrics.jsonl` (curve).
