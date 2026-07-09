# Phase 4, Tokenize + pack + split 99/1

**Goal:** turn the Phase 2 text corpus into the exact binary the trainer reads:
packed `uint16` token windows, split into train and val. CPU only, low cost.

Input:  `/data/corpus/<source>/corpus.txt` + `/data/tokenizer/`
Output: `/data/tokens/train/*.bin`, `/data/tokens/val/*.bin`, `/data/tokens/index.json`

---

## The data format (the concrete answer to "what does training data look like")

1. Each cleaned document is **batch-encoded** to token ids by the Phase 3 tokenizer.
2. An **`<|eos|>` id is appended after every document** as a separator, so the
   model learns where documents end.
3. The flat id stream is **packed into fixed 1024-token windows** (the model's
   context length). No padding inside a window; documents flow across window
   boundaries and the EOS marks the seams.
4. Windows are stored as **`uint16` binary shards** (`*.bin`). `uint16` because a
   16,384 vocab fits in 16 bits, half the disk of `int32`. An `index.json` records
   every shard, its window count, the dtype, and the seq_len.

## The 99/1 split

Deterministic and by-window: **every 100th packed window goes to `val/`**, the
rest to `train/`. Because it is a fixed rule on the window index, it is
reproducible and no validation window can leak into training. Result: ~99% train,
~1% val (~2.6B train + ~26M val tokens).

## Parallelization

Tokenization reads **only the local corpus on the Volume, never HuggingFace**. Each
source is split into several shards by line (`line_index % num_shards`), one Modal
worker per shard, running concurrently. Each worker:
- fast-encodes its docs in batches (the Rust tokenizer uses all cores),
- appends `<|eos|>` after each doc, packs full 1024-token windows,
- routes every 100th of its own windows to val, the rest to train,
- writes its own `train-*.bin` / `val-*.bin` (no shared state), and returns counts.

A final step merges the per-worker counts into `index.json`. Partial trailing
windows (< 1024 tokens) are dropped; the loss is a few hundred tokens per shard,
negligible against ~2.6B.

## Deliverable

`docs/04-tokenize.md` + final token/window counts for train and val, and the
written `index.json`.

## RESULTS (executed 2026-07-09)

`/data/tokens/index.json` (seq_len 1024, dtype uint16):

| Split | Windows | Tokens |
|---|---|---|
| train | 2,138,970 | **2.19B** |
| val | 21,614 | **22.1M** (1.0%) |

Per-source train tokens (real tokenizer counts): case-law ~863M, SEC ~861M,
fineweb-edu ~465M. 14 workers (case-law 4 / SEC 6 / fineweb 4), all parallel.

Note: the real token count (2.19B) landed ~8% below the Phase 2 chars/4 estimate
(2.40B) because the 16K tokenizer compresses legal/financial text better than 4
chars/token. This 2.19B-token set is what Phase 5 trains on (multiple epochs).
