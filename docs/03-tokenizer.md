# Phase 3, Train the tokenizer

**Goal:** train a fresh **16,384-token byte-level BPE** tokenizer on the Phase 2
corpus and save it in HuggingFace form. CPU only, ≈ $0.

Input:  `/data/corpus/<source>/corpus.txt`
Output: `/data/tokenizer/` (a `PreTrainedTokenizerFast`: `tokenizer.json` + configs)

---

## What a tokenizer is (and why we train our own)

The model reads integers, not characters. The tokenizer is the dictionary that
maps text <-> integer ids. We train a **fresh** one on our own corpus so the
vocabulary is tuned to legal/financial language (common terms like "plaintiff",
"pursuant", "the Company" become single tokens), which packs more text into the
same context window.

**Byte-level BPE** means the base alphabet is the 256 raw bytes, so:
- **no out-of-vocabulary token, ever** - any character (odd unicode, OCR garble)
  is representable as a sequence of byte tokens. `<|unk|>` is essentially never used.
- BPE then greedily merges the most frequent adjacent pairs until the vocab
  reaches 16,384 entries.

## Why 16K (small on purpose)

At 125M params the embedding table is `vocab x hidden = 16,384 x 768 ≈ 12.6M`
params, about **10% of the model**. A larger vocab would spend the model's tiny
budget on the embedding table instead of the transformer layers. 16K is a
deliberate trade that leaves more capacity for actual reasoning.

## Special tokens (reserved at train time)

Reserved now so ids are stable forever:
`<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`, plus chat tokens
`<|user|>`, `<|assistant|>`, `<|system|>` (unused in pretraining; reserved so a
later alignment phase needs no tokenizer change).

## How it is trained (implementation)

- HuggingFace `tokenizers`: a `models.BPE` with a `pre_tokenizers.ByteLevel` and a
  `ByteLevelBPETrainer`, `vocab_size=16384`, the special tokens above.
- Trained by streaming the corpus files line-by-line (no full load into RAM). The
  Rust trainer is fast: 16K merges over ~10GB of text is a few minutes on CPU.
- Wrapped as a `transformers.PreTrainedTokenizerFast` with the special-token map
  and saved with `save_pretrained` so Phases 4 to 6 load it with
  `AutoTokenizer.from_pretrained`.

## Known gotcha (write it down now)

`llama.cpp`'s `convert_hf_to_gguf.py` rejects a fresh byte-level BPE
(`BPE pre-tokenizer was not recognized`) and the hash changes every retrain. If we
later export GGUF, patch the converter to map our unknown byte-level BPE to the
`gpt-2` pre-tokenizer, kept idempotent. Not needed for the website deploy.

## Deliverable

`docs/03-tokenizer.md` + a round-trip encode/decode sanity check on a legal and a
financial sentence, plus a few sample merges and the final vocab size.

## RESULTS (executed 2026-07-09)

- Trained a fresh **16,384-token byte-level BPE** on the ~2.40B-token Phase 2
  corpus, saved as a `PreTrainedTokenizerFast` at `/data/tokenizer/`.
- Round-trip sanity check (both exact):
  - "The plaintiff shall bear the burden of proof by a preponderance of the
    evidence." -> **15 tokens** (13 words; tight legal compression), roundtrip OK.
  - "The Company's net revenues increased 12% year over year pursuant to the
    agreement." -> **16 tokens**, roundtrip OK.
- Special tokens reserved: `<|bos|> <|eos|> <|pad|> <|unk|>` + chat tokens
  `<|user|> <|assistant|> <|system|>` (unused in pretraining).
