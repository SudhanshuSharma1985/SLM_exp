# 125M SLM Build, Data Pipeline Summary (Phases 0 to 4)

*What we did to turn three raw HuggingFace datasets into ~2.19B training-ready
tokens, before any GPU is touched. Written 2026-07-09.*

---

## 1. What was built and the data flow

A domain-specific (legal/financial) corpus for a 125M Llama-style model, built
entirely on **CPU** on Modal. Every artifact lives on one persistent Modal Volume
(`slm-125m`, mounted at `/data`). The pipeline:

```
3 HF datasets  ->  clean (rules)  ->  dedup + decontaminate  ->  16K BPE tokenizer  ->  pack uint16 windows, split 99/1
   (stream)        /data/clean         /data/corpus              /data/tokenizer        /data/tokens/{train,val}
```

The corpus target was chosen from measurement, not the original brief. The two
legal sources hold only ~2B unique tokens combined, so a strict 70/20/10 mix at
10B tokens was impossible. We chose **Choice A (legal-first)**: take all the legal
data, add a small web slice, and reach more tokens-seen later via multiple epochs.

---

## 2. How much of each dataset (amounts at every stage)

Datasets streamed (never fully downloaded): `HFforLegal/case-law` (split `us`),
`PleIAs/SEC`, `HuggingFaceFW/fineweb-edu` (`sample-10BT`).

| Source | Docs streamed | Kept after clean (P1) | Kept after dedup/decontam (P2) | Final train tokens (P4) |
|---|---|---|---|---|
| case-law | 238,207 | 232,292 (97.5%) | 206,684 | ~863M |
| SEC | 47,752 | 47,199 (98.8%) | 45,035 | ~861M |
| fineweb-edu | 432,821 | 418,405 (96.7%) | 418,405 | ~465M |
| **Total** | **718,780** | **697,896** | **670,124** | **~2.19B train + 22.1M val** |

Token counts: Phase 1 to 2 use a chars/4 proxy (~2.68B then ~2.40B); Phase 4's
real tokenizer gives the true **2.19B train + 22.1M val** (1.0% val, 99/1 split).
The proxy overestimated by ~8% because the tokenizer compresses legal text well.

Why the legal caps: case-law `us` has only 282K docs (~1.0B tokens) and SEC 49K
docs (~1.1B tokens). fineweb-edu was intentionally capped at 0.5B (it holds ~11B).

---

## 3. What "cleaning" did (Phase 1, deterministic rule chain)

Each streamed document ran through a fixed 6-step chain (`cleaning.py`), cheapest
check first; the first failure drops the doc and the reason is tallied:

1. **filter_lines** - drop lines < 40 chars or > 30% non-alphanumeric; collapse whitespace.
2. **strip_boilerplate** - delete lines matching known regexes: `FORM 10-K`,
   `Page N of M`, SEC cover headers, `/s/` signatures, `Table of Contents`,
   "All rights reserved".
3. **length gate** - drop the doc if < 600 chars survive (`too_short`).
4. **is_repetitive** - drop if the top-10 4-grams cover > 50% of all 4-grams.
5. **is_english** - ASCII-ratio first (near-pure ASCII = English), `langdetect`
   only for the ambiguous 90-99% band (this made streaming ~100x faster on web docs).
6. **OCR gate (case-law only)** - drop if > 20% of a doc's words are not in the
   system English dictionary (`/usr/share/dict/words`). Threshold picked from a
   measured sample (median non-word ratio 4.7%, so > 20% is real OCR garble, not
   legalese). Removed 1,370 badly-scanned opinions.

Drops were almost all `too_short` (tiny stubs). No model scoring, no rewriting, no
PII work; everything is deterministic and reproducible.

**Phase 2 (dedup + decontaminate)** then refined the clean corpus:
- **Near-dedup** case-law with MinHash-LSH (5-word shingles, 32-perm signatures,
  Jaccard 0.8): removed 1,606 near-duplicate opinions.
- **Exact-dedup** all sources (blake2b of normalized text): removed 1,989 SEC and
  62 web verbatim repeats.
- **Contamination strip** (legal sources): dropped any doc sharing a 13-word span
  with the CaseHOLD/LexGLUE benchmarks (480,908 eval 13-grams). Removed **24,002
  case-law docs**, essential because CaseHOLD is built from case-law holdings, so
  training on them would make held-out evaluation dishonest.

---

## 4. How the tokenizer was trained (Phase 3)

- A **fresh 16,384-token byte-level BPE**, trained with HuggingFace `tokenizers`
  (`models.BPE` + `pre_tokenizers.ByteLevel` + `BpeTrainer`) by streaming the whole
  ~2.4B-token corpus line by line. The Rust trainer finished in a few minutes.
- **Byte-level** so there is never an out-of-vocabulary token (any character maps
  to byte tokens); robust to OCR garble and odd unicode.
- **16K on purpose**: the embedding table is `16,384 x 768 ≈ 12.6M` params (~10%
  of the model). A bigger vocab would spend the tiny 125M budget on embeddings
  rather than transformer layers.
- Special tokens reserved at train time: `<|bos|> <|eos|> <|pad|> <|unk|>` plus
  chat tokens `<|user|> <|assistant|> <|system|>` (reserved for later alignment).
- Saved as a `transformers.PreTrainedTokenizerFast` at `/data/tokenizer/`.
- Round-trip check (both exact): "The plaintiff shall bear the burden of proof by a
  preponderance of the evidence." = 15 tokens; "The Company's net revenues
  increased 12% year over year pursuant to the agreement." = 16 tokens.

**Phase 4 (tokenize + pack)** then encoded every doc, appended `<|eos|>` after
each, packed the stream into fixed **1024-token windows**, and wrote them as
**uint16 `.bin` shards** with an `index.json`. The split is deterministic and
by-window (every 100th window to val), so no val window can leak into training.
Result: **2.19B train + 22.1M val tokens** (99/1). 14 workers ran in parallel.

---

## 5. Time and cost

**Cost: ~$0.18 total so far** (all Phases 0 to 4, CPU only, per `modal billing
report`). Effectively free; the only real spend is Phase 5 pretraining on GPU
(~$15 to 25, hard-capped at $40).

**Wall-clock (useful compute, per phase, after fixes):**

| Phase | What | Time |
|---|---|---|
| 0 Setup + smoke test | 30-doc stream + clean | ~2 min |
| 1 Clean | stream + clean 718K docs, 16-wide | ~15 min useful (more with iterations) |
| 2 Dedup + decontaminate | parallel MinHash sigs + LSH + writes | ~6 min |
| 3 Tokenizer | train 16K BPE on 2.4B tokens | ~4 min |
| 4 Tokenize + pack | encode + pack + split, 14-wide | ~10 min |

Real elapsed was longer because of transparent iteration: we measured true dataset
sizes before committing, fixed a `langdetect` throughput bottleneck, replaced a
no-op OCR heuristic with a real dictionary gate, and re-architected Phase 2 from a
slow single container (SHA1/blake2b per shingle, plus Modal preemption restarting
it from zero) into fully parallel per-shard workers. Those lessons are recorded in
`docs/01-data.md` and `docs/02-dedup.md`.

---

## 6. What's next (not yet run)

**Phase 5, pretraining** (the only GPU spend): train the 125M Llama-style model
(12L / 768d / 12h, SwiGLU, RMSNorm, RoPE, tied embeddings, ctx 1024) on the 2.19B
train tokens for several epochs on 8xH100, targeting held-out validation
perplexity. Needs your explicit go-ahead and a `--max-usd` cap before it starts.
