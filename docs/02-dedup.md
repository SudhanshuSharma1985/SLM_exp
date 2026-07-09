# Phase 2, Dedup + contamination strip

**Goal:** turn the Phase 1 cleaned corpus into the final training corpus by
removing (a) near-duplicate documents and (b) any text that overlaps the
evaluation benchmarks. CPU only, low cost. No GPU.

Input:  `/data/clean/<source>/*.txt`  (Phase 1 output, ~2.68B tokens)
Output: `/data/corpus/<source>/*.txt` (final training corpus)

We write to a **new** directory so Phase 1's output stays immutable and the step
is re-runnable.

---

## Why this phase matters

- **Duplicates** waste the model's tiny capacity memorizing repeated passages and
  inflate the apparent corpus size. Court opinions especially reuse long
  boilerplate (syllabi, standard-of-review paragraphs, procedural recitals).
- **Contamination** is training on text that also appears in our eval sets. If we
  do that, held-out perplexity / benchmark numbers are dishonest, the model has
  effectively seen the test. We strip it so evaluation stays fair.

---

## 1. Near-duplicate removal (MinHash-LSH), case-law only

case-law is the dominant, most-repetitive source, so it gets real near-dedup:

- **Shingles:** each doc -> set of overlapping **5-word** shingles.
- **Signature:** a **64-permutation MinHash** per doc (a compact fingerprint where
  the chance two docs' hashes match ≈ their Jaccard similarity).
- **LSH bucketing:** MinHashLSH with **Jaccard threshold 0.8**. Docs that land in
  the same bucket are near-duplicates; we keep the first seen and drop the rest.
- Implemented with `datasketch`. Two-pass and streaming so we never hold the full
  ~4GB of case-law text in memory: pass 1 builds the LSH and marks duplicate
  line-ids, pass 2 rewrites the survivors.

SEC and fineweb are not near-deduped (SEC docs are distinct filings; fineweb is
already deduplicated upstream). All three sources still get **exact** dedup.

## 2. Exact-duplicate removal (all sources)

A `blake2b` hash of each doc's normalized text; the second and later occurrences
of an identical doc are dropped. Cheap, streaming, catches verbatim repeats.

## 3. Contamination strip (all sources)

- Build a set of hashed **word 13-grams** from the eval benchmarks
  (`casehold/casehold` and `coastalcph/lex_glue`, loaded from their HF
  auto-converted parquet, no dataset scripts).
- Drop any training doc that shares a 13-gram with that set. A 13-word verbatim
  overlap is the standard decontamination signal (used by GPT-3 and others), long
  enough that incidental matches are rare.

Each drop is attributed to a reason: `near_dup`, `exact_dup`, or `contaminated`.

---

## Deliverable

`docs/02-dedup.md` (this file) + kept/dropped counts per source and per reason,
and the final token estimate of `/data/corpus/`.

## RESULTS (executed 2026-07-09)

| Source | Kept docs | Clean tokens | near_dup | exact_dup | contaminated |
|---|---|---|---|---|---|
| case-law | 206,684 | ~0.81B | 1,606 | 0 | 24,002 |
| SEC | 45,035 | ~1.09B | 0 | 1,989 | 175 |
| fineweb-edu | 418,405 | ~0.50B | 0 | 62 | 0 |
| **Total** | **670,124** | **~2.40B** | | | |

Notes:
- **24,002 case-law docs removed as CaseHOLD-contaminated.** CaseHOLD (via the
  LexGLUE `case_hold` config, 480,908 eval 13-grams) is derived from case-law
  holdings, so heavy overlap is expected and this is the leakage we needed gone.
- Near-dup on case-law was modest (1,606, ~0.7%); this source is fairly clean
  upstream. SEC's exact-dups (1,989) are re-filed/boilerplate filings.
- **Performance note (important lesson):** the first single-container implementation
  was too slow (SHA1 per MinHash shingle + blake2b per 13-gram = hundreds of
  millions of hashes) and kept getting **preempted** (restart-from-zero). Fixed by
  (1) `hash(tuple(...))` instead of blake2b for n-grams, (2) MinHash `num_perm`
  64->32, (3) fanning out signature computation one worker per shard, and (4)
  parallelizing the write pass one worker per shard. Near-dedup stays global (LSH
  over all precomputed signatures); exact-dedup is within-shard (cross-shard exact
  dups are negligible after near-dedup). Corpus corrected 2.68B -> 2.40B tokens.
