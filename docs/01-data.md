# Phase 1, Download + clean

**Goal:** stream three HuggingFace datasets, clean each document on the fly with a
fixed rule chain, and write the survivors to the Volume as the durable, reusable
corpus. CPU only, low cost. No GPU.

---

## The corpus we are building (decided: Choice A, legal-first)

We measured the real yield of each source (2,000 docs sampled, run through our
actual cleaner) before committing:

| Source | Keep rate | Avg clean doc | Total docs | Unique clean tokens available |
|---|---|---|---|---|
| case-law | 76% | 11,455 chars | 282,390 | **~0.81B** |
| SEC | 98% | 95,371 chars | 48,543 | **~1.16B** |
| fineweb-edu | 96% | 4,827 chars | 9,670,000 | ~11.7B (we take a slice) |

The two legal sources max out at **~2B unique tokens combined**, so we take *all*
of them and add a small web slice for fluency. Final target:

| Source | Token budget | Notes |
|---|---|---|
| case-law | up to 1.0B (takes all ~0.81B) | US judicial opinions; **strict OCR gate ON** (scanned) |
| SEC | up to 1.3B (takes all ~1.16B) | 10-K etc.; born-digital, OCR gate off |
| fineweb-edu | 0.5B (hard cap) | general-English fluency filler |
| **Total** | **~2.5B unique clean tokens (~78% legal)** | |

We reach ~10 to 12B **tokens-seen** by training ~4 to 5 epochs in Phase 5 (the
proven recipe: a compact high-quality corpus seen multiple times).

---

## Download strategy: stream, never hoard

We use HuggingFace `datasets` with `streaming=True`. Documents arrive one at a
time, are cleaned immediately, and kept survivors are appended to disk. We **stop
each source at its token budget**. We never materialize the multi-GB raw parquet.

**Token counting during streaming:** the real 16K tokenizer does not exist yet
(that is Phase 3), so we budget with a **chars/token ≈ 4.0 proxy** (`clean_chars /
4`). It is an estimate for *stopping*, not the final count. Phase 4 reports exact
token counts from the trained tokenizer.

---

## The cleaning chain (fixed, deterministic, cheapest-check-first)

Exactly the chain you saw pass in the Phase 0 smoke test (`cleaning.py`). Each
document runs through it; the first failed check drops the doc and the reason is
tallied:

1. **filter_lines** - drop lines < 40 chars or > 30% non-alphanumeric; collapse whitespace.
2. **strip_boilerplate** - delete lines matching known regexes (FORM 10-K, `Page N of M`,
   SEC headers, `/s/` signatures, Table of Contents, "All rights reserved").
3. **length gate** - drop the doc if < 600 chars survive (`too_short`).
4. **is_repetitive** - drop if the top-10 4-grams cover > 50% of all 4-grams (`repetitive`).
5. **is_english** - `langdetect` on the first 5k chars; ASCII-ratio fallback (`non_english`).
6. **strict OCR gate** - *case-law only*: drop docs where > 3% of words look like OCR
   errors (`ocr`). Off for SEC and web, which are born-digital.

No model-based scoring, no rewriting, no dedup here (dedup is Phase 2).

---

## Parallelization + output format

- Each source's parquet files are split across **CPU workers, one worker per
  parquet shard**, fanned out with Modal `.map`. Workers run concurrently and each
  writes its own output shard, so there is no merge step and no shared-state races.
- Kept documents are written **one per line** (newlines inside a doc are already
  collapsed by the cleaner) to:

```
/data/clean/case-law/shard-XX.txt
/data/clean/sec/shard-XX.txt
/data/clean/fineweb-edu/shard-XX.txt
```

- This cleaned corpus is the **durable, reusable asset**. Future/bigger models
  reuse it; we only ever stream *additional* data, never re-clean what is here.

---

## Cost + time estimate

- CPU containers only. ~5 GB of legal parquet + a 0.5B-token web slice to stream
  and clean. Rough wall-clock **~15 to 40 min** with per-shard fan-out; cost a few
  cents to ~$1 of CPU time. No GPU, no meaningful spend.

---

## Deliverable

`docs/01-data.md` (this file) + a **per-source drop-count report**: for each
source, how many docs were streamed, kept, and dropped by each reason
(`too_short` / `repetitive` / `non_english` / `ocr`), plus the final clean-token
estimate per source and the total.

## RESULTS (executed 2026-07-08/09)

Final consolidated drop report (OCR gate at >20% non-dictionary words, on case-law):

| Source | Streamed | Kept | Keep rate | Drops | Clean tokens |
|---|---|---|---|---|---|
| case-law | 238,207 | 232,292 | 97.5% | too_short 10,460; ocr 1,370 | ~1.00B |
| SEC | 47,752 | 47,199 | 98.8% | too_short 553 | ~1.18B |
| fineweb-edu | 432,821 | 418,467 | 96.7% | too_short 14,348; non_english 6 | ~0.50B |
| **Total** | **718,780** | **697,958** | **97.1%** | | **~2.68B** |

Notes:
- case-law came in larger than the pre-sample projected (~1.0B, all 10 shards hit
  the 100M/shard cap); SEC took essentially everything (~1.18B); fineweb capped at 0.5B.
- Two cleaner improvements were made mid-phase: (1) `is_english` reordered to be
  ASCII-first so `langdetect` only runs on ambiguous docs (killed the fineweb
  throughput bottleneck); (2) the OCR gate replaced with a real English-dictionary
  non-word-ratio check (the original heuristic was a no-op on letter-substitution garble).
- fineweb re-run 5-wide (5 parquet files in parallel) finished in ~2 min vs 10+ single-worker.

## After Phase 1

Phase 2 (dedup + contamination strip) runs on this cleaned corpus.
