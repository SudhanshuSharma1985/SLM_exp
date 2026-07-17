"""Modal App for the from-scratch 125M SLM build.

Phase 0 scope: the App, a cheap CPU image, the persistent Volume mount, and a
``smoke_test`` that streams a handful of documents from each source and runs them
through the deterministic cleaner. Later phases add clean / tokenizer / tokenize /
pretrain functions to this same App.

Run:
    source .env.local && export MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    modal run modal_app.py::smoke_test
"""

from __future__ import annotations

import modal

import config

app = modal.App(config.PROJECT)

# Cheap CPU base for all pre-GPU phases. Pinned, ungated deps only. All build
# steps (pip/apt) must come BEFORE add_local_* (Modal requirement), so the base
# holds every build step and each image adds local source last.
_cpu_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")  # /usr/share/dict/words for the OCR-garble gate
    .pip_install(
        "datasets==3.6.0",
        "huggingface_hub==0.34.4",
        "langdetect==1.0.9",
        "pyarrow==17.0.0",
        "datasketch==1.6.5",
    )
)

# Ship our local source into the container so functions can import them.
cpu_image = _cpu_base.add_local_python_source("config", "cleaning", "dedup")

# The one persistent Volume, mounted at /data in every function.
volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


def _stream_source(source: "config.Source", n: int):
    """Yield up to n raw records from a streamed HF dataset (helper, no I/O)."""
    from datasets import load_dataset

    ds = load_dataset(
        source.hf_id,
        source.config_name,
        split=source.split,
        streaming=True,
    )
    for i, record in enumerate(ds):
        if i >= n:
            break
        yield record


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 15)
def smoke_test(n_per_source: int = 10) -> dict:
    """Stream n docs per source, clean each, print before/after. Stores nothing.

    Proves: network reachability, correct per-source field extraction
    (`document` vs `text`), and that the cleaner behaves before any real run.
    """
    from cleaning import clean_document

    summary: dict[str, dict] = {}

    for source in config.DATA_MIX:
        print("\n" + "=" * 78)
        print(f"SOURCE: {source.name}  ({source.hf_id}, split={source.split}, "
              f"field='{source.text_field}')")
        print("=" * 78)

        kept = 0
        reasons: dict[str, int] = {}
        for i, record in enumerate(_stream_source(source, n_per_source)):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            result = clean_document(text)
            reasons[result.reason] = reasons.get(result.reason, 0) + 1
            kept += int(result.kept)

            excerpt = (result.text[:240] if result.kept else text[:160]).replace("\n", " / ")
            print(f"\n[{source.name} #{i}] raw={result.raw_chars:>7} chars  "
                  f"clean={result.clean_chars:>7} chars  -> {result.reason.upper()}")
            print(f"    {excerpt}")

        summary[source.name] = {
            "streamed": n_per_source,
            "kept": kept,
            "reasons": reasons,
        }

    print("\n" + "#" * 78)
    print("SMOKE TEST SUMMARY")
    for name, s in summary.items():
        print(f"  {name:<12} kept {s['kept']}/{s['streamed']}  reasons={s['reasons']}")
    print("#" * 78)
    return summary


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20)
def measure_sources(n_per_source: int = 2000) -> dict:
    """Stream a real sample per source and project the true clean-token yield.

    Uses known total row counts to turn a sample average into a corpus estimate,
    so the Phase 1 token budget is grounded in fact, not a guessed parquet
    compression ratio. Stores nothing.
    """
    from cleaning import clean_document

    # Known dataset row counts (HF datasets-server, this config/split).
    TOTAL_ROWS = {"case-law": 282_390, "sec": 48_543, "fineweb-edu": 9_670_000}
    CHARS_PER_TOKEN = 4.0

    out: dict[str, dict] = {}
    for source in config.DATA_MIX:
        raw_chars = clean_chars = kept = 0
        reasons: dict[str, int] = {}
        for record in _stream_source(source, n_per_source):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text)
            raw_chars += r.raw_chars
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
            if r.kept:
                kept += 1
                clean_chars += r.clean_chars

        n = n_per_source
        avg_raw = raw_chars / n if n else 0
        avg_clean = clean_chars / n if n else 0  # averaged over ALL sampled (kept-only contribute)
        total = TOTAL_ROWS[source.name]
        est_clean_tokens = total * avg_clean / CHARS_PER_TOKEN
        out[source.name] = {
            "sampled": n,
            "kept": kept,
            "keep_rate": round(kept / n, 3) if n else 0,
            "avg_raw_chars": round(avg_raw),
            "avg_clean_chars_per_doc": round(avg_clean),
            "total_rows": total,
            "est_clean_tokens": int(est_clean_tokens),
            "reasons": reasons,
        }
        print(f"{source.name:<12} keep={out[source.name]['keep_rate']:.0%}  "
              f"avg_raw={avg_raw:>7.0f}  avg_clean={avg_clean:>7.0f} ch/doc  "
              f"rows={total:>9,}  est_clean_tokens={est_clean_tokens/1e9:>5.2f}B")
    print("\nTOTAL est clean tokens: "
          f"{sum(v['est_clean_tokens'] for v in out.values())/1e9:.2f}B")
    return out


# --------------------------------------------------------------------------- #
# Phase 1: stream + clean, one worker per parquet shard
# --------------------------------------------------------------------------- #

_SOURCE_BY_NAME = {s.name: s for s in config.DATA_MIX}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 60)
def clean_shard(source_name: str, url: str, shard_index: int, token_cap: int) -> dict:
    """Stream one parquet shard, clean each doc, append survivors to the Volume.

    Pure w.r.t. inputs; the only side effect is writing this worker's own output
    shard (no other worker touches it, so there is no shared-state race). Stops
    early once ~token_cap clean tokens (chars/proxy) have been written.
    """
    import os

    from datasets import load_dataset

    from cleaning import clean_document

    source = _SOURCE_BY_NAME[source_name]
    out_dir = f"{config.CLEAN_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/shard-{shard_index:03d}.txt"

    ds = load_dataset("parquet", data_files=url, split="train", streaming=True)

    streamed = kept = clean_chars = 0
    reasons: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in ds:
            streamed += 1
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text, strict_ocr=source.strict_ocr)
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
            if r.kept:
                fh.write(r.text.replace("\n", " ").strip() + "\n")
                kept += 1
                clean_chars += r.clean_chars
                if clean_chars / config.CHARS_PER_TOKEN >= token_cap:
                    break

    volume.commit()
    est_tokens = int(clean_chars / config.CHARS_PER_TOKEN)
    print(f"[{source_name} shard {shard_index:03d}] streamed={streamed} kept={kept} "
          f"est_tokens={est_tokens/1e6:.1f}M reasons={reasons}")
    return {
        "source": source_name,
        "shard": shard_index,
        "streamed": streamed,
        "kept": kept,
        "clean_chars": clean_chars,
        "est_tokens": est_tokens,
        "reasons": reasons,
    }


def _parquet_urls(hf_id: str, config_name: str, split: str) -> list[str]:
    """List parquet file URLs for one dataset config/split (helper, local)."""
    import json
    import urllib.request

    api = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "slm-125m"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    files = [
        f["url"]
        for f in data.get("parquet_files", [])
        if f.get("config") == config_name and f.get("split") == split
    ]
    return files


@app.local_entrypoint()
def clean(fineweb_shards: int = 1, only: str = ""):
    """`modal run modal_app.py::clean` -> Phase 1 stream + clean fan-out.

    Builds the work list locally (list parquet shards per source), fans out one
    worker per shard, then prints the per-source drop report. Pass
    `--only <name>` to (re)run a single source without touching the others.
    """
    import json

    # HF auto-convert uses config "default" unless the source sets one.
    def cfg(s: "config.Source") -> str:
        return s.config_name or "default"

    sources = [s for s in config.DATA_MIX if not only or s.name == only]
    work: list[tuple[str, str, int, int]] = []
    for s in sources:
        urls = _parquet_urls(s.hf_id, cfg(s), s.split)
        if s.name == "fineweb-edu":
            urls = urls[:fineweb_shards]  # only need a 0.5B-token slice
        per_shard_cap = s.token_budget // max(1, len(urls))
        for i, url in enumerate(urls):
            work.append((s.name, url, i, per_shard_cap))
        print(f"{s.name:<12} {len(urls)} shard(s), per-shard cap "
              f"~{per_shard_cap/1e6:.0f}M tokens")

    print(f"\nLaunching {len(work)} clean workers...\n")
    results = list(clean_shard.starmap(work))

    # Aggregate per source.
    report: dict[str, dict] = {}
    for r in results:
        agg = report.setdefault(r["source"], {
            "streamed": 0, "kept": 0, "est_tokens": 0, "reasons": {}})
        agg["streamed"] += r["streamed"]
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v

    print("\n" + "#" * 78)
    print("PHASE 1 DROP REPORT")
    total = 0
    for name, a in report.items():
        keep_rate = a["kept"] / a["streamed"] if a["streamed"] else 0
        total += a["est_tokens"]
        print(f"  {name:<12} streamed={a['streamed']:>8}  kept={a['kept']:>8} "
              f"({keep_rate:.0%})  est_tokens={a['est_tokens']/1e9:.2f}B")
        print(f"               drops={a['reasons']}")
    print(f"  {'TOTAL':<12} est_clean_tokens={total/1e9:.2f}B")
    print("#" * 78)

    # Persist the report to the Volume for the record.
    save_report.remote(report)


# The base already carries the wordlist, so the OCR analysis uses the CPU image.
ocr_image = cpu_image


# --------------------------------------------------------------------------- #
# Phase 2: dedup + contamination strip
# --------------------------------------------------------------------------- #

SHINGLE_K = 5
MINHASH_PERM = 32       # 32 is plenty for a 0.8 threshold; halves MinHash cost
MINHASH_THRESHOLD = 0.8
DECONTAM_NGRAM = 13


def _iter_source_docs(source_name: str):
    """Yield (shard_name, line_index, text) for every clean doc of a source."""
    import glob
    import os

    for path in sorted(glob.glob(f"{config.CLEAN_DIR}/{source_name}/*.txt")):
        shard = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.rstrip("\n")
                if line:
                    yield shard, i, line


def _build_contamination_ngrams() -> set:
    """Hashed word-13-grams from the eval benchmarks (parquet, no scripts)."""
    from datasets import load_dataset

    from dedup import word_ngrams, words

    grams: set = set()
    eval_specs = [
        ("casehold/casehold", None),
        ("coastalcph/lex_glue", "case_hold"),
    ]
    for hf_id, cfg_name in eval_specs:
        try:
            urls = _parquet_urls(hf_id, cfg_name or "default", "test")
            if not urls:
                urls = _parquet_urls(hf_id, cfg_name or "default", "train")
            ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                grams |= word_ngrams(words(text), DECONTAM_NGRAM)
        except Exception as e:  # keep going with whatever loaded
            print(f"  [decontam] could not load {hf_id}: {e}")
    print(f"  [decontam] {len(grams):,} eval 13-grams loaded")
    return grams


SIG_DIR = f"{config.DATA_ROOT}/tmp/minhash_sigs"


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=4_096)
def minhash_shard(shard_basename: str) -> dict:
    """Compute MinHash signatures for one case-law clean shard, save to the Volume.

    Signature computation (SHA1 per shingle) is the expensive part, so it is
    fanned out one worker per shard. Short-lived workers also sidestep the
    preemption that kills a long single container. Saves an .npz of the signature
    matrix + line indices, keyed later by (shard, line_index).
    """
    import os

    import numpy as np
    from datasketch import MinHash

    from dedup import shingles, words

    path = f"{config.CLEAN_DIR}/case-law/{shard_basename}"
    sigs: list = []
    idxs: list[int] = []
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            m = MinHash(num_perm=MINHASH_PERM)
            sh = list(shingles(words(line), SHINGLE_K))
            if sh:
                m.update_batch(sh)
            sigs.append(m.hashvalues.astype(np.uint64))
            idxs.append(idx)

    os.makedirs(SIG_DIR, exist_ok=True)
    out = f"{SIG_DIR}/{shard_basename}.npz"
    np.savez(out, sigs=np.vstack(sigs), idxs=np.asarray(idxs, dtype=np.int64))
    volume.commit()
    print(f"[minhash {shard_basename}] {len(idxs):,} docs -> {out}")
    return {"shard": shard_basename, "n": len(idxs)}


NEAR_DUPS_PATH = f"{config.DATA_ROOT}/tmp/near_dups.json"
DECONTAM_SOURCES = {"case-law", "sec"}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, memory=8_192)
def build_near_dups() -> int:
    """LSH over the precomputed case-law signatures; save the near-dup key set.

    Fast (no re-hashing, just LSH insert/query on stored hashvalues), so this
    single-container step is short. Writes {shard: [line_idx, ...]} to the Volume.
    """
    import glob
    import json
    import os

    import numpy as np
    from datasketch import MinHash, MinHashLSH

    near: dict[str, list[int]] = {}
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERM)
    for npz_path in sorted(glob.glob(f"{SIG_DIR}/*.npz")):
        shard = os.path.basename(npz_path)[: -len(".npz")]
        data = np.load(npz_path)
        for row, idx in zip(data["sigs"], data["idxs"]):
            m = MinHash(num_perm=MINHASH_PERM, hashvalues=row)
            if lsh.query(m):
                near.setdefault(shard, []).append(int(idx))
            else:
                lsh.insert(f"{shard}:{int(idx)}", m)

    os.makedirs(os.path.dirname(NEAR_DUPS_PATH), exist_ok=True)
    with open(NEAR_DUPS_PATH, "w", encoding="utf-8") as fh:
        json.dump(near, fh)
    volume.commit()
    total = sum(len(v) for v in near.values())
    print(f"[near-dups] {total:,} case-law near-duplicates")
    return total


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0, memory=8_192)
def write_corpus_shard(source_name: str, shard_basename: str) -> dict:
    """Write one final-corpus shard: drop near-dups (case-law), exact-dups, and
    eval-contaminated docs. Parallelized one worker per clean shard."""
    import json
    import os

    from dedup import exact_hash, word_ngrams, words

    near: set[int] = set()
    if source_name == "case-law":
        with open(NEAR_DUPS_PATH, encoding="utf-8") as fh:
            near = set(json.load(fh).get(shard_basename, []))

    contam = _build_contamination_ngrams() if source_name in DECONTAM_SOURCES else None

    in_path = f"{config.CLEAN_DIR}/{source_name}/{shard_basename}"
    out_dir = f"{config.CORPUS_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{shard_basename}"

    seen: set[str] = set()
    kept = clean_chars = 0
    reasons = {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            text = line.rstrip("\n")
            if not text:
                continue
            if idx in near:
                reasons["near_dup"] += 1
                continue
            h = exact_hash(text)
            if h in seen:
                reasons["exact_dup"] += 1
                continue
            if contam and (word_ngrams(words(text), DECONTAM_NGRAM) & contam):
                reasons["contaminated"] += 1
                continue
            seen.add(h)
            fout.write(text + "\n")
            kept += 1
            clean_chars += len(text)
            reasons["kept"] += 1

    volume.commit()
    print(f"[corpus {source_name}/{shard_basename}] kept={kept} drops={reasons}")
    return {
        "source": source_name,
        "shard": shard_basename,
        "kept": kept,
        "est_tokens": int(clean_chars / config.CHARS_PER_TOKEN),
        "reasons": reasons,
    }


@app.function(image=cpu_image, volumes=VOLUMES)
def write_phase2_report(results: list) -> dict:
    """Aggregate per-shard corpus results into /data/corpus/phase2_report.json."""
    import json

    report: dict[str, dict] = {}
    for r in results:
        if not r:
            continue
        agg = report.setdefault(r["source"], {
            "kept": 0, "est_tokens": 0,
            "reasons": {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}})
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v

    total = sum(v["est_tokens"] for v in report.values())
    print("#" * 70 + "\nPHASE 2 REPORT")
    for name, a in report.items():
        print(f"  {name:<12} kept={a['kept']:>8} est_tokens={a['est_tokens']/1e9:.2f}B "
              f"drops={a['reasons']}")
    print(f"  TOTAL corpus est tokens: {total/1e9:.2f}B\n" + "#" * 70)

    path = f"{config.CORPUS_DIR}/phase2_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    return report


# Clean-shard layout produced by Phase 1 (one output shard per input parquet file).
CLEAN_SHARDS = {"case-law": 10, "sec": 5, "fineweb-edu": 5}


@app.local_entrypoint()
def dedup(compute_sigs: bool = True):
    """`modal run modal_app.py::dedup` -> Phase 2, fully parallel.

    1. MinHash signatures per case-law shard (parallel).  2. LSH near-dup set.
    3. Write final corpus per shard (parallel: near-dup + exact-dup + decontam).
    Pass `--no-compute-sigs` to reuse signatures already on the Volume.
    """
    if compute_sigs:
        names = [f"shard-{i:03d}.txt" for i in range(CLEAN_SHARDS["case-law"])]
        print(f"1/3 MinHash signatures for {len(names)} case-law shards...")
        list(minhash_shard.map(names))

    print("2/3 building near-dup set (LSH)...")
    build_near_dups.remote()

    work = [
        (src, f"shard-{i:03d}.txt")
        for src, n in CLEAN_SHARDS.items()
        for i in range(n)
    ]
    print(f"3/3 writing final corpus ({len(work)} shards, parallel)...")
    results = list(write_corpus_shard.starmap(work))
    write_phase2_report.remote(results)


# --------------------------------------------------------------------------- #
# Phase 3: train the 16K byte-level BPE tokenizer
# --------------------------------------------------------------------------- #

# transformers brings a compatible `tokenizers`; no torch needed to train.
ml_image = _cpu_base.pip_install("transformers==4.46.3").add_local_python_source(
    "config", "cleaning", "dedup"
)


def _corpus_line_iter():
    """Yield every line of the Phase 2 corpus (all sources)."""
    import glob

    for path in sorted(glob.glob(f"{config.CORPUS_DIR}/*/*.txt")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield line


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def train_tokenizer() -> dict:
    """Train a fresh 16,384 byte-level BPE and save it as a PreTrainedTokenizerFast."""
    import os

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = list(config.SPECIAL_TOKENS.values()) + list(config.EXTRA_CHAT_TOKENS)

    tok = Tokenizer(models.BPE(unk_token=config.SPECIAL_TOKENS["unk_token"]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.MODEL.vocab_size,
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print("training BPE...")
    tok.train_from_iterator(_corpus_line_iter(), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=config.SPECIAL_TOKENS["bos_token"],
        eos_token=config.SPECIAL_TOKENS["eos_token"],
        pad_token=config.SPECIAL_TOKENS["pad_token"],
        unk_token=config.SPECIAL_TOKENS["unk_token"],
        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS),
    )
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    fast.save_pretrained(config.TOKENIZER_DIR)
    volume.commit()

    # Round-trip sanity check.
    samples = [
        "The plaintiff shall bear the burden of proof by a preponderance of the evidence.",
        "The Company's net revenues increased 12% year over year pursuant to the agreement.",
    ]
    checks = []
    for s in samples:
        ids = fast.encode(s)
        back = fast.decode(ids)
        checks.append({"text": s, "n_tokens": len(ids), "roundtrip_ok": back.strip() == s})
        print(f"  '{s[:40]}...' -> {len(ids)} tokens | roundtrip={back.strip() == s}")

    out = {"vocab_size": fast.vocab_size, "specials": specials, "checks": checks}
    print(f"vocab_size={fast.vocab_size}")
    return out


@app.local_entrypoint()
def tokenizer():
    """`modal run modal_app.py::tokenizer` -> Phase 3 train the tokenizer."""
    train_tokenizer.remote()


# --------------------------------------------------------------------------- #
# Phase 4: tokenize + pack into uint16 1024-token windows, split 99/1
# --------------------------------------------------------------------------- #

TOKENIZE_SHARDS = {"case-law": 4, "sec": 6, "fineweb-edu": 4}
ENCODE_BATCH = 1_000


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def tokenize_shard(source_name: str, shard_index: int, num_shards: int) -> dict:
    """Encode this shard's docs, pack into 1024-windows, split 99/1, write uint16."""
    import glob
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    seq_len = config.SEQ_LEN

    os.makedirs(config.TRAIN_TOKENS_DIR, exist_ok=True)
    os.makedirs(config.VAL_TOKENS_DIR, exist_ok=True)
    train_path = f"{config.TRAIN_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    val_path = f"{config.VAL_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"

    buf: list[int] = []
    win_count = 0
    n_train = n_val = 0
    corpus_files = sorted(glob.glob(f"{config.CORPUS_DIR}/{source_name}/*.txt"))

    def _doc_iter():
        for path in corpus_files:
            with open(path, encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if idx % num_shards == shard_index:
                        line = line.rstrip("\n")
                        if line:
                            yield line

    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        batch: list[str] = []

        def _flush_batch():
            nonlocal win_count, n_train, n_val
            if not batch:
                return
            encs = tok(batch, add_special_tokens=False)["input_ids"]
            for ids in encs:
                buf.extend(ids)
                buf.append(eos_id)
            # Emit all full windows currently in buf.
            while len(buf) >= seq_len:
                window = np.asarray(buf[:seq_len], dtype=np.uint16)
                del buf[:seq_len]
                if win_count % config.VAL_EVERY_N_WINDOWS == 0:
                    window.tofile(fva)
                    n_val += 1
                else:
                    window.tofile(ftr)
                    n_train += 1
                win_count += 1

        for doc in _doc_iter():
            batch.append(doc)
            if len(batch) >= ENCODE_BATCH:
                _flush_batch()
                batch = []
        _flush_batch()

    volume.commit()
    res = {
        "source": source_name,
        "shard": shard_index,
        "train_windows": n_train,
        "val_windows": n_val,
        "train_tokens": n_train * seq_len,
        "val_tokens": n_val * seq_len,
    }
    print(f"[{source_name} {shard_index:03d}] train_win={n_train} val_win={n_val} "
          f"train_tok={n_train*seq_len/1e6:.1f}M")
    return res


@app.function(image=ml_image, volumes=VOLUMES)
def write_token_index(results: list) -> dict:
    """Merge per-shard results into /data/tokens/index.json."""
    import json

    shards = [r for r in results if r]
    total = {
        "seq_len": config.SEQ_LEN,
        "dtype": config.TOKENS_DTYPE,
        "train_windows": sum(r["train_windows"] for r in shards),
        "val_windows": sum(r["val_windows"] for r in shards),
        "train_tokens": sum(r["train_tokens"] for r in shards),
        "val_tokens": sum(r["val_tokens"] for r in shards),
        "shards": shards,
    }
    path = f"{config.TOKENS_DIR}/index.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(total, fh, indent=2)
    volume.commit()
    print(f"index: train={total['train_tokens']/1e9:.2f}B tok "
          f"({total['train_windows']} win), val={total['val_tokens']/1e6:.1f}M tok "
          f"({total['val_windows']} win)")
    return total


@app.local_entrypoint()
def tokenize():
    """`modal run modal_app.py::tokenize` -> Phase 4 tokenize + pack + split."""
    work = [
        (name, i, n)
        for name, n in TOKENIZE_SHARDS.items()
        for i in range(n)
    ]
    print(f"Launching {len(work)} tokenize workers...")
    results = list(tokenize_shard.starmap(work))
    write_token_index.remote(results)


@app.function(image=ocr_image, timeout=60 * 15)
def ocr_sample(n_docs: int = 3000) -> dict:
    """Measure OCR-garble in case-law via a real English-dictionary non-word ratio.

    For each sampled doc, compute the fraction of alphabetic word-tokens (len>=3)
    that are NOT in the system English wordlist. Report how many docs would be
    dropped at several thresholds so the OCR gate can be chosen with real numbers.
    Reads /usr/share/dict/words; streams live, stores nothing.
    """
    import re

    from cleaning import clean_document

    with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as fh:
        words = {w.strip().lower() for w in fh if w.strip().isalpha()}
    tok = re.compile(r"[A-Za-z]{3,}")

    source = _SOURCE_BY_NAME["case-law"]
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    ratios: list[float] = []
    for record in _stream_source(source, n_docs):
        text = record.get(source.text_field) or ""
        if not isinstance(text, str):
            text = str(text)
        r = clean_document(text)  # only score docs that already pass the base chain
        if not r.kept:
            continue
        toks = [t.lower() for t in tok.findall(r.text)]
        if len(toks) < 50:
            continue
        nonword = sum(1 for t in toks if t not in words)
        ratios.append(nonword / len(toks))

    ratios.sort()
    n = len(ratios)
    drops = {f">{int(t*100)}%": sum(1 for x in ratios if x > t) for t in thresholds}
    pct = {f"p{p}": round(ratios[int(p / 100 * (n - 1))], 3) for p in (50, 75, 90, 95, 99)} if n else {}
    print(f"scored {n} kept case-law docs")
    print(f"non-word-ratio percentiles: {pct}")
    for k, v in drops.items():
        print(f"  drop if non-word ratio {k:<5}: {v:>5} docs ({v/n:.1%})" if n else k)
    return {"scored": n, "percentiles": pct, "drops_at_threshold": drops}


@app.local_entrypoint()
def ocr(n_docs: int = 3000):
    """`modal run modal_app.py::ocr` -> OCR-garble drop-rate analysis."""
    ocr_sample.remote(n_docs)


@app.function(image=cpu_image, volumes=VOLUMES)
def save_report(report: dict) -> None:
    """Write the Phase 1 drop report to the Volume."""
    import json

    path = f"{config.CLEAN_DIR}/phase1_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Phase 5: pretrain the 125M model (GPU, single-node DDP)
# --------------------------------------------------------------------------- #

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==2.1.3",
        "safetensors==0.4.5",
    )
    .add_local_python_source("config", "cleaning", "dedup", "train")
)


def _pretrain_fn(smoke: bool, epochs: int, max_usd: float, gpus: int, resume: bool):
    import torch.multiprocessing as mp

    import train

    args = {
        "smoke": smoke,
        "epochs": epochs,
        "max_usd": max_usd,
        "resume": resume,
        "max_steps": 20 if smoke else None,
    }
    world = gpus
    print(f"[pretrain] spawning {world} rank(s), smoke={smoke}, epochs={epochs}, "
          f"max_usd={max_usd}", flush=True)
    if world == 1:
        train.worker(0, 1, args)
    else:
        mp.spawn(train.worker, args=(world, args), nprocs=world, join=True)
    volume.commit()
    print("[pretrain] committed volume", flush=True)


@app.function(image=gpu_image, volumes=VOLUMES,
              gpu=f"{config.PRETRAIN_GPU}:{config.PRETRAIN_GPU_COUNT}",
              timeout=60 * 60 * 4)
def pretrain_full(epochs: int, max_usd: float, resume: bool = False):
    """Full 8xH100 DDP pretraining run."""
    _pretrain_fn(False, epochs, max_usd, config.PRETRAIN_GPU_COUNT, resume)


@app.function(image=gpu_image, volumes=VOLUMES, gpu=f"{config.PRETRAIN_GPU}:1",
              timeout=60 * 30)
def pretrain_smoke():
    """Single-H100 smoke: ~20 steps + eval + checkpoint write. Near $0."""
    _pretrain_fn(True, 1, config.BUDGET_CAP_USD, 1, False)


@app.local_entrypoint()
def smoke_pretrain():
    """`modal run modal_app.py::smoke_pretrain` -> Phase 5 smoke test."""
    pretrain_smoke.remote()


@app.local_entrypoint()
def pretrain(epochs: int = 0, max_usd: float = 0.0, resume: bool = False):
    """`modal run modal_app.py::pretrain` -> full Phase 5 run (8xH100)."""
    e = epochs or config.PRETRAIN_EPOCHS
    cap = max_usd or config.BUDGET_CAP_USD
    print(f"launching full pretrain: {e} epochs, cap ${cap}, "
          f"{config.PRETRAIN_GPU_COUNT}x{config.PRETRAIN_GPU}")
    pretrain_full.remote(e, cap, resume)


@app.local_entrypoint()
def pretrain_bg(epochs: int = 0, max_usd: float = 0.0, resume: bool = False):
    """Detached launch via .spawn() so a local client/network drop can't cancel
    the GPU job. Run with `modal run --detach modal_app.py::pretrain_bg`."""
    e = epochs or config.PRETRAIN_EPOCHS
    cap = max_usd or config.BUDGET_CAP_USD
    handle = pretrain_full.spawn(e, cap, resume)
    print(f"SPAWNED full pretrain: {e} epochs, cap ${cap}, "
          f"{config.PRETRAIN_GPU_COUNT}x{config.PRETRAIN_GPU}")
    print(f"SPAWN_CALL_ID={handle.object_id}")


@app.function(image=gpu_image, volumes=VOLUMES, gpu="H100:1", timeout=60 * 15)
def generate_samples() -> list:
    """Complete a few legal/financial prefixes with the trained base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.BASE_CKPT_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_CKPT_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()
    eos = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    bos = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["bos_token"])

    prompts = [
        "The plaintiff alleges that the defendant",
        "Pursuant to the terms of this Agreement,",
        "The Company's net revenues for the fiscal year",
        "In determining whether the search was reasonable, the court",
    ]
    outs = []
    for p in prompts:
        ids = torch.tensor([[bos] + tok.encode(p, add_special_tokens=False)]).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                ids, max_new_tokens=90, min_new_tokens=40, do_sample=True,
                temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.3,
                eos_token_id=eos, pad_token_id=eos)
        text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        outs.append({"prompt": p, "completion": text})
        print(f"\n>>> {p}\n{text}", flush=True)
    return outs


@app.local_entrypoint()
def samples():
    """`modal run modal_app.py::samples` -> sample completions from the base model."""
    generate_samples.remote()


# --------------------------------------------------------------------------- #
# Phase 6: inference endpoint (CPU, scale-to-zero) + HF push
# --------------------------------------------------------------------------- #

infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==2.1.3",
        "safetensors==0.4.5",
        "fastapi[standard]==0.115.5",
    )
    .add_local_python_source("config")
)


@app.cls(image=infer_image, volumes=VOLUMES, cpu=2.0, memory=4_096,
         min_containers=0, scaledown_window=300)
class Inference:
    """Loads the base model once per container; serves /generate over HTTP."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.BASE_CKPT_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.BASE_CKPT_DIR, torch_dtype=torch.float32).eval()
        self.eos = self.tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
        self.bos = self.tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["bos_token"])

    def _complete(self, body: dict) -> str:
        torch = self.torch
        prompt = (body.get("prompt") or "").strip()
        ids = [self.bos] + self.tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids])
        with torch.no_grad():
            gen = self.model.generate(
                inp,
                max_new_tokens=int(body.get("max_new_tokens", 90)),
                min_new_tokens=int(body.get("min_new_tokens", 40)),
                do_sample=True,
                temperature=float(body.get("temperature", 0.8)),
                top_k=int(body.get("top_k", 50)),
                top_p=float(body.get("top_p", 0.95)),
                repetition_penalty=float(body.get("repetition_penalty", 1.3)),
                eos_token_id=self.eos, pad_token_id=self.eos)
        return self.tok.decode(gen[0][inp.shape[1]:], skip_special_tokens=True)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        api = FastAPI(title="slm-125m")
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "model": "slm-125m-base", "val_ppl": 8.50}

        @api.post("/generate")
        def generate(body: dict):
            try:
                return {"generated": self._complete(body)}
            except Exception as e:  # never 500 the frontend
                return {"generated": "", "error": str(e)}

        return api


@app.function(image=gpu_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)], timeout=60 * 20)
def push_to_hf(repo: str = ""):
    """Push /data/checkpoints/base to a HuggingFace model repo (canonical home)."""
    import os

    from huggingface_hub import HfApi

    repo = repo or config.HF_REPO
    token = os.environ["HUGGINGFACE_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=config.BASE_CKPT_DIR, repo_id=repo, repo_type="model")
    print(f"pushed {config.BASE_CKPT_DIR} -> https://huggingface.co/{repo}")
    return f"https://huggingface.co/{repo}"


@app.local_entrypoint()
def hf_push(repo: str = ""):
    """`modal run modal_app.py::hf_push` -> push the model to HuggingFace."""
    push_to_hf.remote(repo)


@app.local_entrypoint()
def main(n_per_source: int = 10):
    """Local entrypoint so `modal run modal_app.py` triggers the smoke test."""
    smoke_test.remote(n_per_source)


@app.local_entrypoint()
def measure(n_per_source: int = 2000):
    """`modal run modal_app.py::measure` -> project true clean-token yield."""
    measure_sources.remote(n_per_source)
