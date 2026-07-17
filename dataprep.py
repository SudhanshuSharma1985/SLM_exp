"""Standalone data preparation (Phases 1-4), no Modal.

Turns the three HuggingFace sources in `config.DATA_MIX` into packed uint16
1024-token windows under `config.TOKENS_DIR`, ready for `train_azure.py`. This is
the single-machine port of the Phase 1-4 functions in `modal_app.py`; the actual
cleaning/dedup logic is imported unchanged from `cleaning.py` / `dedup.py`.

  Phase 1  stream + clean each source            -> /data/clean/<source>/shard-*.txt
  Phase 2  MinHash near-dup + exact-dup + decontam -> /data/corpus/<source>/shard-*.txt
  Phase 3  train the 16K byte-level BPE tokenizer  -> /data/tokenizer/
  Phase 4  tokenize + pack + 99/1 split            -> /data/tokens/{train,val}/*.bin

Run it on a CHEAP CPU box (it never touches the GPU). All output lands under
`config.DATA_ROOT` (/data), so point that at a writable disk.

Prereqs:
  pip install datasets==3.6.0 huggingface_hub==0.34.4 langdetect==1.0.9 \
              pyarrow==17.0.0 datasketch==1.6.5 transformers==4.46.3 numpy==2.1.3
  sudo apt-get install -y wamerican   # /usr/share/dict/words, for the OCR gate

Usage:
  python dataprep.py --phase all --fineweb-shards 5 --workers 8
  python dataprep.py --phase 1                 # just clean
  python dataprep.py --phase 4                 # just re-tokenize
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import config

# --------------------------------------------------------------------------- #
# Phase 2/4 knobs (identical to modal_app.py)
# --------------------------------------------------------------------------- #
SHINGLE_K = 5
MINHASH_PERM = 32
MINHASH_THRESHOLD = 0.8
DECONTAM_NGRAM = 13
DECONTAM_SOURCES = {"case-law", "sec"}
ENCODE_BATCH = 1_000
TOKENIZE_SHARDS = {"case-law": 4, "sec": 6, "fineweb-edu": 4}

_SOURCE_BY_NAME = {s.name: s for s in config.DATA_MIX}
SIG_DIR = f"{config.DATA_ROOT}/tmp/minhash_sigs"
NEAR_DUPS_PATH = f"{config.DATA_ROOT}/tmp/near_dups.json"
CONTAM_PATH = f"{config.DATA_ROOT}/tmp/contam_ngrams.pkl"


def _parquet_urls(hf_id: str, config_name: str, split: str) -> list[str]:
    """List parquet file URLs for one dataset config/split via the HF datasets-server."""
    api = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "slm-125m"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return [f["url"] for f in data.get("parquet_files", [])
            if f.get("config") == config_name and f.get("split") == split]


def _run_pool(fn, work: list, workers: int):
    """Map fn over work with a process pool (or inline if workers<=1)."""
    if workers <= 1:
        return [fn(w) for w in work]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, work))


# --------------------------------------------------------------------------- #
# Phase 1: stream + clean, one task per parquet shard
# --------------------------------------------------------------------------- #


def _clean_one(task: tuple) -> dict:
    source_name, url, shard_index, token_cap = task
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

    est = int(clean_chars / config.CHARS_PER_TOKEN)
    print(f"[clean {source_name} {shard_index:03d}] streamed={streamed} kept={kept} "
          f"est_tokens={est/1e6:.1f}M", flush=True)
    return {"source": source_name, "kept": kept, "est_tokens": est, "reasons": reasons}


def phase1_clean(fineweb_shards: int, workers: int) -> None:
    work: list[tuple] = []
    for s in config.DATA_MIX:
        urls = _parquet_urls(s.hf_id, s.config_name or "default", s.split)
        if s.name == "fineweb-edu":
            urls = urls[:fineweb_shards]
        cap = s.token_budget // max(1, len(urls))
        for i, url in enumerate(urls):
            work.append((s.name, url, i, cap))
        print(f"{s.name:<12} {len(urls)} shard(s), ~{cap/1e6:.0f}M tokens/shard")

    print(f"\n[phase1] {len(work)} clean tasks on {workers} workers...\n")
    results = _run_pool(_clean_one, work, workers)

    report: dict[str, dict] = {}
    for r in results:
        agg = report.setdefault(r["source"], {"kept": 0, "est_tokens": 0, "reasons": {}})
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    os.makedirs(config.CLEAN_DIR, exist_ok=True)
    with open(f"{config.CLEAN_DIR}/phase1_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    total = sum(a["est_tokens"] for a in report.values())
    print(f"\n[phase1] est clean tokens: {total/1e9:.2f}B  ({report})")


# --------------------------------------------------------------------------- #
# Phase 2: MinHash near-dup + exact-dup + decontamination
# --------------------------------------------------------------------------- #


def _minhash_one(shard_basename: str) -> dict:
    import numpy as np
    from datasketch import MinHash

    from dedup import shingles, words

    path = f"{config.CLEAN_DIR}/case-law/{shard_basename}"
    sigs, idxs = [], []
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
    print(f"[minhash {shard_basename}] {len(idxs):,} docs", flush=True)
    return {"shard": shard_basename, "n": len(idxs)}


def _build_near_dups() -> int:
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
    total = sum(len(v) for v in near.values())
    print(f"[near-dups] {total:,} case-law near-duplicates")
    return total


def _build_contamination_ngrams() -> set:
    """Hashed word-13-grams from the eval benchmarks (held OUT of training)."""
    from datasets import load_dataset

    from dedup import word_ngrams, words

    grams: set = set()
    for hf_id, cfg_name in [("casehold/casehold", None), ("coastalcph/lex_glue", "case_hold")]:
        try:
            urls = (_parquet_urls(hf_id, cfg_name or "default", "test")
                    or _parquet_urls(hf_id, cfg_name or "default", "train"))
            ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                grams |= word_ngrams(words(text), DECONTAM_NGRAM)
        except Exception as e:
            print(f"  [decontam] could not load {hf_id}: {e}")
    print(f"  [decontam] {len(grams):,} eval 13-grams")
    return grams


def _write_corpus_shard(source_name: str, shard_basename: str, near: set, contam: set | None) -> dict:
    from dedup import exact_hash, word_ngrams, words

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
    print(f"[corpus {source_name}/{shard_basename}] kept={kept} drops={reasons}", flush=True)
    return {"source": source_name, "kept": kept,
            "est_tokens": int(clean_chars / config.CHARS_PER_TOKEN), "reasons": reasons}


def phase2_dedup(workers: int) -> None:
    # 1. MinHash signatures for case-law shards (the expensive part) -> parallel.
    cl_shards = [os.path.basename(p) for p in
                 sorted(glob.glob(f"{config.CLEAN_DIR}/case-law/*.txt"))]
    print(f"[phase2] 1/3 MinHash for {len(cl_shards)} case-law shards...")
    _run_pool(_minhash_one, cl_shards, workers)

    # 2. LSH near-dup set (single process).
    print("[phase2] 2/3 near-dup set (LSH)...")
    _build_near_dups()
    with open(NEAR_DUPS_PATH, encoding="utf-8") as fh:
        near_by_shard = {k: set(v) for k, v in json.load(fh).items()}

    # 3. Build the contamination set ONCE, then write every corpus shard.
    print("[phase2] 3/3 decontaminate + write final corpus...")
    contam = _build_contamination_ngrams()
    results = []
    for src in _SOURCE_BY_NAME:
        for path in sorted(glob.glob(f"{config.CLEAN_DIR}/{src}/*.txt")):
            base = os.path.basename(path)
            near = near_by_shard.get(base, set()) if src == "case-law" else set()
            results.append(_write_corpus_shard(
                src, base, near, contam if src in DECONTAM_SOURCES else None))

    report: dict[str, dict] = {}
    for r in results:
        agg = report.setdefault(r["source"], {"kept": 0, "est_tokens": 0, "reasons": {}})
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    with open(f"{config.CORPUS_DIR}/phase2_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    total = sum(a["est_tokens"] for a in report.values())
    print(f"\n[phase2] corpus est tokens: {total/1e9:.2f}B  ({report})")


# --------------------------------------------------------------------------- #
# Phase 3: train the 16K byte-level BPE tokenizer
# --------------------------------------------------------------------------- #


def phase3_tokenizer() -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    def _corpus_lines():
        for path in sorted(glob.glob(f"{config.CORPUS_DIR}/*/*.txt")):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line:
                        yield line

    specials = list(config.SPECIAL_TOKENS.values()) + list(config.EXTRA_CHAT_TOKENS)
    tok = Tokenizer(models.BPE(unk_token=config.SPECIAL_TOKENS["unk_token"]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.MODEL.vocab_size, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    print("[phase3] training BPE...")
    tok.train_from_iterator(_corpus_lines(), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=config.SPECIAL_TOKENS["bos_token"],
        eos_token=config.SPECIAL_TOKENS["eos_token"],
        pad_token=config.SPECIAL_TOKENS["pad_token"],
        unk_token=config.SPECIAL_TOKENS["unk_token"],
        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS))
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    fast.save_pretrained(config.TOKENIZER_DIR)
    print(f"[phase3] vocab_size={fast.vocab_size} -> {config.TOKENIZER_DIR}")


# --------------------------------------------------------------------------- #
# Phase 4: tokenize + pack into uint16 1024-token windows, split 99/1
# --------------------------------------------------------------------------- #


def _tokenize_one(task: tuple) -> dict:
    source_name, shard_index, num_shards = task
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    seq_len = config.SEQ_LEN

    os.makedirs(config.TRAIN_TOKENS_DIR, exist_ok=True)
    os.makedirs(config.VAL_TOKENS_DIR, exist_ok=True)
    train_path = f"{config.TRAIN_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    val_path = f"{config.VAL_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    corpus_files = sorted(glob.glob(f"{config.CORPUS_DIR}/{source_name}/*.txt"))

    def _doc_iter():
        for path in corpus_files:
            with open(path, encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if idx % num_shards == shard_index:
                        line = line.rstrip("\n")
                        if line:
                            yield line

    buf: list[int] = []
    win_count = n_train = n_val = 0
    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        batch: list[str] = []

        def _flush():
            nonlocal win_count, n_train, n_val
            if not batch:
                return
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
                buf.append(eos_id)
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
                _flush()
                batch = []
        _flush()

    print(f"[tok {source_name} {shard_index:03d}] train_win={n_train} val_win={n_val}", flush=True)
    return {"train_windows": n_train, "val_windows": n_val,
            "train_tokens": n_train * seq_len, "val_tokens": n_val * seq_len}


def phase4_tokenize(workers: int) -> None:
    work = [(name, i, n) for name, n in TOKENIZE_SHARDS.items() for i in range(n)]
    print(f"[phase4] {len(work)} tokenize tasks on {workers} workers...")
    results = _run_pool(_tokenize_one, work, workers)
    total = {
        "seq_len": config.SEQ_LEN, "dtype": config.TOKENS_DTYPE,
        "train_windows": sum(r["train_windows"] for r in results),
        "val_windows": sum(r["val_windows"] for r in results),
        "train_tokens": sum(r["train_tokens"] for r in results),
        "val_tokens": sum(r["val_tokens"] for r in results),
    }
    os.makedirs(config.TOKENS_DIR, exist_ok=True)
    with open(f"{config.TOKENS_DIR}/index.json", "w", encoding="utf-8") as fh:
        json.dump(total, fh, indent=2)
    print(f"[phase4] train={total['train_tokens']/1e9:.2f}B tok ({total['train_windows']} win), "
          f"val={total['val_tokens']/1e6:.1f}M tok ({total['val_windows']} win)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="single-machine data prep (Phases 1-4)")
    ap.add_argument("--phase", choices=["1", "2", "3", "4", "all"], default="all")
    ap.add_argument("--fineweb-shards", type=int, default=5,
                    help="how many fineweb-edu parquet shards to stream (Phase 1)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    if args.phase in ("1", "all"):
        phase1_clean(args.fineweb_shards, args.workers)
    if args.phase in ("2", "all"):
        phase2_dedup(args.workers)
    if args.phase in ("3", "all"):
        phase3_tokenizer()
    if args.phase in ("4", "all"):
        phase4_tokenize(args.workers)


if __name__ == "__main__":
    main()
