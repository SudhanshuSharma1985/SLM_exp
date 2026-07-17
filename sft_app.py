"""Phase 1 of fine-tuning: build the SFT Q&A dataset.

Pipeline (each stage fanned out across Modal workers):

  corpus  ->  chunk+sample passages  ->  Gemini generate  ->  Gemini judge
          ->  ground/dedup/decontaminate  ->  chat JSONL  ->  packed tokens

Everything is tokenized with the BASE MODEL's tokenizer (thesreedath/slm-125m-base),
NOT the tokenizer we trained during pretraining -- their vocabs differ.
"""

from __future__ import annotations

import modal

import config
import sft_config as sc

app = modal.App("slm-125m-sft")

_base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "huggingface_hub==0.34.4",
        "numpy==2.1.3",
        "httpx==0.27.2",
        "datasets==3.6.0",
        "pyarrow==17.0.0",
    )
)
image = _base.add_local_python_source("config", "sft_config", "dedup", "cleaning")

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
TEACHER_SECRET = modal.Secret.from_name(
    sc.OPENAI_SECRET if sc.TEACHER == "openai" else sc.GEMINI_SECRET
)


def _client():
    """An httpx client authenticated for the configured teacher."""
    import os

    import httpx

    if sc.TEACHER == "openai":
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        }
    else:
        headers = {
            "x-goog-api-key": os.environ["GEMINI_API_KEY"],
            "Content-Type": "application/json",
        }
    return httpx.Client(headers=headers)


# --------------------------------------------------------------------------- #
# Stage 1: chunk the corpus into passages the base model can actually fit
# --------------------------------------------------------------------------- #


@app.function(image=image, volumes=VOLUMES, timeout=60 * 45, cpu=8.0, memory=16_384)
def chunk_passages() -> dict:
    """Chunk /data/corpus into ~500-token passages (measured with the BASE
    tokenizer) and sample N_PASSAGES of them per the source + task mix."""
    import glob
    import json
    import os
    import random

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(sc.BASE_MODEL)
    rng = random.Random(1337)

    # How many passages we want from each source.
    want = {s: int(round(sc.N_PASSAGES * f)) for s, f in sc.SOURCE_MIX.items()}
    print(f"[chunk] target passages per source: {want}", flush=True)

    picked: list[dict] = []
    for source, n_want in want.items():
        files = sorted(glob.glob(f"{config.CORPUS_DIR}/{source}/*.txt"))
        # Reservoir-sample documents first so we don't tokenize the whole corpus.
        # Oversample docs (each doc yields >=1 passage) then trim.
        docs: list[str] = []
        doc_target = n_want * 3
        seen = 0
        for path in files:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if len(line) < 800:  # too short to make a 500-token passage
                        continue
                    seen += 1
                    if len(docs) < doc_target:
                        docs.append(line)
                    else:
                        j = rng.randrange(seen)
                        if j < doc_target:
                            docs[j] = line
            if len(docs) >= doc_target and seen > doc_target * 5:
                break

        # Chunk sampled docs into passages of ~PASSAGE_TOKENS tokens.
        out: list[dict] = []
        for doc in docs:
            ids = tok(doc, add_special_tokens=False)["input_ids"]
            for start in range(0, len(ids), sc.PASSAGE_TOKENS):
                chunk = ids[start : start + sc.PASSAGE_TOKENS]
                if len(chunk) < sc.PASSAGE_MIN_TOKENS:
                    continue
                out.append({"source": source, "text": tok.decode(chunk), "n_tok": len(chunk)})
                break  # one passage per doc -> maximizes topical diversity
            if len(out) >= n_want * 2:
                break

        rng.shuffle(out)
        out = out[:n_want]
        print(f"[chunk] {source:<12} docs_seen={seen:>7,} passages={len(out):>5,}", flush=True)
        picked.extend(out)

    # Assign a task type to each passage per TASK_MIX.
    rng.shuffle(picked)
    tasks: list[str] = []
    for task, frac in sc.TASK_MIX.items():
        tasks.extend([task] * int(round(len(picked) * frac)))
    while len(tasks) < len(picked):
        tasks.append("grounded_qa")
    tasks = tasks[: len(picked)]
    rng.shuffle(tasks)
    for p, t in zip(picked, tasks):
        p["task"] = t

    os.makedirs(sc.SFT_DIR, exist_ok=True)
    with open(sc.PASSAGES_PATH, "w", encoding="utf-8") as fh:
        for i, p in enumerate(picked):
            p["pid"] = i
            fh.write(json.dumps(p) + "\n")
    volume.commit()

    by_task: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for p in picked:
        by_task[p["task"]] = by_task.get(p["task"], 0) + 1
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    avg = sum(p["n_tok"] for p in picked) / max(1, len(picked))
    print(f"[chunk] TOTAL {len(picked):,} passages | avg {avg:.0f} tok", flush=True)
    print(f"[chunk] by source: {by_source}", flush=True)
    print(f"[chunk] by task  : {by_task}", flush=True)
    return {"passages": len(picked), "by_source": by_source, "by_task": by_task}


# --------------------------------------------------------------------------- #
# Gemini plumbing
# --------------------------------------------------------------------------- #

_GEN_PROMPTS = {
    "grounded_qa": """You are creating high-quality grounded question-answer pairs to fine-tune a small legal/financial language model.

Read the PASSAGE and write exactly 3 Q&A pairs:
1. type "lookup": a factual question whose answer is stated explicitly in the passage.
2. type "reasoning": a question requiring synthesis across several parts of the passage.
3. type "unanswerable": a plausible-sounding question that is NOT answerable from the passage. Its answer must be exactly: "That is not stated in the context."

Rules:
- Every answer must be fully grounded in the PASSAGE. Invent nothing.
- Answers: 1-3 sentences, precise, in a professional legal/financial register.
- Questions must be self-contained (do not say "in the passage").
- Return ONLY valid JSON: {"pairs":[{"type":"...","question":"...","answer":"..."}]}

PASSAGE:
{passage}""",
    "summarization": """You are creating high-quality summarization training examples for a small legal/financial language model.

Read the PASSAGE and write exactly 3 instruction/response pairs, each a DIFFERENT summarization style:
1. type "summary_short": instruction asks for a concise 1-2 sentence summary.
2. type "summary_bullets": instruction asks for 3-4 bullet points of the key facts.
3. type "summary_key": instruction asks for the single most important holding/finding and why it matters.

Rules:
- Compress only what is in the PASSAGE. Invent nothing; state no fact not present.
- The "question" field holds the instruction; the "answer" field holds the summary.
- Return ONLY valid JSON: {"pairs":[{"type":"...","question":"...","answer":"..."}]}

PASSAGE:
{passage}""",
    "extraction": """You are creating high-quality extraction training examples for a small legal/financial language model.

Read the PASSAGE and write exactly 3 instruction/response pairs that convert prose into STRUCTURED JSON:
1. type "extract_entities": extract the key named entities (parties, companies, courts, people) as JSON.
2. type "extract_dates": extract every date/period mentioned and what happened on it, as JSON.
3. type "extract_facts": extract the key figures/claims/holdings as a JSON object.

Rules:
- The "answer" MUST be a valid JSON object rendered as a string, containing ONLY facts present in the PASSAGE.
- If a field has no value in the passage, omit it. Invent nothing.
- The "question" field holds the instruction (state the exact schema you want).
- Return ONLY valid JSON: {"pairs":[{"type":"...","question":"...","answer":"..."}]}

PASSAGE:
{passage}""",
    "rewriting": """You are creating high-quality rewriting/style-transfer examples for a small legal/financial language model.

Read the PASSAGE and write exactly 3 instruction/response pairs:
1. type "rewrite_plain": rewrite a portion in plain English a non-lawyer understands.
2. type "rewrite_formal": rewrite a portion in precise, formal legal register.
3. type "rewrite_bullets": restructure a portion into clear ordered steps or bullets.

Rules:
- Preserve meaning exactly. Add no facts that are not in the PASSAGE.
- The "question" field holds the instruction and MUST quote or clearly reference the text to rewrite.
- Return ONLY valid JSON: {"pairs":[{"type":"...","question":"...","answer":"..."}]}

PASSAGE:
{passage}""",
}

_JUDGE_PROMPT = """You are a strict quality judge for fine-tuning data for a legal/financial model.

Given the PASSAGE and a list of generated PAIRS, score EACH pair 1-5 and give a verdict.

Score 5 = answer is fully correct, fully grounded in the passage, well-formed, useful.
Score 3 = partially grounded, vague, or awkward.
Score 1 = hallucinated, contradicts the passage, malformed, or trivially degenerate.

Rules:
- An "unanswerable" pair is CORRECT (score 5) only if the question truly cannot be answered from the passage AND the answer is a refusal.
- Any answer asserting a fact not in the PASSAGE must score 1 or 2.
- Be harsh. Most mediocre pairs should score <= 3.

Return ONLY valid JSON: {"scores":[{"i":0,"score":5,"reason":"..."}]}  (one entry per pair, in order)

PASSAGE:
{passage}

PAIRS:
{pairs}"""


def _teacher_call(client, prompt: str, max_out: int, temperature: float) -> tuple[str, int, int]:
    """One teacher-LLM call. Returns (text, in_tokens, out_tokens).

    Retries transient 429/5xx with exponential backoff + jitter. A hard billing
    failure (OpenAI `insufficient_quota`) is NOT retried -- it would just burn
    the backoff budget -- and is surfaced loudly instead.
    """
    import random
    import time

    if sc.TEACHER == "openai":
        url = sc.OPENAI_URL
        body = {
            "model": sc.OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_out,
            "response_format": {"type": "json_object"},
        }
    else:
        url = sc.GEMINI_URL.format(model=sc.GEMINI_MODEL)
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_out,
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

    delay = 2.0
    for attempt in range(sc.MAX_RETRIES):
        try:
            r = client.post(url, json=body, timeout=180.0)
            if r.status_code == 200:
                d = r.json()
                if sc.TEACHER == "openai":
                    u = d.get("usage", {})
                    ch = d.get("choices", [])
                    txt = ch[0]["message"]["content"] if ch else ""
                    return txt or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                u = d.get("usageMetadata", {})
                cand = d.get("candidates", [])
                tin = u.get("promptTokenCount", 0)
                tout = u.get("candidatesTokenCount", 0)
                if not cand or not cand[0].get("content", {}).get("parts"):
                    return "", tin, tout
                return cand[0]["content"]["parts"][0]["text"], tin, tout

            # No credits / no billing: retrying cannot help. Fail fast and loudly.
            if r.status_code == 429 and "insufficient_quota" in r.text:
                raise RuntimeError(
                    "TEACHER BILLING ERROR: the API account has no credits "
                    f"({sc.TEACHER}/{sc.OPENAI_MODEL if sc.TEACHER=='openai' else sc.GEMINI_MODEL}). "
                    "Add credits and re-run."
                )
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay + random.uniform(0, 1.5))
                delay = min(delay * 2, 60)
                continue
            print(f"  [teacher] HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return "", 0, 0
        except RuntimeError:
            raise
        except Exception as e:  # network hiccup -> back off and retry
            if attempt == sc.MAX_RETRIES - 1:
                print(f"  [teacher] giving up after {sc.MAX_RETRIES}: {e}", flush=True)
                return "", 0, 0
            time.sleep(delay + random.uniform(0, 1.5))
            delay = min(delay * 2, 60)
    return "", 0, 0


def _parse_json(text: str) -> dict | None:
    import json

    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # Occasionally the model wraps JSON in prose/fences; salvage the object.
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------- #
# Stage 2: generate raw Q&A pairs (concurrent within a worker, fanned out across)
# --------------------------------------------------------------------------- #


@app.function(image=image, volumes=VOLUMES, secrets=[TEACHER_SECRET],
              timeout=60 * 60, cpu=4.0)
def generate_shard(shard: int, n_shards: int) -> dict:
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor

    client = _client()

    passages = []
    with open(sc.PASSAGES_PATH, encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            if p["pid"] % n_shards == shard:
                passages.append(p)

    print(f"[gen {shard}] {len(passages)} passages", flush=True)

    def one(p: dict) -> dict:
        prompt = _GEN_PROMPTS[p["task"]].replace("{passage}", p["text"])
        # Vary temperature for diversity.
        temp = 0.75 + 0.35 * ((p["pid"] * 7919) % 100) / 100.0
        text, tin, tout = _teacher_call(client, prompt, max_out=1800, temperature=temp)
        d = _parse_json(text)
        pairs = (d or {}).get("pairs", []) if isinstance(d, dict) else []
        clean = []
        for q in pairs:
            if not isinstance(q, dict):
                continue
            question, answer = str(q.get("question", "")).strip(), str(q.get("answer", "")).strip()
            if len(question) < 10 or len(answer) < 2:
                continue
            clean.append({
                "pid": p["pid"], "source": p["source"], "task": p["task"],
                "type": str(q.get("type", "")).strip(), "passage": p["text"],
                "question": question, "answer": answer,
            })
        return {"pairs": clean, "in": tin, "out": tout}

    results = []
    with ThreadPoolExecutor(max_workers=sc.CONCURRENCY) as ex:
        for i, r in enumerate(ex.map(one, passages)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"[gen {shard}] {i+1}/{len(passages)}", flush=True)

    # Retry pass: a transient failure yields 0 pairs. Do NOT silently lose the
    # passage (the smoke test lost 40% this way) -- re-run the empty ones once.
    got = {q["pid"] for r in results for q in r["pairs"]}
    missing = [p for p in passages if p["pid"] not in got]
    if missing:
        print(f"[gen {shard}] retrying {len(missing)} empty passages", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, sc.CONCURRENCY // 2)) as ex:
            results.extend(ex.map(one, missing))
        got2 = {q["pid"] for r in results for q in r["pairs"]}
        still = len(passages) - len(got2)
        print(f"[gen {shard}] after retry, still empty: {still}", flush=True)

    pairs = [q for r in results for q in r["pairs"]]
    tin = sum(r["in"] for r in results)
    tout = sum(r["out"] for r in results)

    os.makedirs(sc.RAW_DIR, exist_ok=True)
    with open(f"{sc.RAW_DIR}/shard-{shard:03d}.jsonl", "w", encoding="utf-8") as fh:
        for q in pairs:
            fh.write(json.dumps(q) + "\n")
    volume.commit()

    usd = (tin / 1e6) * sc.PRICE_IN_PER_M + (tout / 1e6) * sc.PRICE_OUT_PER_M
    print(f"[gen {shard}] pairs={len(pairs)} in={tin:,} out={tout:,} ~${usd:.2f}", flush=True)
    return {"shard": shard, "pairs": len(pairs), "in": tin, "out": tout}


# --------------------------------------------------------------------------- #
# Stage 3: LLM-as-judge (batched per passage so the passage is sent once)
# --------------------------------------------------------------------------- #


@app.function(image=image, volumes=VOLUMES, secrets=[TEACHER_SECRET],
              timeout=60 * 60, cpu=4.0)
def judge_shard(shard: int, n_shards: int) -> dict:
    import json
    import os
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    client = _client()

    by_pid: dict[int, list] = defaultdict(list)
    with open(f"{sc.RAW_DIR}/shard-{shard:03d}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            by_pid[q["pid"]].append(q)

    groups = list(by_pid.items())
    print(f"[judge {shard}] {len(groups)} groups / {sum(len(v) for _, v in groups)} pairs", flush=True)

    def one(item) -> dict:
        pid, pairs = item
        listing = "\n".join(
            f'{i}. [{p["type"]}] Q: {p["question"]}\n   A: {p["answer"]}'
            for i, p in enumerate(pairs)
        )
        prompt = (_JUDGE_PROMPT
                  .replace("{passage}", pairs[0]["passage"])
                  .replace("{pairs}", listing))
        text, tin, tout = _teacher_call(client, prompt, max_out=1000, temperature=0.0)
        d = _parse_json(text)
        scores = (d or {}).get("scores", []) if isinstance(d, dict) else []
        by_i = {int(s["i"]): s for s in scores if isinstance(s, dict) and "i" in s}
        out = []
        for i, p in enumerate(pairs):
            s = by_i.get(i, {})
            p = dict(p)
            p["judge_score"] = int(s.get("score", 0) or 0)
            p["judge_reason"] = str(s.get("reason", ""))[:200]
            out.append(p)
        return {"pairs": out, "in": tin, "out": tout}

    results = []
    with ThreadPoolExecutor(max_workers=sc.CONCURRENCY) as ex:
        for i, r in enumerate(ex.map(one, groups)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"[judge {shard}] {i+1}/{len(groups)}", flush=True)

    pairs = [q for r in results for q in r["pairs"]]
    tin = sum(r["in"] for r in results)
    tout = sum(r["out"] for r in results)

    os.makedirs(sc.JUDGED_DIR, exist_ok=True)
    with open(f"{sc.JUDGED_DIR}/shard-{shard:03d}.jsonl", "w", encoding="utf-8") as fh:
        for q in pairs:
            fh.write(json.dumps(q) + "\n")
    volume.commit()

    kept = sum(1 for p in pairs if p["judge_score"] >= sc.JUDGE_MIN_SCORE)
    usd = (tin / 1e6) * sc.PRICE_IN_PER_M + (tout / 1e6) * sc.PRICE_OUT_PER_M
    print(f"[judge {shard}] scored={len(pairs)} pass={kept} in={tin:,} out={tout:,} ~${usd:.2f}",
          flush=True)
    return {"shard": shard, "scored": len(pairs), "passed": kept, "in": tin, "out": tout}


# --------------------------------------------------------------------------- #
# Stage 4: grounding check + dedup + decontamination + chat formatting
# --------------------------------------------------------------------------- #


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60, cpu=8.0, memory=32_768)
def filter_and_build() -> dict:
    import glob
    import json
    import os
    import re

    import numpy as np
    from transformers import AutoTokenizer

    from dedup import normalize, word_ngrams, words

    tok = AutoTokenizer.from_pretrained(sc.BASE_MODEL)

    pairs = []
    for path in sorted(glob.glob(f"{sc.JUDGED_DIR}/*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                pairs.append(json.loads(line))
    print(f"[filter] loaded {len(pairs):,} judged pairs", flush=True)

    drops = {"judge": 0, "grounding": 0, "too_long": 0, "dup": 0, "contaminated": 0}
    stage = []

    # --- gate 1: judge score ---
    for p in pairs:
        if p.get("judge_score", 0) < sc.JUDGE_MIN_SCORE:
            drops["judge"] += 1
            continue
        stage.append(p)
    print(f"[filter] after judge>={sc.JUDGE_MIN_SCORE}: {len(stage):,}", flush=True)

    # --- gate 2: grounding (answer tokens must overlap the passage) ---
    _REFUSAL = re.compile(r"not stated in the context", re.I)
    kept = []
    for p in stage:
        if _REFUSAL.search(p["answer"]):
            kept.append(p)  # refusals are grounded by construction
            continue
        a = set(words(p["answer"]))
        ctx = set(words(p["passage"]))
        if not a:
            drops["grounding"] += 1
            continue
        overlap = len(a & ctx) / len(a)
        if overlap < sc.GROUNDING_MIN_OVERLAP:
            drops["grounding"] += 1
            continue
        p["grounding"] = round(overlap, 3)
        kept.append(p)
    stage = kept
    print(f"[filter] after grounding: {len(stage):,}", flush=True)

    # --- gate 3: length must fit the 1024 window with the chat template ---
    kept = []
    for p in stage:
        n = len(tok(_render(p), add_special_tokens=False)["input_ids"])
        if n > sc.SEQ_LEN:
            drops["too_long"] += 1
            continue
        p["n_tok"] = n
        kept.append(p)
    stage = kept
    print(f"[filter] after length<= {sc.SEQ_LEN}: {len(stage):,}", flush=True)

    # --- gate 4: dedup on question+answer (char n-gram cosine) ---
    # NOTE: deduping on the QUESTION alone is wrong here. Extraction/summarization
    # instructions are deliberately templated ("Extract the named entities as
    # JSON"), so question-only dedup nukes them as "duplicates" even though each
    # is attached to a different passage. Including the answer (which is
    # passage-specific) keeps those distinct examples and still catches real dups.
    def _vec(text: str) -> dict:
        t = normalize(text)
        grams = [t[i : i + 4] for i in range(max(0, len(t) - 3))]
        v: dict[int, float] = {}
        for g in grams:
            h = hash(g) % 4096
            v[h] = v.get(h, 0.0) + 1.0
        norm = sum(x * x for x in v.values()) ** 0.5 or 1.0
        return {k: x / norm for k, x in v.items()}

    vecs = [_vec(f'{p["question"]} {p["answer"]}') for p in stage]
    kept, keep_vecs = [], []
    for p, v in zip(stage, vecs):
        dup = False
        for kv in keep_vecs:
            if len(v) < len(kv):
                small, big = v, kv
            else:
                small, big = kv, v
            cos = sum(x * big.get(k, 0.0) for k, x in small.items())
            if cos > sc.DEDUP_COSINE_MAX:
                dup = True
                break
        if dup:
            drops["dup"] += 1
            continue
        kept.append(p)
        keep_vecs.append(v)
    stage = kept
    print(f"[filter] after dedup: {len(stage):,}", flush=True)

    # --- gate 5: decontaminate vs CaseHOLD / LexGLUE (same as pretrain phase 2) ---
    contam = _eval_ngrams()
    if contam:
        kept = []
        for p in stage:
            probe = f'{p["question"]} {p["answer"]}'
            if word_ngrams(words(probe), sc.DECONTAM_NGRAM) & contam:
                drops["contaminated"] += 1
                continue
            kept.append(p)
        stage = kept
    print(f"[filter] after decontam: {len(stage):,}", flush=True)

    # --- write filtered + train/val split ---
    rng = np.random.default_rng(1337)
    rng.shuffle(stage)
    n_val = max(1, int(len(stage) * sc.VAL_FRACTION))
    val, train = stage[:n_val], stage[n_val:]

    os.makedirs(sc.SFT_DIR, exist_ok=True)
    for path, rows in ((sc.FILTERED_PATH, stage), (sc.TRAIN_JSONL, train), (sc.VAL_JSONL, val)):
        with open(path, "w", encoding="utf-8") as fh:
            for p in rows:
                fh.write(json.dumps(_to_chat(p)) + "\n")
    volume.commit()

    by_task: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for p in stage:
        by_task[p["task"]] = by_task.get(p["task"], 0) + 1
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1

    report = {
        "final": len(stage), "train": len(train), "val": len(val),
        "drops": drops, "by_task": by_task, "by_source": by_source, "by_type": by_type,
        "avg_tokens": round(sum(p["n_tok"] for p in stage) / max(1, len(stage)), 1),
    }
    print("\n=== PHASE 1 FILTER REPORT ===", flush=True)
    print(f"  final pairs : {len(stage):,}  (train {len(train):,} / val {len(val):,})", flush=True)
    print(f"  drops       : {drops}", flush=True)
    print(f"  by task     : {by_task}", flush=True)
    print(f"  by source   : {by_source}", flush=True)
    print(f"  by type     : {by_type}", flush=True)
    print(f"  avg tokens  : {report['avg_tokens']}", flush=True)
    return report


def _system_for(p: dict) -> str:
    return sc.DEFAULT_SYSTEM


def _user_for(p: dict) -> str:
    """Grounded tasks get the passage as context; all tasks are context-grounded."""
    return f'Context:\n{p["passage"]}\n\n{p["question"]}'


def _render(p: dict) -> str:
    """Flat string form, used only to measure token length."""
    return (
        f'{sc.BOS_TOKEN}{sc.SYS_TOKEN}{_system_for(p)}'
        f'{sc.USER_TOKEN}{_user_for(p)}'
        f'{sc.ASSISTANT_TOKEN}{p["answer"]}{sc.EOS_TOKEN}'
    )


def _to_chat(p: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": _system_for(p)},
            {"role": "user", "content": _user_for(p)},
            {"role": "assistant", "content": p["answer"]},
        ],
        "meta": {
            "source": p["source"], "task": p["task"], "type": p["type"],
            "judge_score": p.get("judge_score"), "grounding": p.get("grounding"),
        },
    }


def _eval_ngrams() -> set:
    """13-grams of the CaseHOLD/LexGLUE eval sets (reuses pretrain Phase 2 logic)."""
    import json
    import urllib.request

    from datasets import load_dataset

    from dedup import word_ngrams, words

    def _parquet_urls(hf_id: str, cfg_name: str, split: str) -> list[str]:
        api = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
        req = urllib.request.Request(api, headers={"User-Agent": "slm-125m"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        return [f["url"] for f in data.get("parquet_files", [])
                if f.get("config") == cfg_name and f.get("split") == split]

    grams: set = set()
    for hf_id, cfg_name in [("coastalcph/lex_glue", "case_hold")]:
        try:
            urls = _parquet_urls(hf_id, cfg_name, "test") or _parquet_urls(hf_id, cfg_name, "train")
            ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                grams |= word_ngrams(words(text), sc.DECONTAM_NGRAM)
        except Exception as e:
            print(f"  [decontam] could not load {hf_id}: {e}", flush=True)
    print(f"  [decontam] {len(grams):,} eval 13-grams", flush=True)
    return grams


# --------------------------------------------------------------------------- #
# Stage 5: tokenize + pack with the BASE MODEL's tokenizer, with loss masks
# --------------------------------------------------------------------------- #


@app.function(image=image, volumes=VOLUMES, timeout=60 * 30, cpu=8.0, memory=16_384)
def tokenize_sft() -> dict:
    """Pack each example to SEQ_LEN. Loss is computed ONLY on assistant tokens."""
    import json
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(sc.BASE_MODEL)
    sid = lambda t: tok.convert_tokens_to_ids(t)  # noqa: E731
    BOS, EOS, PAD = sid(sc.BOS_TOKEN), sid(sc.EOS_TOKEN), sid(sc.PAD_TOKEN)
    SYS, USR, ASST = sid(sc.SYS_TOKEN), sid(sc.USER_TOKEN), sid(sc.ASSISTANT_TOKEN)

    os.makedirs(sc.SFT_TOKENS_DIR, exist_ok=True)
    out: dict[str, dict] = {}

    for split, path in (("train", sc.TRAIN_JSONL), ("val", sc.VAL_JSONL)):
        ids_all, mask_all, n = [], [], 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                m = {x["role"]: x["content"] for x in row["messages"]}
                enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731

                prompt = [BOS, SYS] + enc(m["system"]) + [USR] + enc(m["user"]) + [ASST]
                answer = enc(m["assistant"]) + [EOS]
                ids = prompt + answer
                if len(ids) > sc.SEQ_LEN:
                    continue
                # 1 = compute loss (assistant span only), 0 = ignore
                mask = [0] * len(prompt) + [1] * len(answer)
                pad = sc.SEQ_LEN - len(ids)
                ids += [PAD] * pad
                mask += [0] * pad

                ids_all.append(np.asarray(ids, dtype=np.uint16))
                mask_all.append(np.asarray(mask, dtype=np.uint8))
                n += 1

        if not n:
            continue
        ids_arr = np.vstack(ids_all)
        mask_arr = np.vstack(mask_all)
        ids_arr.tofile(f"{sc.SFT_TOKENS_DIR}/{split}_ids.bin")
        mask_arr.tofile(f"{sc.SFT_TOKENS_DIR}/{split}_mask.bin")
        trainable = int(mask_arr.sum())
        out[split] = {
            "examples": n,
            "total_tokens": int(n * sc.SEQ_LEN),
            "trainable_tokens": trainable,
            "avg_answer_tokens": round(trainable / n, 1),
        }
        print(f"[tok] {split}: {n:,} ex | {trainable:,} trainable tokens "
              f"(avg answer {trainable/n:.0f})", flush=True)

    index = {
        "base_model": sc.BASE_MODEL, "seq_len": sc.SEQ_LEN,
        "dtype_ids": "uint16", "dtype_mask": "uint8",
        "pad_id": PAD, "eos_id": EOS,
        "chat_template": "<|bos|><|system|>SYS<|user|>USER<|assistant|>ANSWER<|eos|>",
        "loss_on": "assistant_span_only",
        "splits": out,
    }
    with open(f"{sc.SFT_TOKENS_DIR}/index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    volume.commit()
    print(f"[tok] wrote {sc.SFT_TOKENS_DIR}/index.json", flush=True)
    return index


# --------------------------------------------------------------------------- #
# Local entrypoints
# --------------------------------------------------------------------------- #


@app.local_entrypoint()
def chunk():
    chunk_passages.remote()


@app.local_entrypoint()
def generate():
    n = sc.GEN_WORKERS
    results = list(generate_shard.starmap([(i, n) for i in range(n)]))
    tin = sum(r["in"] for r in results)
    tout = sum(r["out"] for r in results)
    usd = (tin / 1e6) * sc.PRICE_IN_PER_M + (tout / 1e6) * sc.PRICE_OUT_PER_M
    print(f"\nGENERATED {sum(r['pairs'] for r in results):,} raw pairs")
    print(f"tokens in={tin:,} out={tout:,}  ESTIMATED SPEND ~${usd:.2f}")


@app.local_entrypoint()
def judge():
    n = sc.GEN_WORKERS
    results = list(judge_shard.starmap([(i, n) for i in range(n)]))
    tin = sum(r["in"] for r in results)
    tout = sum(r["out"] for r in results)
    usd = (tin / 1e6) * sc.PRICE_IN_PER_M + (tout / 1e6) * sc.PRICE_OUT_PER_M
    print(f"\nJUDGED {sum(r['scored'] for r in results):,} pairs; "
          f"{sum(r['passed'] for r in results):,} passed")
    print(f"tokens in={tin:,} out={tout:,}  ESTIMATED SPEND ~${usd:.2f}")


@app.local_entrypoint()
def build():
    filter_and_build.remote()
    tokenize_sft.remote()


# --------------------------------------------------------------------------- #
# PHASE 2: fine-tune the base model on the SFT set (single GPU)
# --------------------------------------------------------------------------- #

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==2.1.3",
        "safetensors==0.4.5",
        "huggingface_hub==0.34.4",
        "jinja2==3.1.4",
    )
    .add_local_python_source("config", "sft_config", "finetune")
)


@app.function(image=gpu_image, volumes=VOLUMES,
              gpu=f"{sc.FT_GPU}:{sc.FT_GPU_COUNT}", timeout=60 * 90)
def finetune_run(epochs: int, lr: float, max_usd: float) -> dict:
    import finetune

    result = finetune.run({
        "epochs": epochs,
        "lr": lr,
        "min_lr": sc.FT_MIN_LR,
        "weight_decay": sc.FT_WEIGHT_DECAY,
        "micro_batch": sc.FT_MICRO_BATCH,
        "grad_accum": sc.FT_GRAD_ACCUM,
        "max_usd": max_usd,
        "gpus": sc.FT_GPU_COUNT,
    })
    volume.commit()
    return result


@app.function(image=gpu_image, volumes=VOLUMES, gpu=f"{sc.FT_GPU}:1", timeout=60 * 20)
def sft_samples() -> list:
    """Prompt the fine-tuned model in its trained chat format."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(sc.SFT_CKPT_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        sc.SFT_CKPT_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()

    ctx = (
        "The plaintiff, Marcia DeWitt, filed suit against Northstar Freight Corp. on "
        "March 14, 1994, alleging negligent maintenance of a loading dock ramp. The "
        "district court granted summary judgment for the defendant, holding that DeWitt "
        "failed to establish that Northstar had actual or constructive notice of the "
        "defect. The record shows Northstar inspected the ramp quarterly, with the last "
        "inspection on January 6, 1994, revealing no defects. We affirm."
    )
    questions = [
        "On what date did the plaintiff file suit?",
        "Why did the court affirm summary judgment for the defendant?",
        "What injuries did the plaintiff suffer?",          # unanswerable -> should refuse
        "Summarize this passage in one sentence.",
    ]

    out = []
    for q in questions:
        msgs = [
            {"role": "system", "content": sc.DEFAULT_SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\n{q}"},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        # this tokenizer emits token_type_ids, which LlamaForCausalLM.generate rejects
        enc = {k: v.to("cuda") for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=90, do_sample=False,
                eos_token_id=tok.convert_tokens_to_ids(sc.EOS_TOKEN),
                pad_token_id=tok.convert_tokens_to_ids(sc.PAD_TOKEN),
            )
        ans = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n>>> {q}\n    {ans}", flush=True)
        out.append({"question": q, "answer": ans})
    return out


@app.local_entrypoint()
def finetune_bg(epochs: int = 0, lr: float = 0.0, max_usd: float = 0.0):
    """Detached SFT launch (spawn survives a local network drop)."""
    e = epochs or sc.FT_EPOCHS
    l = lr or sc.FT_LR
    cap = max_usd or sc.FT_MAX_USD
    h = finetune_run.spawn(e, l, cap)
    print(f"SPAWNED finetune: {e} epochs, lr={l}, cap=${cap}, "
          f"{sc.FT_GPU_COUNT}x{sc.FT_GPU}")
    print(f"SPAWN_CALL_ID={h.object_id}")


@app.local_entrypoint()
def samples():
    sft_samples.remote()


# --------------------------------------------------------------------------- #
# Publish: push the fine-tuned model to the Hub
# --------------------------------------------------------------------------- #

HF_SFT_REPO = "Sudhanshu1985/slm-125m-sft"


@app.function(image=gpu_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name("huggingface-token")], timeout=60 * 30)
def push_sft(repo: str = "") -> str:
    import os

    from huggingface_hub import HfApi

    repo = repo or HF_SFT_REPO
    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(repo, exist_ok=True, repo_type="model")

    card = f"""---
license: apache-2.0
language: [en]
library_name: transformers
pipeline_tag: text-generation
base_model: {sc.BASE_MODEL}
tags: [llama, legal, finance, sft, instruction-tuned, grounded-qa]
---

# slm-125m-sft

An **instruction-tuned** 125M legal/financial model: [`{sc.BASE_MODEL}`]
(https://huggingface.co/{sc.BASE_MODEL}) fine-tuned on a synthetic, judged,
deduplicated, decontaminated grounded-QA dataset.

The base model is a *completer*. This one **follows instructions**: it answers
questions about a supplied context, summarizes, extracts to JSON, and rewrites.

## Results

| | Base | **After SFT** |
|---|---|---|
| Held-out loss | 2.66 | **1.63** |
| Held-out perplexity | 14.30 | **5.10** |

## Chat template (required)

It was trained on one exact format, which ships in `tokenizer_config.json`:

```
<|bos|><|system|>SYSTEM<|user|>USER<|assistant|>ANSWER<|eos|>
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}")

msgs = [
    {{"role": "system", "content": "{sc.DEFAULT_SYSTEM}"}},
    {{"role": "user", "content": "Context:\\n<your passage>\\n\\nWhat date was suit filed?"}},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt", add_special_tokens=False)
enc = {{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}}
print(tok.decode(model.generate(**enc, max_new_tokens=90)[0][enc["input_ids"].shape[1]:],
                 skip_special_tokens=True))
```

## Training data

**7,279 pairs** synthesized with `gpt-4.1-mini` from a cleaned legal/financial
corpus (US case law 50%, SEC filings 35%, educational web 15%), then filtered:

- **LLM-as-judge** (1-5 rubric for correctness + grounding); kept >= 4
- **Grounding check** (answer/passage token overlap)
- **Near-duplicate removal** (char n-gram cosine on question+answer)
- **Decontamination** (13-gram overlap vs CaseHOLD / LexGLUE)

Task mix: grounded QA 3,692 (incl. 1,243 "unanswerable" refusals) - summarization
1,457 - extraction-to-JSON 1,043 - rewriting 1,087.

Recipe: 3 epochs, 1xH100, lr 2e-5 cosine, AdamW, bf16, effective batch 32,
**loss on the assistant span only**. ~22M tokens processed.

## Limitations (read this)

- **Refusal is unreliable.** Despite 1,243 "not stated in the context" training
  examples, the model will sometimes **confidently invent** an answer that is not
  in the context instead of refusing. Judging *"is this answerable?"* is hard at
  125M params. **Do not rely on it for factual grounding without verification.**
- It knows **no facts** of its own; it only works over context you supply.
- Domain-biased toward US legal/financial register.
- Not legal or financial advice.
"""
    with open("/tmp/README.md", "w", encoding="utf-8") as fh:
        fh.write(card)
    api.upload_file(path_or_fileobj="/tmp/README.md", path_in_repo="README.md",
                    repo_id=repo, repo_type="model")
    api.upload_folder(folder_path=sc.SFT_CKPT_DIR, repo_id=repo, repo_type="model")
    url = f"https://huggingface.co/{repo}"
    print(f"pushed {sc.SFT_CKPT_DIR} -> {url}", flush=True)
    return url


@app.local_entrypoint()
def hf_push_sft(repo: str = ""):
    push_sft.remote(repo)


# --------------------------------------------------------------------------- #
# Export: ONNX + INT8 dynamic quantization (CPU inference artifact)
# --------------------------------------------------------------------------- #

ONNX_DIR = f"{config.DATA_ROOT}/checkpoints/sft-onnx"
ONNX_INT8_DIR = f"{config.DATA_ROOT}/checkpoints/sft-onnx-int8"

onnx_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "optimum[onnxruntime]==1.23.3",
        "onnx==1.17.0",
        "onnxruntime==1.20.1",
        "numpy==2.1.3",
        "huggingface_hub==0.34.4",
    )
    .add_local_python_source("config", "sft_config")
)


@app.function(image=onnx_image, volumes=VOLUMES, timeout=60 * 60,
              cpu=8.0, memory=32_768)
def export_onnx_int8() -> dict:
    """FP32 ONNX export -> INT8 dynamic quantization -> verify -> report sizes."""
    import glob
    import os
    import shutil

    from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    def _dirsize(d: str) -> float:
        return sum(os.path.getsize(f) for f in glob.glob(f"{d}/**/*", recursive=True)
                   if os.path.isfile(f)) / 1e6

    tok = AutoTokenizer.from_pretrained(sc.SFT_CKPT_DIR)

    # ---- 1. export to ONNX (fp32) ----
    print("[onnx] exporting to ONNX...", flush=True)
    shutil.rmtree(ONNX_DIR, ignore_errors=True)
    os.makedirs(ONNX_DIR, exist_ok=True)
    ort = ORTModelForCausalLM.from_pretrained(sc.SFT_CKPT_DIR, export=True)
    ort.save_pretrained(ONNX_DIR)
    tok.save_pretrained(ONNX_DIR)
    fp32_mb = _dirsize(ONNX_DIR)
    print(f"[onnx] fp32 ONNX = {fp32_mb:.1f} MB", flush=True)

    # ---- 2. INT8 dynamic quantization ----
    # Dynamic (not static): weights -> int8, activations quantized on the fly.
    # No calibration set needed, and it is the standard choice for transformer
    # CPU inference.
    print("[onnx] quantizing to INT8 (dynamic)...", flush=True)
    shutil.rmtree(ONNX_INT8_DIR, ignore_errors=True)
    os.makedirs(ONNX_INT8_DIR, exist_ok=True)

    onnx_files = [os.path.basename(p) for p in glob.glob(f"{ONNX_DIR}/*.onnx")]
    print(f"[onnx] onnx graphs: {onnx_files}", flush=True)

    # CRITICAL for a 125M model with TIED embeddings:
    # quantizing the embedding `Gather` and the tied `lm_head` MatMul to INT8
    # destroys the output distribution (verified: it emits pure gibberish).
    # So: quantize ONLY the transformer MatMuls (attention + MLP), and keep the
    # embedding / lm_head in FP32. Big models tolerate this; small ones do not.
    import onnx as _onnx

    graph = _onnx.load(f"{ONNX_DIR}/{onnx_files[0]}", load_external_data=False).graph
    exclude = [n.name for n in graph.node
               if n.op_type == "MatMul" and (
                   "lm_head" in n.name or "embed" in n.name.lower())]
    print(f"[onnx] excluding {len(exclude)} node(s) from INT8: {exclude}", flush=True)

    qconfig = AutoQuantizationConfig.avx512_vnni(
        is_static=False,
        per_channel=True,
        operators_to_quantize=["MatMul"],   # NOT Gather -> embeddings stay fp32
        nodes_to_exclude=exclude,           # keep the tied lm_head in fp32
    )

    for f in onnx_files:
        q = ORTQuantizer.from_pretrained(ONNX_DIR, file_name=f)
        q.quantize(save_dir=ONNX_INT8_DIR, quantization_config=qconfig)
    # carry the configs/tokenizer across so the dir is self-contained
    for f in glob.glob(f"{ONNX_DIR}/*.json") + glob.glob(f"{ONNX_DIR}/*.txt"):
        shutil.copy(f, ONNX_INT8_DIR)
    tok.save_pretrained(ONNX_INT8_DIR)

    int8_mb = _dirsize(ONNX_INT8_DIR)
    print(f"[onnx] int8 ONNX = {int8_mb:.1f} MB", flush=True)

    # ---- 3. verify the quantized model still answers correctly ----
    print("[onnx] verifying INT8 model...", flush=True)
    q_model = ORTModelForCausalLM.from_pretrained(ONNX_INT8_DIR)
    ctx = (
        "The plaintiff, Marcia DeWitt, filed suit against Northstar Freight Corp. on "
        "March 14, 1994, alleging negligent maintenance of a loading dock ramp. The "
        "district court granted summary judgment for the defendant, holding that DeWitt "
        "failed to establish that Northstar had actual or constructive notice of the defect."
    )
    answers = []
    for q_text in ["On what date did the plaintiff file suit?",
                   "Why did the court grant summary judgment?"]:
        msgs = [{"role": "system", "content": sc.DEFAULT_SYSTEM},
                {"role": "user", "content": f"Context:\n{ctx}\n\n{q_text}"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        out = q_model.generate(
            **enc, max_new_tokens=70, do_sample=False,
            eos_token_id=tok.convert_tokens_to_ids(sc.EOS_TOKEN),
            pad_token_id=tok.convert_tokens_to_ids(sc.PAD_TOKEN),
        )
        ans = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"  Q: {q_text}\n  A: {ans}", flush=True)
        answers.append({"question": q_text, "answer": ans})

    volume.commit()
    st = os.path.getsize(f"{sc.SFT_CKPT_DIR}/model.safetensors") / 1e6
    report = {
        "pytorch_bf16_mb": round(st, 1),
        "onnx_fp32_mb": round(fp32_mb, 1),
        "onnx_int8_mb": round(int8_mb, 1),
        "shrink_vs_pytorch": f"{st/int8_mb:.2f}x",
        "answers": answers,
    }
    print(f"\n[onnx] PyTorch bf16 : {st:7.1f} MB", flush=True)
    print(f"[onnx] ONNX fp32    : {fp32_mb:7.1f} MB", flush=True)
    print(f"[onnx] ONNX INT8    : {int8_mb:7.1f} MB  ({st/int8_mb:.2f}x smaller than PyTorch)",
          flush=True)
    return report


@app.local_entrypoint()
def onnx():
    export_onnx_int8.remote()


@app.function(image=onnx_image, volumes=VOLUMES, timeout=60 * 45,
              cpu=4.0, memory=16_384)
def bench_cpu(new_tokens: int = 64, reps: int = 5, warmup: int = 2) -> dict:
    """INT8 vs FP32 CPU latency. Same prompt, same token budget, greedy decoding,
    so every backend does identical work. Warmup runs are discarded."""
    import statistics
    import time

    import torch
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(sc.SFT_CKPT_DIR)
    eos = tok.convert_tokens_to_ids(sc.EOS_TOKEN)
    pad = tok.convert_tokens_to_ids(sc.PAD_TOKEN)

    ctx = (
        "The plaintiff, Marcia DeWitt, filed suit against Northstar Freight Corp. on "
        "March 14, 1994, alleging negligent maintenance of a loading dock ramp. The "
        "district court granted summary judgment for the defendant, holding that DeWitt "
        "failed to establish that Northstar had actual or constructive notice of the "
        "defect. The record shows Northstar inspected the ramp quarterly, with the last "
        "inspection on January 6, 1994, revealing no defects. We affirm."
    )
    msgs = [{"role": "system", "content": sc.DEFAULT_SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\nWhy did the court affirm?"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    prompt_len = int(enc["input_ids"].shape[1])
    print(f"[bench] prompt={prompt_len} tok | generating {new_tokens} tok, "
          f"greedy | {reps} reps (+{warmup} warmup) | 4 CPU threads\n", flush=True)

    def _gen(model, force_len: bool):
        """One full generation. force_len=True disables EOS so every backend
        emits exactly `new_tokens` (otherwise a backend that stops early would
        look artificially fast)."""
        kw = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                  do_sample=False, pad_token_id=pad)
        if not force_len:
            kw["eos_token_id"] = eos
        t0 = time.perf_counter()
        out = model.generate(**enc, **kw)
        return time.perf_counter() - t0, out

    # the exported ONNX graph takes position_ids as a REQUIRED input (generate()
    # fills it in; a raw forward call must supply it explicitly)
    pos_ids = torch.arange(prompt_len, dtype=torch.long).unsqueeze(0)

    def _prefill(model, needs_pos: bool):
        """Time a single forward over the prompt = time-to-first-token."""
        kw = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if needs_pos:
            kw["position_ids"] = pos_ids
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**kw)
        return time.perf_counter() - t0

    backends = {}
    print("[bench] loading PyTorch fp32...", flush=True)
    backends["PyTorch FP32"] = AutoModelForCausalLM.from_pretrained(
        sc.SFT_CKPT_DIR, torch_dtype=torch.float32).eval()
    print("[bench] loading ONNX fp32...", flush=True)
    backends["ONNX FP32"] = ORTModelForCausalLM.from_pretrained(ONNX_DIR)
    print("[bench] loading ONNX int8...", flush=True)
    backends["ONNX INT8"] = ORTModelForCausalLM.from_pretrained(ONNX_INT8_DIR)

    results: dict[str, dict] = {}
    for name, model in backends.items():
        needs_pos = name.startswith("ONNX")
        for _ in range(warmup):
            _gen(model, force_len=True)
            _prefill(model, needs_pos)

        gens = [_gen(model, force_len=True)[0] for _ in range(reps)]
        pres = [_prefill(model, needs_pos) for _ in range(reps)]

        g_med = statistics.median(gens)
        p_med = statistics.median(pres)
        # decode time = total minus the one prefill pass
        decode = max(1e-6, g_med - p_med)
        results[name] = {
            "total_s": round(g_med, 3),
            "prefill_ms": round(p_med * 1000, 1),
            "decode_tok_per_s": round(new_tokens / decode, 1),
            "ms_per_token": round(decode / new_tokens * 1000, 1),
            "stdev_s": round(statistics.pstdev(gens), 3),
        }
        r = results[name]
        print(f"[bench] {name:<14} total={r['total_s']:.2f}s  "
              f"prefill={r['prefill_ms']:.0f}ms  "
              f"decode={r['decode_tok_per_s']:.1f} tok/s  "
              f"({r['ms_per_token']:.1f} ms/tok)", flush=True)

    base = results["ONNX FP32"]["decode_tok_per_s"]
    pt = results["PyTorch FP32"]["decode_tok_per_s"]
    i8 = results["ONNX INT8"]["decode_tok_per_s"]
    print("\n=== SPEEDUP (decode throughput) ===", flush=True)
    print(f"  ONNX INT8 vs ONNX FP32   : {i8/base:.2f}x", flush=True)
    print(f"  ONNX INT8 vs PyTorch FP32: {i8/pt:.2f}x", flush=True)
    print(f"  ONNX FP32 vs PyTorch FP32: {base/pt:.2f}x", flush=True)

    # Sanity: what does each backend actually SAY? (quality must not be silently lost)
    print("\n=== OUTPUT CHECK (greedy, EOS enabled) ===", flush=True)
    for name, model in backends.items():
        _, out = _gen(model, force_len=False)
        ans = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        print(f"  [{name}] {ans[:150]}", flush=True)
        results[name]["sample"] = ans

    results["_speedup"] = {
        "int8_vs_onnx_fp32": round(i8 / base, 2),
        "int8_vs_pytorch_fp32": round(i8 / pt, 2),
        "onnx_fp32_vs_pytorch_fp32": round(base / pt, 2),
    }
    return results


@app.local_entrypoint()
def bench(new_tokens: int = 64, reps: int = 5):
    bench_cpu.remote(new_tokens, reps)


@app.function(image=onnx_image, volumes=VOLUMES, timeout=60 * 45,
              cpu=8.0, memory=32_768)
def eval_quantized(n: int = 64) -> dict:
    """Ground truth on quality: real cross-entropy loss on the held-out SFT set
    for each backend. Eyeballing a couple of prompts is not evidence; this is."""
    import math

    import numpy as np
    import torch
    import torch.nn.functional as F
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoModelForCausalLM

    ids = np.fromfile(f"{sc.SFT_TOKENS_DIR}/val_ids.bin", dtype=np.uint16).reshape(-1, sc.SEQ_LEN)
    msk = np.fromfile(f"{sc.SFT_TOKENS_DIR}/val_mask.bin", dtype=np.uint8).reshape(-1, sc.SEQ_LEN)
    n = min(n, len(ids))
    print(f"[eval] {n} held-out examples, loss on the assistant span only\n", flush=True)

    backends = {
        "PyTorch FP32": AutoModelForCausalLM.from_pretrained(
            sc.SFT_CKPT_DIR, torch_dtype=torch.float32).eval(),
        "ONNX FP32": ORTModelForCausalLM.from_pretrained(ONNX_DIR),
        "ONNX INT8": ORTModelForCausalLM.from_pretrained(ONNX_INT8_DIR),
    }

    out: dict[str, dict] = {}
    for name, model in backends.items():
        tot, ntok = 0.0, 0
        for i in range(n):
            x = torch.from_numpy(ids[i : i + 1].astype(np.int64))
            m = torch.from_numpy(msk[i : i + 1].astype(np.int64))
            kw = {"input_ids": x, "attention_mask": torch.ones_like(x)}
            if name.startswith("ONNX"):
                kw["position_ids"] = torch.arange(x.shape[1]).unsqueeze(0)
            with torch.no_grad():
                logits = model(**kw).logits.float()
            # next-token CE over the assistant span
            lg = logits[:, :-1, :]
            tgt = x[:, 1:]
            keep = m[:, 1:].bool()
            if keep.sum() == 0:
                continue
            loss = F.cross_entropy(
                lg[keep].view(-1, lg.size(-1)), tgt[keep].view(-1), reduction="sum")
            tot += float(loss)
            ntok += int(keep.sum())

        mean = tot / max(1, ntok)
        ppl = math.exp(min(20, mean))
        out[name] = {"loss": round(mean, 4), "ppl": round(ppl, 2), "tokens": ntok}
        print(f"[eval] {name:<14} loss={mean:.4f}  perplexity={ppl:.2f}", flush=True)

    base = out["PyTorch FP32"]["ppl"]
    print("\n=== QUALITY vs PyTorch FP32 ===", flush=True)
    for name, r in out.items():
        delta = r["ppl"] / base
        verdict = "OK" if delta < 1.10 else ("DEGRADED" if delta < 2 else "BROKEN")
        print(f"  {name:<14} ppl {r['ppl']:>8.2f}  ({delta:>6.2f}x)  {verdict}", flush=True)
    return out


@app.local_entrypoint()
def evalq(n: int = 64):
    eval_quantized.remote(n)


# --------------------------------------------------------------------------- #
# Serve: a chat endpoint for the fine-tuned model (scale-to-zero CPU)
# --------------------------------------------------------------------------- #

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==2.1.3",
        "safetensors==0.4.5",
        "fastapi[standard]==0.115.5",
        "jinja2==3.1.4",
    )
    .add_local_python_source("config", "sft_config")
)


@app.cls(image=serve_image, volumes=VOLUMES, min_containers=0,
         scaledown_window=300, timeout=60 * 10)
class SFTInference:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(sc.SFT_CKPT_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            sc.SFT_CKPT_DIR, torch_dtype=torch.float32).eval()
        self.eos = self.tok.convert_tokens_to_ids(sc.EOS_TOKEN)
        self.pad = self.tok.convert_tokens_to_ids(sc.PAD_TOKEN)
        print("[sft-serve] model loaded", flush=True)

    @modal.asgi_app()
    def web(self):
        from fastapi import Body, FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        api = FastAPI()
        api.add_middleware(CORSMiddleware, allow_origins=["*"],
                           allow_methods=["*"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "model": "slm-125m-sft", "val_ppl": 5.10,
                    "base": sc.BASE_MODEL}

        # NOTE: this module uses `from __future__ import annotations`, so a
        # Pydantic-model annotation would arrive as an unresolvable STRING and
        # FastAPI would mis-read the body as a query param. Take a plain dict.
        @api.post("/chat")
        def chat(payload: dict = Body(...)):
            try:
                context = str(payload.get("context") or "")
                question = str(payload.get("question") or "")
                max_new = int(payload.get("max_new_tokens") or 120)
                do_sample = bool(payload.get("do_sample") or False)
                temperature = float(payload.get("temperature") or 0.3)
                top_p = float(payload.get("top_p") or 0.9)

                user = (f"Context:\n{context}\n\n{question}"
                        if context.strip() else question)
                msgs = [{"role": "system", "content": sc.DEFAULT_SYSTEM},
                        {"role": "user", "content": user}]
                text = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
                enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
                enc = {k: v for k, v in enc.items()
                       if k in ("input_ids", "attention_mask")}
                with self.torch.no_grad():
                    out = self.model.generate(
                        **enc,
                        max_new_tokens=min(256, max(16, max_new)),
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        top_p=top_p if do_sample else None,
                        eos_token_id=self.eos, pad_token_id=self.pad,
                    )
                ans = self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
                return {"answer": ans}
            except Exception as e:  # never 500 the frontend
                return {"answer": "", "error": str(e)}

        return api