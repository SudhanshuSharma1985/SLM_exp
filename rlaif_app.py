"""RLAIF -> DPO alignment of the 125M SFT model, on Modal.

Pipeline (each stage fanned out / staged like sft_app.py):

  SFT prompts  ->  sample_candidates (GPU: K on-policy samples per prompt)
               ->  feedback_shard   (API: score K + pairwise-verify chosen/rejected)
               ->  build_dpo        (CPU: dedup + train/val split)
               ->  dpo_run          (GPU: Direct Preference Optimization)
               ->  dpo_samples      (GPU: before/after side-by-side)

DPO on purpose (not PPO): no reward model, no rollout loop -> fits a 15-min live
lecture window. The labeler is gpt-4.1-mini (the SFT teacher), reusing the same
`openai-token` secret.
"""

from __future__ import annotations

import modal

import config
import rlaif_config as rc
import sft_config as sc

app = modal.App("slm-125m-rlaif")

_base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "huggingface_hub==0.34.4",
        "numpy==2.1.3",
        "httpx==0.27.2",
    )
)
image = _base.add_local_python_source("config", "sft_config", "rlaif_config", "dedup")

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
    .add_local_python_source("config", "sft_config", "rlaif_config", "dpo")
)

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}
LABELER_SECRET = modal.Secret.from_name(rc.OPENAI_SECRET)


# --------------------------------------------------------------------------- #
# Labeler plumbing (openai gpt-4.1-mini; same shape as sft_app._teacher_call)
# --------------------------------------------------------------------------- #


def _client():
    import os

    import httpx

    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }
    return httpx.Client(headers=headers)


def _labeler_call(client, prompt: str, max_out: int, temperature: float) -> tuple[str, int, int]:
    """One labeler call. Returns (text, in_tokens, out_tokens). Transient
    429/5xx are retried with exponential backoff; a hard billing failure
    (insufficient_quota) is surfaced loudly instead of burning the backoff."""
    import random
    import time

    body = {
        "model": rc.OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_out,
        "response_format": {"type": "json_object"},
    }
    delay = 2.0
    for attempt in range(rc.MAX_RETRIES):
        try:
            r = client.post(rc.OPENAI_URL, json=body, timeout=180.0)
            if r.status_code == 200:
                d = r.json()
                u = d.get("usage", {})
                ch = d.get("choices", [])
                txt = ch[0]["message"]["content"] if ch else ""
                return txt or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            if r.status_code == 429 and "insufficient_quota" in r.text:
                raise RuntimeError(
                    "LABELER BILLING ERROR: the OpenAI account has no credits "
                    f"({rc.OPENAI_MODEL}). Add credits and re-run."
                )
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay + random.uniform(0, 1.5))
                delay = min(delay * 2, 60)
                continue
            print(f"  [labeler] HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return "", 0, 0
        except RuntimeError:
            raise
        except Exception as e:
            if attempt == rc.MAX_RETRIES - 1:
                print(f"  [labeler] giving up after {rc.MAX_RETRIES}: {e}", flush=True)
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
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------- #
# Stage 1: sample K on-policy candidates per prompt (GPU, no API cost)
# --------------------------------------------------------------------------- #


@app.function(image=gpu_image, volumes=VOLUMES, gpu=f"{rc.DPO_GPU}:1", timeout=60 * 45)
def sample_candidates(n_prompts: int = 0, k: int = 0) -> dict:
    """Read the SFT training prompts, sample N_PROMPTS of them, and draw K
    diverse completions PER PROMPT from the SFT model itself (on-policy). These
    are the candidates the labeler will rank into chosen/rejected."""
    import json
    import os
    import random
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n_prompts = n_prompts or rc.N_PROMPTS
    k = k or rc.K_CANDIDATES
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(rc.POLICY_MODEL)
    tok.padding_side = "left"  # left-pad so generation continues from the prompt end
    tok.pad_token = sc.PAD_TOKEN
    model = AutoModelForCausalLM.from_pretrained(
        rc.POLICY_MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
    eos = tok.convert_tokens_to_ids(sc.EOS_TOKEN)
    pad = tok.convert_tokens_to_ids(sc.PAD_TOKEN)

    # --- pick prompts from the SFT train split (fall back to val) ---
    src = rc.SFT_TRAIN_JSONL
    if not os.path.exists(src):
        src = rc.SFT_VAL_JSONL
    prompts: list[dict] = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            m = {x["role"]: x["content"] for x in row["messages"]}
            if "user" in m:
                prompts.append({"system": m.get("system", sc.DEFAULT_SYSTEM),
                                "user": m["user"]})
    random.Random(1337).shuffle(prompts)
    prompts = prompts[:n_prompts]
    for i, p in enumerate(prompts):
        p["pid"] = i
    print(f"[sample] {len(prompts)} prompts from {src}, k={k}", flush=True)

    def _fmt(p: dict) -> str:
        msgs = [{"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    out: list[dict] = []
    B = rc.SAMPLE_BATCH_PROMPTS
    for start in range(0, len(prompts), B):
        batch = prompts[start : start + B]
        texts = [_fmt(p) for p in batch]
        enc = tok(texts, return_tensors="pt", add_special_tokens=False, padding=True)
        enc = {kk: v.to("cuda") for kk, v in enc.items()
               if kk in ("input_ids", "attention_mask")}
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=rc.SAMPLE_TEMPERATURE,
                top_p=rc.SAMPLE_TOP_P, max_new_tokens=rc.SAMPLE_MAX_NEW_TOKENS,
                num_return_sequences=k, eos_token_id=eos, pad_token_id=pad,
            )
        plen = enc["input_ids"].shape[1]
        gen = gen[:, plen:]  # keep only the newly generated completion tokens
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for bi, p in enumerate(batch):
            cands = [decoded[bi * k + j].strip() for j in range(k)]
            cands = [c for c in cands if len(c) >= 2]
            if len(cands) < 2:
                continue  # need at least two distinct candidates to form a pair
            out.append({"pid": p["pid"], "system": p["system"], "user": p["user"],
                        "candidates": cands})
        if (start // B + 1) % 5 == 0:
            print(f"[sample] {start + len(batch)}/{len(prompts)} prompts", flush=True)

    os.makedirs(rc.RLAIF_DIR, exist_ok=True)
    with open(rc.PROMPTS_PATH, "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps({k2: p[k2] for k2 in ("pid", "system", "user")}) + "\n")
    with open(rc.CANDIDATES_PATH, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    volume.commit()

    usd = (time.time() - t0) * rc.GPU_RATE
    print(f"[sample] wrote {len(out)} prompts w/ candidates | GPU ~${usd:.2f}", flush=True)
    return {"prompts": len(out), "gpu_usd": round(usd, 2)}


# --------------------------------------------------------------------------- #
# Stage 2: AI feedback -- score K candidates, then pairwise-verify (API)
# --------------------------------------------------------------------------- #

_SCORE_PROMPT = """You are a strict quality judge for a legal/financial assistant.

Given a CONTEXT+QUESTION and several candidate ANSWERS, score EACH answer 1-10.

Score 10 = fully correct, fully grounded in the context, precise, well-formed.
Score 5  = partially grounded, vague, or awkward.
Score 1  = hallucinated, contradicts the context, malformed, or degenerate.

Rules:
- An answer that refuses ("not stated in the context") is CORRECT only if the
  question genuinely cannot be answered from the context.
- Any answer asserting a fact not in the context must score <= 3.
- Be harsh and discriminating: spread the scores.

Return ONLY valid JSON: {"scores":[{"i":0,"score":8},{"i":1,"score":3}, ...]}
(one entry per candidate, in order).

CONTEXT + QUESTION:
{prompt}

CANDIDATES:
{candidates}"""

_VERIFY_PROMPT = """You are comparing two candidate answers to the same question
for a legal/financial assistant. Decide which answer is BETTER (more correct,
better grounded in the context, clearer). If they are equally good, say "tie".

Return ONLY valid JSON: {"winner":"A"}  or  {"winner":"B"}  or  {"winner":"tie"}

CONTEXT + QUESTION:
{prompt}

ANSWER A:
{a}

ANSWER B:
{b}"""


@app.function(image=image, volumes=VOLUMES, secrets=[LABELER_SECRET],
              timeout=60 * 45, cpu=4.0)
def feedback_shard(shard: int, n_shards: int) -> dict:
    """For each prompt in this shard: score the K candidates, take the best as
    `chosen` and the worst as `rejected`, require a score margin, then run a
    SEPARATE pairwise verification that the labeler agrees chosen > rejected."""
    import json
    import os
    import random
    from concurrent.futures import ThreadPoolExecutor

    client = _client()

    rows = []
    with open(rc.CANDIDATES_PATH, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["pid"] % n_shards == shard:
                rows.append(r)
    print(f"[fb {shard}] {len(rows)} prompts", flush=True)

    def one(r: dict) -> dict:
        tin = tout = 0
        prompt_str = f'Context:\n{r["user"]}' if "Context:" not in r["user"] else r["user"]
        cands = r["candidates"]
        listing = "\n".join(f"{i}. {c}" for i, c in enumerate(cands))
        sp = (_SCORE_PROMPT.replace("{prompt}", prompt_str)
              .replace("{candidates}", listing))
        text, i1, o1 = _labeler_call(client, sp, max_out=400, temperature=0.0)
        tin += i1
        tout += o1
        d = _parse_json(text)
        scores = (d or {}).get("scores", []) if isinstance(d, dict) else []
        by_i = {int(s["i"]): int(s.get("score", 0) or 0)
                for s in scores if isinstance(s, dict) and "i" in s}
        if len(by_i) < 2:
            return {"pair": None, "in": tin, "out": tout}

        best_i = max(by_i, key=by_i.get)
        worst_i = min(by_i, key=by_i.get)
        if best_i == worst_i or by_i[best_i] - by_i[worst_i] < rc.SCORE_MIN_MARGIN:
            return {"pair": None, "in": tin, "out": tout}
        chosen, rejected = cands[best_i], cands[worst_i]

        # --- separate pairwise verification (randomize order vs position bias) ---
        if rc.REQUIRE_VERIFY:
            flip = random.random() < 0.5
            a, b = (rejected, chosen) if flip else (chosen, rejected)
            vp = (_VERIFY_PROMPT.replace("{prompt}", prompt_str)
                  .replace("{a}", a).replace("{b}", b))
            vtext, i2, o2 = _labeler_call(client, vp, max_out=20, temperature=0.0)
            tin += i2
            tout += o2
            vd = _parse_json(vtext) or {}
            winner = str(vd.get("winner", "")).strip().upper()
            chosen_won = (winner == "B") if flip else (winner == "A")
            if not chosen_won:
                return {"pair": None, "in": tin, "out": tout}

        return {"pair": {"pid": r["pid"], "system": r["system"], "user": r["user"],
                         "chosen": chosen, "rejected": rejected,
                         "chosen_score": by_i[best_i], "rejected_score": by_i[worst_i]},
                "in": tin, "out": tout}

    results = []
    with ThreadPoolExecutor(max_workers=rc.CONCURRENCY) as ex:
        for i, res in enumerate(ex.map(one, rows)):
            results.append(res)
            if (i + 1) % 50 == 0:
                print(f"[fb {shard}] {i+1}/{len(rows)}", flush=True)

    pairs = [res["pair"] for res in results if res["pair"]]
    tin = sum(res["in"] for res in results)
    tout = sum(res["out"] for res in results)

    os.makedirs(rc.FEEDBACK_DIR, exist_ok=True)
    with open(f"{rc.FEEDBACK_DIR}/shard-{shard:03d}.jsonl", "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    volume.commit()

    usd = rc.api_usd(tin, tout)
    print(f"[fb {shard}] kept={len(pairs)}/{len(rows)} in={tin:,} out={tout:,} ~${usd:.2f}",
          flush=True)
    return {"shard": shard, "kept": len(pairs), "seen": len(rows), "in": tin, "out": tout}


# --------------------------------------------------------------------------- #
# Stage 3: dedup + build train/val (CPU)
# --------------------------------------------------------------------------- #


@app.function(image=image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=8_192)
def build_dpo() -> dict:
    import glob
    import json
    import os

    import numpy as np

    from dedup import normalize

    def _vec(text: str) -> dict:
        t = normalize(text)
        grams = [t[i : i + 4] for i in range(max(0, len(t) - 3))]
        v: dict[int, float] = {}
        for g in grams:
            h = hash(g) % 4096
            v[h] = v.get(h, 0.0) + 1.0
        norm = sum(x * x for x in v.values()) ** 0.5 or 1.0
        return {k: x / norm for k, x in v.items()}

    pairs = []
    for path in sorted(glob.glob(f"{rc.FEEDBACK_DIR}/*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                pairs.append(json.loads(line))
    print(f"[build] loaded {len(pairs):,} verified pairs", flush=True)

    drops = {"degenerate": 0}
    kept = []
    for p in pairs:
        # A pair where chosen ~= rejected carries no learnable signal -> drop.
        cv, rv = _vec(p["chosen"]), _vec(p["rejected"])
        small, big = (cv, rv) if len(cv) < len(rv) else (rv, cv)
        cos = sum(x * big.get(k, 0.0) for k, x in small.items())
        if cos > rc.DEDUP_COSINE_MAX:
            drops["degenerate"] += 1
            continue
        kept.append(p)
    print(f"[build] after dedup: {len(kept):,}", flush=True)

    rng = np.random.default_rng(1337)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * rc.VAL_FRACTION))
    val, train = kept[:n_val], kept[n_val:]

    os.makedirs(rc.RLAIF_DIR, exist_ok=True)
    for path, rows in ((rc.PAIRS_PATH, kept), (rc.TRAIN_JSONL, train), (rc.VAL_JSONL, val)):
        with open(path, "w", encoding="utf-8") as fh:
            for p in rows:
                fh.write(json.dumps(p) + "\n")
    volume.commit()

    avg_margin = round(sum(p["chosen_score"] - p["rejected_score"] for p in kept)
                       / max(1, len(kept)), 2)
    report = {"final": len(kept), "train": len(train), "val": len(val),
              "drops": drops, "avg_score_margin": avg_margin}
    with open(rc.REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    print(f"\n=== DPO DATA REPORT ===\n  {report}", flush=True)
    return report


# --------------------------------------------------------------------------- #
# Stage 4: DPO training (GPU)
# --------------------------------------------------------------------------- #


@app.function(image=gpu_image, volumes=VOLUMES,
              gpu=f"{rc.DPO_GPU}:{rc.DPO_GPU_COUNT}", timeout=60 * 60)
def dpo_run(epochs: int, lr: float, beta: float, max_usd: float) -> dict:
    import dpo

    result = dpo.run({
        "epochs": epochs, "lr": lr, "min_lr": rc.DPO_MIN_LR, "beta": beta,
        "weight_decay": rc.DPO_WEIGHT_DECAY, "grad_clip": rc.DPO_GRAD_CLIP,
        "micro_batch": rc.DPO_MICRO_BATCH, "grad_accum": rc.DPO_GRAD_ACCUM,
        "max_usd": max_usd, "gpus": rc.DPO_GPU_COUNT,
    })
    volume.commit()
    return result


@app.function(image=gpu_image, volumes=VOLUMES, gpu=f"{rc.DPO_GPU}:1", timeout=60 * 20)
def dpo_samples() -> list:
    """Side-by-side: SFT (before) vs DPO (after) on the same held-out prompts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(rc.DPO_CKPT_DIR)
    eos = tok.convert_tokens_to_ids(sc.EOS_TOKEN)
    pad = tok.convert_tokens_to_ids(sc.PAD_TOKEN)

    before = AutoModelForCausalLM.from_pretrained(
        rc.POLICY_MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
    after = AutoModelForCausalLM.from_pretrained(
        rc.DPO_CKPT_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()

    ctx = (
        "The plaintiff, Marcia DeWitt, filed suit against Northstar Freight Corp. on "
        "March 14, 1994, alleging negligent maintenance of a loading dock ramp. The "
        "district court granted summary judgment for the defendant, holding that DeWitt "
        "failed to establish that Northstar had actual or constructive notice of the "
        "defect. The record shows Northstar inspected the ramp quarterly, with the last "
        "inspection on January 6, 1994, revealing no defects. We affirm."
    )
    questions = [
        "Why did the court affirm summary judgment for the defendant?",
        "Summarize this passage in one sentence.",
        "What injuries did the plaintiff suffer?",  # unanswerable -> should refuse
    ]

    def gen(model, q):
        msgs = [{"role": "system", "content": sc.DEFAULT_SYSTEM},
                {"role": "user", "content": f"Context:\n{ctx}\n\n{q}"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to("cuda") for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=90, do_sample=False,
                                  eos_token_id=eos, pad_token_id=pad)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    out = []
    for q in questions:
        b, a = gen(before, q), gen(after, q)
        print(f"\n>>> {q}\n  [SFT ] {b}\n  [DPO ] {a}", flush=True)
        out.append({"question": q, "sft": b, "dpo": a})
    return out


# --------------------------------------------------------------------------- #
# Publish
# --------------------------------------------------------------------------- #


@app.function(image=gpu_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)], timeout=60 * 30)
def push_dpo(repo: str = "") -> str:
    import os

    from huggingface_hub import HfApi

    repo = repo or rc.HF_DPO_REPO
    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(repo, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=rc.DPO_CKPT_DIR, repo_id=repo, repo_type="model")
    url = f"https://huggingface.co/{repo}"
    print(f"pushed {rc.DPO_CKPT_DIR} -> {url}", flush=True)
    return url


# --------------------------------------------------------------------------- #
# Local entrypoints
# --------------------------------------------------------------------------- #


@app.local_entrypoint()
def sample(n_prompts: int = 0, k: int = 0):
    r = sample_candidates.remote(n_prompts, k)
    print(f"\nSAMPLED candidates for {r['prompts']} prompts | GPU ~${r['gpu_usd']}")


@app.local_entrypoint()
def feedback():
    n = rc.FEEDBACK_WORKERS
    results = list(feedback_shard.starmap([(i, n) for i in range(n)]))
    tin = sum(r["in"] for r in results)
    tout = sum(r["out"] for r in results)
    kept = sum(r["kept"] for r in results)
    seen = sum(r["seen"] for r in results)
    print(f"\nFEEDBACK: kept {kept:,}/{seen:,} pairs")
    print(f"tokens in={tin:,} out={tout:,}  ESTIMATED SPEND ~${rc.api_usd(tin, tout):.2f}")


@app.local_entrypoint()
def build():
    build_dpo.remote()


@app.local_entrypoint()
def dpo_bg(epochs: int = 0, lr: float = 0.0, beta: float = 0.0, max_usd: float = 0.0):
    """Detached DPO launch (survives a local network drop)."""
    e = epochs or rc.DPO_EPOCHS
    l = lr or rc.DPO_LR
    b = beta or rc.DPO_BETA
    cap = max_usd or rc.RLAIF_MAX_USD
    h = dpo_run.spawn(e, l, b, cap)
    print(f"SPAWNED dpo: {e} epochs, lr={l}, beta={b}, cap=${cap}, "
          f"{rc.DPO_GPU_COUNT}x{rc.DPO_GPU}")
    print(f"SPAWN_CALL_ID={h.object_id}")


@app.local_entrypoint()
def dpo_fg(epochs: int = 0, lr: float = 0.0, beta: float = 0.0, max_usd: float = 0.0):
    """Blocking DPO run (prints metrics live -- good for the lecture)."""
    r = dpo_run.remote(epochs or rc.DPO_EPOCHS, lr or rc.DPO_LR,
                       beta or rc.DPO_BETA, max_usd or rc.RLAIF_MAX_USD)
    print(f"\nDPO DONE: pref_acc {r['val_pref_acc_before']:.3f} -> "
          f"{r['val_pref_acc_after']:.3f} | spent ${r['usd']}")


@app.local_entrypoint()
def samples():
    dpo_samples.remote()


@app.local_entrypoint()
def hf_push_dpo(repo: str = ""):
    push_dpo.remote(repo)
