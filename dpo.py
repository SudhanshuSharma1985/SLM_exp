"""Direct Preference Optimization of the 125M SFT model.

Runs INSIDE the Modal GPU container (no modal imports here). Single GPU: the
preference set is only a few hundred pairs, far too small for DDP to pay for its
setup + gradient-sync overhead.

DPO in one paragraph: for each (prompt, chosen, rejected) triplet we compare the
policy's log-prob of the two completions against a FROZEN reference (the SFT
model). The loss pushes the policy to raise the relative log-prob of `chosen`
over `rejected`, while `beta` keeps it from drifting far from the reference:

    L = -log sigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )

where pi_* / ref_* are the summed log-probs of the completion tokens only (the
prompt is masked out on both sides).
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

import rlaif_config as rc
import sft_config as sc

IGNORE = -100


# --------------------------------------------------------------------------- #
# Data: turn each JSONL triplet into (prompt_ids, chosen_ids, rejected_ids)
# --------------------------------------------------------------------------- #


def _load_pairs(path: str, tok) -> list[dict]:
    """Tokenize every triplet once. Each completion carries its own loss mask
    (1 on completion tokens + EOS, 0 on the prompt) so DPO scores only the
    answer, exactly like SFT did."""
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    BOS = tok.convert_tokens_to_ids(sc.BOS_TOKEN)
    EOS = tok.convert_tokens_to_ids(sc.EOS_TOKEN)
    SYS = tok.convert_tokens_to_ids(sc.SYS_TOKEN)
    USR = tok.convert_tokens_to_ids(sc.USER_TOKEN)
    ASST = tok.convert_tokens_to_ids(sc.ASSISTANT_TOKEN)

    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            prompt = [BOS, SYS] + enc(row["system"]) + [USR] + enc(row["user"]) + [ASST]

            def _seq(completion: str) -> tuple[list[int], list[int]] | None:
                ans = enc(completion) + [EOS]
                ids = prompt + ans
                if len(ids) > rc.SEQ_LEN:
                    return None
                mask = [0] * len(prompt) + [1] * len(ans)
                return ids, mask

            c = _seq(row["chosen"])
            r = _seq(row["rejected"])
            if c is None or r is None:
                continue
            out.append({"chosen": c[0], "chosen_mask": c[1],
                        "rejected": r[0], "rejected_mask": r[1]})
    return out


def _collate(rows: list[dict], pad_id: int, device):
    """Pad chosen+rejected of a mini-batch into one stacked tensor so a single
    forward covers both. Returns (input_ids, attn, loss_mask, n) where the first
    n rows are the `chosen` sequences and the next n are the `rejected`."""
    import torch

    seqs = [(r["chosen"], r["chosen_mask"]) for r in rows] + \
           [(r["rejected"], r["rejected_mask"]) for r in rows]
    maxlen = max(len(ids) for ids, _ in seqs)

    ids_arr = np.full((len(seqs), maxlen), pad_id, dtype=np.int64)
    attn_arr = np.zeros((len(seqs), maxlen), dtype=np.int64)
    mask_arr = np.zeros((len(seqs), maxlen), dtype=np.int64)
    for i, (ids, m) in enumerate(seqs):
        ids_arr[i, : len(ids)] = ids
        attn_arr[i, : len(ids)] = 1
        mask_arr[i, : len(m)] = m

    return (
        torch.from_numpy(ids_arr).to(device),
        torch.from_numpy(attn_arr).to(device),
        torch.from_numpy(mask_arr).to(device),
        len(rows),
    )


def _seq_logps(model, input_ids, attn, loss_mask):
    """Summed log-prob of the completion tokens for every row in the batch."""
    import torch
    import torch.nn.functional as F

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]            # predict token t+1 from t
    labels = input_ids[:, 1:]
    mask = loss_mask[:, 1:].to(logits.dtype)   # align mask to the shifted labels
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_logp = torch.gather(logp, 2, labels.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum(dim=-1)


def _dpo_loss(policy_logps, ref_logps, n, beta):
    """Standard DPO loss + diagnostics. Rows [0:n]=chosen, [n:2n]=rejected."""
    import torch
    import torch.nn.functional as F

    pi_c, pi_r = policy_logps[:n], policy_logps[n:]
    ref_c, ref_r = ref_logps[:n], ref_logps[n:]
    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        chosen_reward = beta * (pi_c - ref_c)
        rejected_reward = beta * (pi_r - ref_r)
        acc = (chosen_reward > rejected_reward).float().mean()
        margin = (chosen_reward - rejected_reward).mean()
    return loss, float(acc), float(margin)


def lr_at(step: int, total: int, base_lr: float, warmup: int, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #


def run(args: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    torch.manual_seed(1337)

    epochs = args["epochs"]
    micro = args["micro_batch"]
    accum = args["grad_accum"]
    base_lr = args["lr"]
    beta = args["beta"]

    tok = AutoTokenizer.from_pretrained(rc.POLICY_MODEL)
    pad_id = tok.convert_tokens_to_ids(sc.PAD_TOKEN)

    # Policy = trainable; reference = frozen copy of the same SFT weights.
    policy = AutoModelForCausalLM.from_pretrained(
        rc.POLICY_MODEL, torch_dtype=torch.bfloat16).to(device)
    ref = AutoModelForCausalLM.from_pretrained(
        rc.POLICY_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy.gradient_checkpointing_disable()
    policy.train()

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[dpo] policy={rc.POLICY_MODEL} params={n_params:,} "
          f"({n_params/1e6:.1f}M) beta={beta}", flush=True)

    train = _load_pairs(rc.TRAIN_JSONL, tok)
    val = _load_pairs(rc.VAL_JSONL, tok)
    n_train = len(train)
    steps_per_epoch = max(1, n_train // (micro * accum))
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.05 * total_steps))
    print(f"[dpo] train={n_train:,} pairs  val={len(val):,} pairs", flush=True)
    print(f"[dpo] micro={micro} accum={accum} eff_batch={micro*accum} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    decay = [p for _, p in policy.named_parameters() if p.dim() >= 2]
    no_decay = [p for _, p in policy.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=base_lr, betas=(0.9, 0.95),
    )

    @torch.no_grad()
    def _ref_logps(input_ids, attn, mask):
        return _seq_logps(ref, input_ids, attn, mask)

    @torch.no_grad()
    def evaluate() -> dict:
        policy.eval()
        losses, accs = [], []
        for i in range(0, len(val), micro):
            batch = val[i : i + micro]
            if not batch:
                continue
            ids, attn, mask, n = _collate(batch, pad_id, device)
            pol = _seq_logps(policy, ids, attn, mask)
            rf = _ref_logps(ids, attn, mask)
            loss, acc, _ = _dpo_loss(pol, rf, n, beta)
            losses.append(float(loss))
            accs.append(acc)
        policy.train()
        return {"loss": float(np.mean(losses or [0.0])),
                "acc": float(np.mean(accs or [0.0]))}

    rate = rc.GPU_RATE * args["gpus"]
    t0 = time.time()
    spent = lambda: (time.time() - t0) * rate  # noqa: E731

    metrics: list[dict] = []
    v0 = evaluate()
    print(f"[dpo] BEFORE: val_loss={v0['loss']:.4f} val_pref_acc={v0['acc']:.3f}", flush=True)
    metrics.append({"step": 0, **{f"val_{k}": v for k, v in v0.items()}})

    step = 0
    for epoch in range(epochs):
        order = np.random.default_rng(1337 + epoch).permutation(n_train)
        pos = 0
        for _ in range(steps_per_epoch):
            lr = lr_at(step, total_steps, base_lr, warmup, args["min_lr"])
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)

            loss_acc, acc_acc, marg_acc = 0.0, 0.0, 0.0
            for _ in range(accum):
                rows = [train[j] for j in order[pos : pos + micro]]
                pos += micro
                ids, attn, mask, n = _collate(rows, pad_id, device)
                pol = _seq_logps(policy, ids, attn, mask)
                rf = _ref_logps(ids, attn, mask)
                loss, acc, marg = _dpo_loss(pol, rf, n, beta)
                (loss / accum).backward()
                loss_acc += float(loss) / accum
                acc_acc += acc / accum
                marg_acc += marg / accum

            torch.nn.utils.clip_grad_norm_(policy.parameters(), args["grad_clip"])
            opt.step()
            step += 1

            if step % 10 == 0:
                print(f"[step {step}/{total_steps}] loss={loss_acc:.4f} "
                      f"pref_acc={acc_acc:.3f} margin={marg_acc:+.3f} "
                      f"lr={lr:.2e} spent=${spent():.2f}", flush=True)
                metrics.append({"step": step, "train_loss": loss_acc,
                                "train_pref_acc": acc_acc, "lr": lr})

            if spent() > args["max_usd"]:
                print(f"[stop] budget cap ${args['max_usd']} hit", flush=True)
                break

        v = evaluate()
        print(f"[dpo] === epoch {epoch+1}/{epochs} | val_loss={v['loss']:.4f} "
              f"val_pref_acc={v['acc']:.3f} spent=${spent():.2f}", flush=True)
        metrics.append({"step": step, "epoch": epoch + 1,
                        "val_loss": v["loss"], "val_pref_acc": v["acc"]})

    # Save the aligned model in HF form (+ tokenizer + the SFT chat template).
    os.makedirs(rc.DPO_CKPT_DIR, exist_ok=True)
    policy.save_pretrained(rc.DPO_CKPT_DIR, safe_serialization=True)
    tok.chat_template = sc.CHAT_TEMPLATE_JINJA
    tok.save_pretrained(rc.DPO_CKPT_DIR)
    with open(f"{rc.DPO_CKPT_DIR}/metrics.jsonl", "w", encoding="utf-8") as fh:
        for m in metrics:
            fh.write(json.dumps(m) + "\n")

    vf = evaluate()
    print(f"[dpo] DONE. val_pref_acc {v0['acc']:.3f} -> {vf['acc']:.3f} | "
          f"val_loss {v0['loss']:.4f} -> {vf['loss']:.4f} spent=${spent():.2f}", flush=True)
    return {
        "val_loss_before": v0["loss"], "val_loss_after": vf["loss"],
        "val_pref_acc_before": v0["acc"], "val_pref_acc_after": vf["acc"],
        "steps": step, "usd": round(spent(), 2),
    }
