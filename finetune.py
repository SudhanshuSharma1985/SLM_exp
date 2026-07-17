"""Phase 2: supervised fine-tuning (SFT) of the 125M base model.

Runs INSIDE the Modal GPU container (no modal imports here). Single GPU: the SFT
set is ~7.3M tokens/epoch, which is ~100x too small for DDP to pay for itself.

Loads `thesreedath/slm-125m-base` (the BASE model), trains next-token
cross-entropy on the assistant span ONLY (prompt/context is masked to -100),
evaluates held-out loss each epoch, and saves an HF model directory.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

import sft_config as sc

IGNORE = -100


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    ids = np.fromfile(f"{sc.SFT_TOKENS_DIR}/{split}_ids.bin", dtype=np.uint16)
    mask = np.fromfile(f"{sc.SFT_TOKENS_DIR}/{split}_mask.bin", dtype=np.uint8)
    ids = ids.reshape(-1, sc.SEQ_LEN)
    mask = mask.reshape(-1, sc.SEQ_LEN)
    assert ids.shape == mask.shape, (ids.shape, mask.shape)
    return ids, mask


def _batch(ids: np.ndarray, mask: np.ndarray, rows: np.ndarray, device):
    import torch

    x = torch.from_numpy(ids[rows].astype(np.int64)).to(device)
    m = torch.from_numpy(mask[rows].astype(np.int64)).to(device)
    # Loss only where mask==1 (the assistant answer + <|eos|>).
    labels = x.masked_fill(m == 0, IGNORE)
    return x, labels


def lr_at(step: int, total: int, base_lr: float, warmup: int, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def run(args: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    torch.manual_seed(1337)

    epochs = args["epochs"]
    micro = args["micro_batch"]
    accum = args["grad_accum"]
    base_lr = args["lr"]

    tok = AutoTokenizer.from_pretrained(sc.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(sc.BASE_MODEL, torch_dtype=torch.bfloat16)
    model.to(device)
    model.gradient_checkpointing_disable()
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sft] base={sc.BASE_MODEL} params={n_params:,} ({n_params/1e6:.1f}M)", flush=True)

    tr_ids, tr_mask = _load_split("train")
    va_ids, va_mask = _load_split("val")
    n_train = len(tr_ids)
    steps_per_epoch = max(1, n_train // (micro * accum))
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.03 * total_steps))

    trainable = int(tr_mask.sum())
    print(f"[sft] train={n_train:,} ex  val={len(va_ids):,} ex", flush=True)
    print(f"[sft] micro={micro} accum={accum} eff_batch={micro*accum} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} warmup={warmup}", flush=True)
    print(f"[sft] tokens/epoch={n_train*sc.SEQ_LEN/1e6:.1f}M padded "
          f"({trainable/1e3:.0f}K trainable)", flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=base_lr, betas=(0.9, 0.95),
    )

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        tot, cnt = 0.0, 0
        for i in range(0, len(va_ids), micro):
            rows = np.arange(i, min(i + micro, len(va_ids)))
            x, y = _batch(va_ids, va_mask, rows, device)
            out = model(input_ids=x, labels=y)
            tot += float(out.loss)
            cnt += 1
        model.train()
        return tot / max(1, cnt)

    rate = sc.GPU_RATE * args["gpus"]
    t0 = time.time()
    spent = lambda: (time.time() - t0) * rate  # noqa: E731

    metrics: list[dict] = []
    v0 = evaluate()
    print(f"[sft] val_loss BEFORE training = {v0:.4f} (ppl {math.exp(min(20,v0)):.2f})", flush=True)
    metrics.append({"step": 0, "val_loss": v0})

    step = 0
    for epoch in range(epochs):
        perm = np.random.default_rng(1337 + epoch).permutation(n_train)
        pos = 0
        for _ in range(steps_per_epoch):
            lr = lr_at(step, total_steps, base_lr, warmup, args["min_lr"])
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)

            loss_acc = 0.0
            for _ in range(accum):
                rows = perm[pos : pos + micro]
                pos += micro
                x, y = _batch(tr_ids, tr_mask, rows, device)
                out = model(input_ids=x, labels=y)
                (out.loss / accum).backward()
                loss_acc += float(out.loss) / accum

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % 20 == 0:
                print(f"[step {step}/{total_steps}] loss={loss_acc:.4f} lr={lr:.2e} "
                      f"spent=${spent():.2f}", flush=True)
                metrics.append({"step": step, "train_loss": loss_acc, "lr": lr})

            if spent() > args["max_usd"]:
                print(f"[stop] budget cap ${args['max_usd']} hit", flush=True)
                break

        v = evaluate()
        ppl = math.exp(min(20, v))
        print(f"[sft] === epoch {epoch+1}/{epochs} done | val_loss={v:.4f} val_ppl={ppl:.2f} "
              f"spent=${spent():.2f}", flush=True)
        metrics.append({"step": step, "epoch": epoch + 1, "val_loss": v, "val_ppl": ppl})

    # Save the fine-tuned model in HF form (+ tokenizer + a chat template).
    os.makedirs(sc.SFT_CKPT_DIR, exist_ok=True)
    model.save_pretrained(sc.SFT_CKPT_DIR, safe_serialization=True)
    tok.chat_template = sc.CHAT_TEMPLATE_JINJA
    tok.save_pretrained(sc.SFT_CKPT_DIR)
    with open(f"{sc.SFT_CKPT_DIR}/metrics.jsonl", "w", encoding="utf-8") as fh:
        for m in metrics:
            fh.write(json.dumps(m) + "\n")

    vf = evaluate()
    print(f"[sft] DONE. val_loss {v0:.4f} -> {vf:.4f} "
          f"(ppl {math.exp(min(20,v0)):.2f} -> {math.exp(min(20,vf)):.2f}) "
          f"spent=${spent():.2f}", flush=True)
    return {
        "val_loss_before": v0, "val_loss_after": vf,
        "val_ppl_before": math.exp(min(20, v0)), "val_ppl_after": math.exp(min(20, vf)),
        "steps": step, "tokens_processed": step * micro * accum * sc.SEQ_LEN,
        "usd": round(spent(), 2),
    }