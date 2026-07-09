"""Phase 5 pretraining: single-node multi-GPU DDP for the 125M Llama model.

This module runs INSIDE the Modal GPU container. It has no Modal imports so it can
be unit-reasoned in isolation; `modal_app.py::pretrain` mp.spawns `worker` across
the GPUs. Reads packed uint16 windows from the Volume, trains next-token
cross-entropy with AdamW + cosine LR, evaluates val perplexity, and writes a
resumable ckpt.pt plus an HF safetensors model.
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
import time

import numpy as np

import config


# --------------------------------------------------------------------------- #
# Data: memory-mapped uint16 windows
# --------------------------------------------------------------------------- #


class WindowSet:
    """Flat index over all 1024-token windows in a directory of uint16 .bin shards."""

    def __init__(self, directory: str, seq_len: int):
        self.seq_len = seq_len
        self.files = sorted(glob.glob(f"{directory}/*.bin"))
        if not self.files:
            raise FileNotFoundError(f"no .bin shards in {directory}")
        self.arrays = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        self.counts = [len(a) // seq_len for a in self.arrays]
        self.cum = np.cumsum([0] + self.counts)
        self.n = int(self.cum[-1])

    def __len__(self) -> int:
        return self.n

    def gather(self, idxs: np.ndarray) -> np.ndarray:
        """Return an [len(idxs), seq_len] int64 array for the given window ids."""
        out = np.empty((len(idxs), self.seq_len), dtype=np.int64)
        file_of = np.searchsorted(self.cum, idxs, side="right") - 1
        for row, (gi, fi) in enumerate(zip(idxs, file_of)):
            off = (gi - self.cum[fi]) * self.seq_len
            out[row] = self.arrays[fi][off : off + self.seq_len]
        return out


# --------------------------------------------------------------------------- #
# LR schedule (by tokens seen)
# --------------------------------------------------------------------------- #


def lr_at(tokens: int, total_tokens: int, tc: "config.TrainConfig") -> float:
    if tokens < tc.warmup_tokens:
        return tc.lr * tokens / max(1, tc.warmup_tokens)
    if tokens >= total_tokens:
        return tc.min_lr
    prog = (tokens - tc.warmup_tokens) / max(1, total_tokens - tc.warmup_tokens)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1.0 + math.cos(math.pi * prog))


# --------------------------------------------------------------------------- #
# The per-rank worker
# --------------------------------------------------------------------------- #


def worker(rank: int, world_size: int, args: dict) -> None:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import LlamaConfig, LlamaForCausalLM

    is_master = rank == 0
    smoke = args["smoke"]
    tc = config.TRAIN

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(tc.seed + rank)

    # Model (fresh random weights).
    model = LlamaForCausalLM(LlamaConfig(**config.MODEL.to_llama_kwargs())).to(device)
    model = model.to(torch.bfloat16)
    ddp = DDP(model, device_ids=[rank])
    opt = torch.optim.AdamW(
        model.parameters(), lr=tc.lr, betas=(tc.beta1, tc.beta2),
        weight_decay=tc.weight_decay,
    )

    train = WindowSet(config.TRAIN_TOKENS_DIR, tc.seq_len)
    val = WindowSet(config.VAL_TOKENS_DIR, tc.seq_len)

    micro = tc.micro_batch_size
    tokens_per_micro = micro * tc.seq_len * world_size
    accum = max(1, tc.global_batch_tokens // tokens_per_micro)
    total_tokens = args["epochs"] * len(train) * tc.seq_len
    if is_master:
        print(f"[setup] train_windows={len(train):,} val_windows={len(val):,} "
              f"micro={micro} accum={accum} world={world_size} "
              f"tokens/step={tokens_per_micro*accum:,} total_tokens={total_tokens/1e9:.2f}B",
              flush=True)

    # Resume.
    step, tokens_seen, best_ppl = 0, 0, float("inf")
    if args["resume"] and os.path.exists(config.RESUME_CKPT_PATH):
        ck = torch.load(config.RESUME_CKPT_PATH, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, tokens_seen, best_ppl = ck["step"], ck["tokens"], ck.get("best_ppl", float("inf"))
        if is_master:
            print(f"[resume] step={step} tokens={tokens_seen/1e9:.2f}B", flush=True)

    rate = config.GPU_RATE_PER_SEC[config.PRETRAIN_GPU] * world_size
    max_usd = args["max_usd"]
    t0 = time.time()

    def spent() -> float:
        return (time.time() - t0) * rate

    def evaluate() -> float:
        ddp.eval()
        n_eval = min(len(val), 512 if smoke else len(val))
        idxs = np.arange(n_eval)[rank::world_size]
        loss_sum = torch.zeros(1, device=device)
        count = 0
        with torch.no_grad():
            for i in range(0, len(idxs), micro):
                batch = torch.from_numpy(val.gather(idxs[i : i + micro])).to(device)
                out = ddp(input_ids=batch, labels=batch)
                loss_sum += out.loss.float()
                count += 1
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        cnt = torch.tensor([count], device=device)
        dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
        ddp.train()
        return float((loss_sum / cnt).item())

    def commit_volume() -> None:
        """Persist the mounted Volume mid-run so preemption cannot lose progress."""
        try:
            import modal

            modal.Volume.from_name(config.VOLUME_NAME).commit()
        except Exception:
            pass  # end-of-function commit in modal_app is the backstop

    def save_ckpt() -> None:
        if not is_master:
            return
        os.makedirs(config.CKPT_DIR, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "tokens": tokens_seen, "best_ppl": best_ppl},
                   config.RESUME_CKPT_PATH)
        commit_volume()

    def save_hf() -> None:
        if not is_master:
            return
        os.makedirs(config.BASE_CKPT_DIR, exist_ok=True)
        model.save_pretrained(config.BASE_CKPT_DIR, safe_serialization=True)
        for f in glob.glob(f"{config.TOKENIZER_DIR}/*"):
            shutil.copy(f, config.BASE_CKPT_DIR)

    def log_metric(row: dict) -> None:
        if not is_master:
            return
        os.makedirs(config.CKPT_DIR, exist_ok=True)
        with open(config.METRICS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    ddp.train()
    max_steps = args.get("max_steps") or 10**9
    stop = False
    for epoch in range(args["epochs"]):
        perm = np.random.default_rng(tc.seed + epoch).permutation(len(train))
        rank_idxs = perm[rank::world_size]
        pos = 0
        while pos + micro * accum <= len(rank_idxs):
            lr = lr_at(tokens_seen, total_tokens, tc)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                batch = torch.from_numpy(train.gather(rank_idxs[pos : pos + micro])).to(device)
                pos += micro
                out = ddp(input_ids=batch, labels=batch)
                (out.loss / accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            step += 1
            tokens_seen += tokens_per_micro * accum

            if is_master and step % tc.log_every_steps == 0:
                print(f"[step {step}] loss={out.loss.item():.3f} lr={lr:.2e} "
                      f"tokens={tokens_seen/1e9:.2f}B spent=${spent():.2f}", flush=True)
                log_metric({"step": step, "tokens": tokens_seen,
                            "train_loss": out.loss.item(), "lr": lr, "usd": round(spent(), 2)})

            if step % (5 if smoke else tc.eval_every_steps) == 0:
                vloss = evaluate()
                ppl = math.exp(min(20, vloss))
                if is_master:
                    print(f"[eval step {step}] val_loss={vloss:.3f} val_ppl={ppl:.2f}", flush=True)
                    log_metric({"step": step, "val_loss": vloss, "val_ppl": ppl})
                    if ppl < best_ppl:
                        best_ppl = ppl
                        save_hf()

            if step % tc.ckpt_every_steps == 0:
                save_ckpt()

            if spent() >= max_usd * 0.97 or step >= max_steps:
                if is_master:
                    print(f"[stop] step={step} spent=${spent():.2f} cap=${max_usd}", flush=True)
                stop = True
                break
        if stop:
            break

    # Final eval + save.
    vloss = evaluate()
    ppl = math.exp(min(20, vloss))
    save_ckpt()
    save_hf()
    if is_master:
        print(f"[done] step={step} tokens={tokens_seen/1e9:.2f}B final_val_ppl={ppl:.2f} "
              f"best_ppl={min(best_ppl, ppl):.2f} spent=${spent():.2f}", flush=True)
        log_metric({"step": step, "final_val_loss": vloss, "final_val_ppl": ppl,
                    "usd": round(spent(), 2)})
    dist.barrier()
    dist.destroy_process_group()
