"""Single-GPU pretraining launcher for a bare VM (e.g. Azure NC24ads_A100_v4).

This is the Azure equivalent of `modal_app.py::pretrain`, minus Modal. It calls
the SAME `train.worker` the Modal path uses -- for one GPU, worker runs in-process
(world_size=1), so there is no DDP / torchrun / NCCL fan-out to set up.

Prereqs (see the runbook):
  * the packed token windows live under `config.TOKENS_DIR` (/data/tokens/...)
  * the tokenizer lives under `config.TOKENIZER_DIR` (/data/tokenizer)
  * checkpoints are written under `config.CKPT_DIR` (/data/checkpoints) -- put
    that on a PERSISTENT disk so a spot eviction cannot lose the run.

Usage:
  python train_azure.py --smoke                 # ~20 steps, calibrate tok/s
  python train_azure.py --epochs 1              # your ppl~11 recipe
  python train_azure.py --epochs 5 --resume     # full recipe, resumable
"""

from __future__ import annotations

import argparse
import os

import config
import train


def main() -> None:
    ap = argparse.ArgumentParser(description="single-GPU pretrain (no Modal)")
    ap.add_argument("--epochs", type=int, default=config.PRETRAIN_EPOCHS)
    # NOTE: the spent()/max_usd cap in train.py is priced with Modal's H100 rate,
    # which is MEANINGLESS on Azure. Default it sky-high so it never auto-stops a
    # legitimate run; rely on --epochs (and your own Azure billing) instead.
    ap.add_argument("--max-usd", type=float, default=1_000_000.0)
    ap.add_argument("--resume", action="store_true",
                    help="resume from config.RESUME_CKPT_PATH if present")
    ap.add_argument("--smoke", action="store_true",
                    help="20-step smoke test: verify loss falls + ckpt writes")
    args = ap.parse_args()

    # train.worker reads these; harmless single-process defaults.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    worker_args = {
        "smoke": args.smoke,
        "epochs": 1 if args.smoke else args.epochs,
        "max_usd": args.max_usd,
        "resume": args.resume,
        "max_steps": 20 if args.smoke else None,
    }
    print(f"[azure] world_size=1  epochs={worker_args['epochs']}  "
          f"smoke={args.smoke}  resume={args.resume}", flush=True)
    print(f"[azure] tokens={config.TOKENS_DIR}  ckpt={config.CKPT_DIR}", flush=True)

    train.worker(rank=0, world_size=1, args=worker_args)


if __name__ == "__main__":
    main()
