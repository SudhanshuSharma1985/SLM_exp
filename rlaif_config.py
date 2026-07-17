"""Single source of truth for the RLAIF -> DPO alignment run.

We take the frozen SFT model (`Sudhanshu1985/slm-125m-sft`) and nudge it toward
answers a stronger LLM prefers, using **Direct Preference Optimization** (no
reward model, no PPO rollouts -- deliberately, so the whole thing fits a 15-min
live lecture window).

Pipeline (each stage mirrors the SFT build in sft_app.py):

  SFT prompts  ->  sample K on-policy candidates (GPU)
               ->  AI feedback: score + pairwise-verify with gpt-4.1-mini (API)
               ->  dedup + build (prompt, chosen, rejected) triplets
               ->  DPO train the SFT model (GPU)  ->  before/after eval

Everything is tokenized with the SFT model's OWN tokenizer (same one the base
model shipped), and prompts are formatted with the exact chat template the SFT
model was trained on -- both reused from sft_config.
"""

from __future__ import annotations

from dataclasses import dataclass

import config          # GPU economics + volume/paths
import sft_config as sc  # base/SFT tokenizer, chat template, special tokens

# --------------------------------------------------------------------------- #
# The model we are aligning. This is the FROZEN SFT checkpoint; DPO trains a
# copy of it (the "policy") against a frozen copy of it (the "reference").
# Loaded from the Hub so the run is reproducible and the chat template + special
# tokens travel with it. Swap to sc.SFT_CKPT_DIR to load from the Volume instead.
# --------------------------------------------------------------------------- #
POLICY_MODEL = "Sudhanshu1985/slm-125m-sft"

SEQ_LEN = sc.SEQ_LEN  # 1024; hard ceiling for prompt + completion

# --------------------------------------------------------------------------- #
# AI-feedback labeler. Chosen with the user: gpt-4.1-mini (credits already on
# the OpenAI account; Gemini's free tier caps at 20 req/day -> unusable for 500
# prompts x 2 calls). Reuses the SFT teacher plumbing + secret.
# --------------------------------------------------------------------------- #
LABELER = "openai"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_SECRET = "openai-token"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Priced per 1M tokens (gpt-4.1-mini), used only to report estimated spend.
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.40, 1.60

CONCURRENCY = 8          # in-flight labeler requests per worker
FEEDBACK_WORKERS = 6     # Modal workers fanned out over prompt shards
MAX_RETRIES = 8          # transient 429/5xx retried with exponential backoff

# --------------------------------------------------------------------------- #
# Preference-data size (chosen with the user: small + fast for a live demo).
#   500 prompts -> K candidates each -> 1 winning + 1 losing per prompt,
#   minus ties/verification failures/dups -> ~400-450 clean DPO pairs.
# --------------------------------------------------------------------------- #
N_PROMPTS = 500
K_CANDIDATES = 4         # on-policy samples per prompt to choose best/worst from

# On-policy sampling knobs (diversity matters: identical samples give no signal).
SAMPLE_TEMPERATURE = 0.9
SAMPLE_TOP_P = 0.95
SAMPLE_MAX_NEW_TOKENS = 128
SAMPLE_BATCH_PROMPTS = 32  # prompts per generate() call (x K sequences each)

# --------------------------------------------------------------------------- #
# Quality gates on the preference pairs
# --------------------------------------------------------------------------- #
SCORE_MIN_MARGIN = 2       # chosen_score - rejected_score must be >= this (1-10 rubric)
REQUIRE_VERIFY = True      # separate pairwise judge must AGREE chosen > rejected
DEDUP_COSINE_MAX = 0.95    # drop pairs whose chosen ~= rejected (no learnable signal)
VAL_FRACTION = 0.05        # 5% held out for before/after reporting

# --------------------------------------------------------------------------- #
# DPO hyperparameters (single H100 -- the pair set is tiny; DDP never pays off).
#   beta: KL strength vs the reference. 0.1 is the standard DPO default.
#   lr:   ~4x below SFT's 2e-5; DPO is a gentle nudge, not a re-train.
# --------------------------------------------------------------------------- #
DPO_GPU = "H100"
DPO_GPU_COUNT = 1
GPU_RATE = config.GPU_RATE_PER_SEC[DPO_GPU]

DPO_BETA = 0.1
DPO_EPOCHS = 3
DPO_MICRO_BATCH = 8         # pairs per micro-step (each pair = chosen + rejected fwd)
DPO_GRAD_ACCUM = 2          # effective batch = 16 pairs
DPO_LR = 5e-6
DPO_MIN_LR = 5e-7
DPO_WEIGHT_DECAY = 0.0      # DPO typically runs without weight decay
DPO_GRAD_CLIP = 1.0

# Hard cost envelope for the WHOLE run (sampling + feedback + DPO). The estimate
# is ~$1.10; this cap is pure headroom so a stuck loop can never run away.
RLAIF_MAX_USD = 3.0

# --------------------------------------------------------------------------- #
# Chat formatting -- reuse the EXACT tokens/template/system the SFT model knows.
# --------------------------------------------------------------------------- #
SYS_TOKEN = sc.SYS_TOKEN
USER_TOKEN = sc.USER_TOKEN
ASSISTANT_TOKEN = sc.ASSISTANT_TOKEN
BOS_TOKEN = sc.BOS_TOKEN
EOS_TOKEN = sc.EOS_TOKEN
PAD_TOKEN = sc.PAD_TOKEN
DEFAULT_SYSTEM = sc.DEFAULT_SYSTEM

# --------------------------------------------------------------------------- #
# On-Volume paths
# --------------------------------------------------------------------------- #
RLAIF_DIR = f"{config.DATA_ROOT}/rlaif"
PROMPTS_PATH = f"{RLAIF_DIR}/prompts.jsonl"        # the 500 sampled SFT prompts
CANDIDATES_PATH = f"{RLAIF_DIR}/candidates.jsonl"  # prompt + K on-policy samples
FEEDBACK_DIR = f"{RLAIF_DIR}/feedback"             # per-shard scored+verified pairs
PAIRS_PATH = f"{RLAIF_DIR}/pairs.jsonl"            # all clean triplets
TRAIN_JSONL = f"{RLAIF_DIR}/train.jsonl"
VAL_JSONL = f"{RLAIF_DIR}/val.jsonl"
REPORT_PATH = f"{RLAIF_DIR}/report.json"
DPO_CKPT_DIR = f"{config.DATA_ROOT}/checkpoints/dpo"

# Where prompts come from: the SFT training set already on the Volume. Falls back
# to the val split if train is absent.
SFT_TRAIN_JSONL = sc.TRAIN_JSONL
SFT_VAL_JSONL = sc.VAL_JSONL

# Published DPO model repo (Phase: publish).
HF_DPO_REPO = "Sudhanshu1985/slm-125m-dpo"


@dataclass(frozen=True)
class Budget:
    """Rolling token/spend accounting so we never blow past the estimate."""

    in_tokens: int = 0
    out_tokens: int = 0

    def usd(self) -> float:
        return (self.in_tokens / 1e6) * PRICE_IN_PER_M + (
            self.out_tokens / 1e6
        ) * PRICE_OUT_PER_M


def api_usd(tin: int, tout: int) -> float:
    return (tin / 1e6) * PRICE_IN_PER_M + (tout / 1e6) * PRICE_OUT_PER_M


if __name__ == "__main__":
    print(f"policy model : {POLICY_MODEL} (seq_len {SEQ_LEN})")
    print(f"labeler      : {OPENAI_MODEL}")
    print(f"prompts      : {N_PROMPTS:,} x {K_CANDIDATES} candidates")
    print(f"DPO          : beta={DPO_BETA} lr={DPO_LR} epochs={DPO_EPOCHS} "
          f"eff_batch={DPO_MICRO_BATCH * DPO_GRAD_ACCUM} on 1x{DPO_GPU}")
    print(f"cost cap     : ${RLAIF_MAX_USD}")
