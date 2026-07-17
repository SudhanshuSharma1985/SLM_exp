"""Single source of truth for the SFT (fine-tuning) data build.

Phase 1: turn the cleaned pretraining corpus into a high-quality, diverse,
deduplicated, decontaminated Q&A dataset, tokenized with the BASE MODEL's OWN
tokenizer (which differs from the one we trained in pretraining).
"""

from __future__ import annotations

from dataclasses import dataclass

import config  # reuse volume/paths from the pretraining build

# --------------------------------------------------------------------------- #
# The base model we are fine-tuning ON TOP OF.
# Its tokenizer vocab is DIFFERENT from config.TOKENIZER_DIR (verified: vocab
# sha 790fd0b1... vs 371d1fb1...). All SFT data MUST be tokenized with THIS one.
# --------------------------------------------------------------------------- #
BASE_MODEL = "thesreedath/slm-125m-base"

SEQ_LEN = 1_024  # base model's context window; hard ceiling per example

# --------------------------------------------------------------------------- #
# Teacher LLM.
#   "openai" -> gpt-4.1-mini  (chosen: needs credits on the OpenAI account)
#   "gemini" -> gemini-3.5-flash (blocked: free tier caps at 20 requests/DAY)
# --------------------------------------------------------------------------- #
TEACHER = "openai"

OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_SECRET = "openai-token"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_SECRET = "gemini-token"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

# Priced per 1M tokens; used only to report estimated spend as we go.
_PRICES = {
    "openai": (0.40, 1.60),   # gpt-4.1-mini
    "gemini": (0.30, 2.50),   # gemini-3.5-flash
}
PRICE_IN_PER_M, PRICE_OUT_PER_M = _PRICES[TEACHER]

CONCURRENCY = 6          # in-flight teacher requests per worker
GEN_WORKERS = 6          # Modal workers fanned out over passage shards
MAX_RETRIES = 8          # transient 429/5xx are retried with exponential backoff

# --------------------------------------------------------------------------- #
# Dataset size (chosen with the user: LIMA-style, quality over quantity)
# --------------------------------------------------------------------------- #
TARGET_FINAL_PAIRS = 5_000
PAIRS_PER_PASSAGE = 3
N_PASSAGES = 2_500              # 2,500 x 3 = 7,500 raw -> ~5,000 after filtering

PASSAGE_TOKENS = 500            # chunk size, measured with the BASE tokenizer
PASSAGE_MIN_TOKENS = 200        # discard stubby chunks

# Source mix (of passages). Legal-heavy, with a web slice so the model does not
# become brittle/purely-legal.
SOURCE_MIX: dict[str, float] = {
    "case-law": 0.50,
    "sec": 0.35,
    "fineweb-edu": 0.15,
}

# Task mix (of passages). Group-B recipes from the fine-tuning brief.
TASK_MIX: dict[str, float] = {
    "grounded_qa": 0.50,   # incl. unanswerable/refusal negatives
    "summarization": 0.20,
    "extraction": 0.15,
    "rewriting": 0.15,
}

# --------------------------------------------------------------------------- #
# Quality gates
# --------------------------------------------------------------------------- #
JUDGE_MIN_SCORE = 4            # 1-5 rubric; keep >= 4
GROUNDING_MIN_OVERLAP = 0.35   # answer/passage token overlap (skipped for refusals)
DEDUP_COSINE_MAX = 0.95        # drop near-duplicate questions above this
DECONTAM_NGRAM = 13            # vs CaseHOLD / LexGLUE, same as pretrain Phase 2

VAL_FRACTION = 0.02            # 2% held out

# --------------------------------------------------------------------------- #
# Chat template (the base model reserves these tokens but defines NO template)
#   <|bos|> <|system|> sys <|user|> user <|assistant|> answer <|eos|>
# Loss is computed ONLY on the assistant span + <|eos|>.
# --------------------------------------------------------------------------- #
SYS_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"

DEFAULT_SYSTEM = (
    "You are a legal and financial assistant. Answer only from the provided "
    "context. If the answer is not in the context, say so."
)

# --------------------------------------------------------------------------- #
# On-Volume paths
# --------------------------------------------------------------------------- #
SFT_DIR = f"{config.DATA_ROOT}/sft"
PASSAGES_PATH = f"{SFT_DIR}/passages.jsonl"
RAW_DIR = f"{SFT_DIR}/raw"                 # per-shard generated pairs
JUDGED_DIR = f"{SFT_DIR}/judged"           # per-shard judged pairs
FILTERED_PATH = f"{SFT_DIR}/filtered.jsonl"
TRAIN_JSONL = f"{SFT_DIR}/train.jsonl"
VAL_JSONL = f"{SFT_DIR}/val.jsonl"
SFT_TOKENS_DIR = f"{SFT_DIR}/tokens"
REPORT_PATH = f"{SFT_DIR}/report.json"
SFT_CKPT_DIR = f"{config.DATA_ROOT}/checkpoints/sft"

# --------------------------------------------------------------------------- #
# Phase 2: fine-tuning hyperparameters.
# ONE GPU on purpose: ~7.3M tokens/epoch is ~100x too small for DDP to pay for
# its own setup + gradient-sync overhead.
# --------------------------------------------------------------------------- #
FT_GPU = "H100"
FT_GPU_COUNT = 1
GPU_RATE = config.GPU_RATE_PER_SEC[FT_GPU]

FT_EPOCHS = 3
FT_MICRO_BATCH = 16
FT_GRAD_ACCUM = 2            # effective batch = 32 examples
FT_LR = 2e-5                 # ~30x below pretraining's 6e-4: SFT must not wreck the base
FT_MIN_LR = 2e-6
FT_WEIGHT_DECAY = 0.01
FT_MAX_USD = 5.0             # hard cap; the run should cost well under $1

# Jinja chat template saved with the fine-tuned model so downstream users get the
# EXACT format it was trained on (the base model ships with none).
CHAT_TEMPLATE_JINJA = (
    "{{ '<|bos|>' }}"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ '<|system|>' + m['content'] }}"
    "{% elif m['role'] == 'user' %}{{ '<|user|>' + m['content'] }}"
    "{% elif m['role'] == 'assistant' %}{{ '<|assistant|>' + m['content'] + '<|eos|>' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


@dataclass(frozen=True)
class Budget:
    """Rolling token/spend accounting so we never blow past the estimate."""

    in_tokens: int = 0
    out_tokens: int = 0

    def usd(self) -> float:
        return (self.in_tokens / 1e6) * PRICE_IN_PER_M + (
            self.out_tokens / 1e6
        ) * PRICE_OUT_PER_M


if __name__ == "__main__":
    raw = N_PASSAGES * PAIRS_PER_PASSAGE
    print(f"base model : {BASE_MODEL} (seq_len {SEQ_LEN})")
    print(f"teacher    : {GEMINI_MODEL}")
    print(f"passages   : {N_PASSAGES:,} x {PAIRS_PER_PASSAGE} = {raw:,} raw pairs")
    print(f"target     : ~{TARGET_FINAL_PAIRS:,} after judge+dedup+decontam")
    print(f"source mix : {SOURCE_MIX}")
    print(f"task mix   : {TASK_MIX}")