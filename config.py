"""Single source of truth for the from-scratch 125M SLM build.

Every other module imports from here so model geometry, tokenizer vocab, the
data mix, Modal GPU economics, and on-Volume paths can never drift apart. The
model is a vanilla Llama-style decoder on purpose: it maps 1:1 to
``transformers.LlamaConfig`` and so ``convert_hf_to_gguf.py`` recognizes it with
no custom-converter work (the single biggest deployment risk).

Immutable by convention: treat these dataclasses as frozen records. Build a new
one with ``dataclasses.replace(cfg, ...)`` rather than mutating in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# --------------------------------------------------------------------------- #
# Project identity / paths (all paths are relative to the Modal Volume mount)
# --------------------------------------------------------------------------- #

PROJECT = "slm-125m"
# The finished weights are pushed here with the HF write token (user thesreedath,
# write role; org context-course also available).
HF_REPO = "thesreedath/slm-125m-base"

# One persistent Modal Volume holds every durable artifact, mounted at /data.
VOLUME_NAME = "slm-125m"
DATA_ROOT = "/data"
CLEAN_DIR = f"{DATA_ROOT}/clean"          # Phase 1: cleaned .txt shards, per source
CORPUS_DIR = f"{DATA_ROOT}/corpus"        # Phase 2: deduped + decontaminated corpus
TOKENIZER_DIR = f"{DATA_ROOT}/tokenizer"  # Phase 3: the fresh 16K byte-level BPE
TOKENS_DIR = f"{DATA_ROOT}/tokens"        # Phase 4: packed uint16 windows (train/ + val/)
TRAIN_TOKENS_DIR = f"{TOKENS_DIR}/train"
VAL_TOKENS_DIR = f"{TOKENS_DIR}/val"
CKPT_DIR = f"{DATA_ROOT}/checkpoints"     # Phase 5
BASE_CKPT_DIR = f"{CKPT_DIR}/base"        # final HF safetensors
RESUME_CKPT_PATH = f"{CKPT_DIR}/ckpt.pt"  # resumable optimizer + step state
METRICS_PATH = f"{CKPT_DIR}/metrics.jsonl"

# The Modal secret name that carries HUGGINGFACE_TOKEN into containers.
HF_SECRET_NAME = "huggingface-token"

# --------------------------------------------------------------------------- #
# Model: vanilla Llama geometry (SwiGLU, RMSNorm, RoPE, tied embeddings)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """Maps 1:1 to transformers.LlamaConfig so HF -> GGUF export "just works".

    Param count with tied embeddings:
      embed        16,384 * 768                    =  12.58M
      per layer    attn (MHA) 4*768*768 = 2.359M
                   mlp SwiGLU 3*768*3072 = 7.078M  =   9.44M
      12 layers                                    = 113.24M
      total (tied head adds 0)                    ~= 125.8M
    """

    vocab_size: int = 16_384
    hidden_size: int = 768
    intermediate_size: int = 3_072        # SwiGLU inner
    num_hidden_layers: int = 12
    num_attention_heads: int = 12         # head dim 64
    num_key_value_heads: int = 12         # == heads -> MHA (export-safe, bootcamp-faithful)
    max_position_embeddings: int = 1_024  # context length
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"              # SwiGLU
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    def to_llama_kwargs(self) -> dict:
        """kwargs for transformers.LlamaConfig(**...)."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": self.hidden_act,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
        }

    def approx_params(self) -> int:
        e = self.vocab_size * self.hidden_size
        h, i = self.hidden_size, self.intermediate_size
        kv = self.num_key_value_heads * (h // self.num_attention_heads)
        attn = h * h + 2 * (h * kv) + h * h  # q + k + v + o
        mlp = 3 * h * i                      # gate + up + down (SwiGLU)
        per_layer = attn + mlp + 2 * h       # + two RMSNorm vectors
        return e + self.num_hidden_layers * per_layer  # tied head adds 0


MODEL = ModelConfig()

# Special tokens reserved at tokenizer-train time. The chat tokens are reserved
# now (cheap) so a later alignment phase can use them without retraining the
# tokenizer; this build never emits them.
SPECIAL_TOKENS: Mapping[str, str] = {
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",  # byte-level fallback means this is essentially never used
}
# Reserved for later alignment; unused during pretraining.
EXTRA_CHAT_TOKENS: tuple[str, ...] = ("<|user|>", "<|assistant|>", "<|system|>")

# --------------------------------------------------------------------------- #
# Data mix (streamed from HuggingFace, all ungated, all parquet-native)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Source:
    name: str
    hf_id: str
    token_budget: int      # stop streaming this source once ~this many clean tokens land
    text_field: str        # the record field holding the document text
    split: str = "train"   # case-law uses the "us" split, not "train"
    config_name: str | None = None  # HF dataset config/subset (None = default)
    strict_ocr: bool = False  # apply the OCR-garble gate (scanned sources only)


# Choice A (legal-first, measured 2026-07-08). The two legal sources max out at
# ~2B unique clean tokens combined, so we take ALL of them and add a small web
# slice for fluency. We reach more tokens-SEEN via ~5 epochs in Phase 5. Measured
# yields: case-law ~0.81B, SEC ~1.16B, fineweb effectively unlimited. Budgets are
# set a little above the legal ceilings (so we take everything) and capped for web.
DATA_MIX: tuple[Source, ...] = (
    # scanned opinions -> strict OCR gate on
    Source("case-law", "HFforLegal/case-law", 1_000_000_000, "document",
           split="us", strict_ocr=True),
    Source("sec", "PleIAs/SEC", 1_300_000_000, "text", split="train"),
    Source("fineweb-edu", "HuggingFaceFW/fineweb-edu", 500_000_000, "text",
           split="train", config_name="sample-10BT"),
)

# ~2.5B unique clean tokens across the mix (~78% legal). Tokens counted during
# streaming via a chars/token proxy (the real 16K tokenizer does not exist until
# Phase 3); CHARS_PER_TOKEN is the conversion used for budgeting.
TARGET_TOKENS: int = 2_500_000_000
CHARS_PER_TOKEN: float = 4.0

# Benchmarks held OUT of pretraining; Phase 2 strips docs resembling these so
# held-out evaluation stays honest.
EVAL_HOLDOUT: tuple[str, ...] = ("coastalcph/lex_glue", "casehold/casehold")

# --------------------------------------------------------------------------- #
# Cleaning-pipeline thresholds (fixed, rule-based, deterministic)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CleanConfig:
    min_line_chars: int = 40          # drop lines shorter than this
    max_nonalnum_ratio: float = 0.30  # drop lines >30% non-alphanumeric
    min_doc_chars: int = 600          # drop the doc if fewer chars survive
    repetition_top_k: int = 10        # top-K 4-grams for the repetition test
    max_repetition_ratio: float = 0.50  # drop if top-K 4-grams cover >50%
    ngram_n: int = 4
    lang_sample_chars: int = 5_000    # langdetect on the first N chars
    # Strict OCR gate (dictionary-based): drop a doc if more than this fraction
    # of its alphabetic word-tokens are not real English words. 0.20 chosen from
    # a measured case-law sample (median 4.7%, p99 22.7%); >20% is almost always
    # OCR-mangled, not legalese. Only applied to sources with strict_ocr=True.
    nonword_ratio_max: float = 0.20
    ocr_min_tokens: int = 50          # skip the gate on very short docs
    dict_path: str = "/usr/share/dict/words"


CLEAN = CleanConfig()

# --------------------------------------------------------------------------- #
# Tokenization / packing (Phase 4)
# --------------------------------------------------------------------------- #

SEQ_LEN: int = 1_024        # packed window length == model context
VAL_EVERY_N_WINDOWS: int = 100  # deterministic 99/1 split: every 100th window -> val
TOKENS_DTYPE: str = "uint16"    # 16K vocab fits in 16 bits, half the disk of int32

# --------------------------------------------------------------------------- #
# Training budget (Phase 5)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = SEQ_LEN
    micro_batch_size: int = 32          # per-step per-device; tune to H100 VRAM
    global_batch_tokens: int = 524_288  # ~0.5M tok/step
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_tokens: int = 200_000_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    ckpt_every_steps: int = 500
    log_every_steps: int = 20
    eval_every_steps: int = 1_000
    seed: int = 1337


TRAIN = TrainConfig()

# --------------------------------------------------------------------------- #
# Modal GPU economics (mid-2026 modal.com/pricing)
# --------------------------------------------------------------------------- #

GPU_RATE_PER_SEC: Mapping[str, float] = {
    "B200": 0.001736,
    "H200": 0.001261,
    "H100": 0.001097,  # best $/token at this scale -> our pretrain GPU
    "A100-80GB": 0.000694,
    "A10G": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}
PRETRAIN_GPU = "H100"
PRETRAIN_GPU_COUNT = 8      # 8xH100 single-node DDP: same total cost, faster wall-clock
# 5 epochs over 2.19B tokens (~11B seen) is ~$55-60 on 8xH100; cap set with headroom.
BUDGET_CAP_USD = 75.0       # hard --max-usd envelope
PRETRAIN_EPOCHS = 5

# --------------------------------------------------------------------------- #
# Pipeline stages (in execution order; drives docs + status reporting)
# --------------------------------------------------------------------------- #

STAGES: tuple[str, ...] = (
    "setup",        # Phase 0
    "clean",        # Phase 1
    "dedup",        # Phase 2
    "tokenizer",    # Phase 3
    "tokenize",     # Phase 4
    "pretrain",     # Phase 5
    "deploy",       # Phase 6
)


if __name__ == "__main__":
    p = MODEL.approx_params()
    print(f"{PROJECT}")
    print(f"model: {p:,} params (~{p/1e6:.1f}M) | vocab {MODEL.vocab_size} | "
          f"{MODEL.num_hidden_layers}L/{MODEL.hidden_size}d/"
          f"{MODEL.num_attention_heads}h kv={MODEL.num_key_value_heads}")
    print(f"target tokens: {TARGET_TOKENS/1e9:.1f}B (~{TARGET_TOKENS/p:.0f} tok/param)")
    print(f"pretrain: {PRETRAIN_GPU_COUNT}x{PRETRAIN_GPU} "
          f"@ ${GPU_RATE_PER_SEC[PRETRAIN_GPU]*3600:.2f}/hr each | cap ${BUDGET_CAP_USD}")
    print(f"stages: {' -> '.join(STAGES)}")
