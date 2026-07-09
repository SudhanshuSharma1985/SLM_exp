# Phase 6, Deploy the model for inference

**Goal:** make the trained 125M base model usable from a web page. Done for ~$0
(scale-to-zero CPU endpoint + free Vercel static hosting).

## Live URLs

- **Website (Vercel):** https://slm-125m.vercel.app
- **Inference endpoint (Modal):** https://thesreedath--slm-125m-inference-web.modal.run
- **Model weights (HuggingFace):** https://huggingface.co/thesreedath/slm-125m-base

## The artifact

A standard HuggingFace model directory at `/data/checkpoints/base/`, also pushed to
the HF repo: `config.json` (LlamaConfig), `model.safetensors` (~250MB), the
tokenizer files, and `generation_config.json`. Loadable with
`AutoModelForCausalLM.from_pretrained("thesreedath/slm-125m-base")`.

## The inference endpoint

A Modal class (`Inference` in `modal_app.py`) that mounts the Volume, loads the
model once per container (`@modal.enter`), and serves a FastAPI app
(`@modal.asgi_app`) with open CORS:

- `GET /health` -> `{ok, model, val_ppl}`
- `POST /generate {prompt, max_new_tokens, min_new_tokens, temperature, top_p,
  top_k, repetition_penalty}` -> `{generated}`

CPU, `min_containers=0`, `scaledown_window=300`: it costs nothing when idle and
spins up on the first request (~15 to 30s cold start), then serves in a few
seconds. Generation is wrapped in try/except so it never returns a 500 to the
frontend. `min_new_tokens=40` prevents the base-model "emit EOS after one token"
failure; `top_k` + sampling keep output clean.

Deploy / update:
```
modal deploy modal_app.py
```

## The website

A single self-contained `slm-125m-site/index.html` deployed to Vercel (team
vi-zuara), in the Vizuara design system (warm light theme, Figtree headings,
JetBrains Mono numerics, teal/violet/rose gradient):

- a prompt box (prefilled) + example-prompt chips,
- sampling controls (temperature, max tokens, top-p, top-k),
- a stats strip (125.8M params, 16K vocab, 1024 ctx, 2.19B tokens, val ppl 8.50, ~$57),
- a "what this is" panel: base completer, honest metric is perplexity, speaks the
  legal register but does not know facts (needs RAG),
- cold-start handling: a "waking the model" state with auto-retry (3 attempts).

Deploy / update:
```
cd slm-125m-site && vercel deploy --prod --yes --name slm-125m
```

## Deliverable (met)

`docs/06-deploy.md` + live HF repo URL + endpoint URL + Vercel URL, all verified:
site returns 200, calls the live endpoint, and returns coherent legal completions.
