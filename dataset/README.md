# Legal/Financial QnA dataset (SFT)

Synthetic, judged, deduplicated, decontaminated instruction-tuning data for the
125M legal SLM. Generated with `gpt-4.1-mini` from a cleaned legal/financial
corpus (US case law 50%, SEC filings 35%, educational web 15%), then filtered by
an LLM judge (kept ≥ 4/5), a grounding check, near-duplicate removal, and 13-gram
decontamination vs CaseHOLD / LexGLUE.

| File | Pairs | Size |
|---|---|---|
| `train.jsonl` | 7,134 | 21.6 MB |
| `val.jsonl` | 145 | 0.45 MB |
| **Total** | **7,279** | ~22 MB |

## Format (one JSON object per line)

```json
{
  "messages": [
    {"role": "system", "content": "You are a legal and financial assistant. ..."},
    {"role": "user", "content": "Context:\n<passage>\n\n<question>"},
    {"role": "assistant", "content": "<grounded answer>"}
  ],
  "meta": {"source": "case-law", "task": "grounded_qa", "type": "lookup",
           "judge_score": 5, "grounding": 0.62}
}
```

Task mix: grounded QA (incl. "unanswerable" refusals), summarization,
extraction-to-JSON, and rewriting. Answers are grounded in the supplied context.

## Load it

```python
from datasets import load_dataset
ds = load_dataset("json", data_files={"train": "dataset/train.jsonl",
                                      "validation": "dataset/val.jsonl"})
```

> Note: the packed pretraining tokens (~4 GB of uint16 `.bin` windows) are **not**
> in this repo — files that large exceed GitHub's 100 MB/file limit. They live on
> the project's Modal Volume; publish them to HuggingFace Datasets if you need
> them hosted.
