"""FastAPI serving layer for FinBERT classifier and GPT-3.5 summarizer.

WHY WE SERVE FINBERT AND NOT GPT-4 (Learning goal)
----------------------------------------------------
After running all three classifiers, you will find FinBERT fine-tuned achieves
comparable or better macro-F1 than GPT-4 few-shot on PhraseBank. But accuracy
alone is not why we choose FinBERT for serving. The operational reasons are:

  LATENCY   FinBERT runs locally in ~30-80ms per request (CPU) or ~5ms (GPU).
            GPT-4 takes 1-3 seconds per call over the network. A trading
            system classifying 500 headlines at market open cannot wait 25
            minutes for GPT-4 to finish.

  COST      GPT-4 charges per token — roughly $0.01-0.03 per classification
            call. FinBERT costs $0 per call after the one-time training cost.
            At 100k classifications/day, GPT-4 costs $1,000-3,000/day.

  RELIABILITY  FinBERT runs offline. If OpenAI has an outage, GPT-4 returns
            503s. FinBERT never goes down unless your own infrastructure does.

  DATA PRIVACY  Financial text sent to GPT-4 leaves your infrastructure and
            enters OpenAI's servers. Many institutional trading environments
            forbid sending non-public market information to third-party APIs.
            FinBERT runs entirely within your own environment.

  REPRODUCIBILITY  GPT-4 model weights change without notice ('gpt-4o' today
            may behave differently in 6 months). FinBERT's checkpoint is
            frozen — identical inputs produce identical outputs forever.

For the summarization endpoint we do use GPT-3.5 because there is no practical
local alternative that produces comparable summarization quality at low cost.
The cache in gpt_summarizer.py ensures repeated calls are free.

Run with:
    uvicorn serving.app:app --reload
    # API docs at http://localhost:8000/docs
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from features.engineer import ID2LABEL, LABEL_ORDER
from models.gpt_summarizer import VARIANTS, summarize

load_dotenv()

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
ROOT           = Path(__file__).parent.parent
CHECKPOINT_DIR = ROOT / "models" / "checkpoints" / "finbert_best"
SUMMARY_DIR    = ROOT / "artifacts" / "summary"
MAX_TOKENS_BERT = 512   # BERT's hard architectural limit

# ---------------------------------------------------------------------------
# Shared in-memory state (populated at startup, cleared at shutdown)
# ---------------------------------------------------------------------------
_state: dict = {}


def _pick_best_variant() -> str:
    """Read summarization_summary.csv and return the variant with highest BERTScore F1.

    Falls back to variant_a if the summary file doesn't exist yet — this lets
    the app start even before the summarizer has been run.
    """
    summary_path = SUMMARY_DIR / "summarization_summary.csv"
    if not summary_path.exists():
        return "variant_a"
    try:
        import pandas as pd
        df = pd.read_csv(summary_path)
        best = df.loc[df["bertscore_f1"].idxmax(), "variant"]
        return str(best)
    except Exception:
        return "variant_a"


# ---------------------------------------------------------------------------
# Lifespan: load models once at startup, release at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FinBERT and OpenAI client into _state at startup."""
    if not CHECKPOINT_DIR.exists():
        raise RuntimeError(
            f"FinBERT checkpoint not found at {CHECKPOINT_DIR}. "
            "Run: python -m models.finbert_finetune"
        )

    print("[serving] loading FinBERT tokenizer and model ...")
    _state["tokenizer"] = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    _state["model"]     = AutoModelForSequenceClassification.from_pretrained(
        str(CHECKPOINT_DIR)
    )
    _state["model"].eval()

    # Temperature scalar saved during fine-tuning — applied at inference.
    temp_path = CHECKPOINT_DIR / "temperature.json"
    _state["temperature"] = (
        json.loads(temp_path.read_text())["temperature"]
        if temp_path.exists() else 1.0
    )

    _state["openai_client"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _state["best_variant"]  = _pick_best_variant()

    print(f"[serving] ready. temperature={_state['temperature']:.4f}, "
          f"summarization variant={_state['best_variant']}")
    yield                           # server runs here
    _state.clear()                  # clean up on shutdown
    print("[serving] shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Finance NLP Pipeline",
    description="FinBERT financial sentiment classifier + GPT-3.5 earnings summarizer.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    truncated: bool     # True if input exceeded 512 tokens and was cut


class SummarizeRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class SummarizeResponse(BaseModel):
    summary: str
    variant_used: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Liveness check — returns ok when the model is loaded and ready."""
    return {"status": "ok", "model": "finbert-finetuned"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify the sentiment of a financial text string.

    Tokenizes the input, runs a forward pass through fine-tuned FinBERT,
    applies temperature scaling, and returns the predicted label with
    calibrated confidence scores.

    If the input exceeds 512 tokens, it is silently truncated and
    `truncated: true` is returned in the response so the caller knows.
    """
    tokenizer   = _state["tokenizer"]
    model       = _state["model"]
    temperature = _state["temperature"]

    # Tokenize — truncation=True silently cuts to 512 tokens.
    # We detect truncation by comparing token count before and after.
    tokens_full     = tokenizer(request.text, add_special_tokens=True)
    was_truncated   = len(tokens_full["input_ids"]) > MAX_TOKENS_BERT

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS_BERT,
        padding=False,
    )

    with torch.no_grad():
        logits = model(**inputs).logits          # shape [1, 3]

    # Temperature scaling: divide logits by T before softmax.
    # This does not change the argmax (predicted class) — only the confidence.
    scaled = logits / max(temperature, 1e-8)
    probs  = F.softmax(scaled, dim=-1).squeeze().tolist()    # [p_neg, p_neu, p_pos]

    pred_idx    = int(torch.argmax(logits).item())
    label       = ID2LABEL[pred_idx]
    confidence  = float(probs[pred_idx])
    prob_dict   = {cls: float(probs[i]) for i, cls in enumerate(LABEL_ORDER)}

    return ClassifyResponse(
        label=label,
        confidence=confidence,
        probabilities=prob_dict,
        truncated=was_truncated,
    )


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(request: SummarizeRequest) -> SummarizeResponse:
    """Summarize an earnings call transcript using GPT-3.5.

    Uses whichever prompt variant scored highest on BERTScore F1 in the
    evaluation run (read from artifacts/summary/summarization_summary.csv).
    Falls back to variant_a if evaluation has not been run yet.

    Responses are cached to data/cache/ — identical inputs never re-call the API.
    """
    client       = _state["openai_client"]
    variant_id   = _state["best_variant"]
    system_prompt = VARIANTS[variant_id]

    try:
        summary = summarize(client, variant_id, system_prompt, request.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}")

    return SummarizeResponse(summary=summary, variant_used=variant_id)
