"""GPT-3.5 summarization of earnings call transcripts — two prompt variants.

ROUGE vs BERTScore (Learning goal)
-----------------------------------
We evaluate summaries with two metrics that measure quality in fundamentally
different ways — understanding both is essential for any NLP evaluation work.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
  Counts n-gram *overlap* between the generated summary and the reference.
  ROUGE-1 : unigram overlap  (individual words)
  ROUGE-2 : bigram overlap   (two-word phrases)
  ROUGE-L : longest common subsequence (respects word order without requiring
            exact contiguous matches)

  Strength  — fast, deterministic, no model required, widely published
  Weakness  — purely lexical: "revenue increased" and "earnings grew" score
              zero overlap even though they mean the same thing. A fluent,
              accurate paraphrase is punished as if it were wrong.

BERTScore:
  Embeds every token in both the generated and reference text with a
  pre-trained transformer (default: roberta-large). For each token in the
  generation it finds the most similar token in the reference (cosine
  similarity in embedding space), and vice versa. F1 is the harmonic mean
  of precision and recall over these soft matches.

  Strength  — captures semantic equivalence; synonyms and paraphrases score
              high; better correlation with human judgement than ROUGE
  Weakness  — slower, requires a GPU or patience, less interpretable,
              scores are not bounded in [0,1] without baseline rescaling

WHY BOTH MATTER
  ROUGE rewards transcription; BERTScore rewards understanding.
  A summary that copies bullet-pointed numbers verbatim scores high on ROUGE.
  A summary that synthesises the narrative scores high on BERTScore.
  Comparing the two variants across both metrics tells us whether Variant B's
  structured extraction produces genuinely better understanding or just
  different surface phrasing.

PROMPT SENSITIVITY
  The gap between Variant A and Variant B metrics is your *prompt sensitivity
  result*: how much does the framing of the instruction change output quality
  for the same underlying model? This is a core finding in prompt engineering
  research — small wording changes can shift ROUGE by 2–5 points on financial
  text, which is meaningful in a production setting.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from bert_score import score as bert_score_fn
from dotenv import load_dotenv
from openai import OpenAI
from rouge_score import rouge_scorer

from data.load import ROOT

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR    = ROOT / "data" / "cache"
SUMMARY_DIR  = ROOT / "artifacts" / "summary"
PREDS_DIR    = ROOT / "data" / "predictions"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GPT_MODEL      = "gpt-3.5-turbo-0125"
TEMPERATURE    = 0                  # deterministic; same transcript -> same summary
MAX_TOKENS     = 300                # summaries should be concise
MAX_WORDS      = 2500               # truncate very long transcripts to control cost
N_QUALITATIVE  = 5                  # qualitative examples to save
SEED           = 42

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
# Variant A — analyst-facing bullet points
VARIANT_A_SYSTEM = (
    "You are a financial analyst assistant. "
    "Summarize the following earnings call transcript in exactly 3 concise bullet points "
    "suitable for a buy-side analyst. Focus on key financial results, guidance, and tone."
)

# Variant B — structured extraction
VARIANT_B_SYSTEM = (
    "You are a financial analyst assistant. "
    "Extract and structure the following from the earnings call transcript: "
    "(1) Key financial guidance for the next period, "
    "(2) Main risks mentioned by management, "
    "(3) Overall outlook and sentiment. "
    "Be specific and use numbers where available."
)

VARIANTS = {
    "variant_a": VARIANT_A_SYSTEM,
    "variant_b": VARIANT_B_SYSTEM,
}


# ---------------------------------------------------------------------------
# Caching — variant-aware
# ---------------------------------------------------------------------------

def _cache_key(variant_id: str, text: str) -> str:
    """SHA256 of variant + text so each variant has its own cache entry.

    Including the variant in the key is critical: the same transcript produces
    different outputs under different prompts, so they must be cached separately.
    """
    payload = f"{variant_id}:{text}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(variant_id: str, text: str) -> Path:
    return CACHE_DIR / f"sum_{_cache_key(variant_id, text)}.json"


def _load_cache(variant_id: str, text: str) -> str | None:
    path = _cache_path(variant_id, text)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["summary"]
    return None


def _save_cache(variant_id: str, text: str, summary: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(variant_id, text).write_text(
        json.dumps({"summary": summary}, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def truncate(text: str, max_words: int = MAX_WORDS) -> str:
    """Truncate to max_words to keep within GPT context and control cost.

    Earnings call transcripts can exceed 10,000 words. We take the first
    max_words words, which typically covers the prepared remarks and the
    start of Q&A — the most information-dense sections.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [truncated]"


def summarize(
    client: OpenAI,
    variant_id: str,
    system_prompt: str,
    text: str,
    retries: int = 3,
) -> str:
    """Call GPT-3.5 with caching. Returns the summary string."""
    cached = _load_cache(variant_id, text)
    if cached is not None:
        return cached

    truncated = truncate(text)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": truncated},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            summary = response.choices[0].message.content.strip()
            _save_cache(variant_id, text, summary)
            return summary
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise exc
    return ""


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute mean ROUGE-1, ROUGE-2, ROUGE-L F1 across all samples."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)    # (reference, prediction) order in rouge_score
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
    }


def compute_bertscore(
    predictions: list[str],
    references: list[str],
) -> float:
    """Compute mean BERTScore F1 across all samples.

    bert_score returns precision, recall, F1 tensors of shape [n_samples].
    We average F1 across samples.  rescale_with_baseline=True maps scores
    to a more interpretable [0,1]-like range using a pre-computed baseline
    (the score of a random English baseline corpus).
    """
    print("  [bertscore] computing embeddings (may take 1-2 min) ...")
    _, _, F1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )
    return float(F1.mean().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # 1. Load ECTSum test split only
    # ------------------------------------------------------------------
    ectsum = pd.read_csv(ROOT / "data" / "ectsum.csv")
    test_df = ectsum[ectsum["split"] == "test"].reset_index(drop=True)
    print(f"[summarizer] ECTSum test set: {len(test_df)} transcripts")

    # ------------------------------------------------------------------
    # 2. Generate summaries for both variants
    # ------------------------------------------------------------------
    generated: dict[str, list[str]] = {}
    references: list[str] = test_df["summary"].tolist()

    for variant_id, system_prompt in VARIANTS.items():
        print(f"\n[summarizer] running {variant_id} ...")
        summaries = []
        cache_hits = 0

        for i, row in test_df.iterrows():
            was_cached = _load_cache(variant_id, row["text"]) is not None
            summary = summarize(client, variant_id, system_prompt, row["text"])
            summaries.append(summary)
            if was_cached:
                cache_hits += 1
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(test_df)} | cache_hits={cache_hits}")

        generated[variant_id] = summaries
        print(f"[summarizer] {variant_id} done | cache_hits={cache_hits}/{len(test_df)}")

    # ------------------------------------------------------------------
    # 3. Compute ROUGE and BERTScore for each variant
    # ------------------------------------------------------------------
    print("\n[summarizer] evaluating metrics ...")
    results: dict[str, dict] = {}

    for variant_id, summaries in generated.items():
        rouge  = compute_rouge(summaries, references)
        bs_f1  = compute_bertscore(summaries, references)
        results[variant_id] = {**rouge, "bertscore_f1": bs_f1}
        print(
            f"  {variant_id}: "
            f"R1={rouge['rouge1']:.4f}  R2={rouge['rouge2']:.4f}  "
            f"RL={rouge['rougeL']:.4f}  BertScore={bs_f1:.4f}"
        )

    # ------------------------------------------------------------------
    # 4. Report winner per metric
    # ------------------------------------------------------------------
    print("\n[summarizer] prompt sensitivity — winner per metric:")
    metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]
    for metric in metrics:
        scores = {v: results[v][metric] for v in VARIANTS}
        winner = max(scores, key=scores.get)
        delta  = abs(scores["variant_a"] - scores["variant_b"])
        print(f"  {metric:15s} -> winner: {winner}  (Δ={delta:.4f})")

    # ------------------------------------------------------------------
    # 5. Save summary metrics table
    # ------------------------------------------------------------------
    rows = []
    for variant_id, m in results.items():
        rows.append({"variant": variant_id, **m})
    summary_df = pd.DataFrame(rows)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / "summarization_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[summarizer] metrics table -> {summary_path}")

    # ------------------------------------------------------------------
    # 6. Save 5 qualitative examples
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(test_df), size=N_QUALITATIVE, replace=False)
    qual_rows  = []
    for idx in sample_idx:
        row = test_df.iloc[idx]
        qual_rows.append({
            "transcript_excerpt":  " ".join(row["text"].split()[:80]) + " ...",
            "reference_summary":   row["summary"],
            "variant_a_output":    generated["variant_a"][idx],
            "variant_b_output":    generated["variant_b"][idx],
        })
    qual_df = pd.DataFrame(qual_rows)
    qual_path = ROOT / "data" / "qualitative_examples.csv"
    qual_df.to_csv(qual_path, index=False)
    print(f"[summarizer] qualitative examples -> {qual_path}")

    # ------------------------------------------------------------------
    # 7. Log both variants to MLflow as separate runs
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("finance_nlp_summarization")

    for variant_id, m in results.items():
        with mlflow.start_run(run_name=f"gpt35_{variant_id}"):
            mlflow.log_params({
                "model":      GPT_MODEL,
                "variant":    variant_id,
                "max_words":  MAX_WORDS,
                "max_tokens": MAX_TOKENS,
                "n_samples":  len(test_df),
            })
            mlflow.log_metrics({
                "rouge1":        m["rouge1"],
                "rouge2":        m["rouge2"],
                "rougeL":        m["rougeL"],
                "bertscore_f1":  m["bertscore_f1"],
            })
            mlflow.log_artifact(str(summary_path))

    print("\n[summarizer] done.")
    print("  -> View MLflow UI: mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    main()
