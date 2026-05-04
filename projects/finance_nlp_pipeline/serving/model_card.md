# Model Card — FinBERT Financial Sentiment Classifier

<!-- WHAT IS A MODEL CARD AND WHY DOES IT EXIST? (Learning goal)

A model card is a short document that travels with a deployed model and answers
the questions a new user or auditor would ask before trusting it:
  - What was this trained to do?
  - What data was it trained on, and when?
  - How well does it actually perform?
  - Where will it fail?

Model cards were proposed by Mitchell et al. (2019) as a response to a pattern
where ML models were deployed in contexts the authors never intended, producing
harm that would have been obvious to anyone who read a clear limitation notice.

Responsible deployment means: the person operating the model in production
can read this card and make an informed decision about whether the model is
appropriate for their use case — and if it fails, they know exactly where
the failure was documented in advance.

In a regulated industry like finance, model cards are not optional — they
are a paper trail. Model Risk Management (MRM) frameworks (SR 11-7 in the US,
SS1/23 in the UK) require documented evidence of model scope, validation, and
known limitations before a model can be used in a decision-making capacity.
-->

---

## Model details

| Field | Value |
|---|---|
| **Name** | FinBERT Financial Sentiment Classifier |
| **Version** | 1.0.0 |
| **Type** | Transformer-based sequence classifier (fine-tuned BERT) |
| **Base model** | ProsusAI/finbert (110M parameters) |
| **Fine-tuning** | 3 epochs, AdamW lr=2e-5, early stopping on val macro-F1 |
| **Calibration** | Temperature scaling fit on validation set |
| **Output** | 3-class label (negative / neutral / positive) + calibrated probabilities |
| **Serving** | FastAPI REST endpoint — POST /classify |
| **Checkpoint** | `models/checkpoints/finbert_best/` |

---

## Intended use

**Primary use case:** Classifying the sentiment of short financial news sentences
and earnings call excerpts (up to ~512 WordPiece tokens) into three categories:
negative, neutral, or positive.

**Target users:** Quantitative analysts, NLP engineers, and researchers building
financial text processing pipelines.

**Deployment context:** Local or private-cloud REST API serving internal
research or trading signal generation workflows.

---

## Out-of-scope uses

| Use case | Reason |
|---|---|
| Documents longer than 512 tokens | Input is silently truncated; the tail of a long document (often the Q&A and risk section) is ignored |
| Non-English financial text | Trained and evaluated on English only; performance on other languages is unknown |
| Non-financial domains | PhraseBank vocabulary is finance-specific; general news, social media, or product reviews are out of distribution |
| Legal or regulatory decisions | Model errors are not audited for downstream impact; human review required |
| Real-time trading without human oversight | Model confidence is calibrated but not zero-error; autonomous execution without monitoring is inadvisable |

---

## Training data

| Field | Value |
|---|---|
| **Dataset** | Financial PhraseBank — `sentences_allagree` subset |
| **Source** | Malo et al. (2014), loaded via HuggingFace `datasets` |
| **Why `sentences_allagree`** | Only sentences where 100% of annotators agreed on the label — highest-confidence subset |
| **Split** | 70% train / 10% val / 20% test, stratified by label, seed=42 |
| **Train size** | ~2,960 sentences |
| **Val size** | ~420 sentences |
| **Test size** | ~840 sentences |
| **Label distribution** | ~59% neutral, ~28% positive, ~13% negative |
| **Date range** | PhraseBank annotations collected circa 2011–2013 |

---

## Performance — test set

*Fill in from `artifacts/summary/classification_summary.csv` after running
`python -m evaluation.metrics`.*

| Metric | Value |
|---|---|
| Accuracy | [run evaluation] |
| Macro-F1 | [run evaluation] |
| F1 — negative | [run evaluation] |
| F1 — neutral | [run evaluation] |
| F1 — positive | [run evaluation] |
| ECE (15 bins) | [run evaluation] |
| Test set size | ~840 sentences |

**Comparison to baselines** *(from `classification_summary.csv`)*:

| Model | Macro-F1 | ECE |
|---|---|---|
| TF-IDF + LogReg | [run eval] | [run eval] |
| FinBERT fine-tuned | [run eval] | [run eval] |
| GPT-4 few-shot | [run eval] | [run eval] |

---

## Known limitations

**1. Neutral class is the hardest to predict.**
Neutral sentences report facts without directional language ("The company
released its quarterly report"). The model most often confuses neutral with
either positive or negative. Inspect `artifacts/summary/top_errors_finbert.csv`
for examples.

**2. Performance degrades on earnings call language (OOD observation).**
Financial PhraseBank consists of short, edited news headlines. Earnings call
transcripts (ECTSum) are longer, more conversational, and contain hedged
forward-looking language ("we expect to see improvement in the second half
pending macro conditions") that is syntactically different from training data.
This is a documented out-of-distribution (OOD) risk — the model was not
validated on earnings call text, so confidence scores on that domain should
be treated with additional caution.

**3. Calibration is approximate.**
Temperature scaling fits a single scalar to the validation set. It improves
calibration on the PhraseBank test set but does not guarantee calibration on
new domains. Check the reliability diagram at
`artifacts/plots/reliability_finbert.png` before deploying to a new data source.

**4. Label quality reflects 2011–2013 annotator consensus.**
Sentiment norms in financial text can shift over time (e.g., language around
ESG, cryptocurrency, or AI-related guidance). The model may be systematically
mis-calibrated on post-2013 financial vocabulary.

**5. Input truncation.**
Inputs exceeding 512 WordPiece tokens are silently truncated. The `/classify`
endpoint returns `"truncated": true` when this occurs. For long documents,
consider a chunking strategy (classify paragraphs, aggregate by majority or
confidence-weighted vote).

---

## Ethical considerations

- **Bias:** Model performance may vary across industries, geographies, or
  company sizes not well-represented in PhraseBank. No fairness audit has
  been conducted across these subgroups.
- **Misuse risk:** Automated sentiment signals should not be the sole input
  to trading decisions without human oversight and independent validation.
- **Data provenance:** PhraseBank was annotated by researchers at Aalto
  University. See Malo et al. (2014) for full annotation methodology.

---

## How to run

```powershell
# Start the API server
uvicorn serving.app:app --reload

# Classify a sentence
curl -X POST http://localhost:8000/classify \
     -H "Content-Type: application/json" \
     -d '{"text": "The company reported record quarterly earnings."}'

# Summarize an earnings call excerpt
curl -X POST http://localhost:8000/summarize \
     -H "Content-Type: application/json" \
     -d '{"text": "<paste earnings call text here>"}'

# Health check
curl http://localhost:8000/health
```

---

## Citation

```
Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
Good debt or bad debt: Detecting semantic orientations in economic texts.
Journal of the American Society for Information Science and Technology, 65(4), 782-796.

Mitchell, M., et al. (2019). Model cards for model reporting.
Proceedings of the conference on fairness, accountability, and transparency.
```
