# Model card — Multilingual Fake News Detector

## Why this document matters

A model card is the standard way to document an ML system's behaviour, limitations, and ethical considerations. It exists so that anyone considering using the model — a researcher, a product team, an end user — can make an informed decision before relying on its outputs. HuggingFace, Google, and most responsible-AI frameworks recommend or require one.

## Model details

| | |
|---|---|
| **Name** | Multilingual Fake News Detector |
| **Version** | v0.1.0 |
| **Architecture** | Fine-tuned `bert-base-multilingual-cased` with a binary classification head |
| **Format** | TensorFlow SavedModel (`tf.saved_model.load`) |
| **Tokenizer** | `bert-base-multilingual-cased` from HuggingFace Hub |
| **Max sequence length** | 128 tokens |
| **Labels** | `Real` (0), `Fake` (1) |
| **Languages** | English, Urdu, Spanish, German, Chinese, Korean |
| **Framework** | TensorFlow 2.15, Transformers 4.35 |
| **Author** | Ammar Ali ([@ammarali-ai](https://github.com/ammarali-ai)) |
| **License** | MIT |
| **Model repo** | None committed — the SavedModel is produced locally by `train.py` |

### Getting the model

There is **no pre-trained checkpoint** in this repo. You create `./saved_model/` yourself:

1. **Build the corpus:** `python data_prep.py`
2. **Train + export:** `python train.py --csv data/train.csv --epochs 3`
   (or run [`notebooks/train_colab.ipynb`](../notebooks/train_colab.ipynb) on a free Colab GPU
   and unzip the result into `./saved_model/`).

See the [Training section of the README](../README.md#training) for the full workflow. The
optional HF Hub fallback (`HF_MODEL_REPO_ID` / `HF_TOKEN`) is only used if `./saved_model/` is
empty.

## Intended use

### Primary use
Classifying short news text — headlines and short articles up to 128 BERT tokens — as **Real** or **Fake** for screening / triage purposes in any of the six supported languages.

### Intended users
- Researchers studying misinformation across languages
- Newsroom triage tools that surface headlines for human review
- Educational demos illustrating multilingual NLP
- Engineers integrating a fake-news signal into a larger content moderation pipeline (alongside other signals, not as a sole decision maker)

### Out of scope
- **Sole authority for content removal or account action.** Outputs are a probability, not a verdict.
- Long-form documents (>128 tokens after BERT tokenization are truncated).
- Languages beyond the six supported ones (model has limited or no signal for them).
- Audio, video, or image content.
- Real-time fact-checking against external sources — the model only sees text, not the world.

## Performance

Performance depends entirely on **your** training run (dataset size, epochs, hardware), so
no fixed accuracy is claimed here. Measure it after training:

```bash
python evaluate.py --csv data/test.csv
```

A short local-CPU smoke run (small `--subset`, 1 epoch) will score low; a full run on a GPU
over the complete corpus scores much higher. Record your own numbers here.

The script writes `evaluation_results.json` with overall accuracy, precision, recall, F1 (macro + weighted), a per-language breakdown, and a confusion matrix.

## Training data

Multi-source corpus covering English, Urdu, Spanish, German, Chinese, and Korean news text. English/Urdu/Spanish come from public HuggingFace datasets; German/Chinese/Korean ship as small hand-authored seed sets (`data/seed/`). `data_prep.py` normalizes all sources to `text,language,label` (0=Real, 1=Fake), dedupes, and writes a stratified split. See the [README Training section](../README.md#training) for exact sources and how to scale each language up.

## Limitations and biases

- **Domain drift.** Trained on a fixed corpus; performance can degrade as news topics, slang, and adversarial styles evolve. Re-evaluate at least quarterly on a recent sample.
- **Length bias.** Truncation to 128 tokens means long-form context past that boundary is lost. Headlines and short paragraphs are the sweet spot.
- **Language coverage is uneven.** English/Urdu/Spanish have substantial public data; German/Chinese/Korean rely on small seed sets plus mBERT cross-lingual transfer, so their accuracy will lag until you scale them up. Per-language metrics in `evaluation_results.json` make this visible.
- **Topic and source bias.** If certain political topics, sources, or regional dialects dominated training, the model will reflect that. Subgroup audits are recommended before any production use.
- **No factuality check.** "Fake" here means "stylistically consistent with the fake class in training," not "verifiably false." A real-but-poorly-written headline can be misclassified; a well-crafted hoax can slip through.
- **Adversarial robustness is untested.** Small perturbations (synonyms, paraphrases) may flip predictions; do not deploy as an adversary-facing filter.

## Ethical considerations

- **Human in the loop.** Outputs should inform human reviewers, not replace them. Automating high-impact decisions (account bans, content removal) from this signal alone risks both false positives (silencing legitimate speech) and false negatives (letting misinformation through).
- **Transparency to end users.** If you embed this model in a product that surfaces predictions, label them as model-generated and disclose your measured accuracy bounds.
- **Multilingual fairness.** Per-language accuracy may differ. Publishing the per-language breakdown (not just the headline number) is required to use this model responsibly across populations.
- **Privacy.** The model only sees the text you pass it. Don't pass content the user has a reasonable expectation of privacy over without consent.
- **Misuse.** Could be used to censor political speech under the guise of "fact-checking." Operators are responsible for the use cases they enable.

## Evaluation methodology

- Hold-out test split disjoint from training and validation.
- Metrics: accuracy, macro / weighted precision, recall, F1, plus a confusion matrix.
- Per-language slice metrics (all six languages) reported separately so cross-language averages don't hide language-level regressions.
- See [`evaluate.py`](../evaluate.py) for the exact implementation.

## Caveats and recommendations

- Run `evaluate.py` on a sample of your own current data before relying on any headline number — distribution drift between the training corpus and your live traffic is the most common reason published numbers don't hold.
- Periodically retrain or fine-tune on recent labelled data.
- Audit subgroup performance (by topic, source, dialect) and publish the breakdown alongside the headline accuracy.
