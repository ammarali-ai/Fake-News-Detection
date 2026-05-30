# Multilingual Fake News Detector — data_prep.py
"""Download, normalize, and merge multilingual fake-news datasets.

Produces a single labelled corpus with the schema expected by train.py and
evaluate.py:

    text, language, label        (label: 0 = Real, 1 = Fake)

Sources (verified, label polarity normalized to 0=Real / 1=Fake):

  English  HF  ErfanMoosaviMonazzah/fake-news-detection-dataset-English
               (cols text/label, where 0=Fake,1=Real -> flipped)
  Urdu     HF  community-datasets/urdu_fake_news
               (cols news/label, where 0=Fake,1=Real -> flipped)
  Spanish  HF  mariagrandury/fake_news_corpus_spanish
               (cols TEXT/CATEGORY bool, where True=Real -> mapped)
  German   seed data/seed/de_seed.csv   (hand-authored, balanced)
  Chinese  seed data/seed/zh_seed.csv   (hand-authored, balanced)
  Korean   seed data/seed/ko_seed.csv   (hand-authored, balanced)

For German/Chinese/Korean the large research corpora (FANG-COVID, CHECKED,
AI-Hub) are not single-file downloads, so a small committed seed set guarantees
coverage offline. To scale any language up, drop extra CSV files with
text/language/label columns into data/raw/<code>/ and they are merged in.

Every loader is wrapped so a single failing source logs a warning and is
skipped rather than aborting the whole run.

Usage:
    python data_prep.py                     # full corpus
    python data_prep.py --subset 200        # cap each language to 200 rows
    python data_prep.py --languages en,ur   # only some languages
"""

import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
SEED_DIR = os.path.join(DATA_DIR, "seed")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# code -> full language name used across the app (model_loader.SUPPORTED_LANGUAGES)
LANG_FULL = {
    "en": "English",
    "ur": "Urdu",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ko": "Korean",
}
ALL_CODES = list(LANG_FULL)

MAX_CHARS = 2000  # keep CSV/tokenizer input sane; BERT truncates to 128 tokens anyway


def log(msg: str) -> None:
    """Print a progress line to stderr so stdout stays parseable."""
    print(msg, file=sys.stderr, flush=True)


def _standardize(texts, labels, code: str) -> pd.DataFrame:
    """Build a clean text/language/label frame, dropping empty/dup rows."""
    df = pd.DataFrame({"text": texts, "label": labels})
    df["text"] = df["text"].astype(str).str.strip().str.slice(0, MAX_CHARS)
    df = df[df["text"].str.len() > 0]
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    df = df.drop_duplicates(subset=["text"])
    df["language"] = LANG_FULL[code]
    return df[["text", "language", "label"]]


def _pick_column(columns, candidates):
    """Return the first matching column name (case-insensitive)."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# --------------------------------------------------------------------------- #
# Per-language loaders
# --------------------------------------------------------------------------- #
def _load_hf_all_splits(repo_id: str):
    """Load every split of an HF dataset and return a single pandas frame."""
    from datasets import load_dataset

    ds = load_dataset(repo_id)
    frames = [split.to_pandas() for split in ds.values()]
    return pd.concat(frames, ignore_index=True)


def load_english() -> pd.DataFrame:
    """English: cols text/label where 0=Fake, 1=Real -> flip to 0=Real,1=Fake."""
    raw = _load_hf_all_splits("ErfanMoosaviMonazzah/fake-news-detection-dataset-English")
    text_col = _pick_column(raw.columns, ["text", "title"])
    return _standardize(raw[text_col], 1 - raw["label"].astype(int), "en")


def load_urdu() -> pd.DataFrame:
    """Urdu: cols news/label where 0=Fake, 1=Real -> flip to 0=Real,1=Fake."""
    raw = _load_hf_all_splits("community-datasets/urdu_fake_news")
    text_col = _pick_column(raw.columns, ["news", "text"])
    return _standardize(raw[text_col], 1 - raw["label"].astype(int), "ur")


def load_spanish() -> pd.DataFrame:
    """Spanish: cols TEXT/CATEGORY(bool) where True=Real -> 0=Real,1=Fake."""
    raw = _load_hf_all_splits("mariagrandury/fake_news_corpus_spanish")
    text_col = _pick_column(raw.columns, ["TEXT", "text", "HEADLINE", "headline"])
    cat_col = _pick_column(raw.columns, ["CATEGORY", "category", "label"])
    cat = raw[cat_col]
    # True / "true" / 1  -> Real (0);  else Fake (1)
    real_mask = cat.astype(str).str.strip().str.lower().isin(["true", "1", "real"])
    labels = (~real_mask).astype(int)
    return _standardize(raw[text_col], labels, "es")


def load_seed(code: str) -> pd.DataFrame:
    """Load a committed hand-authored seed CSV (text/language/label)."""
    path = os.path.join(SEED_DIR, f"{code}_seed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"seed file missing: {path}")
    raw = pd.read_csv(path)
    return _standardize(raw["text"], raw["label"], code)


HF_LOADERS = {"en": load_english, "ur": load_urdu, "es": load_spanish}
SEED_LANGS = {"de", "zh", "ko"}


def load_raw_overrides(code: str) -> pd.DataFrame:
    """Merge any user-supplied CSVs from data/raw/<code>/ (scale-up path)."""
    folder = os.path.join(RAW_DIR, code)
    if not os.path.isdir(folder):
        return pd.DataFrame(columns=["text", "language", "label"])
    frames = []
    for csv_path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        try:
            raw = pd.read_csv(csv_path)
            tcol = _pick_column(raw.columns, ["text", "news", "article", "content"])
            lcol = _pick_column(raw.columns, ["label"])
            if tcol is None or lcol is None:
                log(f"  [raw] skipping {csv_path}: needs text + label columns")
                continue
            frames.append(_standardize(raw[tcol], raw[lcol], code))
            log(f"  [raw] merged {csv_path} ({len(frames[-1])} rows)")
        except Exception as exc:  # noqa: BLE001
            log(f"  [raw] failed to read {csv_path}: {exc}")
    if not frames:
        return pd.DataFrame(columns=["text", "language", "label"])
    return pd.concat(frames, ignore_index=True)


def load_language(code: str) -> pd.DataFrame:
    """Load one language from its primary source plus any raw overrides."""
    frames = []
    try:
        if code in HF_LOADERS:
            frames.append(HF_LOADERS[code]())
        elif code in SEED_LANGS:
            frames.append(load_seed(code))
    except Exception as exc:  # noqa: BLE001
        log(f"  [{code}] primary source failed, skipping it: {exc}")
    try:
        overrides = load_raw_overrides(code)
        if len(overrides):
            frames.append(overrides)
    except Exception as exc:  # noqa: BLE001
        log(f"  [{code}] raw override read failed: {exc}")
    if not frames:
        return pd.DataFrame(columns=["text", "language", "label"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])


# --------------------------------------------------------------------------- #
# Subsetting + splitting
# --------------------------------------------------------------------------- #
def cap_balanced(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Cap a frame to `cap` rows, keeping the real/fake balance even."""
    if cap <= 0 or len(df) <= cap:
        return df
    per_class = cap // 2
    parts = []
    for lbl in (0, 1):
        sub = df[df["label"] == lbl]
        parts.append(sub.sample(n=min(per_class, len(sub)), random_state=seed))
    capped = pd.concat(parts, ignore_index=True)
    return capped.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_split(df: pd.DataFrame, test_size: float, seed: int):
    """Split into train/test stratified by (language, label) when possible."""
    from sklearn.model_selection import train_test_split

    strata = df["language"] + "_" + df["label"].astype(str)
    # train_test_split needs >= 2 members per stratum; fall back if not.
    if strata.value_counts().min() < 2:
        log("  warning: a (language,label) group has <2 rows; using random split")
        strata = None
    return train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True, stratify=strata
    )


def main() -> None:
    """Entry point for the data preparation CLI."""
    parser = argparse.ArgumentParser(description="Build the multilingual fake-news corpus.")
    parser.add_argument("--languages", default=",".join(ALL_CODES),
                        help="comma-separated language codes (default: all six)")
    parser.add_argument("--subset", type=int, default=0,
                        help="cap rows per language (0 = use all)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=DATA_DIR)
    args = parser.parse_args()

    codes = [c.strip() for c in args.languages.split(",") if c.strip()]
    bad = [c for c in codes if c not in LANG_FULL]
    if bad:
        parser.error(f"unknown language code(s): {bad}; valid: {ALL_CODES}")

    log(f"Preparing languages: {[LANG_FULL[c] for c in codes]}")
    frames = []
    for code in codes:
        log(f"[{code}] loading {LANG_FULL[code]} ...")
        df = load_language(code)
        df = cap_balanced(df, args.subset, args.seed)
        n_fake = int((df["label"] == 1).sum())
        n_real = int((df["label"] == 0).sum())
        log(f"[{code}] {len(df)} rows (real={n_real}, fake={n_fake})")
        if len(df):
            frames.append(df)

    if not frames:
        log("ERROR: no data loaded for any language; aborting.")
        sys.exit(1)

    corpus = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
    corpus = corpus.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    train_df, test_df = stratified_split(corpus, args.test_size, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.csv")
    test_path = os.path.join(args.out_dir, "test.csv")
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    log("\n=== Corpus summary ===")
    summary = (
        corpus.groupby(["language", "label"]).size().unstack(fill_value=0)
        .rename(columns={0: "real", 1: "fake"})
    )
    log(summary.to_string())
    log(f"\nTotal: {len(corpus)} rows -> train {len(train_df)} / test {len(test_df)}")
    log(f"Wrote {train_path} and {test_path}")


if __name__ == "__main__":
    main()
