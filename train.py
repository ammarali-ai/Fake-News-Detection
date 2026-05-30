# Multilingual Fake News Detector — train.py
"""Fine-tune bert-base-multilingual-cased and export a TensorFlow SavedModel.

The exported SavedModel is written with a ``serving_default`` signature that
matches model_loader.predict() exactly: it accepts ``input_ids`` and
``attention_mask`` (int32, shape [None, MAX_LEN]) and returns ``{"logits": ...}``.

The same script runs locally (CPU) and on Colab (GPU) — only the flags differ.

Usage (local CPU smoke test):
    python train.py --csv data/train.csv --subset 200 --epochs 1

Usage (full / Colab GPU):
    python train.py --csv data/train.csv --epochs 3 --batch-size 16
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFBertForSequenceClassification

MODEL_NAME = "bert-base-multilingual-cased"
DEFAULT_OUT = "./saved_model"
MAX_LEN = 128  # must match model_loader.MAX_LENGTH
NUM_LABELS = 2  # 0 = Real, 1 = Fake (matches model_loader.LABELS)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune mBERT for fake-news detection.")
    p.add_argument("--csv", default="data/train.csv",
                   help="training CSV with columns text,language,label")
    p.add_argument("--out", default=DEFAULT_OUT, help="SavedModel output directory")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--subset", type=int, default=0,
                   help="cap total training rows (0 = use all)")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_frame(csv_path: str, subset: int, seed: int) -> pd.DataFrame:
    """Load and validate the training CSV."""
    if not os.path.exists(csv_path):
        log(f"ERROR: training CSV not found: {csv_path}")
        log("Run `python data_prep.py` first to build data/train.csv.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        log(f"ERROR: CSV missing columns: {sorted(missing)}")
        sys.exit(1)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    if subset > 0 and len(df) > subset:
        df = df.sample(n=subset, random_state=seed)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_dataset(encodings, labels, batch_size: int, shuffle: bool):
    """Build a tf.data.Dataset of ({input_ids, attention_mask}, label)."""
    ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        },
        labels,
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(labels), 2048), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def export_savedmodel(model, out_dir: str, max_len: int) -> None:
    """Save the model with a serving signature matching model_loader.predict()."""
    @tf.function(input_signature=[
        tf.TensorSpec([None, max_len], tf.int32, name="input_ids"),
        tf.TensorSpec([None, max_len], tf.int32, name="attention_mask"),
    ])
    def serving_fn(input_ids, attention_mask):
        outputs = model(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            training=False,
        )
        return {"logits": outputs.logits}

    os.makedirs(out_dir, exist_ok=True)
    tf.saved_model.save(model, out_dir, signatures={"serving_default": serving_fn})
    log(f"Exported SavedModel to {out_dir}")


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    df = load_frame(args.csv, args.subset, args.seed)
    n_real = int((df["label"] == 0).sum())
    n_fake = int((df["label"] == 1).sum())
    log(f"Training rows: {len(df)} (real={n_real}, fake={n_fake})")
    if n_real == 0 or n_fake == 0:
        log("ERROR: training data must contain both classes (real and fake).")
        sys.exit(1)

    log(f"Loading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TFBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    texts = df["text"].astype(str).tolist()
    labels = df["label"].to_numpy(dtype=np.int32)
    encodings = tokenizer(
        texts,
        max_length=args.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    encodings = {
        "input_ids": encodings["input_ids"].astype(np.int32),
        "attention_mask": encodings["attention_mask"].astype(np.int32),
    }

    # train / validation split
    n_val = max(1, int(len(labels) * args.val_split))
    val_enc = {k: v[:n_val] for k, v in encodings.items()}
    val_lab = labels[:n_val]
    tr_enc = {k: v[n_val:] for k, v in encodings.items()}
    tr_lab = labels[n_val:]

    train_ds = make_dataset(tr_enc, tr_lab, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_enc, val_lab, args.batch_size, shuffle=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    log(f"Training for {args.epochs} epoch(s), batch size {args.batch_size} ...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    export_savedmodel(model, args.out, args.max_len)
    log("Done. You can now run evaluate.py / app.py / api.py.")


if __name__ == "__main__":
    main()
