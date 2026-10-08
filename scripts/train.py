"""Train and compare three ways of deciding "important or not".

  A. Zero-shot + rules   - no training at all (the default app behaviour)
  B. TF-IDF + logistic regression   - classic ML baseline, trained in < 1 second
  C. Fine-tuned DistilBERT  (--distilbert)  - a small transformer trained on our labels

Train set: data/synthetic_train.csv  (run scripts/make_dataset.py first)
Test set:  data/sample/messages.json - 30 hand-written messages the models never saw, written
           in a different style from the templates. Testing on differently-generated data is a
           more honest check than a random split of the synthetic set (which would be ~100%).

We care most about RECALL on important messages: missing your landlord's text is worse than
showing you one extra promo.

Usage:
    python scripts/make_dataset.py
    python scripts/train.py                 # A + B
    python scripts/train.py --distilbert    # A + B + C (~5-10 min on a laptop CPU)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inbox_signal import config, load_any  # noqa: E402
from inbox_signal.personal import build_pipeline  # noqa: E402
from inbox_signal.scorer import score_messages  # noqa: E402


def report(name: str, y_true, y_pred) -> dict:
    r = {"model": name,
         "accuracy": accuracy_score(y_true, y_pred),
         "precision": precision_score(y_true, y_pred, zero_division=0),
         "recall": recall_score(y_true, y_pred, zero_division=0),
         "f1": f1_score(y_true, y_pred, zero_division=0)}
    print(f"  {name:28s} acc {r['accuracy']:.2f}  precision {r['precision']:.2f}  "
          f"recall {r['recall']:.2f}  F1 {r['f1']:.2f}")
    return r


def as_input(sender: str, text: str) -> str:
    return f"from {sender} : {text}"


def train_distilbert(train: pd.DataFrame, epochs: int = 2):
    """A plain PyTorch fine-tuning loop - every step visible, no Trainer magic."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(config.FINETUNE_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FINETUNE_BASE, num_labels=2, id2label={0: "not_important", 1: "important"},
        label2id={"not_important": 0, "important": 1})
    texts = [as_input(s, t) for s, t in zip(train["sender_name"], train["text"])]
    enc = tok(texts, truncation=True, max_length=96, padding=True, return_tensors="pt")
    labels = torch.tensor(train["important"].to_numpy())
    data = list(zip(enc["input_ids"], enc["attention_mask"], labels))
    loader = DataLoader(data, batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for ids, mask, y in loader:
            out = model(input_ids=ids, attention_mask=mask, labels=y)  # loss = cross-entropy
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total += out.loss.item()
        print(f"    epoch {epoch + 1}: mean loss {total / len(loader):.4f}")
    model.eval()
    out_dir = ROOT / config.FINETUNED_DIR
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"    saved -> {out_dir.relative_to(ROOT)} (set BLEND_FINETUNED = True in config.py to use it in the app)")
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilbert", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    train_path = ROOT / "data/synthetic_train.csv"
    if not train_path.exists():
        sys.exit("Run  python scripts/make_dataset.py  first.")
    train = pd.read_csv(train_path)
    test_raw = json.loads((ROOT / "data/sample/messages.json").read_text())
    msgs = load_any(ROOT / "data/sample/messages.json")
    y_test = np.array([m["expected_important"] for m in test_raw])
    test_inputs = [as_input(m.sender_name, m.text) for m in msgs]
    print(f"Train: {len(train)} synthetic messages | Test: {len(msgs)} hand-written messages\n")

    results = []
    print("A. Zero-shot + rules (no training)")
    scored = score_messages(msgs, use_personal=False).set_index("id").loc[[m.id for m in msgs]]
    results.append(report("zero-shot + rules", y_test, (scored["tier"] == "important").astype(int)))

    print("B. TF-IDF + logistic regression")
    clf = build_pipeline().fit([as_input(s, t) for s, t in zip(train["sender_name"], train["text"])],
                               train["important"])
    results.append(report("tf-idf + logreg", y_test, clf.predict(test_inputs)))

    if args.distilbert:
        import torch

        print("C. Fine-tuning DistilBERT")
        model, tok = train_distilbert(train, args.epochs)
        with torch.no_grad():
            enc = tok(test_inputs, truncation=True, max_length=96, padding=True, return_tensors="pt")
            pred = model(**enc).logits.argmax(-1).numpy()
        results.append(report("fine-tuned distilbert", y_test, pred))

    out = ROOT / "docs/results.md"
    table = pd.DataFrame(results).round(2)
    out.write_text("# Model comparison (test = 30 hand-written messages)\n\n" + table.to_markdown(index=False) + "\n")
    print(f"\nSaved table -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
