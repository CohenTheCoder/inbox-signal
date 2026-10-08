"""Load the Hugging Face zero-shot model once and classify messages in batches."""
from functools import lru_cache
from pathlib import Path

from . import config

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def zero_shot():
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=config.ZERO_SHOT_MODEL)


def classify(texts: list[str], batch_size: int = 8) -> list[dict[str, float]]:
    """For each text: {category_key: probability}. Probabilities sum to 1 across categories."""
    if not texts:
        return []
    keys = list(config.CATEGORIES)
    labels = [config.CATEGORIES[k][0] for k in keys]
    label_to_key = dict(zip(labels, keys))
    out = zero_shot()(list(texts), candidate_labels=labels,
                      hypothesis_template=config.HYPOTHESIS, batch_size=batch_size)
    if isinstance(out, dict):
        out = [out]
    return [{label_to_key[l]: float(s) for l, s in zip(o["labels"], o["scores"])} for o in out]


@lru_cache(maxsize=1)
def finetuned():
    """The DistilBERT importance model from scripts/train.py --distilbert, if you trained one."""
    path = ROOT / config.FINETUNED_DIR
    if not path.exists():
        return None
    from transformers import pipeline

    return pipeline("text-classification", model=str(path), truncation=True, max_length=256)
