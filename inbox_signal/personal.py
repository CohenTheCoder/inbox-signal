"""Step 4 (optional): learn YOUR definition of important from thumbs up / down.

Every click in the app appends a row to data/feedback.csv. Once there are enough labels, we
train a TF-IDF + logistic regression model on them. It is tiny, trains in under a second, and
gets blended into the score - so the system adapts to you over time (human-in-the-loop ML).
"""
from pathlib import Path

import pandas as pd

from . import config

ROOT = Path(__file__).resolve().parent.parent


def feedback_path() -> Path:
    return ROOT / config.FEEDBACK_PATH


def add_feedback(msg_id: str, text: str, sender: str, important: bool) -> None:
    path = feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"id": msg_id, "sender": sender, "text": text, "important": int(important)}])
    if path.exists():
        df = pd.concat([pd.read_csv(path), row]).drop_duplicates("id", keep="last")
    else:
        df = row
    df.to_csv(path, index=False)


def load_feedback() -> pd.DataFrame:
    path = feedback_path()
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["id", "sender", "text", "important"])


def build_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    # sender is prepended so the model can learn "messages from Mom matter"
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000),
    )


def train_personal_model() -> dict:
    import joblib
    from sklearn.model_selection import cross_val_score

    df = load_feedback()
    if len(df) < config.MIN_FEEDBACK_TO_TRAIN or df["important"].nunique() < 2:
        return {"trained": False, "reason": f"Need {config.MIN_FEEDBACK_TO_TRAIN}+ labels with both "
                                            f"important and not-important examples (have {len(df)})."}
    X = ("from " + df["sender"].fillna("") + " : " + df["text"].fillna("")).tolist()
    y = df["important"].astype(int).to_numpy()
    model = build_pipeline()
    folds = min(5, int(min(y.sum(), len(y) - y.sum())))
    cv = cross_val_score(model, X, y, cv=folds).mean() if folds >= 2 else float("nan")
    model.fit(X, y)
    out = ROOT / config.PERSONAL_MODEL_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    return {"trained": True, "n": len(df), "cv_accuracy": cv}


def load_personal_model():
    import joblib

    path = ROOT / config.PERSONAL_MODEL_PATH
    return joblib.load(path) if path.exists() else None


def personal_probability(model, sender: str, text: str) -> float:
    return float(model.predict_proba([f"from {sender} : {text}"])[0, 1])
