"""Step 3: combine the model's category guess with the rule signals into a 0-100 score.

    base     = sum over categories of P(category) * how-important-that-category-usually-is
    score    = base + boosts (VIP, urgent, deadline, question to you, money)
                    - penalties (bulk mail, no-reply sender, short-code, link shorteners)
    if you've trained a personal model:  score = 50% rules + 50% your model
    if you've fine-tuned DistilBERT and set config.BLEND_FINETUNED: blended the same way

Every boost/penalty is recorded as a human-readable reason.
"""
import numpy as np
import pandas as pd

from . import config, features, personal
from .ingest import Message
from .models import classify, finetuned


def _rule_score(cat_probs: dict, f: dict) -> tuple[float, list[str]]:
    base = sum(p * config.CATEGORIES[c][1] for c, p in cat_probs.items())
    top = max(cat_probs, key=cat_probs.get)
    reasons = [f"Looks like: {top} ({cat_probs[top]:.0%})"]
    s = base
    if f["vip"]:
        s += 0.30; reasons.append(f"From a VIP: {f['vip']}")
    if f["urgent"]:
        s += 0.15; reasons.append(f"Urgent wording: “{f['urgent']}”")
    if f["when"] and top not in ("promo", "social", "scam", "delivery"):
        s += 0.08; reasons.append(f"Mentions a time: “{f['when']}”")
    if f["question"]:
        s += 0.10; reasons.append("Asks you a direct question")
    if f["money"] and top in ("bill", "work", "personal"):
        s += 0.05; reasons.append(f"Money involved: {f['money']}")
    if f["otp"]:
        s -= 0.15; reasons.append("One-time code (only matters if you just asked for it)")
    if f["bulk"]:
        s -= 0.25; reasons.append(f"Bulk-mail sign: “{f['bulk']}”")
    if f["noreply"]:
        s -= 0.10; reasons.append("Sent from a no-reply / notifications address")
    if f["shortcode"]:
        s -= 0.10; reasons.append("Automated short-code sender")
    if f["shortener"]:
        s -= 0.20; reasons.append(f"Shortened link ({f['shortener']}) - common in scams")
    return float(np.clip(s, 0, 1)), reasons


def score_messages(messages: list[Message], vips: list[str] | None = None,
                   use_personal: bool = True) -> pd.DataFrame:
    vips = [v.strip() for v in (vips or []) if v.strip()]
    texts = [m.text for m in messages]
    cats = classify(texts)
    pmodel = personal.load_personal_model() if use_personal else None
    ft = finetuned() if config.BLEND_FINETUNED else None
    ft_out = ft(texts, batch_size=16) if ft is not None else None

    rows = []
    for i, (m, cp) in enumerate(zip(messages, cats)):
        f = features.extract(m.text, m.sender, m.sender_name, vips)
        rule, reasons = _rule_score(cp, f)
        blend = [rule]
        if ft_out is not None:
            o = ft_out[i]
            p_imp = o["score"] if o["label"] in ("LABEL_1", "important") else 1 - o["score"]
            blend.append(p_imp)
            reasons.append(f"Fine-tuned DistilBERT: {p_imp:.0%} important")
        if pmodel is not None:
            p = personal.personal_probability(pmodel, m.sender, m.text)
            blend.append(p)
            reasons.append(f"Your personal model: {p:.0%} important")
        score = 100 * float(np.mean(blend))
        tier = ("important" if score >= config.IMPORTANT_THRESHOLD
                else "maybe" if score >= config.MAYBE_THRESHOLD else "skip")
        top = max(cp, key=cp.get)
        rows.append({
            "id": m.id, "channel": m.channel, "from": m.sender_name or m.sender, "sender": m.sender,
            "subject": m.subject, "timestamp": m.timestamp, "category": top,
            "category_conf": cp[top], "score": round(score, 1), "tier": tier,
            "gist": features.gist(m.subject, m.body), "reasons": reasons,
            "text": m.text, **{f"p_{k}": v for k, v in cp.items()},
        })
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).reset_index(drop=True) if len(df) else df
