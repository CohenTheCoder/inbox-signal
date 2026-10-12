"""Export the scored sample inbox for the browser playground (playground/data.json).

The playground runs a smaller zero-shot model live in the browser for messages you type.
This adds the full-size model's scores for the 30-message sample, plus the hand labels,
so the page can show the digest and a precision/recall threshold lab.

    python scripts/export_playground.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inbox_signal import config, load_any, score_messages  # noqa: E402


def main() -> None:
    path = ROOT / "data/sample/messages.json"
    msgs = load_any(path)
    expected = {m["id"]: m["expected_important"] for m in json.loads(path.read_text())}
    scored = score_messages(msgs, vips=["Mom"], use_personal=False)
    rows = []
    for _, r in scored.iterrows():
        rows.append({
            "id": r["id"], "channel": r["channel"], "from": r["from"], "subject": r["subject"],
            "text": r["text"], "gist": r["gist"], "category": r["category"], "score": r["score"],
            "tier": r["tier"], "reasons": r["reasons"], "expected": expected[r["id"]],
            "probs": {k[2:]: round(v, 4) for k, v in r.items() if k.startswith("p_")},
        })
    data = {
        "model": config.ZERO_SHOT_MODEL,
        "categories": {k: {"label": v[0], "weight": v[1]} for k, v in config.CATEGORIES.items()},
        "hypothesis": config.HYPOTHESIS,
        "important_threshold": config.IMPORTANT_THRESHOLD,
        "maybe_threshold": config.MAYBE_THRESHOLD,
        "messages": rows,
    }
    out = ROOT / "playground/data.json"
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e3:.0f} kB, {len(rows)} messages)")


if __name__ == "__main__":
    main()
