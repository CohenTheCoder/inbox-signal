"""Step 5: build the digest and send it to you.

Delivery options (pick with --deliver or in the app):
- file:  writes digests/digest-YYYY-MM-DD-HHMM.md  (default, nothing leaves your machine)
- ntfy:  free push notification to your phone. Install the ntfy app, subscribe to a secret
         topic name, and set NTFY_TOPIC=that-name.  https://ntfy.sh
- email: SMTP. Set SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, DIGEST_TO.
- slack: an incoming-webhook URL in SLACK_WEBHOOK_URL.
"""
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
ICONS = {"email": "✉️", "sms": "💬"}


def build_markdown(scored: pd.DataFrame, include_maybe: bool = True) -> str:
    imp = scored[scored["tier"] == "important"]
    maybe = scored[scored["tier"] == "maybe"]
    skipped = len(scored) - len(imp) - (len(maybe) if include_maybe else 0)
    lines = [f"# Inbox Signal · {datetime.now():%a %b %d, %I:%M %p}",
             f"**{len(imp)} important** out of {len(scored)} messages · {skipped} filtered out", ""]
    if len(imp):
        lines.append("## Needs your attention")
        for _, r in imp.iterrows():
            lines.append(f"- {ICONS.get(r['channel'], '')} **{r['from']}** ({r['category']}, {r['score']:.0f}) - {r['gist']}")
            lines.append(f"  - _why: {'; '.join(r['reasons'][1:4]) or r['reasons'][0]}_")
    if include_maybe and len(maybe):
        lines += ["", "## Worth a glance"]
        for _, r in maybe.iterrows():
            lines.append(f"- {ICONS.get(r['channel'], '')} {r['from']} - {r['gist']}")
    return "\n".join(lines)


def build_short(scored: pd.DataFrame, max_items: int = 8) -> str:
    """Compact version for phone push notifications."""
    imp = scored[scored["tier"] == "important"].head(max_items)
    if imp.empty:
        return f"Nothing important in {len(scored)} new messages."
    return "\n".join(f"• {r['from']}: {r['gist'][:90]}" for _, r in imp.iterrows())


def deliver(scored: pd.DataFrame, method: str = "file") -> str:
    """Send the digest. Returns a short status string."""
    md = build_markdown(scored)
    n = int((scored["tier"] == "important").sum())
    title = f"Inbox Signal: {n} important"
    if method == "file":
        out = ROOT / "digests" / f"digest-{datetime.now():%Y-%m-%d-%H%M}.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text(md)
        return f"Saved {out.relative_to(ROOT)}"
    if method == "ntfy":
        topic = os.environ["NTFY_TOPIC"]
        requests.post(f"https://ntfy.sh/{topic}", data=build_short(scored).encode(),
                      headers={"Title": title, "Priority": "high" if n else "low"}, timeout=15).raise_for_status()
        return f"Pushed to ntfy topic {topic}"
    if method == "email":
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = title, os.environ["SMTP_USER"], os.environ["DIGEST_TO"]
        msg.set_content(md)
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", 587))) as s:
            s.starttls()
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            s.send_message(msg)
        return f"Emailed to {os.environ['DIGEST_TO']}"
    if method == "slack":
        requests.post(os.environ["SLACK_WEBHOOK_URL"], json={"text": md}, timeout=15).raise_for_status()
        return "Posted to Slack"
    raise ValueError(f"Unknown delivery method {method!r}")
