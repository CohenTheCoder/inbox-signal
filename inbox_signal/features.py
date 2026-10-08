"""Step 2: cheap, explainable signals that don't need a model.

Each function returns a value plus the evidence, so the app can say *why* a message scored high.
"""
import re

URGENT = re.compile(
    r"\b(urgent|asap|immediately|right away|emergency|time[- ]sensitive|action required|"
    r"overdue|past due|final notice|last chance to respond|deadline|call me|need you|"
    r"by (?:eod|end of day|tonight|today|tomorrow))\b", re.I)
WHEN = re.compile(
    r"\b(today|tonight|tomorrow|this (?:morning|afternoon|evening|week)|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2}|"
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d{1,2}(?::\d{2})? ?(?:am|pm))\b", re.I)
MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
BULK = re.compile(
    r"\b(unsubscribe|view (?:this email )?in (?:your )?browser|manage (?:your )?preferences|"
    r"email preferences|% off|promo code|limited time|shop now|newsletter|you're receiving this|"
    r"reply stop|txt stop|text stop|msg&data rates)\b", re.I)
NOREPLY = re.compile(r"(no-?reply|donotreply|notifications?@|news(?:letter)?@|marketing@|promo)", re.I)
SHORTCODE = re.compile(r"^\+?\d{5,6}$")
OTP = re.compile(r"\b(code|passcode|otp)\b[^.]{0,40}\b\d{4,8}\b|\b\d{4,8}\b[^.]{0,40}\b(code|passcode)\b", re.I)
QUESTION_TO_YOU = re.compile(r"\b(can|could|would|will|are|do|did|have) you\b[^?]{0,120}\?", re.I)
LINK_SHORTENER = re.compile(r"\b(bit\.ly|tinyurl|t\.co|goo\.gl|rb\.gy|cutt\.ly)/", re.I)


def _first(rx: re.Pattern, text: str) -> str:
    m = rx.search(text)
    return m.group(0) if m else ""


def extract(text: str, sender: str, sender_name: str, vips: list[str]) -> dict:
    """All rule-based signals for one message."""
    low_sender = f"{sender} {sender_name}".lower()
    vip_hit = next((v for v in vips if v and v.lower() in low_sender), "")
    return {
        "urgent": _first(URGENT, text),
        "when": _first(WHEN, text),
        "money": _first(MONEY, text),
        "question": _first(QUESTION_TO_YOU, text),
        "bulk": _first(BULK, text),
        "noreply": bool(NOREPLY.search(sender)),
        "shortcode": bool(SHORTCODE.match(sender.strip())),
        "otp": bool(OTP.search(text)),
        "shortener": _first(LINK_SHORTENER, text),
        "vip": vip_hit,
    }


def gist(subject: str, body: str, max_len: int = 160) -> str:
    """One-line preview: subject + the first real sentence of the body."""
    body = re.sub(r"\s+", " ", body).strip()
    body = re.sub(r"^(hi|hey|hello|dear)\b[^,.!]{0,30}[,.!]\s*", "", body, flags=re.I)
    first = re.split(r"(?<=[.!?])\s", body, maxsplit=1)[0]
    line = f"{subject} - {first}" if subject else first
    return line if len(line) <= max_len else line[: max_len - 1].rstrip() + "…"
