"""Step 1: load messages from wherever they live into one common shape.

Supported sources
- JSON list of messages (the sample data)
- .eml files / a folder of them, and .mbox archives (Gmail "Takeout", Apple Mail export)
- Any IMAP inbox (Gmail, Outlook, iCloud...) with an *app password*
- Text messages: a CSV (timestamp, sender, body), an Android "SMS Backup & Restore" XML file,
  or the macOS Messages database (~/Library/Messages/chat.db)
"""
import csv
import email
import imaplib
import json
import mailbox
import re
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

from bs4 import BeautifulSoup


@dataclass
class Message:
    id: str
    channel: str        # "email" | "sms"
    sender: str         # address or phone number
    sender_name: str
    subject: str
    body: str
    timestamp: str      # ISO 8601

    @property
    def text(self) -> str:
        """What the models read: subject + body, trimmed."""
        t = f"{self.subject}. {self.body}" if self.subject else self.body
        return re.sub(r"\s+", " ", t).strip()[:1500]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------- JSON ----------------
def load_json(path: str | Path) -> list[Message]:
    rows = json.loads(Path(path).read_text())
    fields = Message.__dataclass_fields__
    return [Message(**{k: str(r.get(k, "")) for k in fields}) for r in rows]


# ---------------- email ----------------
def _decode(value) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return str(value)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ")


def _email_body(msg: email.message.Message) -> str:
    plain, html = [], []
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.get_content_maintype() == "multipart" or part.get_filename():
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        text = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
        (html if part.get_content_type() == "text/html" else plain).append(text)
    body = "\n".join(plain) if plain else _html_to_text("\n".join(html))
    return re.sub(r"\s+", " ", body).strip()


def parse_email(raw: bytes | email.message.Message, msg_id: str = "") -> Message:
    msg = raw if isinstance(raw, email.message.Message) else email.message_from_bytes(raw)
    name, addr = parseaddr(_decode(msg.get("From")))
    try:
        ts = parsedate_to_datetime(msg.get("Date")).isoformat()
    except Exception:
        ts = ""
    return Message(
        id=msg_id or _decode(msg.get("Message-ID")) or addr + ts,
        channel="email", sender=addr.lower(), sender_name=name or addr,
        subject=_decode(msg.get("Subject")), body=_email_body(msg), timestamp=ts,
    )


def load_eml(path: str | Path) -> list[Message]:
    path = Path(path)
    files = sorted(path.glob("*.eml")) if path.is_dir() else [path]
    return [parse_email(f.read_bytes(), f.stem) for f in files]


def load_mbox(path: str | Path, limit: int = 500) -> list[Message]:
    box = mailbox.mbox(str(path))
    out = []
    for i, msg in enumerate(box):
        if i >= limit:
            break
        out.append(parse_email(msg, f"mbox-{i}"))
    return out


def fetch_imap(host: str, user: str, password: str, days: int = 2, limit: int = 100,
               folder: str = "INBOX") -> list[Message]:
    """Read recent mail over IMAP without marking it as read (BODY.PEEK).

    Gmail: turn on 2-step verification, create an app password at
    https://myaccount.google.com/apppasswords, and use host imap.gmail.com.
    """
    since = (datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")
    with imaplib.IMAP4_SSL(host) as imap:
        imap.login(user, password)
        imap.select(folder, readonly=True)
        _, data = imap.search(None, f'(SINCE "{since}")')
        ids = data[0].split()[-limit:]
        out = []
        for mid in reversed(ids):
            _, parts = imap.fetch(mid, "(BODY.PEEK[])")
            raw = next((p[1] for p in parts if isinstance(p, tuple)), None)
            if raw:
                out.append(parse_email(raw, f"imap-{mid.decode()}"))
    return out


# ---------------- text messages ----------------
def load_sms_csv(path: str | Path) -> list[Message]:
    """CSV with columns: timestamp, sender, body  (optional: sender_name)."""
    out = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            out.append(Message(f"sms-{i}", "sms", r.get("sender", ""),
                               r.get("sender_name") or r.get("sender", ""), "",
                               r.get("body", ""), r.get("timestamp", "")))
    return out


def load_android_xml(path: str | Path) -> list[Message]:
    """Backup file from the Android app 'SMS Backup & Restore'. Keeps received messages only."""
    out = []
    for i, sms in enumerate(ET.parse(path).getroot().iter("sms")):
        if sms.get("type") != "1":  # 1 = received
            continue
        ts = datetime.fromtimestamp(int(sms.get("date", "0")) / 1000, tz=timezone.utc).isoformat()
        name = sms.get("contact_name") or ""
        out.append(Message(f"android-{i}", "sms", sms.get("address", ""),
                           "" if name == "(Unknown)" else name, "", sms.get("body", ""), ts))
    return out


_APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


def _decode_attributed_body(blob: bytes | None) -> str:
    """Newer macOS stores message text in a binary 'attributedBody' blob instead of `text`."""
    if not blob or b"NSString" not in blob:
        return ""
    try:
        s = blob.split(b"NSString", 1)[1][5:]
        if s[0] == 0x81:
            length, s = int.from_bytes(s[1:3], "little"), s[3:]
        else:
            length, s = s[0], s[1:]
        return s[:length].decode("utf-8", errors="ignore")
    except Exception:
        return ""


def load_imessage(db_path: str | Path = "~/Library/Messages/chat.db", days: int = 2,
                  limit: int = 300) -> list[Message]:
    """Read received iMessages/SMS from the Mac Messages database (read-only).

    Your terminal (or the app running Streamlit) needs Full Disk Access:
    System Settings -> Privacy & Security -> Full Disk Access.
    """
    db = Path(db_path).expanduser()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days) - _APPLE_EPOCH).total_seconds() * 1e9
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        """SELECT m.ROWID, m.text, m.attributedBody, m.date, h.id
           FROM message m LEFT JOIN handle h ON m.handle_id = h.ROWID
           WHERE m.is_from_me = 0 AND m.date > ? ORDER BY m.date DESC LIMIT ?""",
        (cutoff, limit)).fetchall()
    con.close()
    out = []
    for rowid, text, blob, date, handle in rows:
        body = text or _decode_attributed_body(blob)
        if not body.strip():
            continue
        ts = (_APPLE_EPOCH + timedelta(seconds=date / 1e9)).isoformat()
        out.append(Message(f"imsg-{rowid}", "sms", handle or "", handle or "", "", body, ts))
    return out


def load_any(path: str | Path) -> list[Message]:
    """Pick a loader from the file extension."""
    p = Path(path)
    if p.is_dir():
        return load_eml(p)
    ext = p.suffix.lower()
    loaders = {".json": load_json, ".eml": load_eml, ".mbox": load_mbox, ".csv": load_sms_csv,
               ".xml": load_android_xml, ".db": load_imessage}
    if ext not in loaders:
        raise ValueError(f"Don't know how to read {ext} files")
    return loaders[ext](p)
