"""Fast tests for loaders and rule features (no model downloads needed)."""
import sqlite3
from pathlib import Path

from inbox_signal import features, ingest
from inbox_signal.scorer import _rule_score

ROOT = Path(__file__).resolve().parent.parent


def test_load_sample():
    msgs = ingest.load_any(ROOT / "data/sample/messages.json")
    assert len(msgs) == 30 and {m.channel for m in msgs} == {"email", "sms"}


def test_parse_eml(tmp_path):
    raw = (b"From: Jane Doe <Jane@Example.com>\r\nSubject: Lunch?\r\n"
           b"Date: Tue, 07 Oct 2026 10:00:00 -0400\r\nContent-Type: text/html\r\n\r\n"
           b"<p>Can you do <b>lunch</b> tomorrow?</p>")
    (tmp_path / "a.eml").write_bytes(raw)
    m = ingest.load_any(tmp_path)[0]
    assert m.sender == "jane@example.com" and m.sender_name == "Jane Doe"
    assert "lunch" in m.body and m.timestamp.startswith("2026-10-07")


def test_sms_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("timestamp,sender,body\n2026-10-07,+15550100,call me asap\n")
    assert ingest.load_any(p)[0].body == "call me asap"


def test_imessage_reader(tmp_path):
    db = tmp_path / "chat.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT)")
    con.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB,"
                " date INTEGER, is_from_me INTEGER, handle_id INTEGER)")
    con.execute("INSERT INTO handle VALUES (1, '+15550142')")
    # a far-future date so it's always inside the "days back" window
    con.execute("INSERT INTO message VALUES (1, 'dinner sunday?', NULL, 9e17, 0, 1)")
    con.execute("INSERT INTO message VALUES (2, 'sent by me', NULL, 9e17, 1, 1)")
    con.commit()
    con.close()
    msgs = ingest.load_imessage(db, days=1)
    assert [m.body for m in msgs] == ["dinner sunday?"]


def test_features():
    f = features.extract("URGENT: can you send the deck by 2pm today? Invoice $40.00",
                         "boss@work.com", "Boss", vips=["boss@work.com"])
    assert f["urgent"] and f["when"] and f["question"] and f["money"] and f["vip"]
    g = features.extract("50% off! Unsubscribe here", "no-reply@shop.com", "Shop", vips=[])
    assert g["bulk"] and g["noreply"]


def test_rule_score_orders_sensibly():
    work = {"work": 0.9, "promo": 0.1}
    promo = {"work": 0.1, "promo": 0.9}
    no = features.extract("", "", "", [])
    assert _rule_score(work, no)[0] > _rule_score(promo, no)[0]


def test_gist_strips_greeting():
    assert features.gist("", "Hi Sam, the meeting moved to 3pm. See you there.") == "the meeting moved to 3pm."
