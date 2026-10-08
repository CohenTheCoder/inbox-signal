"""Streamlit front end.  Run:  streamlit run app.py"""
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from inbox_signal import config, ingest, personal
from inbox_signal.digest import build_markdown, deliver
from inbox_signal.scorer import score_messages

ROOT = Path(__file__).parent
TIER_COLORS = {"important": "#dc2626", "maybe": "#f59e0b", "skip": "#94a3b8"}

st.set_page_config(page_title="Inbox Signal", page_icon="📬", layout="wide")
ss = st.session_state

# ---------- sidebar ----------
st.sidebar.title("📬 Inbox Signal")
st.sidebar.caption("Reads your emails + texts, shows you only what matters")
page = st.sidebar.radio("Steps", ["1 · Load messages", "2 · Triage", "3 · Digest", "4 · Teach it", "How it works"])
st.sidebar.divider()
vips_text = st.sidebar.text_area("VIPs (one per line: names, emails, numbers)",
                                 value=ss.get("vips_text", "Mom"), height=90)
ss.vips_text = vips_text
vips = [v.strip() for v in vips_text.splitlines() if v.strip()]
st.sidebar.markdown(f"**Model (Hugging Face)**\n\n`{config.ZERO_SHOT_MODEL}`")
st.sidebar.caption("Runs 100% locally. Your messages never leave this computer unless you choose a delivery option.")


def run_scoring():
    with st.spinner(f"Classifying {len(ss.messages)} messages with the zero-shot model (~1-2 s each on a laptop CPU)…"):
        ss.scored = score_messages(ss.messages, vips=vips)


# ======================= 1. LOAD =======================
if page.startswith("1"):
    st.header("Step 1 · Load messages")
    st.caption("Everything is converted to one format: channel, sender, subject, body, time.")
    src = st.radio("Source", ["Sample inbox (30 fictional messages)", "Upload a file", "Email via IMAP", "Mac Messages (iMessage/SMS)"])

    if src.startswith("Sample"):
        if st.button("Load sample", type="primary"):
            ss.messages = ingest.load_json(ROOT / "data/sample/messages.json")
            ss.pop("scored", None)
    elif src.startswith("Upload"):
        st.markdown("Accepted: `.json` · `.eml` · `.mbox` (Gmail Takeout) · `.csv` texts (timestamp, sender, body) · "
                    "`.xml` (Android *SMS Backup & Restore*)")
        up = st.file_uploader("File", type=["json", "eml", "mbox", "csv", "xml"])
        if up is not None and st.button("Load file", type="primary"):
            with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
                tmp.write(up.getbuffer())
            try:
                ss.messages = ingest.load_any(tmp.name)
                ss.pop("scored", None)
            except Exception as exc:
                st.error(f"Couldn't read that file: {exc}")
            finally:
                os.unlink(tmp.name)
    elif src.startswith("Email"):
        st.info("Gmail: turn on 2-step verification, then create an **app password** at "
                "myaccount.google.com/apppasswords. Mail is read with BODY.PEEK so nothing gets marked as read.")
        c1, c2 = st.columns(2)
        host = c1.text_input("IMAP host", value=os.environ.get("IMAP_HOST", "imap.gmail.com"))
        user = c2.text_input("Email address", value=os.environ.get("IMAP_USER", ""))
        pw = c1.text_input("App password", type="password", value=os.environ.get("IMAP_PASSWORD", ""))
        days = c2.number_input("Days back", 1, 30, 2)
        if st.button("Fetch mail", type="primary", disabled=not (user and pw)):
            try:
                with st.spinner("Connecting…"):
                    ss.messages = ingest.fetch_imap(host, user, pw, days=int(days))
                ss.pop("scored", None)
            except Exception as exc:
                st.error(f"IMAP error: {exc}")
    else:
        st.info("Reads `~/Library/Messages/chat.db` read-only. Give your terminal **Full Disk Access** "
                "(System Settings → Privacy & Security) and restart Streamlit.")
        days = st.number_input("Days back", 1, 30, 2)
        if st.button("Read Messages", type="primary"):
            try:
                ss.messages = ingest.load_imessage(days=int(days))
                ss.pop("scored", None)
            except Exception as exc:
                st.error(f"Couldn't open the Messages database: {exc}")

    if ss.get("messages"):
        df = pd.DataFrame([m.to_dict() for m in ss.messages])
        st.success(f"Loaded {len(df)} messages ({(df['channel'] == 'email').sum()} emails, "
                   f"{(df['channel'] == 'sms').sum()} texts). Next: **2 · Triage** in the sidebar.")
        st.dataframe(df[["channel", "sender_name", "subject", "body"]], use_container_width=True, height=360)


# ======================= 2. TRIAGE =======================
elif page.startswith("2"):
    st.header("Step 2 · Triage")
    if not ss.get("messages"):
        st.info("Load messages first (step 1).")
        st.stop()
    if st.button("Score messages" if "scored" not in ss else "Re-score", type="primary"):
        run_scoring()
    sc = ss.get("scored")
    if sc is None:
        st.stop()

    k = st.columns(4)
    k[0].metric("Messages", len(sc))
    k[1].metric("Important", int((sc["tier"] == "important").sum()))
    k[2].metric("Worth a glance", int((sc["tier"] == "maybe").sum()))
    k[3].metric("Filtered out", int((sc["tier"] == "skip").sum()))

    a, b = st.columns([2, 1])
    fig = px.histogram(sc, x="score", color="tier", nbins=20, color_discrete_map=TIER_COLORS,
                       title="Importance scores", range_x=[0, 100])
    fig.add_vline(x=config.IMPORTANT_THRESHOLD, line_dash="dash")
    a.plotly_chart(fig, use_container_width=True)
    b.plotly_chart(px.pie(sc, names="category", title="What the model thinks they are", hole=0.5),
                   use_container_width=True)

    show = st.multiselect("Show tiers", ["important", "maybe", "skip"], default=["important", "maybe"])
    for _, r in sc[sc["tier"].isin(show)].iterrows():
        icon = "✉️" if r["channel"] == "email" else "💬"
        with st.expander(f"{icon} {r['score']:.0f} · {r['from']} — {r['gist'][:90]}"):
            st.write(r["text"])
            st.markdown("**Why this score**")
            for reason in r["reasons"]:
                st.markdown(f"- {reason}")
            probs = pd.Series({k[2:]: v for k, v in r.items() if k.startswith("p_")}).sort_values()
            st.plotly_chart(px.bar(probs, orientation="h", labels={"value": "probability", "index": ""},
                                   height=260).update_layout(showlegend=False, margin=dict(t=10, b=0)),
                            use_container_width=True, key=f"p-{r['id']}")


# ======================= 3. DIGEST =======================
elif page.startswith("3"):
    st.header("Step 3 · Your digest")
    sc = ss.get("scored")
    if sc is None:
        st.info("Score messages first (step 2).")
        st.stop()
    md = build_markdown(sc)
    st.markdown(md)
    st.divider()
    st.subheader("Send it to yourself")
    method = st.selectbox("Delivery", ["file", "ntfy", "email", "slack"],
                          format_func={"file": "Save a Markdown file (local)", "ntfy": "Phone push via ntfy.sh",
                                       "email": "Email (SMTP)", "slack": "Slack webhook"}.get)
    needs = {"ntfy": ["NTFY_TOPIC"], "email": ["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "DIGEST_TO"],
             "slack": ["SLACK_WEBHOOK_URL"]}.get(method, [])
    missing = [v for v in needs if not os.environ.get(v)]
    if missing:
        st.warning(f"Set these environment variables before starting Streamlit: {', '.join(missing)}")
    if st.button("Send digest", type="primary", disabled=bool(missing)):
        try:
            st.success(deliver(sc, method))
        except Exception as exc:
            st.error(f"Delivery failed: {exc}")
    st.caption("Automate it: `python scripts/run_digest.py --imap --deliver ntfy` on a cron schedule (see README).")


# ======================= 4. TEACH IT =======================
elif page.startswith("4"):
    st.header("Step 4 · Teach it what *you* think is important")
    st.markdown("Label messages 👍 / 👎. After a dozen labels, train a personal model "
                "(TF-IDF + logistic regression). It gets blended into every future score.")
    sc = ss.get("scored")
    fb = personal.load_feedback()
    labeled = set(fb["id"].astype(str))
    c1, c2, c3 = st.columns(3)
    c1.metric("Your labels", len(fb))
    c2.metric("Marked important", int(fb["important"].sum()) if len(fb) else 0)
    c3.metric("Personal model", "trained" if personal.load_personal_model() is not None else "not yet")

    if st.button(f"Train my model (needs {config.MIN_FEEDBACK_TO_TRAIN}+ labels)"):
        res = personal.train_personal_model()
        if res["trained"]:
            st.success(f"Trained on {res['n']} labels · cross-validated accuracy {res['cv_accuracy']:.0%}. "
                       "Re-score in step 2 to use it.")
        else:
            st.warning(res["reason"])

    if sc is None:
        st.info("Score messages in step 2 to start labeling.")
        st.stop()
    todo = sc[~sc["id"].astype(str).isin(labeled)]
    st.caption(f"{len(todo)} unlabeled messages, lowest-confidence first (active learning: the "
               "labels that teach the model the most are the ones it's least sure about).")
    todo = todo.assign(uncertainty=(todo["score"] - config.IMPORTANT_THRESHOLD).abs()).sort_values("uncertainty")
    for _, r in todo.head(10).iterrows():
        cols = st.columns([6, 1, 1])
        cols[0].markdown(f"**{r['from']}** · score {r['score']:.0f}  \n{r['gist']}")
        if cols[1].button("👍", key=f"up-{r['id']}", help="Important"):
            personal.add_feedback(r["id"], r["text"], r["from"], True)
            st.rerun()
        if cols[2].button("👎", key=f"down-{r['id']}", help="Not important"):
            personal.add_feedback(r["id"], r["text"], r["from"], False)
            st.rerun()


# ======================= HOW IT WORKS =======================
else:
    st.header("How it works")
    st.markdown((ROOT / "docs/PROCESS.md").read_text())
    res = ROOT / "docs/results.md"
    if res.exists():
        st.markdown(res.read_text())
