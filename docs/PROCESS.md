### The pipeline in 6 steps

```
Gmail / Outlook (IMAP) ─┐
.eml / .mbox files ─────┤                ┌─► 2. rule signals (urgent? deadline? VIP? unsubscribe link?)
Mac Messages (chat.db) ─┼─► 1. ingest ───┤
Android SMS backup ─────┤   (one format) └─► 3. zero-shot model: which of 9 categories?
CSV of texts ───────────┘                              │
                                                       ▼
                       4. score 0-100 + reasons  ◄── (optional) your personal model / fine-tuned DistilBERT
                                                       │
                                                       ▼
                       5. digest: only "important" + "worth a glance" ──► file · phone push · email · Slack
                                                       │
                       6. you click 👍/👎 ──► retrain personal model ──┘
```

**Step 1 · Ingest** (`inbox_signal/ingest.py`)
Every source gets converted into the same `Message` object: channel, sender, name, subject,
body, timestamp. HTML emails are stripped to text with BeautifulSoup. IMAP uses `BODY.PEEK`
so nothing gets marked as read. Mac Messages are read from the local SQLite database in
read-only mode.

**Step 2 · Rule signals** (`inbox_signal/features.py`)
Regular expressions catch things that are easy to spot without ML: urgent wording ("ASAP",
"past due"), times and dates ("tomorrow at 2pm"), money ("$142.80"), direct questions ("can you…?"),
bulk-mail signs ("unsubscribe", "view in browser"), no-reply senders, SMS short codes, link
shorteners (a classic scam tell), and your VIP list.

**Step 3 · Load the model from Hugging Face** (`inbox_signal/models.py`)
```python
from transformers import pipeline
zs = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
zs("Your electricity bill of $142.80 is past due",
   candidate_labels=["a bill, payment, or banking notice", "a newsletter or marketing promotion", ...],
   hypothesis_template="This message is {}.")
```
**How zero-shot works:** the model was trained on *natural language inference*: given a
premise and a hypothesis, is the hypothesis true? We turn each category into a hypothesis
("This message is a bill, payment, or banking notice.") and ask the model how likely each one is.
That means **no labeled training data is needed**, and you can add a category by writing one
sentence in `config.py`.

**Step 4 · Score** (`inbox_signal/scorer.py`)
- `base = Σ P(category) × typical importance of that category` (work 0.85, personal 0.80, bill
  0.70 … promo 0.05, scam 0)
- add boosts (VIP +0.30, urgent +0.15, question +0.10…) and subtract penalties (bulk −0.25,
  link shortener −0.20…)
- if a personal model or fine-tuned DistilBERT exists, average its probability in
- score ≥ 55 → **important**, 35–55 → **worth a glance**, below → filtered out

Every adjustment is saved as a plain-English reason, so the app can show *why*.

**Step 5 · Digest** (`inbox_signal/digest.py`, `scripts/run_digest.py`)
Builds a short Markdown summary and sends it the way you choose. `run_digest.py` is a
command-line version you can schedule with cron to run every morning.

**Step 6 · Learn from you** (`inbox_signal/personal.py`)
👍/👎 clicks are saved to `data/feedback.csv`. With 12+ labels you can train a TF-IDF + logistic
regression model in under a second. The app asks about the messages it's **least sure about**
first (*uncertainty sampling*, a form of active learning), because those labels teach the
model the most.

**Model comparison** (`scripts/make_dataset.py`, `scripts/train.py`)
Real inboxes are private, so we generate 1,200 synthetic labeled messages from templates
(including traps like promos that say "URGENT"). Then we compare three approaches on 30
**hand-written** messages the models never saw:
A) zero-shot + rules with no training, B) TF-IDF + logistic regression, and C) DistilBERT
fine-tuned with a plain PyTorch training loop. The main metric is **recall** on important
messages, because missing a real one costs more than seeing an extra promo.
