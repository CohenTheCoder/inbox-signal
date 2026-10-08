"""Generate a labeled training set of fake emails and texts.

Real inboxes are private, so we build a synthetic one from templates with random names,
amounts, times and stores. Labels: important = 1 / 0 plus a category.

The templates deliberately include tricky cases so a model can't just memorize keywords:
promos that SAY "urgent", personal chit-chat that is NOT important, work FYIs that need no reply.

Usage:  python scripts/make_dataset.py --n 1200
Writes: data/synthetic_train.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

NAMES = ["Alex", "Jordan", "Priya", "Marcus", "Elena", "Sam", "Taylor", "Chris", "Nina", "Omar",
         "Grace", "Leo", "Fatima", "Ben", "Hannah", "Diego", "Mei", "Isaac", "Rosa", "Kofi"]
FAMILY = ["Mom", "Dad", "Grandma", "Aunt Lisa", "Uncle Ray", "my sister", "your brother"]
STORES = ["ShopWave", "UrbanThreads", "GadgetHub", "FreshCart", "StyleBox", "HomeNest"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow", "today", "tonight"]
TIMES = ["9am", "10:30am", "noon", "2pm", "3:15pm", "4pm", "6pm", "7:30pm"]
TASKS = ["the Q3 deck", "the budget sheet", "the client proposal", "your slides", "the contract",
         "the onboarding doc", "the SQL query results", "the dashboard link"]


def money() -> str:
    return f"${random.randint(12, 2400)}.{random.randint(0, 99):02d}"


# (category, important, template).  {n}=name {f}=family {d}=day {t}=time {s}=store {m}=money {k}=task
TEMPLATES = [
    ("personal", 1, "Hey it's {f}, can you call me {d}? Need to talk about something."),
    ("personal", 1, "{n} here - are you still coming to my place {d} at {t}? Let me know."),
    ("personal", 1, "Can you pick up the kids at {t} {d}? I'm stuck at work."),
    ("personal", 1, "{f} is in the hospital, nothing serious but please call when you see this"),
    ("personal", 1, "Are we still on for dinner {d}? I made a reservation for {t}."),
    ("personal", 1, "Would you be able to help me move {d}? Starting around {t}."),
    ("personal", 0, "lol did you see that video {n} posted"),
    ("personal", 0, "haha that's hilarious"),
    ("personal", 0, "Happy birthday!! Hope you have an amazing day 🎉"),
    ("personal", 0, "{n} says hi, we're watching the game"),
    ("personal", 0, "ok sounds good 👍"),
    ("work", 1, "Hi, can you send me {k} before the {t} meeting {d}? Thanks"),
    ("work", 1, "Quick question - did you get a chance to review {k}? Client is asking {d}."),
    ("work", 1, "Urgent: production dashboard is down, can you take a look ASAP?"),
    ("work", 1, "Could you review my pull request by {d}? It's blocking the release."),
    ("work", 1, "Are you available for a quick call at {t} {d} to go over {k}?"),
    ("work", 1, "Interview request: are you available {d} at {t} for a 30 minute phone screen?"),
    ("work", 0, "FYI - I updated {k}. No action needed on your end."),
    ("work", 0, "Thanks {n}, got it!"),
    ("work", 0, "Reminder: the office kitchen will be cleaned {d}. Please remove your food."),
    ("work", 0, "Weekly digest: 14 updates in your workspace this week."),
    ("bill", 1, "Your payment of {m} is past due. Pay by {d} to avoid a late fee."),
    ("bill", 1, "Final notice: your account balance of {m} is overdue."),
    ("bill", 1, "Rent of {m} is due {d}. Please pay through the resident portal."),
    ("bill", 1, "Your credit card statement is ready. Minimum payment {m} due {d}."),
    ("bill", 0, "Thanks for your payment of {m}. Your balance is $0.00."),
    ("bill", 0, "Your receipt from {s}: total {m}. Thanks for shopping with us!"),
    ("bill", 0, "Your monthly statement is available. No payment is due."),
    ("security", 1, "Unusual sign-in to your account from a new device. If this wasn't you, secure your account now."),
    ("security", 1, "Your password was changed. If you didn't do this, contact us immediately."),
    ("security", 0, "Your verification code is {c}. It expires in 10 minutes."),
    ("security", 0, "Use {c} as your login code. Don't share it with anyone."),
    ("appointment", 1, "Reminder: your appointment with Dr. {n} is {d} at {t}. Reply C to confirm."),
    ("appointment", 1, "Meeting moved: team sync is now {d} at {t}."),
    ("appointment", 1, "Your flight tomorrow has been delayed to {t}. Check the app for gate info."),
    ("appointment", 0, "Thanks for visiting! Rate your appointment with Dr. {n}."),
    ("delivery", 0, "Your order from {s} has shipped and arrives {d}."),
    ("delivery", 0, "Out for delivery: your package arrives {d} by {t}."),
    ("delivery", 0, "Your {s} order was delivered. Enjoy!"),
    ("delivery", 1, "Delivery attempt failed - signature required. Reschedule by {d} or it will be returned."),
    ("promo", 0, "URGENT: only hours left! 50% off everything at {s}. Shop now."),
    ("promo", 0, "{s}: your exclusive deal expires tonight. Use code SAVE20. Unsubscribe."),
    ("promo", 0, "New arrivals just for you at {s}. Free shipping over $50."),
    ("promo", 0, "Your weekly newsletter: 7 stories you missed. View in browser."),
    ("promo", 0, "Don't miss out! Last chance to get {m} off your next order."),
    ("social", 0, "{n} liked your photo."),
    ("social", 0, "{n} commented on your post: 'so cool!'"),
    ("social", 0, "You have 5 new notifications. See what you missed."),
    ("social", 0, "{n} started following you."),
    ("social", 1, "{n} sent you a message: 'can you call me back? it's important'"),
    ("scam", 0, "Congratulations! You won a {m} gift card. Claim now at bit.ly/claim-{c}"),
    ("scam", 0, "Your package is on hold. Update your address at tinyurl.com/redeliver{c}"),
    ("scam", 0, "IRS notice: you owe {m}. Pay with gift cards immediately to avoid arrest."),
    ("scam", 0, "Your account is suspended! Verify now: bit.ly/verify{c}"),
]

SENDERS = {
    "personal": lambda: (f"+1555{random.randint(1000000, 9999999)}", random.choice(NAMES + FAMILY)),
    "work": lambda: (f"{random.choice(NAMES).lower()}@company.example", random.choice(NAMES)),
    "bill": lambda: ("billing@utility.example", "Billing"),
    "security": lambda: ("alerts@bank.example", "Bank Alerts"),
    "appointment": lambda: ("reminders@clinic.example", "Reminders"),
    "delivery": lambda: ("no-reply@shipping.example", "Shipping"),
    "promo": lambda: (f"deals@{random.choice(STORES).lower()}.example", random.choice(STORES)),
    "social": lambda: ("notifications@social.example", "Social"),
    "scam": lambda: (f"+1888{random.randint(1000000, 9999999)}", "Unknown"),
}


def make(n: int, seed: int = 7) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for _ in range(n):
        cat, imp, tpl = random.choice(TEMPLATES)
        text = tpl.format(n=random.choice(NAMES), f=random.choice(FAMILY), d=random.choice(DAYS),
                          t=random.choice(TIMES), s=random.choice(STORES), m=money(),
                          k=random.choice(TASKS), c=random.randint(100000, 999999))
        if random.random() < 0.3:
            text = text.lower()  # people text in lowercase
        sender, name = SENDERS[cat]()
        rows.append({"sender": sender, "sender_name": name, "text": text,
                     "category": cat, "important": imp})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    args = ap.parse_args()
    df = make(args.n)
    out = ROOT / "data/synthetic_train.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out.relative_to(ROOT)}  ({df['important'].mean():.0%} important)")
    print(df["category"].value_counts().to_string())
