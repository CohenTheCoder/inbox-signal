"""Settings in one place: the Hugging Face model, categories, and score thresholds."""

# Zero-shot classifier: a DeBERTa-v3 model trained on natural-language-inference data. It can
# sort text into ANY labels you write in plain English - no training data needed.
# https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0   (~370 MB)
ZERO_SHOT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
HYPOTHESIS = "This message is {}."

# Model fine-tuned by scripts/train.py --distilbert (optional)
FINETUNE_BASE = "distilbert-base-uncased"
FINETUNED_DIR = "models/distilbert-importance"
# Off by default: trained on synthetic templates, it becomes overconfident (95%+ on idle chit-chat)
# and blending it didn't improve accuracy on the hand-written test set. Flip on to experiment.
BLEND_FINETUNED = False

# category -> (plain-English label the model reads, how important this kind of message usually is)
CATEGORIES = {
    "personal":    ("a personal message from a friend or family member", 0.80),
    "work":        ("a work request or a question that needs a reply", 0.85),
    "bill":        ("a bill, payment, or banking notice", 0.70),
    "security":    ("a security alert or account verification code", 0.55),
    "appointment": ("an appointment, meeting, or schedule change", 0.75),
    "delivery":    ("a package delivery or order update", 0.30),
    "promo":       ("a newsletter or marketing promotion", 0.05),
    "social":      ("a social media notification", 0.10),
    "scam":        ("a scam or spam message", 0.00),
}

IMPORTANT_THRESHOLD = 55   # score >= this -> goes in your digest
MAYBE_THRESHOLD = 35       # between MAYBE and IMPORTANT -> "worth a glance" section

# Personal model trained from your thumbs up/down in the app
FEEDBACK_PATH = "data/feedback.csv"
PERSONAL_MODEL_PATH = "models/personal_model.joblib"
MIN_FEEDBACK_TO_TRAIN = 12
