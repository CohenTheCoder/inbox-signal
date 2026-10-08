"""inbox-signal: surface the few messages that matter.

Pipeline:  ingest.py -> features.py + models.py -> scorer.py -> personal.py -> digest.py
"""
from .ingest import Message, load_any
from .scorer import score_messages

__all__ = ["Message", "load_any", "score_messages"]
