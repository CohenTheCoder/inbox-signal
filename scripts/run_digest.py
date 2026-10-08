"""Score new messages and send yourself a digest of only the important ones.

Examples:
    python scripts/run_digest.py --source data/sample/messages.json
    python scripts/run_digest.py --imap --days 1 --deliver ntfy
    python scripts/run_digest.py --imessage --deliver email --vip "Mom" "boss@company.com"

Run it automatically every morning at 8am with cron (`crontab -e`):
    0 8 * * * cd /path/to/inbox-signal && .venv/bin/python scripts/run_digest.py --imap --deliver ntfy

IMAP settings come from env vars: IMAP_HOST (default imap.gmail.com), IMAP_USER, IMAP_PASSWORD.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inbox_signal import ingest, score_messages  # noqa: E402
from inbox_signal.digest import build_markdown, deliver  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="file: .json .eml .mbox .csv .xml, or a folder of .eml")
    ap.add_argument("--imap", action="store_true", help="read from an IMAP inbox (env vars)")
    ap.add_argument("--imessage", action="store_true", help="read macOS Messages (needs Full Disk Access)")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--vip", nargs="*", default=os.environ.get("VIPS", "").split(","))
    ap.add_argument("--deliver", default="file", choices=["file", "ntfy", "email", "slack", "print"])
    args = ap.parse_args()

    msgs = []
    if args.source:
        msgs += ingest.load_any(args.source)
    if args.imap:
        msgs += ingest.fetch_imap(os.environ.get("IMAP_HOST", "imap.gmail.com"),
                                  os.environ["IMAP_USER"], os.environ["IMAP_PASSWORD"], days=args.days)
    if args.imessage:
        msgs += ingest.load_imessage(days=args.days)
    if not msgs:
        sys.exit("No messages. Pass --source, --imap or --imessage.")

    print(f"Scoring {len(msgs)} messages...")
    scored = score_messages(msgs, vips=args.vip)
    if args.deliver == "print":
        print(build_markdown(scored))
    else:
        print(deliver(scored, args.deliver))


if __name__ == "__main__":
    main()
