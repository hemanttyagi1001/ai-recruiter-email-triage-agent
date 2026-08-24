"""
One-shot OAuth bootstrap. Run this once to authorize the app against your
Gmail account; it writes token.json to disk. All subsequent runs of ingest
use the persisted token silently.

Usage:
    python -m app.gmail.auth

Scope note (corrected in D40): the pipeline needs BOTH `gmail.readonly`
(ingest lists and fetches inbox messages) and `gmail.compose` (act creates
drafts, auto_send sends). They are disjoint capability sets — neither
subsumes the other. Phase 2 wrongly swapped readonly *for* compose, which
left ingest returning 403 while auth appeared to succeed.

GOTCHA: any change to SCOPES invalidates token.json. Google issues tokens
against the scopes actually consented to, so a stale file keeps failing with
insufficientPermissions no matter what the code now requests. Delete
token.json and re-run this to get a fresh consent screen.

Read the printed output: this authenticates AND lists 5 real messages. The
listing step is the part that proves the read scope — getProfile() succeeds
under compose alone and will happily print your address even when the read
scope is missing.
"""

import logging

from app.gmail.client import GmailClient


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    client = GmailClient.create()
    profile = client.profile()
    print(
        f"Authenticated as {profile['emailAddress']} — "
        f"{profile.get('messagesTotal', '?')} messages total, "
        f"{profile.get('threadsTotal', '?')} threads."
    )
    # Quick smoke: read a handful of message ids from the configured label.
    from app.config import settings

    ids = client.list_message_ids(settings.gmail_label, max_results=5)
    print(f"\nSample of last 5 message ids in label={settings.gmail_label!r}:")
    for gid in ids:
        raw = client.get_message(gid)
        subject = next(
            (h["value"] for h in raw["payload"]["headers"] if h["name"].lower() == "subject"),
            "(no subject)",
        )
        print(f"  {gid}  {subject[:80]}")


if __name__ == "__main__":
    main()
