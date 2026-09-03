"""
The single Gmail surface for the whole pipeline. Wraps the OAuth2 install
flow, the Gmail v1 REST API, and label-name → label-id resolution so callers
just say "list from INBOX, give me 200 message ids, get me this one's
payload, put this draft in the Drafts folder." It exists as one module so
that every mailbox capability the system holds is enumerable by reading a
single file.

Capability, stated honestly: this client can READ any message, can WRITE
(create drafts, send replies), and since D68 can also remove the UNREAD
label. It requests `gmail.readonly` + `gmail.compose` + `gmail.modify`. An
earlier version of this docstring claimed the client was "structurally
incapable of sending"; that stopped being true when Phase 2 added
create_draft and Phase 5 added send_reply, and its claim that no scope
granted label mutation stopped being true at D68. The restraint is now a
CODE property enforced at the call sites and by the kill switch, NOT an
OAuth property. See the SCOPES block below, D33, D36, D40 and D69.

The three scopes are NOT equally load-bearing: readonly and compose are
required, modify is optional and degrades to a no-op. D69 explains why that
distinction is enforced in code rather than left to convention.

Trust boundary note: this module reads untrusted email content but has no LLM
dependency and does not itself act on the content. It only turns Gmail API
JSON into typed Python. The LLM-facing nodes (classify, extract) receive the
already-parsed text.
"""

from __future__ import annotations

import base64
import html
import logging
import re
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.config import settings
from app.retry import RetryPolicy, retry_external

log = logging.getLogger(__name__)

# CONCEPT: Gmail OAuth scopes are DISJOINT capability sets, not a ladder.
#   This is the single most expensive misreading available here, and D19
#   made it: Phase 2 replaced `gmail.readonly` with `gmail.compose` on the
#   assumption that the write scope subsumed the read one. It does not.
#     - gmail.readonly : list/get any message. No writes at all.
#     - gmail.compose  : create/update drafts, and send. Can read DRAFTS,
#                        but CANNOT list or get inbox messages.
#   Neither contains the other, so this pipeline needs both: ingest reads
#   the inbox, act writes drafts, auto_send sends. With compose alone,
#   users().getProfile() still succeeds — which makes the failure look like
#   an auth problem rather than a scope problem, because "Authenticated as
#   you@gmail.com" prints happily right before messages.list returns 403.
#   Corrected in D40.
#
# CONCEPT: scope granularity vs autonomy discipline.
#   Gmail is coarser than we'd like: `compose` grants CREATE-draft AND SEND
#   together. There is no "drafts-only" scope.
#
#   The autonomy discipline is a CODE property, not an OAuth-level one:
#     - create_draft is called from `act` (human-approved path).
#     - send_reply is called from `auto_send` (rule-based decline only,
#       per D33's blast-radius × reversibility criterion). No other node
#       in the pipeline invokes send_reply.
#     - Both create_draft and send_reply consult app.kill_switch
#       IMMEDIATELY before their Gmail HTTP call. Flipping the switch
#       halts both without a deploy.
#
# ALTERNATIVE: a single `gmail.modify`, which covers read and write in one
#   scope. Rejected: it additionally grants label mutation and trashing
#   messages — capability no node here uses. Two narrow scopes state the
#   actual requirement; one broad scope would make the trust-boundary
#   claim in the README weaker than it needs to be.
# GOTCHA: changing this list invalidates any existing token.json. Google
# issues tokens against the scopes consented to, not the scopes requested
# later, so a stale token keeps failing with insufficientPermissions until
# the file is deleted and the consent flow re-run.
# CONCEPT: required vs. optional scopes, and why the distinction is load-bearing.
#   D69 splits this list in two. A scope is REQUIRED when a pipeline stage
#   cannot run without it, and OPTIONAL when its absence costs one feature
#   and nothing else. The split exists because the flat list caused a
#   three-day total outage on 2026-08-29:
#     D68 appended gmail.modify for mark_read — a cosmetic nicety. Every
#     existing token.json had been consented to only the first two scopes,
#     and load_credentials passed this list into from_authorized_user_file,
#     which made the hourly token REFRESH request a scope the refresh token
#     never held. Google answered `invalid_scope: Bad Request`, so no
#     credentials were produced at all. Ingest, drafting and sending — none
#     of which need modify — were down until someone re-ran the consent flow.
#   A convenience feature must never be able to take down the read path.
#   Hence: missing REQUIRED → refuse to start. Missing OPTIONAL → warn and
#   degrade the one call that needs it.
REQUIRED_SCOPES: list[str] = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

OPTIONAL_SCOPES: list[str] = [
    # D68: needed by mark_read — removing the UNREAD label is a messages.modify
    # call, which neither readonly nor compose grants. Optional by D69: without
    # it mark_read returns False and logs, and the only visible consequence is
    # that replied-to mail stays bold in the inbox.
    "https://www.googleapis.com/auth/gmail.modify",
]

# What the consent flow ASKS for. Always everything — there is no reason to
# request less than the full set when a human is already looking at the screen.
# GOTCHA: adding to this list does NOT invalidate an existing token.json any
# more (that was the D68 failure). It does mean the extra scope stays missing
# until the consent flow is re-run; check_scopes() below is what tells you.
SCOPES: list[str] = REQUIRED_SCOPES + OPTIONAL_SCOPES

# System label IDs Gmail hard-codes. Names == IDs for these; for user-created
# labels we have to hit labels.list to resolve name → id.
SYSTEM_LABEL_IDS: frozenset[str] = frozenset(
    {"INBOX", "SENT", "SPAM", "TRASH", "IMPORTANT", "STARRED", "UNREAD",
     "CATEGORY_PERSONAL", "CATEGORY_SOCIAL", "CATEGORY_PROMOTIONS",
     "CATEGORY_UPDATES", "CATEGORY_FORUMS"}
)


def _css_font_stack(font_family: str) -> str:
    """Turn a configured font name into a safe CSS font-family value.

    GOTCHA: this string is interpolated into a style attribute in mail we send
    to third parties. The value comes from candidate.toml rather than from
    email, so it is not attacker-controlled — but stripping quotes and the
    characters that could close the attribute or the declaration costs nothing
    and means a typo in config can never produce broken or hostile markup.
    Names with spaces ("Trebuchet MS") must be quoted in CSS to parse.
    """
    cleaned = re.sub(r"[^A-Za-z0-9 \-]", "", font_family).strip()
    if not cleaned:
        cleaned = "Trebuchet MS"
    return f"'{cleaned}', Helvetica, Arial, sans-serif"


def build_mime(
    to: str,
    subject: str,
    body_text: str,
    in_reply_to_message_id: str | None = None,
    font_family: str | None = None,
    attachment_path: Path | None = None,
) -> "MIMEMultipart":
    """Build the RFC 5322 message sent for both drafts and replies.

    CONCEPT: multipart/alternative — one message, two renderings.
      The plain-text part is the canonical body: it is what the outbound
      validator inspected, what we store in `drafts.body_text`, and what any
      text-only client shows. The HTML part exists solely to carry typography.
      Per RFC 2046 the LAST part is the preferred one, so the HTML must be
      attached second or clients will show the unstyled version.

    SECURITY: the body is HTML-escaped before it reaches the markup, and that
    is load-bearing rather than tidy. `body_text` embeds extractor output —
    `role_title` and `company` came out of an LLM reading untrusted recruiter
    email. A role title of `<img src=x onerror=...>` would otherwise become
    live markup in a message WE send to a third party, turning this system
    into a delivery vehicle for content it was supposed to be quarantining.
    Escaping keeps the HTML part a faithful rendering of the validated text
    and nothing more.

    WHY white-space: pre-wrap instead of converting newlines to <br>: the
    templates use two-space indentation for bullet lists, and HTML collapses
    leading whitespace. pre-wrap preserves both the line breaks and the
    indentation with no transformation of the text at all — fewer moving
    parts, and the HTML stays a pure restyling of the plain part.
    """
    # CONCEPT: the MIME tree changes shape when there is an attachment.
    #   Without one:  multipart/alternative [ text/plain, text/html ]
    #   With one:     multipart/mixed [ multipart/alternative[...], application/pdf ]
    #   The alternative node must stay intact and become a CHILD of mixed —
    #   flattening the two into a single multipart/mixed would make some
    #   clients render the plain and HTML bodies one after the other, so the
    #   recruiter reads the same email twice.
    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(body_text, "plain", "utf-8"))
    stack = _css_font_stack(font_family or settings.email_font_family)
    alternative.attach(MIMEText(
        f'<div style="font-family: {stack}; font-size: 14px; '
        f'white-space: pre-wrap;">{html.escape(body_text)}</div>',
        "html", "utf-8",
    ))

    if attachment_path is not None:
        msg: MIMEMultipart = MIMEMultipart("mixed")
        msg.attach(alternative)
        msg.attach(_pdf_part(attachment_path))
    else:
        msg = alternative

    msg["To"] = to
    msg["Subject"] = subject
    if in_reply_to_message_id:
        # WHY both headers: In-Reply-To identifies the direct parent;
        # References carries the full ancestry chain. For a one-hop reply
        # they're the same. Setting References is what makes the threading
        # survive if the recipient's client is stricter than Gmail's.
        msg["In-Reply-To"] = in_reply_to_message_id
        msg["References"] = in_reply_to_message_id

    return msg


def _pdf_part(path: Path) -> MIMEApplication:
    """Read a PDF from disk into an attachment part.

    WHY read at send time rather than caching at import: the resume is a bind
    mount. Caching it would mean replacing the file on the host has no effect
    until the container restarts, and the failure would be silent — recruiters
    receiving a stale CV with no error anywhere.
    GOTCHA: the filename the recruiter sees comes from the path, so name the
    file on disk the way you want it to appear in their inbox.
    """
    data = path.read_bytes()
    part = MIMEApplication(data, _subtype="pdf")
    part.add_header("Content-Disposition", "attachment", filename=path.name)
    return part


def encode_mime(msg: "MIMEMultipart") -> str:
    """base64url-encode a message for Gmail's `raw` field."""
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")


def load_credentials(credentials_path: Path, token_path: Path) -> Credentials:
    """Return a valid Credentials object, running the install flow if needed.

    Rewrites token.json every time a refresh happens.
    """
    # CONCEPT: OAuth2 for installed apps. Flow:
    #   1. First run: credentials.json (client id/secret from GCP) drives
    #      InstalledAppFlow which opens a browser to Google's consent screen,
    #      then catches the redirect on a localhost port and exchanges the
    #      auth code for (access_token, refresh_token).
    #   2. Access tokens live ~1 hour. The refresh_token is long-lived and
    #      lets us mint new access tokens without re-prompting the user.
    #   3. We persist both to token.json. On subsequent runs we load token.json,
    #      and if access is expired but refresh is valid, we refresh silently.
    # GOTCHA: refresh tokens can be revoked (user changed password, admin
    # disabled the app, 6-month inactivity). On refresh failure the caller
    # gets a RefreshError and needs to delete token.json and re-consent.
    creds: Credentials | None = None
    if token_path.exists():
        # CONCEPT: load with the scopes the token WAS GRANTED, not the scopes
        #   this code wants. Passing scopes=None makes google-auth fall back to
        #   the `scopes` key inside token.json (see from_authorized_user_info:
        #   `if scopes is None and "scopes" in info`).
        # WHY this matters more than it looks: the scopes on a Credentials
        #   object are sent with the token-refresh request. Handing it a
        #   LARGER set than was consented to makes Google reject the refresh
        #   outright with `invalid_scope`, which yields no credentials — so a
        #   missing cosmetic scope breaks reading, drafting and sending too.
        #   Loading the granted set means the refresh always succeeds, and a
        #   missing scope instead surfaces as a 403 on the ONE call that needs
        #   it (mark_read already treats that as non-fatal).
        # ALTERNATIVE: keep passing SCOPES and catch RefreshError to trigger
        #   re-consent automatically. Rejected — the consent flow needs a
        #   browser and a human, neither of which exists in the container at
        #   3am. Degrading beats blocking on an interaction nobody is there
        #   to perform.
        creds = Credentials.from_authorized_user_file(str(token_path), None)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Refreshing expired Gmail token")
            creds.refresh(Request())
        else:
            log.info("No valid token; running install flow (browser will open)")
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"Gmail OAuth client secrets not found at {credentials_path}. "
                    "Download from Google Cloud Console → APIs & Services → Credentials → "
                    "OAuth 2.0 Client IDs (Desktop app)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            # port=0 lets the OS pick a free port; the redirect URI Google
            # accepts is http://localhost with no fixed port for desktop apps.
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
        log.info("Saved token to %s", token_path)

    return creds


class MissingRequiredScopeError(RuntimeError):
    """token.json lacks a scope the pipeline cannot function without.

    WHY its own type rather than a bare RuntimeError: this is the one auth
    failure a human must physically fix (re-run the consent flow in a
    browser). Retrying it, dead-lettering it, or waiting for the next cycle
    all accomplish nothing, so callers need to be able to tell it apart from
    the transient failures that surround it.
    GOTCHA: this lives in app.*, so retry_external's _is_domain_error check
    passes it straight through instead of wrapping it in a
    PermanentExternalError. That is deliberate — it must not become a
    dead-letter row that looks like 57 other dead-letter rows.
    """


def check_scopes(creds: Credentials) -> tuple[list[str], list[str]]:
    """Return (missing_required, missing_optional) for a loaded credential.

    Reads what Google actually granted — `creds.scopes` comes from token.json,
    not from this module's SCOPES list — so it answers "what can this token
    really do", which is the only question worth asking at startup.
    """
    granted = set(creds.scopes or [])
    missing_required = [s for s in REQUIRED_SCOPES if s not in granted]
    missing_optional = [s for s in OPTIONAL_SCOPES if s not in granted]
    return missing_required, missing_optional


def _short(scope: str) -> str:
    """`.../auth/gmail.modify` → `gmail.modify`, for log lines humans read."""
    return scope.rsplit("/", 1)[-1]


RECONSENT_HINT = (
    "Delete token.json and re-run `python -m app.gmail.auth` "
    "(needs a browser, so run it on the host, not in the container)."
)


def assert_required_scopes(creds: Credentials) -> list[str]:
    """Raise if a required scope is missing; return the missing optional ones.

    TRACE: called once per GmailClient.create(), i.e. once per ingest cycle,
    BEFORE the first Gmail HTTP call. The failure therefore lands at the top
    of the run with a message naming the fix, rather than as a 403 from
    whichever node happened to touch Gmail first.
    """
    missing_required, missing_optional = check_scopes(creds)
    if missing_required:
        raise MissingRequiredScopeError(
            f"token.json is missing required Gmail scope(s): "
            f"{', '.join(_short(s) for s in missing_required)}. "
            f"Ingest and drafting cannot run without them. {RECONSENT_HINT}"
        )
    return missing_optional


def preflight_scopes(token_path: Path) -> bool:
    """Report at startup what the persisted token can actually do.

    Returns True when every required scope is present.

    WHY this exists separately from assert_required_scopes: a long-running
    supervisor should state its capabilities at boot, in one line an operator
    can find, rather than letting the first cycle discover them 15 minutes
    later inside a traceback. D69's outage was invisible for three days partly
    because nothing ever announced what the token could do.

    GOTCHA: this deliberately does NOT call load_credentials. That function
    runs InstalledAppFlow.run_local_server() when no valid token exists, which
    opens a browser and blocks until someone completes a consent screen — in a
    container that is an indefinite hang at startup with no explanation. Here
    we only read the file that already exists.
    """
    if not token_path.exists():
        log.error("No token.json at %s — the agent cannot authenticate. %s",
                  token_path, RECONSENT_HINT)
        return False

    creds = Credentials.from_authorized_user_file(str(token_path), None)
    missing_required, missing_optional = check_scopes(creds)

    if missing_required:
        log.error(
            "token.json is missing REQUIRED scope(s): %s. Ingest and drafting "
            "will fail every cycle until this is fixed. %s",
            ", ".join(_short(s) for s in missing_required), RECONSENT_HINT,
        )
        return False

    if missing_optional:
        # WHY warning and not error: this is the D68 situation, and the entire
        # point of D69 is that it is survivable. Say exactly what degrades so
        # nobody goes hunting for a failure that is working as designed.
        log.warning(
            "token.json is missing optional scope(s): %s. Everything runs; "
            "mark_read will no-op, so replied-to mail stays unread in the "
            "inbox. %s",
            ", ".join(_short(s) for s in missing_optional), RECONSENT_HINT,
        )
    else:
        log.info("Gmail scopes OK: %s",
                 ", ".join(_short(s) for s in (creds.scopes or [])))
    return True


class GmailClient:
    """Thin wrapper around the Gmail v1 discovery client."""

    def __init__(self, service: Any) -> None:
        self.service = service
        self._label_cache: dict[str, str] = {}

    @classmethod
    def create(cls) -> GmailClient:
        creds = load_credentials(settings.gmail_credentials_path, settings.gmail_token_path)
        # Fail fast and loudly on a missing REQUIRED scope; a missing optional
        # one is the caller's problem to degrade around (mark_read does).
        assert_required_scopes(creds)
        # WHY cache_discovery=False: the discovery cache lives in a tempdir and
        # emits a noisy warning on Python 3.10+ ("file_cache is only supported
        # with oauth2client<4.0.0"). Turning it off costs a network roundtrip
        # per boot but silences the warning and avoids stale-cache surprises.
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return cls(service)

    def _resolve_label(self, label: str) -> str:
        """Turn a label name into a Gmail label id."""
        if label in SYSTEM_LABEL_IDS:
            return label
        if label in self._label_cache:
            return self._label_cache[label]
        resp = self.service.users().labels().list(userId="me").execute()
        for entry in resp.get("labels", []):
            self._label_cache[entry["name"]] = entry["id"]
        if label not in self._label_cache:
            raise ValueError(
                f"Label {label!r} not found. Available: "
                f"{sorted(self._label_cache) + sorted(SYSTEM_LABEL_IDS)}"
            )
        return self._label_cache[label]

    @retry_external(node="gmail_list")
    def list_message_ids(self, label: str, max_results: int) -> list[str]:
        """Return the most recent up to `max_results` gmail message ids from `label`.

        Handles pagination via nextPageToken. Gmail's per-page cap is 500.
        """
        # TRACE: called once per run from cli/ingest. Emits N/500 HTTP calls
        # where N = ceil(max_results / 500). Order is Gmail's default: newest first.
        label_id = self._resolve_label(label)
        ids: list[str] = []
        page_token: str | None = None

        while len(ids) < max_results:
            remaining = max_results - len(ids)
            resp = (
                self.service.users()
                .messages()
                .list(
                    userId="me",
                    labelIds=[label_id],
                    maxResults=min(remaining, 500),
                    pageToken=page_token,
                )
                .execute()
            )
            for m in resp.get("messages", []):
                ids.append(m["id"])
                if len(ids) >= max_results:
                    break
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return ids

    @retry_external(node="gmail_get")
    def get_message(self, gmail_id: str) -> dict:
        """Return the full-format Gmail message JSON for `gmail_id`.

        format=full returns headers + MIME parts with body data (base64url).
        """
        # ALTERNATIVE: format=raw returns the whole RFC 5322 blob for us to
        # parse ourselves with `email.parser`. Cleaner in theory, but Gmail
        # already parsed the MIME tree for us in `full` — reusing their parser
        # eliminates a class of MIME-corner-case bugs.
        return (
            self.service.users()
            .messages()
            .get(userId="me", id=gmail_id, format="full")
            .execute()
        )

    def profile(self) -> dict:
        return self.service.users().getProfile(userId="me").execute()

    def mark_read(self, gmail_id: str) -> bool:
        """Remove the UNREAD label from a message. True if Gmail confirmed it.

        CONCEPT: this is the only method here that NEVER raises (D68).
          Every other call on this client is load-bearing — a failure to fetch
          or send is a real failure and the caller must hear about it. Marking
          read is bookkeeping that happens AFTER an email has already been
          delivered, and delivery cannot be undone. If this threw, the
          exception would climb into the ingest CLI's per-message handler,
          which dead-letters the message; a later re-ingest would then see no
          record and send the recruiter a second copy. That trades a cosmetic
          problem for the exact duplicate-reply failure this project has been
          burned by.

        TRACE: one POST to users().messages().modify per successful send, and
        only on the send path — skipped messages keep their unread state by
        deliberate choice, so the inbox still shows what the agent ignored.

        GOTCHA: requires the gmail.modify scope. On a token issued before D68
        this returns False and logs, rather than failing the run — which is
        exactly what you want during the window between deploying the code and
        re-running the consent flow.
        """
        try:
            self.service.users().messages().modify(
                userId="me", id=gmail_id, body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            return True
        except Exception as exc:
            log.warning(
                "mark_read failed for gmail_id=%s (%s: %s); the reply was "
                "already sent, so continuing. If this says "
                "insufficientPermissions, delete token.json and re-run "
                "`python -m app.gmail.auth` to consent to gmail.modify.",
                gmail_id, type(exc).__name__, exc,
            )
            return False

    def trash_message(self, gmail_id: str) -> bool:
        """Move a message to Gmail's Trash. True if Gmail confirmed it.

        CONCEPT: trash is not delete, and the difference is the whole safety
          argument. `messages.trash` sets the TRASH label — the message leaves
          the inbox, stops counting against the fetch window, and Gmail purges
          it after 30 days. Until then it is recoverable by the user with one
          click. `messages.delete` is the irreversible one, and it requires the
          full `https://mail.google.com/` scope which this client deliberately
          does not request (see SCOPES above). So the narrow scope is not just
          documentation of intent — it makes the irreversible call impossible
          rather than merely unused.

        TRACE: called only from the inbox_cleanup node, only for messages
        ingest already classified as non-delivery reports by
        app/rules/undeliverable.py, and only when INBOX_CLEANUP_MODE=trash and
        the kill switch is off. Never on the reply path.

        GOTCHA: does not raise, for the same reason as mark_read. This runs
        AFTER persist_terminal has written the message row (D78 ordering), so
        a failure here leaves an accurate database and an untidy inbox — the
        harmless direction. Raising would climb into the ingest CLI's
        per-message handler and dead-letter a message that was processed
        perfectly well.

        GOTCHA: requires the gmail.modify scope, same as mark_read. Returns
        False and logs on a token that predates it.
        """
        try:
            self.service.users().messages().trash(userId="me", id=gmail_id).execute()
            log.info("trashed gmail_id=%s (non-delivery report)", gmail_id)
            return True
        except Exception as exc:
            log.warning(
                "trash_message failed for gmail_id=%s (%s: %s); the message "
                "row is already written, so continuing. If this says "
                "insufficientPermissions, delete token.json and re-run "
                "`python -m app.gmail.auth` to consent to gmail.modify.",
                gmail_id, type(exc).__name__, exc,
            )
            return False

    @retry_external(node="gmail_create_draft")
    def create_draft(
        self,
        to: str,
        subject: str,
        body_text: str,
        in_reply_to_message_id: str | None = None,
        gmail_thread_id: str | None = None,
        attachment_path: Path | None = None,
    ) -> str:
        """Create a Gmail draft in the user's Drafts folder. Returns the
        Gmail draft id (not the RFC 5322 Message-ID — Gmail assigns its
        own id to the draft).

        This is the ONLY Gmail-mutating operation exposed. There is no
        send_message method on this class — that's the code-discipline
        half of D19.

        Params:
            in_reply_to_message_id: the RFC 5322 Message-ID of the mail
                being replied to (with angle brackets, e.g. "<abc@x.com>").
                Sets both In-Reply-To and References headers so the reply
                threads correctly in the recruiter's mail client.
            gmail_thread_id: Gmail's threadId for the original message.
                Setting it makes the draft appear inside the original
                thread in the user's Drafts folder.
        """
        # CONCEPT: Gmail's drafts.create expects a full RFC 5322 message,
        # base64url-encoded, in the `raw` field. build_mime assembles it.
        raw = encode_mime(
            build_mime(to, subject, body_text, in_reply_to_message_id,
                       attachment_path=attachment_path)
        )
        body_dict: dict = {"message": {"raw": raw}}
        if gmail_thread_id:
            body_dict["message"]["threadId"] = gmail_thread_id

        # TRACE: one HTTP POST to Gmail. On success returns {id, message: {...}}.
        # GOTCHA: if the persisted token was consented without gmail.compose
        # — an old readonly-only token, or one from before D40 widened the
        # list — this raises googleapiclient.errors.HttpError 403
        # ("insufficient authentication scopes") even though the code now
        # requests the right scopes. Tokens carry what was granted, not what
        # is asked for later. Delete token.json and re-run the auth flow.
        result = self.service.users().drafts().create(userId="me", body=body_dict).execute()
        return result["id"]

    # WHY max_attempts=1 (no retry) on send_reply: sending a reply is
    # NOT idempotent. If Gmail's send call times out after the request
    # bytes are on the wire, the recruiter may have already received a
    # copy — retrying would send a duplicate. The correct behaviour on
    # ambiguous outcome is to fail loudly (PermanentExternalError →
    # dead_letter → human investigates), NOT to try again. All other
    # Gmail methods are idempotent enough that retry is a net win; send
    # is the singular exception. See D35.
    @retry_external(node="gmail_send", policy=RetryPolicy(max_attempts=1))
    def send_reply(
        self,
        to: str,
        subject: str,
        body_text: str,
        in_reply_to_message_id: str | None = None,
        gmail_thread_id: str | None = None,
        attachment_path: Path | None = None,
    ) -> str:
        """Send a reply message directly. Returns Gmail's assigned message id.

        Called ONLY from the autonomy path (app.pipeline.auto_send_node).
        Every caller must have consulted app.kill_switch.is_send_halted()
        immediately beforehand.

        Threading semantics identical to create_draft: In-Reply-To +
        References make the reply thread correctly in the recruiter's
        mail client; threadId on the message body makes it live in the
        original thread inside the user's Sent folder.

        GOTCHA on retry: intentionally single-attempt. See the class-
        level decorator note above.
        """
        # Same MIME shape as create_draft — a sent reply and a drafted one
        # should be byte-identical apart from which endpoint receives them.
        # Sharing one builder is what guarantees that, so a `draft`-mode soak
        # is honest evidence about what `on` will put in a recruiter's inbox.
        raw = encode_mime(
            build_mime(to, subject, body_text, in_reply_to_message_id,
                       attachment_path=attachment_path)
        )
        body_dict: dict = {"raw": raw}
        if gmail_thread_id:
            body_dict["threadId"] = gmail_thread_id

        # TRACE: single POST to users().messages().send. On success
        # returns {id, threadId, labelIds}. Failure classes are
        # standard Gmail HttpError codes — 403 (scope missing), 400
        # (malformed threadId), 429/5xx (rate/transient — but retry is
        # off, so these become one-shot PermanentExternalError).
        result = (
            self.service.users()
            .messages()
            .send(userId="me", body=body_dict)
            .execute()
        )
        return result["id"]
