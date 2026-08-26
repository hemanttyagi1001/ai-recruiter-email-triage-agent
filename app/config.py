"""
Central configuration for the triage agent. All environment-sourced settings
load through pydantic-settings so we get typed access, validation on startup,
and a single source of truth. Anything that needs a config value imports
`settings` from here rather than reading os.environ directly — this keeps
misconfigured deployments loud (fail at import) instead of silent (fail on
first use of a missing var, halfway through a run).

Loaded once at import time. Instantiating Settings() at module scope means a
missing required var raises before any Gmail / DB / LLM code even boots.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Database ---
    database_url: str = Field(..., description="Primary Postgres connection string")
    test_database_url: str | None = Field(
        default=None,
        description="Separate DB used by pytest. conftest creates/drops tables here.",
    )

    # --- Gmail ---
    # CONCEPT: OAuth2 installed-application flow uses two files.
    # credentials.json is issued by Google Cloud Console for the OAuth *client*
    # itself — it identifies our app to Google but grants no mailbox access.
    # token.json is the per-user access + refresh token pair, created on the
    # first consent screen and rewritten each time the access token refreshes.
    # WHY refresh token matters: access tokens expire in ~1 hour; the refresh
    # token lets us mint a new one without re-prompting the user. Both files
    # are secrets — .gitignore covers them.
    gmail_credentials_path: Path = Field(default=Path("credentials.json"))
    gmail_token_path: Path = Field(default=Path("token.json"))
    gmail_label: str = Field(default="INBOX")
    gmail_max_messages: int = Field(default=200)

    # --- Azure OpenAI ---
    # WHY single deployment for classify + extract in Phase 0: cost > quality
    # trade-off for a learning phase, and the classifier's job is trivial
    # enough for the cheapest model. Splitting into a stronger extract
    # deployment is a deliberate later decision documented in DECISIONS.md.
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = Field(default="2024-10-21")
    azure_openai_deployment: str

    # CONCEPT: model families differ in which sampling knobs they expose.
    # The GPT-5 family are reasoning models — they run an internal
    # deliberation pass before emitting an answer, and they reject any
    # `temperature` other than the default with HTTP 400
    # (code=unsupported_value). GPT-4-class models accept the full 0..2
    # range. Same API, same SDK, incompatible request shapes.
    # WHY a config flag rather than sniffing the deployment name: Azure
    # deployment names are user-chosen. Ours happen to be named after their
    # models (see pricing.py), but nothing enforces that, so the name is not
    # a trustworthy signal of which family we're pointed at.
    # ALTERNATIVE: a `startswith("gpt-5")` check inside client.py — rejected
    # because it breaks silently the first time someone names a deployment
    # something reasonable like "cheap-classifier".
    azure_openai_supports_temperature: bool = Field(default=True)

    # --- Embeddings (Phase 4) ---
    # WHY a separate deployment: Azure requires a distinct deployment per
    # model, and text-embedding-3-small has different pricing + a different
    # response shape than the chat completions deployment. Kept optional at
    # the type level so a dedup-disabled run doesn't force provisioning; the
    # embed() call raises loudly if it fires with this unset.
    azure_openai_embedding_deployment: str | None = Field(default=None)
    # WHY a toggle at all: lets us A/B ingest with and without the dedup
    # pass — same graph shape, dedup nodes are no-ops when this is false.
    # Useful for the Phase 4 benchmark script and for turning the whole
    # feature off if the embedding deployment breaks.
    dedup_enabled: bool = Field(default=True)

    # --- FastAPI (Phase 2) ---
    # WHY 127.0.0.1 default: no auth on Phase 2 endpoints. Binding to a
    # non-loopback interface would expose approve/reject to the local
    # network. Change to 0.0.0.0 only when you've added a reverse proxy
    # with authn/z in front. See DECISIONS.md (no auth is deliberate for
    # this phase — the agent runs on your workstation, one user).
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)

    # --- Autonomy (Phase 6) ---
    # CONCEPT: three states, because "should we send" and "are we allowed to
    # send" are different questions and collapsing them into a bool loses one.
    #   off     — no autonomous send at all. Every draft waits at /pending.
    #             This is the Phase 5 behaviour and the safest setting.
    #   dry_run — walk the entire autonomous path, decide exactly what would
    #             be sent, log it, then DON'T call Gmail. The message lands in
    #             awaiting_approval so you can compare what the agent chose
    #             against what you would have chosen.
    #   draft   — create the reply as a Gmail DRAFT, with no approval step.
    #             Nothing is sent; you review in Gmail and press send yourself.
    #             The intended soak setting: identical recipient resolution,
    #             body and signature to `on`, differing only in which Gmail
    #             API call fires, so days spent here are days of evidence
    #             about what `on` would have done. See D49.
    #   on      — actually send. No human between the classifier and the
    #             recruiter's inbox.
    # WHY dry_run is the default: this setting's failure mode is irreversible
    # (email has no unsend) and its blast radius is your professional
    # reputation. A default that requires a deliberate edit to arm is the only
    # defensible one. See D45.
    # GOTCHA: the kill switch (D36) gates SENDS only. `auto_send_mode=on` with
    # sends_halted=true sends nothing — but `draft` mode creates its Gmail
    # draft regardless of the switch, because a draft is not outbound mail and
    # nothing leaves the mailbox. The asymmetry is deliberate; see D49.
    auto_send_mode: Literal["off", "dry_run", "draft", "on"] = Field(default="dry_run")

    # WHY minutes not seconds: recruiter mail is not time-critical, and a unit
    # that makes "300" mean five hours instead of five minutes is a footgun.
    poll_interval_minutes: int = Field(default=15, gt=0)

    # Typeface for the HTML part of outbound mail.
    # WHY here and not in candidate.toml alongside the signature: this is
    # presentation, not profile, and app/gmail/client.py already imports
    # `settings`. Putting it in the profile would mean threading a
    # CandidateProfile through make_act_node and make_auto_send_node purely to
    # carry one string — churn in two node factories for a font.
    # GOTCHA: the value is interpolated into a CSS style attribute in mail
    # sent to third parties. _css_font_stack strips anything that could close
    # the attribute; a font name is not a place to be clever.
    email_font_family: str = Field(default="Trebuchet MS")

    # CONCEPT: who writes the reply body.
    #   template — str.format over committed .txt files. Cannot hallucinate,
    #             cannot be steered by the email it answers, cannot answer a
    #             question either. This is D12, and it is what the autonomous
    #             send path trusts.
    #   llm      — a tool-free structured_completion writes the body, so the
    #             reply can actually respond to what the recruiter asked.
    # GOTCHA: an LLM-written body is NEVER auto-sent, whatever AUTO_SEND_MODE
    # says. It goes to Gmail Drafts for a human. The safety argument for
    # letting untrusted email into a drafting prompt is the review step, not
    # the prompt wording. See D56.
    draft_mode: Literal["template", "llm"] = Field(default="template")

    # Resume attached to replies, but only when the recruiter asked for one
    # (app/rules/resume_request.py decides that).
    # WHY a path rather than the bytes: the file is bind-mounted into the
    # container and read at send time, so replacing it on the host takes effect
    # on the next reply with no rebuild and no restart.
    # GOTCHA: optional. If unset or missing, replies still go out — just with
    # nothing attached, and a warning in the log. A missing CV must never be
    # the reason a reply fails to reach a recruiter.
    resume_path: Path | None = Field(default=None)

    # --- Observability (LangSmith) ---
    # CONCEPT: this flag arms TRACING. It does not, and must not, control
    # REDACTION. What gets stripped before upload is decided in
    # app/observability/redaction.py — in code, with tests — because "recruiter
    # PII must not reach a vendor" is the kind of constraint this project keeps
    # out of configuration on purpose (same argument as D11 and D14: a rule
    # that must not be violated is a validator, not a setting).
    # WHY default false: traces are a copy of other people's mail leaving the
    # process. Off unless someone deliberately turns it on.
    # GOTCHA: read once at boot and cached (see tracing.get_client). Flipping
    # this needs a restart — unlike the kill switch, which must take effect
    # mid-run.
    langsmith_tracing: bool = Field(default=False)
    # Secret. .env only — never committed, same handling as the Azure key.
    langsmith_api_key: str | None = Field(default=None)
    langsmith_project: str = Field(default="recruiter-triage")

    # --- Logging ---
    log_dir: Path = Field(default=Path("logs"))


# GOTCHA: import-time instantiation. If a required env var is missing, the very
# first `from app.config import settings` in any module (including alembic)
# raises pydantic.ValidationError before doing anything else. That is the
# desired behaviour — do not wrap this in a try/except.
settings = Settings()
