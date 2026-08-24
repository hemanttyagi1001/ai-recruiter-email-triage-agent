"""
Daily digest — summarises the last 24 hours of activity.

CONCEPT: this exists because Phase 5 introduces autonomy.
  Once the agent starts sending on its own, the human's role shifts
  from "review every draft" to "supervise the aggregate." The digest
  is the aggregate: what did the agent do, what's still waiting for
  you, what infra failures piled up while you weren't looking.

  Reading a digest is a habit. Missing a digest for two weeks and
  then finding 40 dead-letters is a much worse experience than seeing
  a running "1 today, 0 today, 3 today" and reacting on the first
  spike. The digest exists so the habit is cheap.

USAGE:
    python -m app.cli.digest                # last 24 hours to stdout
    python -m app.cli.digest --hours 168    # last week
    python -m app.cli.digest --json         # machine-readable
    python -m app.cli.digest --file logs/digest-2026-08-20.txt

CONCEPT: why plain text by default.
  This is meant to be READ. A wall of JSON demands the reader run
  jq mentally; a formatted text block reads at a glance. --json is
  there for the day we wire this into anything automated.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select

from app.config import settings
from app.db.engine import session_scope
from app.db.models import DeadLetter, Draft, DraftStatus, DuplicateFlag, Message, Run
from app.kill_switch import HALT_KEY, is_send_halted


@dataclass
class Digest:
    """The digest payload — one struct so the text and JSON formatters
    read from a single source of truth."""

    period_hours: int
    period_start: str
    period_end: str
    auto_declines_sent: int
    auto_declines_by_rule: dict = field(default_factory=dict)
    awaiting_approval: int = 0
    awaiting_oldest_hours: float | None = None
    dead_lettered: int = 0
    dead_letter_top: list[dict] = field(default_factory=list)
    cost_usd: str = "0.0000"
    cost_breakdown: dict = field(default_factory=dict)
    duplicate_flags_raised: int = 0
    runs_in_period: int = 0
    kill_switch_on: bool = False


def build_digest(hours: int) -> Digest:
    """Query the DB and assemble the digest struct."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    with session_scope() as s:
        # --- Auto-declines sent ---
        # WHY filter on both auto_actioned AND status: defence in depth.
        # A row with auto_actioned=true but status != AUTO_SENT would
        # be a data-inconsistency worth investigating; counting the
        # intersection means the digest is conservative (undercount on
        # inconsistency, not overcount).
        auto_rows = s.execute(
            select(Draft.rule_name).where(
                Draft.auto_actioned.is_(True),
                Draft.status == DraftStatus.AUTO_SENT,
                Draft.resolved_at >= cutoff,
            )
        ).all()
        auto_count = len(auto_rows)
        by_rule: dict[str, int] = {}
        for (rule_name,) in auto_rows:
            key = rule_name or "unknown"
            by_rule[key] = by_rule.get(key, 0) + 1

        # --- Awaiting approval ---
        # WHY status not draft.status: message.status is the
        # single-source-of-truth for the workflow state (drafts.status
        # matches but message is the authoritative row).
        awaiting = s.execute(
            select(func.count()).select_from(Message).where(
                Message.status == "awaiting_approval",
            )
        ).scalar_one()
        # Oldest awaiting — useful for spotting neglected reviews.
        oldest_row = s.execute(
            select(Message.received_at).where(
                Message.status == "awaiting_approval",
            ).order_by(Message.received_at.asc()).limit(1)
        ).scalar_one_or_none()
        oldest_hours = None
        if oldest_row is not None:
            # Normalise both to tz-aware UTC before subtracting; a
            # naive/aware mix raises TypeError.
            if oldest_row.tzinfo is None:
                oldest_row = oldest_row.replace(tzinfo=timezone.utc)
            oldest_hours = round((now - oldest_row).total_seconds() / 3600, 1)

        # --- Dead-lettered ---
        dl_count = s.execute(
            select(func.count()).select_from(DeadLetter).where(
                DeadLetter.occurred_at >= cutoff,
            )
        ).scalar_one()
        # Top-line examples (up to 3) — enough to eyeball what failed
        # without a wall of text.
        dl_top_rows = s.execute(
            select(
                DeadLetter.node, DeadLetter.error_class,
                DeadLetter.error_message, DeadLetter.occurred_at,
            ).where(
                DeadLetter.occurred_at >= cutoff,
            ).order_by(DeadLetter.occurred_at.desc()).limit(3)
        ).all()
        dl_top = [
            {
                "node": r.node,
                "error_class": r.error_class,
                "error_message": (r.error_message or "")[:200],
                "occurred_at": r.occurred_at.isoformat() if r.occurred_at else None,
            }
            for r in dl_top_rows
        ]

        # --- Cost for the period ---
        cost_rows = s.execute(
            select(
                func.coalesce(func.sum(Run.classify_tokens_in), 0),
                func.coalesce(func.sum(Run.classify_tokens_out), 0),
                func.coalesce(func.sum(Run.extract_tokens_in), 0),
                func.coalesce(func.sum(Run.extract_tokens_out), 0),
                func.coalesce(func.sum(Run.score_tokens_in), 0),
                func.coalesce(func.sum(Run.score_tokens_out), 0),
                func.coalesce(func.sum(Run.embed_tokens_in), 0),
                func.coalesce(func.sum(Run.estimated_cost_usd), 0),
                func.count(),
            ).select_from(Run).where(
                Run.finished_at >= cutoff,
            )
        ).one()
        cost_usd = Decimal(str(cost_rows[7] or "0"))
        runs_ct = int(cost_rows[8] or 0)
        cost_breakdown = {
            "classify_tokens_in": int(cost_rows[0]),
            "classify_tokens_out": int(cost_rows[1]),
            "extract_tokens_in": int(cost_rows[2]),
            "extract_tokens_out": int(cost_rows[3]),
            "score_tokens_in": int(cost_rows[4]),
            "score_tokens_out": int(cost_rows[5]),
            "embed_tokens_in": int(cost_rows[6]),
        }

        # --- Duplicate flags raised ---
        dupes = s.execute(
            select(func.count()).select_from(DuplicateFlag).where(
                DuplicateFlag.flagged_at >= cutoff,
            )
        ).scalar_one()

        # --- Kill switch state ---
        # WHY read fresh via is_send_halted rather than SELECT here:
        # gives the digest the same fail-safe semantics as the pipeline.
        # If the row is missing / DB blip, digest reports halted=true,
        # which is honest — the pipeline would refuse to send in that
        # state too.
        halted = is_send_halted(session=s)

    return Digest(
        period_hours=hours,
        period_start=cutoff.isoformat(),
        period_end=now.isoformat(),
        auto_declines_sent=auto_count,
        auto_declines_by_rule=by_rule,
        awaiting_approval=int(awaiting),
        awaiting_oldest_hours=oldest_hours,
        dead_lettered=int(dl_count),
        dead_letter_top=dl_top,
        cost_usd=f"{cost_usd:.4f}",
        cost_breakdown=cost_breakdown,
        duplicate_flags_raised=int(dupes),
        runs_in_period=runs_ct,
        kill_switch_on=halted,
    )


def format_text(d: Digest) -> str:
    """Human-readable formatting. Terminal-friendly width."""
    lines: list[str] = []
    header = f"Digest for the {d.period_hours}h ending {d.period_end[:16]}Z"
    lines.append(header)
    lines.append("-" * len(header))

    rule_str = ""
    if d.auto_declines_by_rule:
        parts = [f"{k}={v}" for k, v in sorted(d.auto_declines_by_rule.items())]
        rule_str = "  (" + " ".join(parts) + ")"
    lines.append(f"Auto-declines sent          : {d.auto_declines_sent}{rule_str}")

    old_str = ""
    if d.awaiting_oldest_hours is not None:
        old_str = f"  (oldest: {d.awaiting_oldest_hours}h)"
    lines.append(f"Awaiting human approval     : {d.awaiting_approval}{old_str}")

    lines.append(f"Dead-lettered failures      : {d.dead_lettered}")
    for i, dl in enumerate(d.dead_letter_top, start=1):
        lines.append(f"  ({i}) [{dl['node']}] {dl['error_class']}: {dl['error_message']}")

    cb = d.cost_breakdown
    cost_detail = (
        f"classify ${_est_cost(cb['classify_tokens_in'], cb['classify_tokens_out']):.4f} "
        f"extract ${_est_cost(cb['extract_tokens_in'], cb['extract_tokens_out']):.4f} "
        f"score ${_est_cost(cb['score_tokens_in'], cb['score_tokens_out']):.4f} "
        f"embed ${_est_cost(cb['embed_tokens_in'], 0, is_embed=True):.4f}"
    )
    lines.append(f"Cost for the period         : ${d.cost_usd}  ({cost_detail})")

    lines.append(f"Duplicate flags raised      : {d.duplicate_flags_raised}")
    lines.append(f"Runs in period              : {d.runs_in_period}")
    kill_str = "ON  (sends halted)" if d.kill_switch_on else "OFF"
    lines.append(f"Kill switch                 : {kill_str}")
    return "\n".join(lines) + "\n"


def _est_cost(tokens_in: int, tokens_out: int, *, is_embed: bool = False) -> float:
    """Rough per-node cost estimate for the digest breakdown.

    WHY not import estimate_cost from app.llm.pricing: pricing is per
    deployment/model name, and the aggregated tokens in the runs table
    don't retain the model name. For the digest a rough estimate is
    enough; the authoritative number is Run.estimated_cost_usd, which
    is what the top-line "Cost for the period" reports.
    """
    # gpt-4o-mini reference pricing (2026-08 approx): $0.15/M in, $0.60/M out
    # text-embedding-3-small: $0.02/M in
    if is_embed:
        return tokens_in * 0.02 / 1_000_000
    return tokens_in * 0.15 / 1_000_000 + tokens_out * 0.60 / 1_000_000


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Daily digest of triage activity.")
    p.add_argument("--hours", type=int, default=24, help="Lookback window (default: 24).")
    p.add_argument("--json", action="store_true", help="Machine-readable output.")
    p.add_argument("--file", type=Path, default=None,
                   help="Also write to this file (append-safe: overwrites).")
    p.add_argument("--auto-file", action="store_true",
                   help="Write to logs/digest-YYYY-MM-DD.txt alongside stdout.")
    args = p.parse_args(argv)

    d = build_digest(args.hours)
    if args.json:
        output = json.dumps(asdict(d), indent=2, default=str) + "\n"
    else:
        output = format_text(d)

    sys.stdout.write(output)

    # Optional file sink — makes the digest a rolling history without
    # a mail server or scheduler. cron / Task Scheduler runs this
    # daily; the file archive answers "what did last Tuesday look like?"
    target: Path | None = args.file
    if args.auto_file and target is None:
        settings.log_dir.mkdir(exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target = settings.log_dir / f"digest-{today}.txt"
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
