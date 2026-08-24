"""
Report CLI — plain-text summary of a run. Phase 1 adds:
  - decision-layer counts (rules-declined, scored, needs-review)
  - draft counts (created, quarantined) with per-draft reasons
  - score-node tokens and cost

Usage:
    python -m app.cli.report --latest
    python -m app.cli.report --run <uuid>
"""

from __future__ import annotations

import argparse
import uuid
from decimal import Decimal

from sqlalchemy import func, select

from app.config import settings
from app.db.engine import session_scope
from app.db.models import Draft, DraftStatus, Message, MessageStatus, Run
from app.llm.pricing import estimate_cost


def main() -> None:
    ap = argparse.ArgumentParser(description="Report on a triage run")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=str, help="Run UUID")
    g.add_argument("--latest", action="store_true", help="Most recent run")
    args = ap.parse_args()

    with session_scope() as s:
        run = _load_run(s, args)
        if run is None:
            print("No matching run.")
            return

        category_rows = s.execute(
            select(Message.category, func.count().label("n"))
            .where(Message.run_id == run.id)
            .group_by(Message.category)
            .order_by(func.count().desc())
        ).all()

        extracted = (
            s.query(Message)
            .filter_by(run_id=run.id, status=MessageStatus.EXTRACTED)
            .count()
        )
        failed_rows = (
            s.query(Message)
            .filter_by(run_id=run.id, status=MessageStatus.EXTRACTION_FAILED)
            .all()
        )

        # Draft breakdown — join messages to keep the query scoped to this run.
        draft_rows = (
            s.query(Draft, Message.subject)
            .join(Message, Draft.message_id == Message.message_id)
            .filter(Message.run_id == run.id)
            .all()
        )

        snapshot = _Snapshot(
            run_id=run.id,
            started_at=run.started_at,
            finished_at=run.finished_at,
            status=run.status,
            category_rows=[(c, n) for c, n in category_rows],
            extracted=extracted,
            failures=[(m.subject, m.extraction_error) for m in failed_rows],
            classify_in=run.classify_tokens_in,
            classify_out=run.classify_tokens_out,
            extract_in=run.extract_tokens_in,
            extract_out=run.extract_tokens_out,
            score_in=run.score_tokens_in,
            score_out=run.score_tokens_out,
            total_cost=Decimal(str(run.estimated_cost_usd or 0)),
            rules_declined=run.rules_declined,
            messages_scored=run.messages_scored,
            messages_needs_review=run.messages_needs_review,
            drafts_created=run.drafts_created,
            drafts_quarantined=run.drafts_quarantined,
            drafts=[
                (
                    d.draft_type,
                    d.status,
                    d.quarantine_reason,
                    d.rule_name,
                    d.fit_score,
                    d.fit_uncertain,
                    subj,
                )
                for d, subj in draft_rows
            ],
        )

    _print_report(snapshot)


class _Snapshot:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _load_run(session, args) -> Run | None:
    if args.run:
        try:
            run_uuid = uuid.UUID(args.run)
        except ValueError:
            print(f"Not a valid UUID: {args.run}")
            return None
        return session.get(Run, run_uuid)
    return session.execute(
        select(Run).order_by(Run.started_at.desc()).limit(1)
    ).scalar_one_or_none()


def _print_report(snap: _Snapshot) -> None:
    deployment = settings.azure_openai_deployment
    classify_cost = estimate_cost(deployment, snap.classify_in, snap.classify_out)
    extract_cost = estimate_cost(deployment, snap.extract_in, snap.extract_out)
    score_cost = estimate_cost(deployment, snap.score_in, snap.score_out)

    rule = "─" * 68
    duration = (
        str(snap.finished_at - snap.started_at).split(".")[0]
        if snap.finished_at else "in-progress"
    )
    print(f"Run {snap.run_id}")
    print(f"  started {snap.started_at:%Y-%m-%d %H:%M %Z}   status {snap.status}   duration {duration}")
    print(rule)

    print(f"{'Category':<40}{'Count':>10}")
    for cat, n in snap.category_rows:
        print(f"  {(cat or '(unclassified)'):<38}{n:>10}")
    if not snap.category_rows:
        print("  (none)")
    print(rule)

    attempted = snap.extracted + snap.drafts_created + len(snap.failures)
    print(
        f"Decision layer: rules-declined {snap.rules_declined}, "
        f"scored {snap.messages_scored}, "
        f"needs-review {snap.messages_needs_review}"
    )
    print(
        f"Drafts: {snap.drafts_created} created, "
        f"{snap.drafts_quarantined} quarantined, "
        f"{snap.drafts_created - snap.drafts_quarantined} pending-review"
    )
    if snap.drafts:
        print("Draft breakdown:")
        for dtype, dstatus, qreason, rname, fscore, funcertain, subj in snap.drafts:
            marker = "!" if dstatus == DraftStatus.QUARANTINED else " "
            detail = _draft_detail(dtype, rname, fscore, funcertain, qreason)
            print(f"  {marker} [{dtype:<10}][{dstatus:<15}] {(subj or '')[:36]:<36} {detail}")
    print(rule)

    if snap.failures:
        print(f"Extraction failures: {len(snap.failures)}")
        for subject, error in snap.failures:
            first_line = (error or "").splitlines()[0] if error else ""
            print(f"  {(subject or '')[:50]:<50}  {first_line[:60]}")
        print(rule)

    print(f"{'Tokens':<12}{'in':>12}{'out':>12}")
    print(f"  {'classify':<10}{snap.classify_in:>12}{snap.classify_out:>12}")
    print(f"  {'extract':<10}{snap.extract_in:>12}{snap.extract_out:>12}")
    print(f"  {'score':<10}{snap.score_in:>12}{snap.score_out:>12}")
    print(
        f"Cost:  classify ${classify_cost:.4f}   extract ${extract_cost:.4f}   "
        f"score ${score_cost:.4f}   total ${snap.total_cost:.4f}"
    )


def _draft_detail(dtype, rname, fscore, funcertain, qreason) -> str:
    if qreason:
        return f"quarantine={qreason[:40]}"
    if rname:
        return f"rule={rname}"
    if fscore is not None:
        return f"score={fscore}{' (uncertain)' if funcertain else ''}"
    return ""


if __name__ == "__main__":
    main()
