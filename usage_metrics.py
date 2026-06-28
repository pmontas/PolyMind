"""
Usage metrics — lightweight event logging for owner analytics.
All public functions are fail-safe; errors never propagate to the bot.
"""
import datetime
import logging
import os
import random

from sqlalchemy import Column, Integer, String, DateTime, func

log = logging.getLogger("polymind.metrics")

METRICS_ENABLED = os.getenv("USAGE_METRICS_ENABLED", "true").lower() in ("1", "true", "yes")
RETENTION_DAYS = int(os.getenv("USAGE_METRICS_RETENTION_DAYS", "90"))
_PRUNE_EVERY_N = 100
_record_count = 0

UsageEvent = None

VALID_EVENTS = frozenset({
    "ask",
    "ask_dm",
    "ask_channel",
    "ask_channel_blocked",
    "summarize_thread_blocked",
    "channel_digest_blocked",
    "quiz_blocked",
    "story_blocked",
    "rate_limited",
    "task_create",
    "task_run",
    "link_shared",
})


def setup_models(base, engine):
    """Register UsageEvent table on the shared Base."""
    global UsageEvent

    class _UsageEvent(base):
        __tablename__ = "usage_events"
        id = Column(Integer, primary_key=True)
        event_type = Column(String(64), index=True)
        guild_id = Column(String, index=True, nullable=True)
        user_id = Column(String, index=True, nullable=True)
        created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    UsageEvent = _UsageEvent
    base.metadata.create_all(engine)
    log.info("UsageEvent table ready.")


def _app():
    import app
    return app


def _maybe_prune(session):
    """Occasionally delete events older than RETENTION_DAYS."""
    global _record_count
    _record_count += 1
    if _record_count % _PRUNE_EVERY_N != 0 or not UsageEvent:
        return
    try:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=RETENTION_DAYS)
        deleted = session.query(UsageEvent).filter(UsageEvent.created_at < cutoff).delete()
        if deleted:
            session.commit()
            log.info("Pruned %s usage events older than %s days.", deleted, RETENTION_DAYS)
    except Exception as e:
        log.debug("Usage metrics prune skipped: %s", e)
        session.rollback()


def record(event_type: str, *, guild_id=None, user_id=None) -> None:
    """Record one usage event. Never raises."""
    if not METRICS_ENABLED or not UsageEvent:
        return
    if event_type not in VALID_EVENTS:
        return
    try:
        session = _app().db_session
        session.add(UsageEvent(
            event_type=event_type,
            guild_id=str(guild_id) if guild_id is not None else None,
            user_id=str(user_id) if user_id is not None else None,
        ))
        session.commit()
        if random.randint(1, _PRUNE_EVERY_N) == 1:
            _maybe_prune(session)
    except Exception as e:
        log.debug("Usage metrics record failed (%s): %s", event_type, e)
        try:
            _app().db_session.rollback()
        except Exception:
            pass


def _since(days: int) -> datetime.datetime:
    return datetime.datetime.utcnow() - datetime.timedelta(days=max(1, min(365, days)))


def get_summary(days: int = 7) -> dict:
    """Aggregate stats for the owner usage report."""
    if not UsageEvent:
        return {}
    session = _app().db_session
    since = _since(days)
    try:
        rows = (
            session.query(UsageEvent.event_type, func.count(UsageEvent.id))
            .filter(UsageEvent.created_at >= since)
            .group_by(UsageEvent.event_type)
            .all()
        )
        by_event = {etype: count for etype, count in rows}

        guild_rows = (
            session.query(UsageEvent.guild_id, func.count(UsageEvent.id))
            .filter(UsageEvent.created_at >= since, UsageEvent.guild_id.isnot(None))
            .group_by(UsageEvent.guild_id)
            .order_by(func.count(UsageEvent.id).desc())
            .limit(10)
            .all()
        )

        user_rows = (
            session.query(func.count(func.distinct(UsageEvent.user_id)))
            .filter(UsageEvent.created_at >= since, UsageEvent.user_id.isnot(None))
            .scalar()
        ) or 0

        guild_user_rows = (
            session.query(UsageEvent.guild_id, func.count(func.distinct(UsageEvent.user_id)))
            .filter(UsageEvent.created_at >= since, UsageEvent.guild_id.isnot(None))
            .group_by(UsageEvent.guild_id)
            .all()
        )
        users_by_guild = {gid: u for gid, u in guild_user_rows}

        premium_blocked = sum(
            v for k, v in by_event.items() if k.endswith("_blocked")
        )

        summary = {
            "days": days,
            "by_event": by_event,
            "top_guilds": [(gid, cnt, users_by_guild.get(gid, 0)) for gid, cnt in guild_rows],
            "unique_users": user_rows,
            "premium_blocked": premium_blocked,
            "total_events": sum(by_event.values()),
            "task_period": {
                "creates": by_event.get("task_create", 0),
                "runs": by_event.get("task_run", 0),
            },
        }
        try:
            import scheduled_tasks
            summary["tasks"] = scheduled_tasks.get_task_stats()
        except Exception as e:
            log.debug("Task stats unavailable: %s", e)
            summary["tasks"] = {}
        return summary
    except Exception as e:
        log.error("get_summary failed: %s", e)
        session.rollback()
        return {}


def get_guild_detail(guild_id: str, days: int = 7) -> dict:
    if not UsageEvent:
        return {}
    session = _app().db_session
    since = _since(days)
    gid = str(guild_id)
    try:
        rows = (
            session.query(UsageEvent.event_type, func.count(UsageEvent.id))
            .filter(UsageEvent.created_at >= since, UsageEvent.guild_id == gid)
            .group_by(UsageEvent.event_type)
            .all()
        )
        users = (
            session.query(func.count(func.distinct(UsageEvent.user_id)))
            .filter(UsageEvent.created_at >= since, UsageEvent.guild_id == gid)
            .scalar()
        ) or 0
        by_event = {etype: count for etype, count in rows}
        detail = {
            "guild_id": gid,
            "days": days,
            "by_event": by_event,
            "unique_users": users,
            "total_events": sum(c for _, c in rows),
            "task_period": {
                "creates": by_event.get("task_create", 0),
                "runs": by_event.get("task_run", 0),
            },
        }
        try:
            import scheduled_tasks
            detail["tasks"] = scheduled_tasks.get_guild_task_stats(gid)
        except Exception as e:
            log.debug("Guild task stats unavailable: %s", e)
            detail["tasks"] = {}
        return detail
    except Exception as e:
        log.error("get_guild_detail failed: %s", e)
        session.rollback()
        return {}


def format_report(summary: dict, bot=None) -> str:
    """Format summary dict as Discord message text."""
    if not summary:
        return "Usage metrics unavailable."

    days = summary.get("days", 7)
    lines = [
        f"**Usage report (last {days} days)**",
        f"Total events: **{summary.get('total_events', 0)}** · "
        f"Unique users: **{summary.get('unique_users', 0)}** · "
        f"Premium gate hits: **{summary.get('premium_blocked', 0)}**",
        "",
        "**Events**",
    ]
    by_event = summary.get("by_event") or {}
    if by_event:
        for etype in sorted(by_event.keys()):
            lines.append(f"• `{etype}`: {by_event[etype]}")
    else:
        lines.append("• No events recorded.")

    lines.extend(_format_task_section(summary, days, bot))

    top = summary.get("top_guilds") or []
    if top:
        lines.extend(["", f"**Top servers**"])
        for i, (gid, cnt, users) in enumerate(top, 1):
            name = gid
            if bot:
                g = bot.get_guild(int(gid))
                if g:
                    name = g.name
            lines.append(f"{i}. **{name}** (`{gid}`) — {cnt} events, {users} user(s)")

    lines.append("")
    return "\n".join(lines)


def format_guild_report(detail: dict, bot=None) -> str:
    if not detail:
        return "No data for that server."
    gid = detail.get("guild_id", "?")
    name = gid
    if bot:
        g = bot.get_guild(int(gid))
        if g:
            name = g.name
    lines = [
        f"**Server: {name}** (`{gid}`) — last {detail.get('days', 7)} days",
        f"Total events: **{detail.get('total_events', 0)}** · "
        f"Unique users: **{detail.get('unique_users', 0)}**",
        "",
    ]
    by_event = detail.get("by_event") or {}
    if by_event:
        for etype in sorted(by_event.keys()):
            lines.append(f"• `{etype}`: {by_event[etype]}")
    else:
        lines.append("• No events recorded.")
    lines.extend(_format_task_section(detail, detail.get("days", 7), bot, guild_only=True))
    return "\n".join(lines)


def _format_task_section(data: dict, days: int, bot=None, guild_only: bool = False) -> list[str]:
    """Format scheduled task stats for Discord reports."""
    tp = data.get("task_period") or {}
    tasks = data.get("tasks") or {}
    has_period = bool(tp.get("creates") or tp.get("runs"))
    has_inventory = bool(tasks.get("total"))
    if not has_period and not has_inventory:
        return []

    lines = ["", "**Scheduled tasks (BETA)**"]
    if tp.get("creates") or tp.get("runs"):
        lines.append(
            f"• Last {days} days: **{tp.get('creates', 0)}** created · **{tp.get('runs', 0)}** runs"
        )
    if not tasks:
        return lines

    if guild_only:
        if tasks.get("total"):
            lines.append(
                f"• This server: **{tasks.get('enabled', 0)}** active · "
                f"**{tasks.get('paused', 0)}** paused · **{tasks.get('total', 0)}** total"
            )
            if tasks.get("total_runs"):
                lines.append(f"• Lifetime runs: **{tasks['total_runs']}**")
            by_type = tasks.get("by_type") or {}
            if by_type:
                lines.append("• By type: " + ", ".join(f"`{k}`: {v}" for k, v in sorted(by_type.items())))
        return lines

    if tasks.get("total"):
        lines.append(
            f"• Live inventory: **{tasks.get('enabled', 0)}** active · "
            f"**{tasks.get('paused', 0)}** paused · **{tasks.get('total', 0)}** total · "
            f"**{tasks.get('guilds_with_tasks', 0)}** server(s)"
        )
        if tasks.get("total_runs"):
            lines.append(f"• Lifetime runs (all tasks): **{tasks['total_runs']}**")
        by_type = tasks.get("by_type") or {}
        if by_type:
            lines.append("• By type: " + ", ".join(f"`{k}`: {v}" for k, v in sorted(by_type.items())))
        top = tasks.get("top_guilds") or []
        if top:
            lines.append("• Top servers by tasks:")
            for i, (gid, cnt) in enumerate(top, 1):
                name = gid
                if bot:
                    g = bot.get_guild(int(gid))
                    if g:
                        name = g.name
                lines.append(f"  {i}. **{name}** — {cnt} task(s)")
    return lines
