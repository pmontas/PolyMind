"""
Scheduled Tasks (BETA) — isolated module for /task automation.
Loaded lazily from app.py on_ready; failures here must not crash the core bot.
"""
import asyncio
import datetime
import logging
import os
import re
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp
import discord
from discord import app_commands
import feedparser
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import Column, Integer, String, Boolean, DateTime

log = logging.getLogger("polymind.tasks")

# --- Config ---
TASK_LIMIT_DEFAULT = int(os.getenv("TASK_LIMIT_DEFAULT", "5"))
TASK_LIMIT_PREMIUM = int(os.getenv("TASK_LIMIT_PREMIUM", "10"))
DEFAULT_TZ = os.getenv("DEFAULT_TZ", "UTC")
TASK_API_BUDGET_HOUR = int(os.getenv("TASK_API_BUDGET_HOUR", "200"))
VALID_TASK_TYPES = frozenset({"ai", "rss", "digest", "research"})
MAX_CONSECUTIVE_FAILURES = 3

ScheduledTask = None
_task_scheduler = None
_commands_registered = False
_task_api_usage: list[float] = []
_task_failures: dict[int, int] = {}  # task_id -> consecutive failure count

TIME_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})$")


def setup_models(base, engine):
    """Register ScheduledTask on the shared SQLAlchemy Base and ensure table exists."""
    global ScheduledTask

    class _ScheduledTask(base):
        __tablename__ = "scheduled_tasks"
        id = Column(Integer, primary_key=True)
        user_id = Column(String, index=True)
        guild_id = Column(String, index=True)
        channel_id = Column(String, index=True)
        task_type = Column(String)
        payload = Column(String)
        hour = Column(Integer)
        minute = Column(Integer)
        timezone = Column(String, default="UTC")
        enabled = Column(Boolean, default=True)
        last_run = Column(DateTime, nullable=True)
        run_count = Column(Integer, default=0)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)

    ScheduledTask = _ScheduledTask
    base.metadata.create_all(engine)
    log.info("ScheduledTask table ready.")
    return ScheduledTask


def _app():
    import app
    return app


def _cleanup_task_api_usage():
    global _task_api_usage
    now = time.time()
    _task_api_usage = [t for t in _task_api_usage if now - t < 3600]


def can_make_task_api_call() -> bool:
    """Task-specific hourly API budget (in addition to global limiter in app)."""
    _cleanup_task_api_usage()
    if TASK_API_BUDGET_HOUR <= 0:
        return True
    return len(_task_api_usage) < TASK_API_BUDGET_HOUR


def record_task_api_call():
    _task_api_usage.append(time.time())


def parse_time(time_str: str) -> tuple[int, int] | None:
    m = TIME_PATTERN.match((time_str or "").strip())
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def validate_timezone(tz_name: str) -> str | None:
    name = (tz_name or DEFAULT_TZ or "UTC").strip()
    try:
        ZoneInfo(name)
        return name
    except ZoneInfoNotFoundError:
        return None


def guild_has_premium_subscriber(guild: discord.Guild) -> bool:
    """True if any entitled premium user is a member of this guild."""
    if not guild:
        return False
    app = _app()
    try:
        rows = app.db_session.query(app.UserEntitlement).all()
        for ent in rows:
            if app.has_premium(ent.user_id) and guild.get_member(int(ent.user_id)):
                return True
    except Exception as e:
        log.error("guild_has_premium_subscriber error: %s", e)
    return False


def get_task_limit_for_guild(guild: discord.Guild) -> int:
    if guild_has_premium_subscriber(guild):
        return TASK_LIMIT_PREMIUM
    return TASK_LIMIT_DEFAULT


def count_active_tasks(guild_id: str) -> int:
    app = _app()
    if not ScheduledTask:
        return 0
    return (
        app.db_session.query(ScheduledTask)
        .filter(ScheduledTask.guild_id == str(guild_id), ScheduledTask.enabled == True)
        .count()
    )


def is_task_due(task, now_utc: datetime.datetime | None = None) -> bool:
    now_utc = now_utc or datetime.datetime.utcnow()
    tz_name = validate_timezone(task.timezone) or "UTC"
    tz = ZoneInfo(tz_name)
    local_now = now_utc.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
    if local_now.hour != task.hour or local_now.minute != task.minute:
        return False
    if task.last_run:
        last_utc = task.last_run.replace(tzinfo=datetime.timezone.utc)
        if last_utc.astimezone(tz).date() == local_now.date():
            return False
    return True


def get_due_tasks() -> list:
    app = _app()
    if not ScheduledTask:
        return []
    now = datetime.datetime.utcnow()
    try:
        tasks = (
            app.db_session.query(ScheduledTask)
            .filter(ScheduledTask.enabled == True)
            .all()
        )
        return [t for t in tasks if is_task_due(t, now)]
    except Exception as e:
        log.error("get_due_tasks error: %s", e)
        app.db_session.rollback()
        return []


def _pause_task(task_id: int, reason: str):
    app = _app()
    try:
        task = app.db_session.query(ScheduledTask).filter(ScheduledTask.id == task_id).first()
        if task:
            task.enabled = False
            app.db_session.commit()
            log.warning("Task #%s auto-paused: %s", task_id, reason)
    except Exception as e:
        log.error("Failed to pause task #%s: %s", task_id, e)
        app.db_session.rollback()


def _record_task_success(task_id: int):
    app = _app()
    _task_failures.pop(task_id, None)
    try:
        task = app.db_session.query(ScheduledTask).filter(ScheduledTask.id == task_id).first()
        if task:
            task.last_run = datetime.datetime.utcnow()
            task.run_count = (task.run_count or 0) + 1
            app.db_session.commit()
    except Exception as e:
        log.error("Failed to update task #%s after run: %s", task_id, e)
        app.db_session.rollback()


def _record_task_failure(task_id: int, reason: str):
    count = _task_failures.get(task_id, 0) + 1
    _task_failures[task_id] = count
    log.error("Task #%s failed (%s/%s): %s", task_id, count, MAX_CONSECUTIVE_FAILURES, reason)
    if count >= MAX_CONSECUTIVE_FAILURES:
        _pause_task(task_id, reason)


async def fetch_rss_items(feed_urls: list[str]) -> str:
    """Fetch and format RSS entries from URLs. Returns combined text or empty string."""
    items = []
    async with aiohttp.ClientSession() as session:
        for url in feed_urls[:10]:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        continue
                    body = await resp.text()
            except Exception as e:
                log.error("RSS fetch error %s: %s", url, e)
                continue
            try:
                feed = await asyncio.to_thread(feedparser.parse, body)
            except Exception as e:
                log.error("RSS parse error %s: %s", url, e)
                continue
            for entry in feed.entries[:5]:
                title = getattr(entry, "title", "") or ""
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
                summary = re.sub(r"<[^>]+>", "", summary)[:300]
                items.append(f"- {title}\n  {summary}")
    return "\n".join(items)[:4000]


async def run_scheduled_task(task) -> None:
    """Execute one scheduled task. Errors are contained — never propagate to caller."""
    app = _app()
    bot = app.bot

    if not app.can_make_api_call():
        log.warning("Task #%s skipped: global API limit reached", task.id)
        return
    if not can_make_task_api_call():
        log.warning("Task #%s skipped: task API budget exhausted", task.id)
        return

    channel = bot.get_channel(int(task.channel_id))
    if not channel or not isinstance(channel, discord.TextChannel):
        _record_task_failure(task.id, "channel not found")
        return

    perms = channel.permissions_for(channel.guild.me)
    if not perms.send_messages:
        _record_task_failure(task.id, "no send permission")
        return

    task_type = (task.task_type or "").lower()
    payload = (task.payload or "").strip()
    result = None
    header = f"**Scheduled task #{task.id}** ({task_type})\n\n"

    try:
        if task_type == "rss":
            urls = [u.strip() for u in payload.split(",") if u.strip()]
            if not urls:
                _record_task_failure(task.id, "no feed URLs")
                return
            text = await fetch_rss_items(urls)
            if not text:
                _record_task_failure(task.id, "no RSS items fetched")
                return
            instruction = (
                "Summarize the top 3 most important or interesting items from this list "
                "in one short paragraph each. Be concise."
            )
            result = await app.get_summary_response(instruction, text)
            record_task_api_call()

        elif task_type == "digest":
            if not payload:
                payload = "Summarize today's main topics in 5 bullet points."
            transcript, err = await app.fetch_channel_transcript(channel, days_back=1)
            if err:
                _record_task_failure(task.id, err)
                return
            result = await app.get_summary_response(payload, transcript)
            record_task_api_call()

        elif task_type == "ai":
            if not payload:
                _record_task_failure(task.id, "empty prompt")
                return
            result = await app.get_gemini_response(payload)
            record_task_api_call()

        elif task_type == "research":
            if not payload:
                _record_task_failure(task.id, "empty research prompt")
                return
            result = await app.get_gemini_grounded_response(payload)
            record_task_api_call()

        else:
            _record_task_failure(task.id, f"unknown type {task_type}")
            return

        if not result or result.startswith("Something went wrong") or result.startswith("The bot is currently"):
            _record_task_failure(task.id, "AI returned error or empty")
            return

        for chunk in app.split_message(header + result):
            await channel.send(chunk[:2000])
        _record_task_success(task.id)
        try:
            if hasattr(app, "track"):
                app.track("task_run", guild_id=task.guild_id, user_id=task.user_id)
        except Exception:
            pass

    except Exception as e:
        log.exception("Task #%s execution error: %s", task.id, e)
        _record_task_failure(task.id, str(e))


async def process_due_tasks():
    """Run all tasks due this minute, each isolated."""
    due = get_due_tasks()
    if not due:
        return
    log.info("Processing %s due scheduled task(s)", len(due))
    for task in due:
        try:
            await run_scheduled_task(task)
        except Exception as e:
            log.exception("Unhandled error in task #%s: %s", task.id, e)
            _record_task_failure(task.id, str(e))


def _task_tick_job():
    """APScheduler job: dispatch due tasks on the bot event loop."""
    app = _app()
    bot = app.bot
    if not bot or not bot.loop:
        return
    try:
        fut = asyncio.run_coroutine_threadsafe(process_due_tasks(), bot.loop)
        fut.result(timeout=300)
    except Exception as e:
        log.error("Task tick job error: %s", e)


def start_task_scheduler(bot) -> None:
    global _task_scheduler
    if _task_scheduler is not None:
        return
    try:
        _task_scheduler = BackgroundScheduler()
        _task_scheduler.add_job(_task_tick_job, "interval", minutes=1, id="task_tick", replace_existing=True)
        _task_scheduler.start()
        log.info("Scheduled tasks tick scheduler started (every 60s).")
    except Exception as e:
        log.error("Failed to start task scheduler: %s", e)


def _can_manage_task(interaction: discord.Interaction, task) -> bool:
    if str(interaction.user.id) == str(task.user_id):
        return True
    if interaction.guild and interaction.user.guild_permissions.manage_guild:
        return True
    return False


def _task_type_label(t: str) -> str:
    return {
        "rss": "RSS news digest",
        "ai": "AI prompt",
        "digest": "Channel digest",
        "research": "Live web research",
    }.get(t, t)


# --- Slash command group ---
task_group = app_commands.Group(
    name="task",
    description="Scheduled Tasks (BETA) — automate daily posts to a channel",
)


@task_group.command(
    name="create",
    description="Schedule a daily post. Types: rss (feeds), ai (prompt), digest (channel summary), research (live web).",
)
@app_commands.describe(
    type="Task type: rss, ai, digest, or research",
    time="Daily run time in HH:MM (24-hour, e.g. 08:00)",
    channel="Channel where the bot will post",
    prompt="Prompt for ai, digest, or research tasks",
    feeds="Comma-separated RSS feed URLs (rss tasks only)",
    timezone="IANA timezone (e.g. America/New_York). Default: UTC",
)
@app_commands.choices(type=[
    app_commands.Choice(name="RSS news digest", value="rss"),
    app_commands.Choice(name="AI prompt (no live web)", value="ai"),
    app_commands.Choice(name="Channel digest", value="digest"),
    app_commands.Choice(name="Live web research", value="research"),
])
async def task_create_slash(
    interaction: discord.Interaction,
    type: app_commands.Choice[str],
    time: str,
    channel: discord.TextChannel,
    prompt: str = "",
    feeds: str = "",
    timezone: str = "",
):
    await interaction.response.defer(ephemeral=True)
    if not interaction.guild:
        await interaction.followup.send("Scheduled tasks only work in a server.", ephemeral=True)
        return

    task_type = type.value
    parsed = parse_time(time)
    if not parsed:
        await interaction.followup.send(
            "Invalid time format. Use **HH:MM** in 24-hour format (e.g. `08:00`, `18:30`).",
            ephemeral=True,
        )
        return
    hour, minute = parsed

    tz = validate_timezone(timezone or DEFAULT_TZ)
    if not tz:
        await interaction.followup.send(
            f"Invalid timezone `{timezone or DEFAULT_TZ}`. Use an IANA name like `America/New_York` or `UTC`.",
            ephemeral=True,
        )
        return

    perms = channel.permissions_for(interaction.guild.me)
    if not perms.send_messages:
        await interaction.followup.send(
            f"I don't have **Send Messages** permission in {channel.mention}. Fix permissions and try again.",
            ephemeral=True,
        )
        return

    if task_type in ("ai", "digest", "research") and not (prompt or "").strip():
        await interaction.followup.send(
            f"**{task_type}** tasks require a **prompt**. Describe what to post each day.",
            ephemeral=True,
        )
        return
    if task_type == "rss" and not (feeds or "").strip():
        await interaction.followup.send(
            "**rss** tasks require **feeds** — comma-separated RSS URLs (e.g. BBC, Hacker News).",
            ephemeral=True,
        )
        return

    payload = (feeds if task_type == "rss" else prompt).strip()[:2000]
    limit = get_task_limit_for_guild(interaction.guild)
    active = count_active_tasks(interaction.guild.id)
    if active >= limit:
        await interaction.followup.send(
            f"This server has **{active}/{limit}** active tasks (BETA). "
            "Delete or pause a task, or upgrade to Premium for up to **10** tasks per server.",
            ephemeral=True,
        )
        return

    app = _app()
    if not ScheduledTask:
        await interaction.followup.send("Scheduled tasks are not available right now.", ephemeral=True)
        return

    try:
        row = ScheduledTask(
            user_id=str(interaction.user.id),
            guild_id=str(interaction.guild.id),
            channel_id=str(channel.id),
            task_type=task_type,
            payload=payload,
            hour=hour,
            minute=minute,
            timezone=tz,
            enabled=True,
        )
        app.db_session.add(row)
        app.db_session.commit()
        task_id = row.id
    except Exception as e:
        log.error("task create DB error: %s", e)
        app.db_session.rollback()
        await interaction.followup.send("Could not save the task. Try again later.", ephemeral=True)
        return

    try:
        app_mod = _app()
        if hasattr(app_mod, "track"):
            app_mod.track("task_create", guild_id=interaction.guild.id, user_id=interaction.user.id)
    except Exception:
        pass

    msg = (
        f"✅ **Task #{task_id} created** ({_task_type_label(task_type)})\n"
        f"Runs daily at **{hour:02d}:{minute:02d}** ({tz}) → {channel.mention}\n"
        f"Active tasks: **{active + 1}/{limit}** (BETA)"
    )
    if task_type == "research":
        msg += (
            "\n\n⚠️ Live research uses more API quota and may require paid Gemini billing. "
            "If you have specific news sources, **rss** with feed URLs is cheaper."
        )
    await interaction.followup.send(msg, ephemeral=True)


@task_group.command(name="list", description="List this server's scheduled tasks and active count.")
async def task_list_slash(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    if not interaction.guild:
        await interaction.followup.send("Use this in a server.", ephemeral=True)
        return

    app = _app()
    if not ScheduledTask:
        await interaction.followup.send("Scheduled tasks are not available.", ephemeral=True)
        return

    limit = get_task_limit_for_guild(interaction.guild)
    tasks = (
        app.db_session.query(ScheduledTask)
        .filter(ScheduledTask.guild_id == str(interaction.guild.id))
        .order_by(ScheduledTask.id)
        .all()
    )
    active = sum(1 for t in tasks if t.enabled)

    lines = [f"**Scheduled Tasks (BETA)** — **{active}/{limit}** active\n"]
    if not tasks:
        lines.append("No tasks yet. Use `/task create` to schedule one.")
    else:
        for t in tasks:
            ch = interaction.guild.get_channel(int(t.channel_id))
            ch_name = ch.mention if ch else f"`#{t.channel_id}`"
            status = "🟢 active" if t.enabled else "⏸ paused"
            tz = t.timezone or "UTC"
            lines.append(
                f"**#{t.id}** · `{t.task_type}` · {t.hour:02d}:{t.minute:02d} {tz} → {ch_name} · {status}"
            )
            if t.run_count:
                lines.append(f"   ↳ ran **{t.run_count}** time(s)")

    await interaction.followup.send("\n".join(lines)[:2000], ephemeral=True)


@task_group.command(name="delete", description="Delete a scheduled task (frees a slot). Creator or Manage Server only.")
@app_commands.describe(task_id="Task ID from /task list")
async def task_delete_slash(interaction: discord.Interaction, task_id: int):
    await interaction.response.defer(ephemeral=True)
    if not interaction.guild:
        await interaction.followup.send("Use this in a server.", ephemeral=True)
        return

    app = _app()
    task = app.db_session.query(ScheduledTask).filter(ScheduledTask.id == task_id).first() if ScheduledTask else None
    if not task or str(task.guild_id) != str(interaction.guild.id):
        await interaction.followup.send(f"Task **#{task_id}** not found in this server.", ephemeral=True)
        return
    if not _can_manage_task(interaction, task):
        await interaction.followup.send("Only the task creator or someone with **Manage Server** can delete this.", ephemeral=True)
        return

    try:
        app.db_session.delete(task)
        app.db_session.commit()
        _task_failures.pop(task_id, None)
    except Exception as e:
        log.error("task delete error: %s", e)
        app.db_session.rollback()
        await interaction.followup.send("Could not delete the task.", ephemeral=True)
        return

    await interaction.followup.send(f"✅ Task **#{task_id}** deleted.", ephemeral=True)


@task_group.command(name="pause", description="Pause a scheduled task (does not count toward the active limit).")
@app_commands.describe(task_id="Task ID from /task list")
async def task_pause_slash(interaction: discord.Interaction, task_id: int):
    await interaction.response.defer(ephemeral=True)
    if not interaction.guild:
        await interaction.followup.send("Use this in a server.", ephemeral=True)
        return

    app = _app()
    task = app.db_session.query(ScheduledTask).filter(ScheduledTask.id == task_id).first() if ScheduledTask else None
    if not task or str(task.guild_id) != str(interaction.guild.id):
        await interaction.followup.send(f"Task **#{task_id}** not found.", ephemeral=True)
        return
    if not _can_manage_task(interaction, task):
        await interaction.followup.send("Only the task creator or someone with **Manage Server** can pause this.", ephemeral=True)
        return
    if not task.enabled:
        await interaction.followup.send(f"Task **#{task_id}** is already paused.", ephemeral=True)
        return

    task.enabled = False
    app.db_session.commit()
    await interaction.followup.send(f"⏸ Task **#{task_id}** paused.", ephemeral=True)


@task_group.command(name="resume", description="Resume a paused task (blocked if server is at the active task cap).")
@app_commands.describe(task_id="Task ID from /task list")
async def task_resume_slash(interaction: discord.Interaction, task_id: int):
    await interaction.response.defer(ephemeral=True)
    if not interaction.guild:
        await interaction.followup.send("Use this in a server.", ephemeral=True)
        return

    app = _app()
    task = app.db_session.query(ScheduledTask).filter(ScheduledTask.id == task_id).first() if ScheduledTask else None
    if not task or str(task.guild_id) != str(interaction.guild.id):
        await interaction.followup.send(f"Task **#{task_id}** not found.", ephemeral=True)
        return
    if not _can_manage_task(interaction, task):
        await interaction.followup.send("Only the task creator or someone with **Manage Server** can resume this.", ephemeral=True)
        return
    if task.enabled:
        await interaction.followup.send(f"Task **#{task_id}** is already active.", ephemeral=True)
        return

    limit = get_task_limit_for_guild(interaction.guild)
    active = count_active_tasks(interaction.guild.id)
    if active >= limit:
        await interaction.followup.send(
            f"This server has **{active}/{limit}** active tasks. Pause or delete another task first.",
            ephemeral=True,
        )
        return

    channel = interaction.guild.get_channel(int(task.channel_id))
    if channel and isinstance(channel, discord.TextChannel):
        perms = channel.permissions_for(interaction.guild.me)
        if not perms.send_messages:
            await interaction.followup.send(
                f"I still can't send messages in the task's channel. Fix permissions first.",
                ephemeral=True,
            )
            return

    task.enabled = True
    _task_failures.pop(task_id, None)
    app.db_session.commit()
    await interaction.followup.send(f"🟢 Task **#{task_id}** resumed.", ephemeral=True)


@task_group.command(name="help", description="How scheduled tasks work, limits, and examples.")
async def task_help_slash(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Scheduled Tasks (BETA)",
        description="Schedule PolyMind to post to a channel **every day** at a set time.",
        color=discord.Color.blue(),
    )
    embed.add_field(
        name="Task types",
        value=(
            "**rss** — Fetch RSS feed URLs, AI summarizes, posts to channel. *Cheapest for news.*\n"
            "**ai** — Run your custom prompt daily (no live web).\n"
            "**digest** — Summarize the channel's last 24h of messages and post the recap.\n"
            "**research** — Gemini searches the web for current info. *Uses more API quota.*"
        ),
        inline=False,
    )
    embed.add_field(
        name="Limits",
        value=(
            f"**{TASK_LIMIT_DEFAULT}** active tasks per server (BETA) — **{TASK_LIMIT_PREMIUM}** with Premium.\n"
            "Once per day per task. Paused tasks don't count toward the cap.\n"
            f"Default timezone: **{DEFAULT_TZ}** (override with `timezone` on create)."
        ),
        inline=False,
    )
    embed.add_field(
        name="Examples",
        value=(
            "`/task create` type **rss** time `08:00` channel #news "
            "feeds `https://feeds.bbci.co.uk/news/technology/rss.xml`\n"
            "`/task create` type **ai** time `09:00` channel #general "
            "prompt `Post one short motivational quote for developers.`\n"
            "`/task list` · `/task pause 2` · `/task resume 2` · `/task delete 1`"
        ),
        inline=False,
    )
    embed.set_footer(text="BETA — feedback welcome via /feedback or /suggestfeature. Limits may change.")
    await interaction.response.send_message(embed=embed, ephemeral=True)


def register_task_commands(bot) -> None:
    global _commands_registered
    if _commands_registered:
        return
    try:
        bot.tree.add_command(task_group)
        _commands_registered = True
        log.info("Registered /task command group.")
    except Exception as e:
        log.error("Failed to register /task commands: %s", e)
