import discord
from discord.ext import commands
import os
import re
import aiohttp
import asyncio
import logging
from dotenv import load_dotenv
import datetime
import google.generativeai as genai
from flask import Flask, render_template, jsonify, request
from threading import Thread
import traceback
import time
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import feedparser
from apscheduler.schedulers.background import BackgroundScheduler

# Load environment variables
load_dotenv()

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("polymind")

# --- DATABASE SETUP ---
Base = declarative_base()


class BotFeature(Base):
    __tablename__ = 'bot_features'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    description = Column(String)
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    category = Column(String)


class UserEntitlement(Base):
    __tablename__ = 'user_entitlements'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    sku_id = Column(String)
    ends_at = Column(DateTime, nullable=True)


class UserMemory(Base):
    __tablename__ = 'user_memories'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    value = Column(String)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)


class UserPersona(Base):
    __tablename__ = 'user_personas'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, unique=True)
    persona_id = Column(String, default="helpful")


class StorySession(Base):
    __tablename__ = 'story_sessions'
    id = Column(Integer, primary_key=True)
    channel_id = Column(String, index=True)
    is_dm = Column(Boolean, default=False)
    state_summary = Column(String)
    last_response = Column(String)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)


# Use /home/ on Azure for persistent storage; fallback to local for dev
DB_PATH = os.getenv("DB_PATH", "polymind.db")
if os.path.isdir("/home"):
    DB_PATH = "/home/polymind.db"

engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={'check_same_thread': False})
Base.metadata.create_all(engine)
session_factory = sessionmaker(bind=engine)
db_session = scoped_session(session_factory)


def init_db():
    """Initialize default features if database is empty."""
    if db_session.query(BotFeature).count() == 0:
        features = [
            BotFeature(name="Gemini 3 Intelligence", description="Standard AI powered by Google Gemini 3 Flash.", category="AI", is_premium=False),
            BotFeature(name="Grok Premium Brain", description="Unlock the power of xAI Grok for specialized insights.", category="AI", is_premium=True),
            BotFeature(name="High Speed Mode", description="Increased rate limits for faster, more frequent responses.", category="Utility", is_premium=True),
            BotFeature(name="Smart Message Splitting", description="Automatically handles long AI responses without crashing.", category="Utility", is_premium=False),
            BotFeature(name="User Rate Limiting", description="Protects the bot from spam with configurable per-user limits.", category="Safety", is_premium=False),
            BotFeature(name="Owner-Only Controls", description="Secure management commands locked to the bot creator.", category="Safety", is_premium=False),
            BotFeature(name="Channel Memory", description="Ask about this channel's recent messages (e.g. what did Carolyn say Friday?) with /ask_channel.", category="Utility", is_premium=False),
            BotFeature(name="Personas & Memory", description="Set your style (helpful, sarcastic, pirate, ELI5) with /mode. Remember facts with /remember and /recall.", category="Utility", is_premium=False),
            BotFeature(name="Quiz & Story", description="Start trivia with /quiz start and adventures with /story start. Great for community nights.", category="Fun", is_premium=False),
            BotFeature(name="Summaries & Digest", description="Summarize a thread with /summarize_thread or channel topics with /channel_digest.", category="Utility", is_premium=False),
            BotFeature(name="Moderation & Support", description="Mods can right-click a message and Check message for toxicity. Support channels get AI reply suggestions.", category="Safety", is_premium=False),
            BotFeature(name="RSS Digest", description="Optional scheduled or on-demand AI summaries of RSS feeds posted to a channel.", category="Integration", is_premium=False),
        ]
        db_session.add_all(features)
        db_session.commit()
        log.info("Database initialized with default features.")


def ensure_channel_memory_feature():
    """Add Channel Memory feature if missing (for existing deployments)."""
    if db_session.query(BotFeature).filter(BotFeature.name == "Channel Memory").first():
        return
    db_session.add(BotFeature(
        name="Channel Memory",
        description="Ask about this channel's recent messages (e.g. what did Carolyn say Friday?) with /ask_channel.",
        category="Utility",
        is_premium=False,
    ))
    db_session.commit()
    log.info("Added 'Channel Memory' feature to database.")


def ensure_new_features():
    """Add new feature rows if missing (for existing deployments)."""
    new_features = [
        ("Personas & Memory", "Set your style (helpful, sarcastic, pirate, ELI5) with /mode. Remember facts with /remember and /recall.", "Utility"),
        ("Quiz & Story", "Start trivia with /quiz start and adventures with /story start. Great for community nights.", "Fun"),
        ("Summaries & Digest", "Summarize a thread with /summarize_thread or channel topics with /channel_digest.", "Utility"),
        ("Moderation & Support", "Mods can right-click a message and Check message for toxicity. Support channels get AI reply suggestions.", "Safety"),
        ("RSS Digest", "Optional scheduled or on-demand AI summaries of RSS feeds posted to a channel.", "Integration"),
        ("Link Reader", "Share any URL in chat and I'll read and summarize its content using Jina AI Reader. Perfect for discussing articles, docs, or web content.", "Utility"),
    ]
    for name, desc, category in new_features:
        if db_session.query(BotFeature).filter(BotFeature.name == name).first():
            continue
        db_session.add(BotFeature(name=name, description=desc, category=category, is_premium=False))
    db_session.commit()
    log.info("Ensured new features in database.")


init_db()
ensure_channel_memory_feature()
ensure_new_features()

# --- WEB SERVER (FLASK) ---
flask_app = Flask(__name__)

# OAuth2 invite URL (bot + applications.commands)
BOT_CLIENT_ID = os.getenv("BOT_CLIENT_ID", "1472960650870259763")
BOT_INVITE_URL = f"https://discord.com/oauth2/authorize?client_id={BOT_CLIENT_ID}&permissions=2147609600&integration_type=0&scope=bot%20applications.commands"


@flask_app.route('/')
def home():
    session = db_session()
    features = session.query(BotFeature).filter(BotFeature.is_active == True).all()
    return render_template('index.html', features=features, year=datetime.datetime.now().year, invite_url=BOT_INVITE_URL)


@flask_app.route('/privacy')
def privacy():
    return render_template('privacy.html')


@flask_app.route('/tos')
def tos():
    return render_template('tos.html')


@flask_app.route('/health')
def health():
    return jsonify({"status": "ok", "bot_online": bot.is_ready() if bot else False})


RSS_CRON_SECRET = os.getenv("RSS_CRON_SECRET")


@flask_app.route('/cron/rss-digest', methods=["POST"])
def cron_rss_digest():
    """Trigger RSS digest via HTTP (e.g. Azure Timer). Requires header X-Cron-Secret: RSS_CRON_SECRET."""
    if RSS_CRON_SECRET and request.headers.get("X-Cron-Secret") != RSS_CRON_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    if not bot.loop:
        return jsonify({"error": "bot not ready"}), 503
    try:
        fut = asyncio.run_coroutine_threadsafe(run_rss_digest(), bot.loop)
        fut.result(timeout=120)
    except Exception as e:
        log.error(f"RSS digest cron error: {e}")
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "ok"})


# Expose Flask app for gunicorn as 'app'
app = flask_app

# --- CONFIGURATION ---
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
PREMIUM_SKU_ID = os.getenv("PREMIUM_SKU_ID")
if not PREMIUM_SKU_ID:
    log.warning("PREMIUM_SKU_ID is not set. Premium detection will not work.")

MOD_ALERTS_CHANNEL_ID = os.getenv("MOD_ALERTS_CHANNEL_ID")
SUPPORT_CHANNEL_IDS = [x.strip() for x in os.getenv("SUPPORT_CHANNEL_IDS", "").split(",") if x.strip()]
SUPPORT_DOC_URLS = [x.strip() for x in os.getenv("SUPPORT_DOC_URLS", "").split(",") if x.strip()]
_support_last_suggestion: dict[int, float] = {}
_SUPPORT_COOLDOWN_SEC = 120

RSS_FEEDS = [x.strip() for x in os.getenv("RSS_FEEDS", "").split(",") if x.strip()]
RSS_POST_CHANNEL_ID = os.getenv("RSS_POST_CHANNEL_ID")
RSS_CRON_HOURS = float(os.getenv("RSS_CRON_HOURS", "6") or "6")

# API keys from environment only (never hardcode)
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
XAI_KEY = os.getenv("XAI_API_KEY")

if not XAI_KEY:
    log.warning("XAI_API_KEY is not set. Grok will not be available.")

gemini_model = None
if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel('gemini-3-flash-preview')
        log.info("Gemini AI configured successfully.")
    except Exception as e:
        log.error(f"Error configuring Gemini: {e}")

# --- CONFIGURABLE SETTINGS ---
settings = {
    "provider": "gemini",
    "rate_limit_free": 5,
    "rate_limit_premium": 20,
    "max_prompt_length": 2000,
    "cooldown_msg": "Slow down! You've reached your limit of {limit} messages per minute. Upgrade to Premium for higher limits!",
}

# --- RATE LIMIT TRACKER (with automatic cleanup) ---
_user_usage: dict[int, list[float]] = {}
_RATE_LIMIT_WINDOW = 60  # seconds
_CLEANUP_INTERVAL = 300  # clean stale entries every 5 min
_last_cleanup = time.time()


def _cleanup_rate_limits():
    """Remove stale entries from rate limit tracker to prevent memory leak."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return
    _last_cleanup = now
    stale_keys = [uid for uid, timestamps in _user_usage.items() if not timestamps or now - timestamps[-1] > _RATE_LIMIT_WINDOW]
    for key in stale_keys:
        del _user_usage[key]


# AI SYSTEM PROMPT
AI_SYSTEM_PROMPT = (
    "You are PolyMind, a cool, knowledgeable assistant. "
    "You're like a smart friend who knows a lot about everything. "
    "Talk naturally and casually, but be very helpful. "
    "Use your extensive internal knowledge to answer questions thoroughly. "
    "Keep your responses engaging and informative."
)

# PERSONA MODES (system prompt overrides)
PERSONA_PROMPTS = {
    "helpful": AI_SYSTEM_PROMPT,
    "sarcastic": (
        "You are PolyMind in sarcastic mode. You're still helpful and knowledgeable, "
        "but you reply with dry wit, light sarcasm, and playful teasing. Keep it fun, not mean."
    ),
    "pirate": (
        "You are PolyMind in pirate mode. Answer all questions in character as a friendly pirate. "
        "Use nautical terms, say 'arr', 'aye', 'matey', and keep answers helpful but entertaining."
    ),
    "eli5": (
        "You are PolyMind in ELI5 (Explain Like I'm 5) mode. Answer every question in simple, "
        "friendly language as if explaining to a curious child. Use short sentences and avoid jargon."
    ),
}

# BANNED KEYWORDS (case-insensitive regex patterns)
BANNED_PATTERNS = [
    re.compile(r"rm\s+-rf", re.IGNORECASE),
    re.compile(r"format\s+c:", re.IGNORECASE),
    re.compile(r"del\s+/s\s+/q\s+c:", re.IGNORECASE),
    re.compile(r"shutdown\s+/s", re.IGNORECASE),
    re.compile(r"system32", re.IGNORECASE),
]

# URL detection and SSRF protection
URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)
BLOCKED_URL_PATTERNS = [
    re.compile(r'localhost', re.IGNORECASE),
    re.compile(r'127\.0\.0\.', re.IGNORECASE),
    re.compile(r'192\.168\.', re.IGNORECASE),
    re.compile(r'10\.', re.IGNORECASE),
    re.compile(r'172\.(1[6-9]|2[0-9]|3[01])\.', re.IGNORECASE),
    re.compile(r'169\.254\.', re.IGNORECASE),
    re.compile(r'file://', re.IGNORECASE),
]

# Channel memory constants
CHANNEL_MEMORY_MAX_DAYS = 14
CHANNEL_MEMORY_MAX_MESSAGES = 200
CHANNEL_MEMORY_MAX_CHARS = 4000

# Quiz state: channel_id -> { "questions": [{"q","a"}], "created_at": float }. TTL 30 min.
_quiz_state: dict[int, dict] = {}
_QUIZ_TTL_SEC = 30 * 60

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Track whether we've synced slash commands this session
_commands_synced = False


# --- HELPER FUNCTIONS ---
def split_message(text, limit=2000):
    """Split a message into chunks that fit Discord's character limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind('\n', 0, limit)
        if split_at == -1:
            split_at = text.rfind(' ', 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks


def _contains_banned(text: str) -> bool:
    """Check if text matches any banned pattern (case-insensitive)."""
    for pattern in BANNED_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _is_safe_url(url: str) -> bool:
    """Check if URL is safe (not localhost, private IPs, file:// etc)."""
    for pattern in BLOCKED_URL_PATTERNS:
        if pattern.search(url):
            return False
    return True


def _extract_urls(text: str) -> list[str]:
    """Extract and validate URLs from text. Returns up to 3 safe URLs."""
    urls = URL_PATTERN.findall(text)
    safe_urls = [u for u in urls if _is_safe_url(u)][:3]
    return safe_urls


async def fetch_url_content(url: str, max_chars: int = 3000) -> str | None:
    """Fetch and extract text from a URL using Jina AI Reader. Returns None on error."""
    try:
        jina_url = f"https://r.jina.ai/{url}"
        async with aiohttp.ClientSession() as session:
            async with session.get(jina_url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    log.warning(f"Jina AI Reader failed for {url}: HTTP {response.status}")
                    return None
                text = await response.text()
                text = text.strip()[:max_chars]
                return text if text else None
    except asyncio.TimeoutError:
        log.warning(f"URL fetch timeout: {url}")
        return None
    except Exception as e:
        log.error(f"URL fetch error {url}: {e}")
        return None


def has_premium(user_id) -> bool:
    """Checks if a user has an active premium entitlement."""
    try:
        ent = db_session.query(UserEntitlement).filter(UserEntitlement.user_id == str(user_id)).first()
        if ent is None:
            return False
        if ent.ends_at and ent.ends_at < datetime.datetime.utcnow():
            return False
        return True
    except Exception as e:
        log.error(f"Error checking premium for {user_id}: {e}")
        return False


def get_membership_display(user_id) -> str:
    """Returns what membership the bot is detecting for this user."""
    try:
        ent = db_session.query(UserEntitlement).filter(UserEntitlement.user_id == str(user_id)).first()
        if not ent:
            return "Free"
        if PREMIUM_SKU_ID and ent.sku_id == str(PREMIUM_SKU_ID):
            return "PolyMind Premium"
        return f"Premium (SKU: {ent.sku_id})"
    except Exception:
        return "Unknown"


def get_user_memories(user_id) -> str:
    """Return a single string of all stored facts for the user for injection into AI context."""
    try:
        rows = (
            db_session.query(UserMemory)
            .filter(UserMemory.user_id == str(user_id))
            .order_by(UserMemory.updated_at.desc())
            .limit(20)
            .all()
        )
        if not rows:
            return ""
        return "; ".join(r.value for r in rows if r.value)
    except Exception as e:
        log.error(f"get_user_memories error: {e}")
        return ""


def add_user_memory(user_id, value: str) -> None:
    """Append a fact for the user. Value truncated to 500 chars."""
    try:
        value = (value or "").strip()[:500]
        if not value:
            return
        db_session.add(UserMemory(user_id=str(user_id), value=value))
        db_session.commit()
    except Exception as e:
        log.error(f"add_user_memory error: {e}")


def get_user_persona(user_id) -> str:
    """Return the persona_id for the user, default 'helpful'."""
    try:
        row = db_session.query(UserPersona).filter(UserPersona.user_id == str(user_id)).first()
        return row.persona_id if row and row.persona_id in PERSONA_PROMPTS else "helpful"
    except Exception as e:
        log.error(f"get_user_persona error: {e}")
        return "helpful"


def set_user_persona(user_id, persona_id: str) -> None:
    """Set the user's persona. persona_id must be in PERSONA_PROMPTS."""
    try:
        row = db_session.query(UserPersona).filter(UserPersona.user_id == str(user_id)).first()
        if row:
            row.persona_id = persona_id
        else:
            db_session.add(UserPersona(user_id=str(user_id), persona_id=persona_id))
        db_session.commit()
    except Exception as e:
        log.error(f"set_user_persona error: {e}")


def get_ai_feature_context() -> str:
    """Fetch all active bot features to make the AI aware of its capabilities."""
    try:
        features = db_session.query(BotFeature).filter(BotFeature.is_active == True).all()
        if not features:
            return ""
        lines = ["You have the following features enabled (as listed on your website):"]
        for f in features:
            premium_tag = " (Premium only)" if f.is_premium else ""
            lines.append(f"- {f.name}: {f.description}{premium_tag}")
        return "\n".join(lines)
    except Exception as e:
        log.error(f"Error fetching feature context: {e}")
        return ""


def is_rate_limited(user_id) -> bool:
    """Checks if a user has exceeded their per-minute limit."""
    _cleanup_rate_limits()
    now = time.time()
    limit = settings["rate_limit_premium"] if has_premium(user_id) else settings["rate_limit_free"]

    if user_id not in _user_usage:
        _user_usage[user_id] = []

    _user_usage[user_id] = [t for t in _user_usage[user_id] if now - t < _RATE_LIMIT_WINDOW]
    if len(_user_usage[user_id]) >= limit:
        return True

    _user_usage[user_id].append(now)
    return False


# --- AI INTEGRATIONS ---
def _build_full_prompt(system_prompt: str, prompt: str, extra_context: str | None = None) -> str:
    """Build the single string sent to Gemini (system + optional context + user)."""
    base = f"{system_prompt}\n\nUser: {prompt}"
    if extra_context and extra_context.strip():
        base = f"{system_prompt}\n\nUser context (use this when relevant): {extra_context.strip()}\n\nUser: {prompt}"
    return base


async def get_gemini_response(prompt: str, system_prompt: str | None = None, extra_context: str | None = None) -> str:
    if not gemini_model:
        return "Gemini is not configured. Please contact the bot owner."
    sys = system_prompt if system_prompt else AI_SYSTEM_PROMPT
    full = _build_full_prompt(sys, prompt, extra_context)
    try:
        response = await asyncio.to_thread(gemini_model.generate_content, full)
        return response.text if response and response.text else "Gemini returned an empty response."
    except Exception as e:
        log.error(f"Gemini error: {e}")
        return "Something went wrong with the AI. Try again in a moment."


# Current xAI chat model (grok-beta is deprecated). Use grok-3-mini, grok-3, or grok-4-1-fast-reasoning.
XAI_MODEL = os.getenv("XAI_MODEL", "grok-3-mini")


def _grok_system_content(system_prompt: str, extra_context: str | None) -> str:
    out = system_prompt
    if extra_context and extra_context.strip():
        out = f"{out}\n\nUser context (use when relevant): {extra_context.strip()}"
    return out


async def get_grok_response(prompt: str, system_prompt: str | None = None, extra_context: str | None = None) -> str:
    if not XAI_KEY:
        return "Grok is not configured. Please contact the bot owner."
    sys = system_prompt if system_prompt else AI_SYSTEM_PROMPT
    system_content = _grok_system_content(sys, extra_context)
    headers = {"Authorization": f"Bearer {XAI_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": XAI_MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(XAI_API_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                body = await response.text()
                log.error(f"Grok API error: {response.status} - {body[:500]}")
                return "Grok encountered an error. Try again in a moment."
    except asyncio.TimeoutError:
        return "Grok took too long to respond. Try again."
    except Exception as e:
        log.error(f"Grok exception: {e}")
        return "Could not reach Grok. Try again in a moment."


async def get_ai_response(
    prompt: str,
    user_id,
    system_prompt: str | None = None,
    extra_context: str | None = None,
) -> str:
    """Route a prompt to the configured AI provider with input validation."""
    if len(prompt) > settings["max_prompt_length"]:
        return f"Your message is too long (max {settings['max_prompt_length']} characters). Please shorten it."

    if _contains_banned(prompt):
        return "I can't help with that request."

    if extra_context is None:
        memories = get_user_memories(user_id)
        feature_context = get_ai_feature_context()
        
        ctx_parts = []
        if feature_context:
            ctx_parts.append(feature_context)
        if memories:
            ctx_parts.append(f"Things this user has asked you to remember: {memories}")
            
        if ctx_parts:
            extra_context = "\n\n".join(ctx_parts)

    # Extract and fetch content from URLs in prompt
    urls = _extract_urls(prompt)
    if urls:
        url_contents = []
        for url in urls:
            content = await fetch_url_content(url)
            if content:
                url_contents.append(f"[Content from {url}]\n{content}")
        
        if url_contents:
            url_context = "\n\n".join(url_contents)
            if extra_context:
                extra_context = f"{extra_context}\n\n{url_context}"
            else:
                extra_context = url_context

    if system_prompt is None:
        persona_id = get_user_persona(user_id)
        system_prompt = PERSONA_PROMPTS.get(persona_id, AI_SYSTEM_PROMPT)

    if settings["provider"] == "grok" and not has_premium(user_id):
        return "**Grok is a Premium feature.** Upgrade your subscription to unlock the Grok brain!"

    if settings["provider"] == "grok":
        return await get_grok_response(prompt, system_prompt=system_prompt, extra_context=extra_context)
    return await get_gemini_response(prompt, system_prompt=system_prompt, extra_context=extra_context)


# --- CHANNEL MEMORY ---
async def fetch_channel_transcript(channel, days_back: int, max_chars: int = CHANNEL_MEMORY_MAX_CHARS):
    """Fetch recent messages from a text channel and return a transcript string."""
    if not isinstance(channel, discord.TextChannel):
        return None, "This command only works in a server text channel."
    after = datetime.datetime.utcnow() - datetime.timedelta(days=days_back)
    try:
        lines = []
        async for msg in channel.history(limit=CHANNEL_MEMORY_MAX_MESSAGES, after=after, oldest_first=True):
            if msg.author.bot:
                continue
            name = msg.author.display_name or msg.author.name
            ts = msg.created_at.strftime("%Y-%m-%d %H:%M")
            content = (msg.content or "").strip()
            if not content:
                continue
            if len(content) > 300:
                content = content[:297] + "..."
            lines.append(f"{name} ({ts}): {content}")
        transcript = "\n".join(lines)
        if len(transcript) > max_chars:
            transcript = transcript[-max_chars:].strip()
            if "\n" in transcript:
                transcript = "..." + transcript[transcript.index("\n") + 1:]
        if not transcript.strip():
            return None, "No messages found in this channel for that time range."
        return transcript, None
    except discord.Forbidden:
        return None, "I don't have permission to read this channel's message history."
    except Exception as e:
        log.error(f"Channel history error: {e}")
        return None, "Could not read channel history. Try again later."


async def fetch_thread_transcript(thread, limit: int = 50, max_chars: int = CHANNEL_MEMORY_MAX_CHARS):
    """Fetch recent messages from a Discord thread. Returns (transcript, None) or (None, error)."""
    if not isinstance(thread, discord.Thread):
        return None, "This command only works inside a thread."
    try:
        lines = []
        async for msg in thread.history(limit=limit, oldest_first=True):
            if msg.author.bot:
                continue
            name = msg.author.display_name or msg.author.name
            ts = msg.created_at.strftime("%Y-%m-%d %H:%M")
            content = (msg.content or "").strip()
            if not content:
                continue
            if len(content) > 300:
                content = content[:297] + "..."
            lines.append(f"{name} ({ts}): {content}")
        transcript = "\n".join(lines)
        if len(transcript) > max_chars:
            transcript = transcript[-max_chars:].strip()
            if "\n" in transcript:
                transcript = "..." + transcript[transcript.index("\n") + 1:]
        if not transcript.strip():
            return None, "No messages found in this thread."
        return transcript, None
    except discord.Forbidden:
        return None, "I don't have permission to read this thread's history."
    except Exception as e:
        log.error(f"Thread history error: {e}")
        return None, "Could not read thread history. Try again later."


def _channel_memory_prompt(transcript: str, question: str) -> str:
    return (
        "The user is asking about messages in this Discord channel. "
        "Below is a recent transcript (Author (date time): content). "
        "Answer their question based ONLY on this transcript. If the answer isn't there, say so briefly.\n\n"
        "--- Transcript ---\n" + transcript + "\n--- End ---\n\nUser question: " + question
    )


async def get_channel_aware_response(transcript: str, question: str, user_id) -> str:
    """Answer a question using channel transcript as context."""
    if len(question) > settings["max_prompt_length"]:
        return f"Your question is too long (max {settings['max_prompt_length']} characters)."

    if _contains_banned(question):
        return "I can't help with that request."

    if settings["provider"] == "grok" and not has_premium(user_id):
        return "**Grok is a Premium feature.** Upgrade to use channel memory with Grok!"

    feature_context = get_ai_feature_context()
    memories = get_user_memories(user_id)
    
    extra_info = ""
    if feature_context:
        extra_info += f"\n\n{feature_context}"
    if memories:
        extra_info += f"\n\nThings this user has asked you to remember: {memories}"

    prompt = _channel_memory_prompt(transcript, question) + extra_info
    if settings["provider"] == "grok":
        return await get_grok_response(prompt)
    return await get_gemini_response(prompt)


async def get_summary_response(prompt_instruction: str, transcript: str) -> str:
    """Call AI to summarize or digest a transcript (no user context). Uses current provider."""
    full_prompt = f"{prompt_instruction}\n\n--- Transcript ---\n{transcript}\n--- End ---"
    if settings["provider"] == "grok":
        return await get_grok_response(full_prompt)
    return await get_gemini_response(full_prompt)


def _normalize_answer(s: str) -> str:
    """Lowercase, strip, remove extra spaces and common punctuation for answer comparison."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def _parse_quiz_response(text: str) -> list[dict]:
    """Parse AI response into list of {q, a}. Expects lines like 'Q1: ... | A1: ...'."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if "|" in line:
            parts = line.split("|", 1)
            if len(parts) == 2:
                q = re.sub(r"^Q\d+:\s*", "", parts[0].strip(), flags=re.IGNORECASE).strip()
                a = re.sub(r"^A\d+:\s*", "", parts[1].strip(), flags=re.IGNORECASE).strip()
                if q and a:
                    out.append({"q": q, "a": a})
        if len(out) >= 5:
            break
    return out[:5]


async def _generate_quiz_questions(topic: str) -> list[dict]:
    """Ask AI for 5 trivia Q&A pairs. Returns list of {q, a}."""
    prompt = (
        f"Generate exactly 5 trivia questions about: {topic}. "
        "Format each line exactly as: Q1: question text | A1: short answer. "
        "Use Q2, A2, ... Q5, A5. One line per question. Only output the 5 lines."
    )
    if settings["provider"] == "grok":
        raw = await get_grok_response(prompt)
    else:
        raw = await get_gemini_response(prompt)
    return _parse_quiz_response(raw)


def _get_story_session(channel_id: int, is_dm: bool):
    try:
        return (
            db_session.query(StorySession)
            .filter(StorySession.channel_id == str(channel_id), StorySession.is_dm == is_dm)
            .first()
        )
    except Exception as e:
        log.error(f"_get_story_session error: {e}")
        return None


def _update_story_session(channel_id: int, is_dm: bool, state_summary: str, last_response: str):
    state_summary = (state_summary or "")[:1000]
    last_response = (last_response or "")[:1000]
    try:
        row = _get_story_session(channel_id, is_dm)
        if row:
            row.state_summary = state_summary
            row.last_response = last_response
            row.updated_at = datetime.datetime.utcnow()
        else:
            db_session.add(
                StorySession(channel_id=str(channel_id), is_dm=is_dm, state_summary=state_summary, last_response=last_response)
            )
        db_session.commit()
    except Exception as e:
        log.error(f"_update_story_session error: {e}")


async def _story_ai(prompt: str) -> str:
    """Call AI for story (no user context). Uses current provider."""
    if settings["provider"] == "grok":
        return await get_grok_response(prompt)
    return await get_gemini_response(prompt)


TOXICITY_PROMPT = (
    "You are a moderation assistant. Rate the following message for harassment, toxicity, or clear rule-breaking. "
    "Reply with exactly one line in this format: score|reason. "
    "Score is 0-10 (0=harmless, 10=severe). Reason is one short sentence. No other output."
)


async def _support_suggestion_task(message: discord.Message):
    """Background task: get AI suggestion for a support question and reply."""
    try:
        content = (message.content or "").strip()[:500]
        urls = ", ".join(SUPPORT_DOC_URLS[:10]) if SUPPORT_DOC_URLS else "no specific docs"
        prompt = (
            f"This is a support question in Discord. Suggest a brief helpful reply (1-3 sentences). "
            f"If relevant, say 'This might be answered by: [link]' from: {urls}. "
            "Only output the suggestion, no preamble."
        )
        if settings["provider"] == "grok":
            suggestion = await get_grok_response(prompt + "\n\nQuestion: " + content)
        else:
            suggestion = await get_gemini_response(prompt + "\n\nQuestion: " + content)
        suggestion = (suggestion or "").strip()[:500]
        if suggestion:
            await message.reply(f"**Suggestion:** {suggestion}")
    except Exception as e:
        log.error(f"Support suggestion error: {e}")


async def run_toxicity_check(content: str) -> tuple[int, str]:
    """Return (score, reason). Score 0-10. Reason is one sentence from AI."""
    if not content or not content.strip():
        return 0, "Empty message."
    prompt = f"{TOXICITY_PROMPT}\n\nMessage to rate:\n{content[:1500]}"
    try:
        if settings["provider"] == "grok":
            raw = await get_grok_response(prompt)
        else:
            raw = await get_gemini_response(prompt)
        raw = (raw or "").strip()
        if "|" in raw:
            score_str, reason = raw.split("|", 1)
            score_str = re.sub(r"\D", "", score_str)
            score = int(score_str) if score_str else 5
            score = max(0, min(10, score))
            return score, reason.strip()[:300]
    except Exception as e:
        log.error(f"run_toxicity_check error: {e}")
    return -1, "Could not analyze."


async def run_rss_digest():
    """Fetch RSS feeds, summarize with AI, post to configured channel."""
    if not RSS_FEEDS or not RSS_POST_CHANNEL_ID:
        return
    channel = bot.get_channel(int(RSS_POST_CHANNEL_ID))
    if not channel:
        log.warning("RSS_POST_CHANNEL_ID channel not found.")
        return
    items = []
    async with aiohttp.ClientSession() as session:
        for url in RSS_FEEDS[:10]:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        continue
                    body = await resp.text()
            except Exception as e:
                log.error(f"RSS fetch error {url}: {e}")
                continue
            try:
                feed = await asyncio.to_thread(feedparser.parse, body)
            except Exception as e:
                log.error(f"RSS parse error {url}: {e}")
                continue
            for entry in feed.entries[:5]:
                title = getattr(entry, "title", "") or ""
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
                summary = re.sub(r"<[^>]+>", "", summary)[:300]
                items.append(f"- {title}\n  {summary}")
    if not items:
        log.info("RSS digest: no items found.")
        return
    text = "\n".join(items)[:4000]
    try:
        prompt = "Summarize the top 3 most important or interesting items from this list in one short paragraph each. Be concise."
        full = f"{prompt}\n\n---\n{text}\n---"
        if settings["provider"] == "grok":
            summary = await get_grok_response(full)
        else:
            summary = await get_gemini_response(full)
        if summary:
            await channel.send("**RSS digest**\n\n" + summary[:1900])
    except Exception as e:
        log.error(f"RSS digest AI or send error: {e}")


def _rss_job():
    """Scheduler job: run run_rss_digest on the bot's event loop."""
    if not bot.loop or not RSS_FEEDS or not RSS_POST_CHANNEL_ID:
        return
    try:
        fut = asyncio.run_coroutine_threadsafe(run_rss_digest(), bot.loop)
        fut.result(timeout=120)
    except Exception as e:
        log.error(f"RSS job error: {e}")


_rss_scheduler = None


def start_rss_scheduler():
    global _rss_scheduler
    if not RSS_FEEDS or not RSS_POST_CHANNEL_ID or _rss_scheduler is not None:
        return
    _rss_scheduler = BackgroundScheduler()
    _rss_scheduler.add_job(_rss_job, "interval", hours=max(1, RSS_CRON_HOURS))
    _rss_scheduler.start()
    log.info("RSS scheduler started.")


# --- PREFIX COMMANDS ---
@bot.command(name='addfeature')
@commands.is_owner()
async def add_feature(ctx, name: str, category: str, is_premium: bool, *, description: str):
    """Add a new feature. Usage: !addfeature "Name" "Category" True/False Description"""
    if len(name) > 100 or len(description) > 500 or len(category) > 50:
        await ctx.send("Name (100), category (50), or description (500) too long.")
        return
    new_feat = BotFeature(name=name, category=category, is_premium=is_premium, description=description)
    db_session.add(new_feat)
    db_session.commit()
    await ctx.send(f"Feature **{name}** added! (Premium: {is_premium})")


@bot.command(name='assistant')
@commands.is_owner()
async def set_assistant(ctx, provider: str):
    """Switch the AI provider. Usage: !assistant gemini or !assistant grok"""
    provider = provider.lower()
    if provider not in ("gemini", "grok"):
        await ctx.send("Invalid provider. Use `gemini` or `grok`.")
        return
    settings["provider"] = provider
    await ctx.send(f"Brain switched to **{provider.capitalize()}**.")


@bot.command(name='addpremium')
@commands.is_owner()
async def add_test_premium(ctx, user_id: str):
    """(Owner only) Grant test premium to a user. Usage: !addpremium <discord_user_id>"""
    user_id = user_id.strip()
    if not user_id.isdigit():
        await ctx.send("Invalid user ID. Use a numeric Discord user ID (e.g. `!addpremium 249640288996294657`).")
        return
    sku_id = str(PREMIUM_SKU_ID) if PREMIUM_SKU_ID else "test_premium"
    try:
        db_session.query(UserEntitlement).filter(UserEntitlement.user_id == user_id).delete()
        db_session.add(UserEntitlement(user_id=user_id, sku_id=sku_id, ends_at=None))
        db_session.commit()
        await ctx.send(f"Premium granted for user ID **{user_id}** (sku: {sku_id}). They can use !status to verify.")
    except Exception as e:
        log.error(f"add_test_premium failed: {e}")
        await ctx.send("Failed to update database. Check logs.")


@bot.command(name='status')
async def bot_status(ctx):
    """Show bot status, premium, and rate limit info."""
    premium = "Active" if has_premium(ctx.author.id) else "Inactive"
    membership = get_membership_display(ctx.author.id)
    limit = settings["rate_limit_premium"] if has_premium(ctx.author.id) else settings["rate_limit_free"]
    persona = get_user_persona(ctx.author.id)
    await ctx.send(
        f"**PolyMind System Report**\n"
        f"Brain: **{settings['provider'].upper()}**\n"
        f"Mode: **{persona}**\n"
        f"Your Premium: {premium}\n"
        f"Membership detected: **{membership}**\n"
        f"Rate Limit: {limit} msg/min"
    )


@bot.command(name='mode')
async def mode_prefix(ctx, persona: str = None):
    """Set or show your persona. Usage: !mode [helpful|sarcastic|pirate|eli5]"""
    persona = (persona or "").lower().strip()
    if not persona:
        current = get_user_persona(ctx.author.id)
        await ctx.send(f"Your current mode is **{current}**. Use `!mode helpful`, `!mode sarcastic`, `!mode pirate`, or `!mode eli5` to change.")
        return
    if persona not in PERSONA_PROMPTS:
        await ctx.send("Invalid mode. Use: helpful, sarcastic, pirate, or eli5.")
        return
    set_user_persona(ctx.author.id, persona)
    await ctx.send(f"Mode set to **{persona}**.")


# --- SLASH COMMANDS ---
@bot.tree.command(name="ask", description="Ask PolyMind anything. Get an AI-powered answer.")
@discord.app_commands.describe(question="Your question or message for PolyMind")
async def ask_slash(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)

    if is_rate_limited(interaction.user.id):
        limit = settings["rate_limit_premium"] if has_premium(interaction.user.id) else settings["rate_limit_free"]
        await interaction.followup.send(settings["cooldown_msg"].format(limit=limit), ephemeral=True)
        return

    try:
        response = await get_ai_response(question, interaction.user.id)
        chunks = split_message(response)
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
    except Exception as e:
        log.error(f"/ask error for {interaction.user.id}: {e}")
        await interaction.followup.send("Something went wrong. Try again!", ephemeral=True)


@bot.tree.command(name="ask_channel", description="Ask about this channel's recent messages.")
@discord.app_commands.describe(
    question="Your question about the channel (e.g. What did Carolyn say Friday?)",
    days_back="How many days of history to look at (1-14, default 7)"
)
@discord.app_commands.choices(days_back=[
    discord.app_commands.Choice(name="Last 1 day", value=1),
    discord.app_commands.Choice(name="Last 3 days", value=3),
    discord.app_commands.Choice(name="Last 7 days", value=7),
    discord.app_commands.Choice(name="Last 14 days", value=14),
])
async def ask_channel_slash(interaction: discord.Interaction, question: str, days_back: int = 7):
    await interaction.response.defer(thinking=True)

    if not interaction.guild or not isinstance(interaction.channel, discord.TextChannel):
        await interaction.followup.send("Use this command in a server text channel.", ephemeral=True)
        return

    if is_rate_limited(interaction.user.id):
        limit = settings["rate_limit_premium"] if has_premium(interaction.user.id) else settings["rate_limit_free"]
        await interaction.followup.send(settings["cooldown_msg"].format(limit=limit), ephemeral=True)
        return

    if days_back is None:
        days_back = 7
    days_back = max(1, min(CHANNEL_MEMORY_MAX_DAYS, days_back))

    transcript, err = await fetch_channel_transcript(interaction.channel, days_back)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return

    try:
        response = await get_channel_aware_response(transcript, question, interaction.user.id)
        chunks = split_message(response)
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
    except Exception as e:
        log.error(f"/ask_channel error for {interaction.user.id}: {e}")
        await interaction.followup.send("Something went wrong. Try again!", ephemeral=True)


@bot.tree.command(name="remember", description="Store a fact for PolyMind to remember about you.")
@discord.app_commands.describe(fact="Something you want me to remember (e.g. I prefer Python)")
async def remember_slash(interaction: discord.Interaction, fact: str):
    fact = (fact or "").strip()
    if not fact:
        await interaction.response.send_message("Please provide something to remember (e.g. `/remember I prefer Python`).", ephemeral=True)
        return
    if len(fact) > 500:
        await interaction.response.send_message("That's too long (max 500 characters). Shorten it.", ephemeral=True)
        return
    add_user_memory(interaction.user.id, fact)
    await interaction.response.send_message(f"Got it, I'll remember: **{fact[:100]}{'…' if len(fact) > 100 else ''}**", ephemeral=True)


@bot.tree.command(name="recall", description="See what PolyMind has remembered about you.")
async def recall_slash(interaction: discord.Interaction):
    memories = get_user_memories(interaction.user.id)
    if not memories:
        await interaction.response.send_message("I don't have any memories stored for you yet. Use `/remember <fact>` to add some!", ephemeral=True)
        return
    parts = memories.split("; ")
    text = "**What I remember about you:**\n" + "\n".join(f"• {p}" for p in parts)
    chunks = split_message(text)
    await interaction.response.send_message(chunks[0], ephemeral=True)
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk, ephemeral=True)


@bot.tree.command(name="mode", description="Set how PolyMind talks to you (persona).")
@discord.app_commands.describe(persona="Persona: helpful, sarcastic, pirate, or eli5")
@discord.app_commands.choices(persona=[
    discord.app_commands.Choice(name="Helpful (default)", value="helpful"),
    discord.app_commands.Choice(name="Sarcastic", value="sarcastic"),
    discord.app_commands.Choice(name="Pirate", value="pirate"),
    discord.app_commands.Choice(name="ELI5", value="eli5"),
])
async def mode_slash(interaction: discord.Interaction, persona: str):
    if persona not in PERSONA_PROMPTS:
        await interaction.response.send_message("Invalid persona. Choose: helpful, sarcastic, pirate, eli5.", ephemeral=True)
        return
    set_user_persona(interaction.user.id, persona)
    await interaction.response.send_message(f"Mode set to **{persona}**. I'll talk to you in that style from now on!", ephemeral=True)


@bot.tree.command(name="summarize_thread", description="Summarize this thread in 3-5 bullet points.")
async def summarize_thread_slash(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)

    if not isinstance(interaction.channel, discord.Thread):
        await interaction.followup.send("Use this command inside a **thread**.", ephemeral=True)
        return

    if is_rate_limited(interaction.user.id):
        limit = settings["rate_limit_premium"] if has_premium(interaction.user.id) else settings["rate_limit_free"]
        await interaction.followup.send(settings["cooldown_msg"].format(limit=limit), ephemeral=True)
        return

    transcript, err = await fetch_thread_transcript(interaction.channel)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return

    try:
        instruction = "Summarize this Discord thread in 3-5 bullet points. Be neutral and factual."
        response = await get_summary_response(instruction, transcript)
        chunks = split_message(response)
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
    except Exception as e:
        log.error(f"/summarize_thread error: {e}")
        await interaction.followup.send("Something went wrong. Try again!", ephemeral=True)


# Quiz command group
quiz_group = discord.app_commands.Group(name="quiz", description="Trivia quiz: start a quiz or submit an answer")


@quiz_group.command(name="start", description="Start a trivia quiz on a topic.")
@discord.app_commands.describe(topic="Topic for the quiz (e.g. Science, History)")
async def quiz_start_slash(interaction: discord.Interaction, topic: str):
    await interaction.response.defer(thinking=True)
    if is_rate_limited(interaction.user.id):
        limit = settings["rate_limit_premium"] if has_premium(interaction.user.id) else settings["rate_limit_free"]
        await interaction.followup.send(settings["cooldown_msg"].format(limit=limit), ephemeral=True)
        return
    topic = (topic or "general knowledge").strip()[:100]
    try:
        questions = await _generate_quiz_questions(topic)
        if len(questions) < 3:
            await interaction.followup.send("Could not generate enough questions. Try another topic!", ephemeral=True)
            return
        channel_id = interaction.channel_id
        _quiz_state[channel_id] = {"questions": questions, "created_at": time.time()}
        lines = [f"**Q{i+1}.** {q['q']}" for i, q in enumerate(questions)]
        body = "**Quiz: " + topic + "**\n\n" + "\n\n".join(lines) + "\n\nUse `/quiz answer <number> <your answer>` to answer!"
        chunks = split_message(body)
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
    except Exception as e:
        log.error(f"/quiz start error: {e}")
        await interaction.followup.send("Something went wrong. Try again!", ephemeral=True)


@quiz_group.command(name="answer", description="Submit your answer to a quiz question.")
@discord.app_commands.describe(
    question_number="Question number (1-5)",
    answer="Your answer",
)
async def quiz_answer_slash(interaction: discord.Interaction, question_number: int, answer: str):
    channel_id = interaction.channel_id
    now = time.time()
    if channel_id in _quiz_state and now - _quiz_state[channel_id]["created_at"] > _QUIZ_TTL_SEC:
        del _quiz_state[channel_id]
    if channel_id not in _quiz_state:
        await interaction.response.send_message("No active quiz in this channel. Start one with `/quiz start <topic>`.", ephemeral=True)
        return
    questions = _quiz_state[channel_id]["questions"]
    if not (1 <= question_number <= len(questions)):
        await interaction.response.send_message(f"Question number must be 1 to {len(questions)}.", ephemeral=True)
        return
    idx = question_number - 1
    correct = _normalize_answer(questions[idx]["a"])
    given = _normalize_answer(answer)
    if given == correct:
        await interaction.response.send_message(f"**Correct!** The answer to Q{question_number} was: **{questions[idx]['a']}**")
    else:
        await interaction.response.send_message(f"Not quite. The answer to Q{question_number} was: **{questions[idx]['a']}**")


bot.tree.add_command(quiz_group)


# Story command group
story_group = discord.app_commands.Group(name="story", description="Multi-turn story adventure")


@story_group.command(name="start", description="Start or restart a story adventure.")
@discord.app_commands.describe(premise="Optional starting premise (e.g. You are in a haunted castle)")
async def story_start_slash(interaction: discord.Interaction, premise: str = ""):
    await interaction.response.defer(thinking=True)
    if is_rate_limited(interaction.user.id):
        limit = settings["rate_limit_premium"] if has_premium(interaction.user.id) else settings["rate_limit_free"]
        await interaction.followup.send(settings["cooldown_msg"].format(limit=limit), ephemeral=True)
        return
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    channel_id = interaction.channel_id
    premise = (premise or "a short adventure").strip()[:300]
    try:
        prompt = f"You are a storyteller. Start a short adventure in 2-3 paragraphs. Premise: {premise}. End with a situation where the reader can choose what to do next."
        response = await _story_ai(prompt)
        if not response or len(response) < 50:
            await interaction.followup.send("Could not start the story. Try again!", ephemeral=True)
            return
        state_summary = response[-400:] if len(response) > 400 else response
        _update_story_session(channel_id, is_dm, state_summary, response)
        chunks = split_message(response)
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
    except Exception as e:
        log.error(f"/story start error: {e}")
        await interaction.followup.send("Something went wrong. Try again!", ephemeral=True)


@story_group.command(name="continue", description="Continue the story with your action.")
@discord.app_commands.describe(action="What you do next (e.g. Open the door)")
async def story_continue_slash(interaction: discord.Interaction, action: str):
    await interaction.response.defer(thinking=True)
    if is_rate_limited(interaction.user.id):
        limit = settings["rate_limit_premium"] if has_premium(interaction.user.id) else settings["rate_limit_free"]
        await interaction.followup.send(settings["cooldown_msg"].format(limit=limit), ephemeral=True)
        return
    is_dm = isinstance(interaction.channel, discord.DMChannel)
    channel_id = interaction.channel_id
    session = _get_story_session(channel_id, is_dm)
    if not session or not session.last_response:
        await interaction.followup.send("No active story here. Start one with `/story start [premise]`.", ephemeral=True)
        return
    action = (action or "they wait").strip()[:300]
    try:
        prompt = (
            f"Ongoing story state/summary:\n{session.state_summary}\n\n"
            f"Last turn:\n{session.last_response}\n\n"
            f"User action: {action}\n\n"
            "Continue the story in 2-3 short paragraphs based on this action. End with a new situation for the reader to respond to."
        )
        response = await _story_ai(prompt)
        if not response or len(response) < 30:
            await interaction.followup.send("Could not continue. Try again!", ephemeral=True)
            return
        state_summary = response[-400:] if len(response) > 400 else response
        _update_story_session(channel_id, is_dm, state_summary, response)
        chunks = split_message(response)
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
    except Exception as e:
        log.error(f"/story continue error: {e}")
        await interaction.followup.send("Something went wrong. Try again!", ephemeral=True)


bot.tree.add_command(story_group)


# Reset command group
reset_group = discord.app_commands.Group(name="reset", description="Reset quiz, persona, or both")


@reset_group.command(name="quiz", description="Cancel the active quiz in this channel.")
async def reset_quiz_slash(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    if channel_id not in _quiz_state:
        await interaction.response.send_message("No active quiz in this channel to reset.", ephemeral=True)
        return

    del _quiz_state[channel_id]
    await interaction.response.send_message("✅ Quiz cancelled! Start a new one with `/quiz start <topic>`.", ephemeral=True)


@reset_group.command(name="persona", description="Reset your persona back to the default (helpful).")
async def reset_persona_slash(interaction: discord.Interaction):
    current_persona = get_user_persona(interaction.user.id)
    if current_persona == "helpful":
        await interaction.response.send_message("Your persona is already set to the default (helpful).", ephemeral=True)
        return

    set_user_persona(interaction.user.id, "helpful")
    await interaction.response.send_message("✅ Persona reset to **helpful**! I'll talk to you in my default style from now on.", ephemeral=True)


@reset_group.command(name="all", description="Reset both active quiz and your persona.")
async def reset_all_slash(interaction: discord.Interaction):
    channel_id = interaction.channel_id
    quiz_reset = channel_id in _quiz_state
    if quiz_reset:
        del _quiz_state[channel_id]

    persona_reset = False
    current_persona = get_user_persona(interaction.user.id)
    if current_persona != "helpful":
        set_user_persona(interaction.user.id, "helpful")
        persona_reset = True

    if not quiz_reset and not persona_reset:
        await interaction.response.send_message("Nothing to reset - no active quiz in this channel and your persona is already default (helpful).", ephemeral=True)
        return

    message_parts = []
    if quiz_reset:
        message_parts.append("quiz cancelled")
    if persona_reset:
        message_parts.append("persona reset to helpful")

    message = f"✅ {', '.join(message_parts)}!"
    if quiz_reset:
        message += " Start a new quiz with `/quiz start <topic>`."
    await interaction.response.send_message(message, ephemeral=True)


bot.tree.add_command(reset_group)


@bot.tree.command(name="channel_digest", description="Summarize key topics and questions in this channel.")
@discord.app_commands.describe(hours_back="How many hours of history (default 24)")
async def channel_digest_slash(interaction: discord.Interaction, hours_back: int = 24):
    await interaction.response.defer(thinking=True)

    if not interaction.guild or not isinstance(interaction.channel, discord.TextChannel):
        await interaction.followup.send("Use this command in a server text channel.", ephemeral=True)
        return

    if is_rate_limited(interaction.user.id):
        limit = settings["rate_limit_premium"] if has_premium(interaction.user.id) else settings["rate_limit_free"]
        await interaction.followup.send(settings["cooldown_msg"].format(limit=limit), ephemeral=True)
        return

    hours_back = max(1, min(168, hours_back))
    days_back = max(1, (hours_back + 23) // 24)

    transcript, err = await fetch_channel_transcript(interaction.channel, days_back)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return

    try:
        instruction = f"Summarize the key topics, decisions, and questions in this channel in the last {hours_back} hours. Output 1-2 short paragraphs."
        response = await get_summary_response(instruction, transcript)
        chunks = split_message(response)
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)
    except Exception as e:
        log.error(f"/channel_digest error: {e}")
        await interaction.followup.send("Something went wrong. Try again!", ephemeral=True)


@bot.tree.command(name="help", description="See everything PolyMind can do.")
async def help_slash(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    embed = discord.Embed(
        title="PolyMind - What I Can Do",
        description="I'm an AI assistant powered by Gemini and Grok. Here's how to use me.",
        color=discord.Color.blue(),
    )
    embed.add_field(
        name="Slash commands",
        value=(
            "**`/ask`** *question* - Ask me anything.\n"
            "**`/ask_channel`** *question* [*days_back*] - Ask about this channel's recent messages.\n"
            "**`/mode`** *persona* - Set style: helpful, sarcastic, pirate, eli5.\n"
            "**`/remember`** *fact* - Store something I'll remember about you.\n"
            "**`/recall`** - See what I've remembered about you.\n"
            "**`/summarize_thread`** - Summarize this thread (use inside a thread).\n"
            "**`/channel_digest`** [*hours_back*] - Summarize key topics in this channel.\n"
            "**`/quiz start`** *topic* - Start a trivia quiz. **`/quiz answer`** *number* *answer* - Submit answer.\n"
            "**`/story start`** [*premise*] - Start an adventure. **`/story continue`** *action* - Continue.\n"
            "**`/reset quiz`** - Cancel active quiz. **`/reset persona`** - Reset to default style. **`/reset all`** - Reset both.\n"
            "**`/help`** - Show this message."
        ),
        inline=False,
    )
    embed.add_field(
        name="Other",
        value=(
            "**DM me** or **@mention me** - I'll reply with AI.\n"
            "**Share a URL** in your message - I'll read and summarize the web page content.\n"
            "**Right-click a message** → Apps → **Check message** (mods) - Toxicity check.\n"
            "In **support channels** (if configured), I may suggest replies or doc links."
        ),
        inline=False,
    )
    embed.add_field(
        name="Prefix commands (optional)",
        value=(
            "**`!status`** - Brain, mode, premium, rate limit.\n"
            "**`!mode`** [*persona*] - Set or show persona.\n"
            "**`!assistant`** *gemini|grok* - (Owner only) Switch AI brain.\n"
            "**`!addfeature`** - (Owner only) Add a feature to the website.\n"
            "**`!addpremium <user_id>`** - (Owner only) Grant test premium."
        ),
        inline=False,
    )
    embed.set_footer(text="Premium: higher limits + Grok brain. Subscribe via Discord.")
    await interaction.followup.send(embed=embed)


@bot.tree.context_menu(name="Check message")
async def check_message_ctx(interaction: discord.Interaction, message: discord.Message):
    await interaction.response.defer(ephemeral=True)
    if interaction.guild and not (interaction.user.guild_permissions.moderate_members or interaction.user.guild_permissions.kick_members):
        await interaction.followup.send("You need mod permissions to use this.", ephemeral=True)
        return
    content = (message.content or "").strip()
    if not content:
        await interaction.followup.send("That message has no text to check.", ephemeral=True)
        return
    score, reason = await run_toxicity_check(content)
    if score < 0:
        await interaction.followup.send("Could not analyze the message. Try again later.", ephemeral=True)
        return
    author_name = message.author.display_name or message.author.name
    result_text = f"**Toxicity check** (by {interaction.user.display_name})\nMessage from **{author_name}**: \"{content[:200]}{'…' if len(content) > 200 else ''}\"\n**Score:** {score}/10\n**Reason:** {reason}"
    if MOD_ALERTS_CHANNEL_ID:
        try:
            channel = bot.get_channel(int(MOD_ALERTS_CHANNEL_ID))
            if channel:
                await channel.send(result_text)
        except Exception as e:
            log.error(f"Failed to post to mod channel: {e}")
    await interaction.followup.send(f"**Score:** {score}/10\n**Reason:** {reason}", ephemeral=True)


# --- ENTITLEMENT EVENTS ---
@bot.event
async def on_entitlement_create(entitlement: discord.Entitlement):
    """Fires when a user subscribes."""
    log.info(f"NEW SUBSCRIPTION: User {entitlement.user_id} -> SKU {entitlement.sku_id}")
    new_ent = UserEntitlement(user_id=str(entitlement.user_id), sku_id=str(entitlement.sku_id))
    db_session.add(new_ent)
    db_session.commit()

    try:
        user = await bot.fetch_user(entitlement.user_id)
        if user:
            await user.send("**Thank you for subscribing to PolyMind Premium!** You've unlocked higher rate limits and the Grok AI brain. Enjoy!")
    except discord.Forbidden:
        log.info(f"Could not DM user {entitlement.user_id} (DMs disabled).")
    except Exception as e:
        log.error(f"Error sending subscription DM to {entitlement.user_id}: {e}")


@bot.event
async def on_entitlement_delete(entitlement: discord.Entitlement):
    """Fires when a subscription ends."""
    log.info(f"SUBSCRIPTION ENDED: User {entitlement.user_id}")
    db_session.query(UserEntitlement).filter(UserEntitlement.user_id == str(entitlement.user_id)).delete()
    db_session.commit()


# --- GLOBAL ERROR HANDLERS ---
@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: discord.app_commands.AppCommandError):
    """Global handler for slash command errors."""
    log.error(f"Slash command error in /{interaction.command.name if interaction.command else '?'}: {error}")
    try:
        if interaction.response.is_done():
            await interaction.followup.send("An unexpected error occurred. Try again later.", ephemeral=True)
        else:
            await interaction.response.send_message("An unexpected error occurred. Try again later.", ephemeral=True)
    except Exception:
        pass


@bot.event
async def on_command_error(ctx, error):
    """Global handler for prefix command errors."""
    if isinstance(error, commands.NotOwner):
        await ctx.send("You don't have permission to use this command.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing required argument: `{error.param.name}`. Use `!help` for usage.")
    elif isinstance(error, commands.CommandNotFound):
        pass
    else:
        log.error(f"Prefix command error: {error}")
        await ctx.send("An unexpected error occurred.")


# --- EVENTS ---
@bot.event
async def on_ready():
    global _commands_synced
    log.info(f"PolyMind logged in as {bot.user} (ID: {bot.user.id})")
    log.info(f"Connected to {len(bot.guilds)} guild(s)")

    if not _commands_synced:
        try:
            synced = await bot.tree.sync()
            log.info(f"Slash commands synced: {len(synced)} command(s)")
            _commands_synced = True
        except Exception as e:
            log.error(f"Failed to sync slash commands: {e}")

    start_rss_scheduler()


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    if is_dm or bot.user.mentioned_in(message):
        content = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()

        if content.startswith('!'):
            await bot.process_commands(message)
            return
        if not content and not is_dm:
            return

        if is_rate_limited(message.author.id):
            limit = settings["rate_limit_premium"] if has_premium(message.author.id) else settings["rate_limit_free"]
            await message.reply(settings["cooldown_msg"].format(limit=limit))
            return

        async with message.channel.typing():
            try:
                response = await get_ai_response(content, message.author.id)
                for chunk in split_message(response):
                    await message.reply(chunk)
            except Exception as e:
                log.error(f"on_message AI error for {message.author.id}: {e}")
                await message.reply("Something went wrong. Try again!")
    else:
        if SUPPORT_CHANNEL_IDS and str(message.channel.id) in SUPPORT_CHANNEL_IDS:
            content = (message.content or "").strip()
            if content and not content.startswith("!"):
                now = time.time()
                last = _support_last_suggestion.get(message.channel.id, 0)
                if now - last >= _SUPPORT_COOLDOWN_SEC:
                    _support_last_suggestion[message.channel.id] = now
                    asyncio.create_task(_support_suggestion_task(message))
        await bot.process_commands(message)


# --- START BOT ---
def run_bot():
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token or not token.strip():
        log.error('DISCORD_BOT_TOKEN is not set. Bot will not connect.')
        return
    log.info('Starting Discord bot...')
    try:
        bot.run(token.strip(), log_handler=None)
    except Exception as e:
        log.error(f'Bot failed to connect: {e}')
        traceback.print_exc()


bot_thread = Thread(target=run_bot, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    flask_app.run(host='0.0.0.0', port=port)
