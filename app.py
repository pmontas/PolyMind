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
from flask import Flask, render_template, jsonify
from threading import Thread
import traceback
import time
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

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


init_db()
ensure_channel_memory_feature()

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


# Expose Flask app for gunicorn as 'app'
app = flask_app

# --- CONFIGURATION ---
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
PREMIUM_SKU_ID = os.getenv("PREMIUM_SKU_ID")
if not PREMIUM_SKU_ID:
    log.warning("PREMIUM_SKU_ID is not set. Premium detection will not work.")

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

# BANNED KEYWORDS (case-insensitive regex patterns)
BANNED_PATTERNS = [
    re.compile(r"rm\s+-rf", re.IGNORECASE),
    re.compile(r"format\s+c:", re.IGNORECASE),
    re.compile(r"del\s+/s\s+/q\s+c:", re.IGNORECASE),
    re.compile(r"shutdown\s+/s", re.IGNORECASE),
    re.compile(r"system32", re.IGNORECASE),
]

# Channel memory constants
CHANNEL_MEMORY_MAX_DAYS = 14
CHANNEL_MEMORY_MAX_MESSAGES = 200
CHANNEL_MEMORY_MAX_CHARS = 4000

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
async def get_gemini_response(prompt: str) -> str:
    if not gemini_model:
        return "Gemini is not configured. Please contact the bot owner."
    try:
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            f"{AI_SYSTEM_PROMPT}\n\nUser: {prompt}"
        )
        return response.text if response and response.text else "Gemini returned an empty response."
    except Exception as e:
        log.error(f"Gemini error: {e}")
        return "Something went wrong with the AI. Try again in a moment."


# Current xAI chat model (grok-beta is deprecated). Use grok-3-mini, grok-3, or grok-4-1-fast-reasoning.
XAI_MODEL = os.getenv("XAI_MODEL", "grok-3-mini")


async def get_grok_response(prompt: str) -> str:
    if not XAI_KEY:
        return "Grok is not configured. Please contact the bot owner."
    headers = {"Authorization": f"Bearer {XAI_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": XAI_MODEL,
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
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


async def get_ai_response(prompt: str, user_id) -> str:
    """Route a prompt to the configured AI provider with input validation."""
    if len(prompt) > settings["max_prompt_length"]:
        return f"Your message is too long (max {settings['max_prompt_length']} characters). Please shorten it."

    if _contains_banned(prompt):
        return "I can't help with that request."

    if settings["provider"] == "grok" and not has_premium(user_id):
        return "**Grok is a Premium feature.** Upgrade your subscription to unlock the Grok brain!"

    if settings["provider"] == "grok":
        return await get_grok_response(prompt)
    return await get_gemini_response(prompt)


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

    prompt = _channel_memory_prompt(transcript, question)
    if settings["provider"] == "grok":
        return await get_grok_response(prompt)
    return await get_gemini_response(prompt)


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
    await ctx.send(
        f"**PolyMind System Report**\n"
        f"Brain: **{settings['provider'].upper()}**\n"
        f"Your Premium: {premium}\n"
        f"Membership detected: **{membership}**\n"
        f"Rate Limit: {limit} msg/min"
    )


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
            "**`/help`** - Show this message."
        ),
        inline=False,
    )
    embed.add_field(
        name="Other ways to talk to me",
        value=(
            "**DM me** - Send any message and I'll reply.\n"
            "**@mention me** in a channel - Type `@PolyMind` and your question."
        ),
        inline=False,
    )
    embed.add_field(
        name="Prefix commands (optional)",
        value=(
            "**`!status`** - Your brain, premium status, and rate limit.\n"
            "**`!assistant`** *gemini|grok* - (Owner only) Switch AI brain.\n"
            "**`!addfeature`** - (Owner only) Add a feature to the website.\n"
            "**`!addpremium <user_id>`** - (Owner only) Grant test premium to a user (for Azure/testing)."
        ),
        inline=False,
    )
    embed.set_footer(text="Premium: higher limits + Grok brain. Subscribe via Discord.")
    await interaction.followup.send(embed=embed)


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
