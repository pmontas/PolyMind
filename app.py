import discord
from discord.ext import commands
import os
import aiohttp
import asyncio
from dotenv import load_dotenv
import datetime
import google.generativeai as genai
from flask import Flask, render_template, jsonify
from threading import Thread
import traceback
import time
from collections import defaultdict
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

# --- DATABASE SETUP ---
Base = declarative_base()

class BotFeature(Base):
    __tablename__ = 'bot_features'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    description = Column(String)
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False) # NEW: Gating
    category = Column(String) # e.g., "AI", "Utility", "Safety"

class UserEntitlement(Base):
    __tablename__ = 'user_entitlements'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    sku_id = Column(String)
    ends_at = Column(DateTime, nullable=True)

# Create database engine
engine = create_engine('sqlite:///polymind.db', connect_args={'check_same_thread': False})
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db_session = Session()

# Initialize default features if database is empty
def init_db():
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
        print("✅ Database initialized with default features.")


def ensure_channel_memory_feature():
    """Add Channel Memory feature if missing (e.g. after deploy so the website shows it)."""
    if db_session.query(BotFeature).filter(BotFeature.name == "Channel Memory").first():
        return
    db_session.add(BotFeature(
        name="Channel Memory",
        description="Ask about this channel's recent messages (e.g. what did Carolyn say Friday?) with /ask_channel.",
        category="Utility",
        is_premium=False,
    ))
    db_session.commit()
    print("✅ Added 'Channel Memory' feature to database.")


init_db()
ensure_channel_memory_feature()

# --- WEB SERVER (FLASK) ---
app = Flask(__name__)

@app.route('/')
def home():
    features = db_session.query(BotFeature).filter(BotFeature.is_active == True).all()
    return render_template('index.html', features=features, year=datetime.datetime.now().year)

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/tos')
def tos():
    return render_template('tos.html')

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting web server on port {port}...")
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run_web_server)
    t.daemon = True
    t.start()

# --- CONFIGURATION ---
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
PREMIUM_SKU_ID = os.getenv("PREMIUM_SKU_ID", "YOUR_SKU_ID_HERE") # Set this in Azure

# Configure Gemini
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
XAI_KEY = os.getenv("XAI_API_KEY", "xai-XgUOgSxJIyOJhJY5njHg7k7X4YXk9MxLxF9hOw2NrNFomvDBvv8VF76WaJDlnTDDkp6Z7B83nj6CS8Yj")

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel('gemini-3-flash-preview')
        print("✅ Gemini AI configured successfully.")
    except Exception as e:
        print(f"❌ Error configuring Gemini: {e}")
        gemini_model = None
else:
    gemini_model = None

# --- CONFIGURABLE SETTINGS ---
settings = {
    "provider": "gemini",
    "xai_token": XAI_KEY,
    "rate_limit_free": 5,
    "rate_limit_premium": 20,
    "max_chars_response": 1800,
    "cooldown_msg": "Slow down! You've reached your limit of {limit} messages per minute. Upgrade to Premium for higher limits! 🚀"
}

# --- RATE LIMIT TRACKER ---
user_usage = defaultdict(list)

# AI SYSTEM PROMPT
AI_SYSTEM_PROMPT = (
    "You are PolyMind, a cool, knowledgeable assistant. "
    "You're like a smart friend who knows a lot about everything. "
    "Talk naturally and casually, but be very helpful. "
    "Use your extensive internal knowledge to answer questions thoroughly. "
    "Keep your responses engaging and informative."
)

# BANNED KEYWORDS
BANNED_KEYWORDS = ["rm -rf", "format c:", "del /s /q c:", "shutdown /s", "system32"]

# Channel memory (ask about channel history)
CHANNEL_MEMORY_MAX_DAYS = 14
CHANNEL_MEMORY_MAX_MESSAGES = 200
CHANNEL_MEMORY_MAX_CHARS = 4000

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

# --- HELPER FUNCTIONS ---
def split_message(text, limit=2000):
    if len(text) <= limit: return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text); break
        split_at = text.rfind('\n', 0, limit)
        if split_at == -1: split_at = text.rfind(' ', 0, limit)
        if split_at == -1: split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks

def has_premium(user_id):
    """Checks if a user has an active premium entitlement"""
    ent = db_session.query(UserEntitlement).filter(UserEntitlement.user_id == str(user_id)).first()
    return ent is not None


def get_membership_display(user_id):
    """Returns what membership the bot is detecting for this user (for status)."""
    ent = db_session.query(UserEntitlement).filter(UserEntitlement.user_id == str(user_id)).first()
    if not ent:
        return "Free"
    if ent.sku_id == str(PREMIUM_SKU_ID):
        return "PolyMind Premium"
    return f"Premium (SKU: {ent.sku_id})"

def is_rate_limited(user_id):
    """Checks if a user has exceeded their minute limit"""
    now = time.time()
    limit = settings["rate_limit_premium"] if has_premium(user_id) else settings["rate_limit_free"]
    
    user_usage[user_id] = [t for t in user_usage[user_id] if now - t < 60]
    if len(user_usage[user_id]) >= limit: return True
    
    user_usage[user_id].append(now)
    return False

# --- AI INTEGRATIONS ---
async def get_gemini_response(prompt):
    if not gemini_model: return "❌ Gemini is not configured."
    try:
        response = await asyncio.to_thread(gemini_model.generate_content, f"{AI_SYSTEM_PROMPT}\n\nUser: {prompt}")
        return response.text if response and response.text else "⚠️ Gemini returned empty."
    except Exception as e:
        return f"❌ Gemini error: {str(e)[:100]}"

async def get_grok_response(prompt):
    headers = {"Authorization": f"Bearer {settings['xai_token']}", "Content-Type": "application/json"}
    payload = {"model": "grok-beta", "messages": [{"role": "system", "content": AI_SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "stream": False}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(XAI_API_URL, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                return f"❌ Grok error {response.status}."
    except Exception: return "❌ Grok unreachable."

async def get_ai_response(prompt, user_id):
    clean_prompt = prompt.lower()
    for kw in BANNED_KEYWORDS:
        if kw in clean_prompt: return "Whoa, chill out! 😅"
    
    # Gate Grok for Premium users only
    if settings["provider"] == "grok" and not has_premium(user_id):
        return "✨ **Grok is a Premium feature.** Upgrade your subscription to unlock the Grok brain! ✨"
        
    if settings["provider"] == "grok": return await get_grok_response(prompt)
    return await get_gemini_response(prompt)

# --- CHANNEL MEMORY (ask about channel history) ---
async def fetch_channel_transcript(channel, days_back: int, max_chars: int = CHANNEL_MEMORY_MAX_CHARS):
    """Fetch recent messages from a text channel and return a transcript string. Returns (transcript, error_msg)."""
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
        return None, f"Could not read channel history: {str(e)[:80]}"

def _channel_memory_prompt(transcript: str, question: str) -> str:
    return (
        "The user is asking about messages in this Discord channel. "
        "Below is a recent transcript (Author (date time): content). "
        "Answer their question based ONLY on this transcript. If the answer isn't there, say so briefly.\n\n"
        "--- Transcript ---\n" + transcript + "\n--- End ---\n\nUser question: " + question
    )

async def get_channel_aware_response(transcript: str, question: str, user_id):
    """Answer a question using channel transcript as context. Banned check on question only; respects provider/premium."""
    q_clean = question.lower()
    for kw in BANNED_KEYWORDS:
        if kw in q_clean:
            return "Whoa, chill out! 😅"
    if settings["provider"] == "grok" and not has_premium(user_id):
        return "✨ **Grok is a Premium feature.** Upgrade to use channel memory with Grok! ✨"
    prompt = _channel_memory_prompt(transcript, question)
    if settings["provider"] == "grok":
        return await get_grok_response(prompt)
    return await get_gemini_response(prompt)

# --- COMMANDS ---
@bot.command(name='addfeature')
@commands.is_owner()
async def add_feature(ctx, name: str, category: str, is_premium: bool, *, description: str):
    """Add a new feature. Usage: !addfeature "Name" "Category" True/False Description"""
    new_feat = BotFeature(name=name, category=category, is_premium=is_premium, description=description)
    db_session.add(new_feat)
    db_session.commit()
    await ctx.send(f"✅ Feature **{name}** added! (Premium: {is_premium})")

@bot.command(name='assistant')
@commands.is_owner()
async def set_assistant(ctx, provider: str):
    settings["provider"] = provider.lower()
    await ctx.send(f"✅ Brain: **{provider.capitalize()}**")

def make_help_embed():
    """Shared help content for /help and !help."""
    embed = discord.Embed(
        title="PolyMind – What I Can Do",
        description="I'm an AI assistant powered by Gemini and Grok. Here's how to use me.",
        color=discord.Color.blue(),
    )
    embed.add_field(
        name="Slash commands",
        value=(
            "**`/ask`** *question* — Ask me anything. I'll reply with an AI answer.\n"
            "**`/ask_channel`** *question* [*days_back*] — Ask about this channel's recent messages (e.g. *What did Carolyn say Friday?*). Use in a server channel.\n"
            "**`/help`** — Show this message."
        ),
        inline=False,
    )
    embed.add_field(
        name="Other ways to talk to me",
        value=(
            "• **DM me** — Send any message and I'll reply.\n"
            "• **@mention me** in a channel — Type `@PolyMind` and your question."
        ),
        inline=False,
    )
    embed.add_field(
        name="Prefix commands (optional)",
        value=(
            "**`!status`** — Your brain, premium status, and rate limit.\n"
            "**`!assistant`** *gemini|grok* — (Owner only) Switch the default AI brain.\n"
            "**`!addfeature`** — (Owner only) Add a feature to the website list."
        ),
        inline=False,
    )
    embed.set_footer(text="Premium: higher limits + Grok brain • Subscribe via Discord or visit your server's bot link.")
    return embed


@bot.command(name='help')
async def help_prefix(ctx):
    """Show everything PolyMind can do (same as /help)."""
    await ctx.send(embed=make_help_embed())


@bot.command(name='status')
async def bot_status(ctx):
    premium = "✅ Active" if has_premium(ctx.author.id) else "❌ Inactive"
    membership = get_membership_display(ctx.author.id)
    limit = settings["rate_limit_premium"] if has_premium(ctx.author.id) else settings["rate_limit_free"]
    await ctx.send(
        f"**PolyMind System Report**\n"
        f"🧠 Brain: **{settings['provider'].upper()}**\n"
        f"💎 Your Premium: {premium}\n"
        f"📋 Membership detected: **{membership}**\n"
        f"🛡️ Rate Limit: {limit} msg/min"
    )

# --- SLASH COMMANDS (for App Directory requirement) ---
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
        await interaction.followup.send("Oops, something went wrong. Try again!", ephemeral=True)


@bot.tree.command(name="ask_channel", description="Ask about this channel's recent messages (e.g. what did Carolyn say Friday?).")
@discord.app_commands.describe(
    question="Your question about the channel (e.g. What did Carolyn say Friday?)",
    days_back="How many days of history to look at (1–14, default 7)"
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
        await interaction.followup.send("Use this command in a server text channel so I can read recent messages.", ephemeral=True)
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
    except Exception:
        await interaction.followup.send("Something went wrong answering from channel history. Try again!", ephemeral=True)


@bot.tree.command(name="help", description="See everything PolyMind can do.")
async def help_slash(interaction: discord.Interaction):
    await interaction.response.send_message(embed=make_help_embed(), ephemeral=True)


# --- ENTITLEMENT EVENTS ---
@bot.event
async def on_entitlement_create(entitlement: discord.Entitlement):
    """Fires when a user subscribes"""
    print(f"💎 NEW SUBSCRIPTION: User {entitlement.user_id} subscribed to SKU {entitlement.sku_id}")
    new_ent = UserEntitlement(user_id=str(entitlement.user_id), sku_id=str(entitlement.sku_id))
    db_session.add(new_ent)
    db_session.commit()
    
    # Try to send a thank you DM
    try:
        user = await bot.fetch_user(entitlement.user_id)
        if user:
            await user.send("💎 **Thank you for subscribing to PolyMind Premium!** You have unlocked higher rate limits and the Grok AI brain. Enjoy! 🚀")
    except: pass

@bot.event
async def on_entitlement_delete(entitlement: discord.Entitlement):
    """Fires when a subscription ends"""
    print(f"❌ SUBSCRIPTION ENDED: User {entitlement.user_id}")
    db_session.query(UserEntitlement).filter(UserEntitlement.user_id == str(entitlement.user_id)).delete()
    db_session.commit()

# --- EVENTS ---
@bot.event
async def on_ready():
    print(f'✅ SUCCESS: PolyMind logged in as {bot.user}')
    try:
        synced = await bot.tree.sync()
        print(f'✅ Slash commands synced: {len(synced)} command(s)')
    except Exception as e:
        print(f'❌ Failed to sync slash commands: {e}')

@bot.event
async def on_message(message):
    if message.author == bot.user: return
    is_dm = isinstance(message.channel, discord.DMChannel)
    if is_dm or bot.user.mentioned_in(message):
        content = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()
        if content.startswith('!'):
            await bot.process_commands(message)
            return
        if not content and not is_dm: return
        
        # --- RATE LIMIT CHECK ---
        if is_rate_limited(message.author.id):
            limit = settings["rate_limit_premium"] if has_premium(message.author.id) else settings["rate_limit_free"]
            await message.reply(settings["cooldown_msg"].format(limit=limit))
            return

        async with message.channel.typing():
            try:
                response = await get_ai_response(content, message.author.id)
                for chunk in split_message(response): await message.reply(chunk)
            except Exception: await message.reply("Oops, error!")
    else:
        await bot.process_commands(message)

# --- START BOT ---
def run_bot():
    token = os.getenv('DISCORD_BOT_TOKEN')
    if token: bot.run(token)

bot_thread = Thread(target=run_bot, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
