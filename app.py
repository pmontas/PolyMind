import discord
from discord.ext import commands
import os
import aiohttp
import asyncio
from dotenv import load_dotenv
import datetime
import google.generativeai as genai
from flask import Flask
from threading import Thread
import traceback
import time
from collections import defaultdict

# Load environment variables
load_dotenv()

# --- WEB SERVER FOR RENDER/AZURE (KEEP-ALIVE) ---
app = Flask(__name__)

@app.route('/')
def home():
    return "PolyMind AI Bot is alive and running!"

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
    "rate_limit_per_min": 5,      # Messages per user per minute
    "max_chars_response": 1800,   # Target response length
    "cooldown_msg": "Slow down! You've reached your limit of {limit} messages per minute. Try again in a bit! ⏳"
}

# --- RATE LIMIT TRACKER ---
# Format: {user_id: [timestamp1, timestamp2, ...]}
user_usage = defaultdict(list)

# AI SYSTEM PROMPT
AI_SYSTEM_PROMPT = (
    "You are PolyMind, a cool, knowledgeable assistant. "
    "You're like a smart friend who knows a lot about everything. "
    "Talk naturally and casually, but be very helpful. "
    "Use your extensive internal knowledge to answer questions thoroughly. "
    "Keep your responses engaging and informative. "
    f"IMPORTANT: Keep your responses concise and under {settings['max_chars_response']} characters whenever possible."
)

# BANNED KEYWORDS (Safety layer)
BANNED_KEYWORDS = [
    "rm -rf", "format c:", "del /s /q c:", "shutdown /s", "system32"
]

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

# --- HELPER FUNCTIONS ---

def split_message(text, limit=2000):
    """Splits a string into chunks of a specific size"""
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

def is_rate_limited(user_id):
    """Checks if a user has exceeded their minute limit"""
    now = time.time()
    # Remove timestamps older than 60 seconds
    user_usage[user_id] = [t for t in user_usage[user_id] if now - t < 60]
    
    if len(user_usage[user_id]) >= settings["rate_limit_per_min"]:
        return True
    
    # Record this usage
    user_usage[user_id].append(now)
    return False

# --- AI INTEGRATIONS ---

async def get_gemini_response(prompt):
    """Connects to Google Gemini API"""
    if not gemini_model:
        return "❌ Gemini is not configured correctly."
    try:
        response = await asyncio.to_thread(gemini_model.generate_content, f"{AI_SYSTEM_PROMPT}\n\nUser: {prompt}")
        return response.text if response and response.text else "⚠️ Gemini returned an empty response."
    except Exception as e:
        print(f"[GEMINI] Error: {e}")
        return "❌ Gemini is having a moment. Try again shortly."

async def get_grok_response(prompt):
    """Connects to xAI Grok API"""
    headers = {"Authorization": f"Bearer {settings['xai_token']}", "Content-Type": "application/json"}
    payload = {
        "model": "grok-beta",
        "messages": [{"role": "system", "content": AI_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "stream": False
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(XAI_API_URL, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                return f"❌ Grok error (Status {response.status})."
    except Exception as e:
        print(f"[GROK] Error: {e}")
        return "❌ Can't reach Grok servers right now."

async def get_ai_response(prompt):
    """Main AI router"""
    clean_prompt = prompt.lower()
    for keyword in BANNED_KEYWORDS:
        if keyword in clean_prompt:
            return "Whoa, chill out with those commands. I'm just here to chat, not break things. 😅"
    
    if settings["provider"] == "grok":
        return await get_grok_response(prompt)
    return await get_gemini_response(prompt)

# --- COMMANDS ---

@bot.command(name='assistant')
@commands.is_owner()
async def set_assistant(ctx, provider: str, token: str = None):
    """Switch providers. Usage: !assistant grok [token] or !assistant gemini"""
    provider = provider.lower()
    if provider not in ["grok", "gemini"]:
        await ctx.send("❌ Use `grok` or `gemini`.")
        return
    settings["provider"] = provider
    if provider == "grok" and token:
        settings["xai_token"] = token
    await ctx.send(f"✅ Switched brain to **{provider.capitalize()}**.")

@bot.command(name='limit')
@commands.is_owner()
async def set_limit(ctx, new_limit: int):
    """Set the rate limit per minute. Usage: !limit 10"""
    settings["rate_limit_per_min"] = new_limit
    await ctx.send(f"✅ Rate limit updated to **{new_limit}** messages per minute.")

@bot.command(name='status')
async def bot_status(ctx):
    """Bot health report"""
    await ctx.send(
        f"**PolyMind System Report**\n"
        f"🧠 Active Brain: **{settings['provider'].upper()}**\n"
        f"🛡️ Rate Limit: **{settings['rate_limit_per_min']} msg/min**\n"
        f"⚡ Status: **Online & Healthy**"
    )

# --- EVENTS ---
@bot.event
async def on_ready():
    print(f'✅ SUCCESS: PolyMind logged in as {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mention = bot.user.mentioned_in(message)

    if is_dm or is_mention:
        content = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()
        
        if content.startswith('!'):
            await bot.process_commands(message)
            return

        if not content and not is_dm:
            return

        # --- RATE LIMIT CHECK ---
        if is_rate_limited(message.author.id):
            await message.reply(settings["cooldown_msg"].format(limit=settings["rate_limit_per_min"]))
            return

        async with message.channel.typing():
            try:
                response = await get_ai_response(content)
                chunks = split_message(response)
                for chunk in chunks:
                    await message.reply(chunk)
            except Exception as e:
                print(f"[ERROR] {e}")
                await message.reply("Oops, I ran into an error. Please try again!")
    else:
        await bot.process_commands(message)

# --- START BOT ---
def run_bot():
    token = os.getenv('DISCORD_BOT_TOKEN')
    if token:
        bot.run(token)

bot_thread = Thread(target=run_bot, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
