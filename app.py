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

# Load environment variables
load_dotenv()

# --- WEB SERVER FOR RENDER (KEEP-ALIVE) ---
app = Flask(__name__)

@app.route('/')
def home():
    return "PolyMind AI Bot is alive and running!"

# --- CONFIGURATION ---
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

# Configure Gemini
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
XAI_KEY = os.getenv("XAI_API_KEY", "xai-XgUOgSxJIyOJhJY5njHg7k7X4YXk9MxLxF9hOw2NrNFomvDBvv8VF76WaJDlnTDDkp6Z7B83nj6CS8Yj")

print(f"GOOGLE_API_KEY present: {bool(GEMINI_KEY)}")
print(f"XAI_API_KEY present: {bool(XAI_KEY)}")

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        # Use gemini-3-flash-preview (the current fast model in 2026)
        gemini_model = genai.GenerativeModel('gemini-3-flash-preview')
        print("✅ Gemini AI configured successfully (using gemini-3-flash-preview).")
    except Exception as e:
        print(f"❌ Error configuring Gemini: {e}")
        gemini_model = None
else:
    print("⚠️ WARNING: GOOGLE_API_KEY not found. Gemini will not work.")
    gemini_model = None

# AI SYSTEM PROMPT
AI_SYSTEM_PROMPT = (
    "You are PolyMind, a cool, knowledgeable assistant. "
    "You're like a smart friend who knows a lot about everything. "
    "Talk naturally and casually, but be very helpful. "
    "Use your extensive internal knowledge to answer questions thoroughly. "
    "Keep your responses engaging and informative."
)

# BANNED KEYWORDS (Safety layer)
BANNED_KEYWORDS = [
    "rm -rf", "format c:", "del /s /q c:", "shutdown /s", "system32"
]

# --- STATE MANAGEMENT (Global, persistent) ---
settings = {
    "provider": "gemini",  # Default to Gemini
    "xai_token": XAI_KEY
}

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

# --- AI INTEGRATIONS ---

async def get_gemini_response(prompt):
    """Connects to Google Gemini API (Free Tier)"""
    print(f"[GEMINI] Attempting to use Gemini for prompt: {prompt[:50]}...")
    
    if not GEMINI_KEY:
        error = "❌ Gemini API Key is missing in environment variables!"
        print(f"[GEMINI] {error}")
        return error
    
    if not gemini_model:
        error = "❌ Gemini model not initialized!"
        print(f"[GEMINI] {error}")
        return error
    
    try:
        full_prompt = f"{AI_SYSTEM_PROMPT}\n\nUser: {prompt}"
        print(f"[GEMINI] Sending request to Gemini API...")
        response = await asyncio.to_thread(gemini_model.generate_content, full_prompt)
        
        if response and response.text:
            print(f"[GEMINI] ✅ Success! Response length: {len(response.text)}")
            return response.text
        else:
            error = "⚠️ Gemini returned an empty response (likely safety filter)."
            print(f"[GEMINI] {error}")
            return error
    except Exception as e:
        error_msg = str(e)
        print(f"[GEMINI] ❌ CRITICAL Error: {error_msg}")
        
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            return "❌ Your Gemini API Key is invalid."
        elif "location" in error_msg.lower() or "region" in error_msg.lower():
            return "❌ Gemini is not available in this Azure region. Try switching to Grok with `!assistant grok`"
        elif "quota" in error_msg.lower():
            return "❌ Gemini quota exceeded. Try again later or switch to Grok."
        
        return f"❌ Gemini error: {error_msg[:100]}"

async def get_grok_response(prompt):
    """Connects to xAI Grok API"""
    print(f"[GROK] Attempting to use Grok for prompt: {prompt[:50]}...")
    
    if not settings['xai_token']:
        error = "❌ Grok API Key is missing!"
        print(f"[GROK] {error}")
        return error
    
    headers = {
        "Authorization": f"Bearer {settings['xai_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "grok-beta",
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "temperature": 0.8
    }
    
    try:
        print(f"[GROK] Sending request to xAI API...")
        async with aiohttp.ClientSession() as session:
            async with session.post(XAI_API_URL, json=payload, headers=headers, timeout=30) as response:
                status = response.status
                print(f"[GROK] Response status: {status}")
                
                if status == 200:
                    data = await response.json()
                    result = data['choices'][0]['message']['content']
                    print(f"[GROK] ✅ Success! Response length: {len(result)}")
                    return result
                elif status == 401:
                    return "❌ Grok API Key is invalid or expired."
                elif status == 429:
                    return "❌ Grok rate limit exceeded. Try again in a moment."
                else:
                    error_text = await response.text()
                    print(f"[GROK] ❌ Error response: {error_text[:200]}")
                    return f"❌ Grok error (Status {status}): {error_text[:100]}"
    except asyncio.TimeoutError:
        error = "❌ Grok request timed out. Try again."
        print(f"[GROK] {error}")
        return error
    except Exception as e:
        error_msg = str(e)
        print(f"[GROK] ❌ Connection error: {error_msg}")
        return f"❌ Can't reach Grok servers: {error_msg[:100]}"

async def get_ai_response(prompt):
    """Main AI router"""
    clean_prompt = prompt.lower()
    
    # Safety check
    for keyword in BANNED_KEYWORDS:
        if keyword in clean_prompt:
            return "Whoa, chill out with those commands. I'm just here to chat, not break things. 😅"

    # Log which provider we're using
    current_provider = settings["provider"]
    print(f"[ROUTER] Current provider: {current_provider}")
    
    if current_provider == "grok":
        return await get_grok_response(prompt)
    else:
        return await get_gemini_response(prompt)

# --- COMMANDS ---

@bot.command(name='assistant')
@commands.has_permissions(administrator=True)
async def set_assistant(ctx, provider: str, token: str = None):
    """Switch between providers."""
    provider = provider.lower()
    print(f"[COMMAND] Received !assistant command: provider={provider}, token={'<provided>' if token else '<none>'}")
    
    if provider not in ["grok", "gemini"]:
        await ctx.send("❌ Invalid provider. Use `!assistant grok` or `!assistant gemini`.")
        return

    settings["provider"] = provider
    print(f"[COMMAND] Provider switched to: {settings['provider']}")
    
    if provider == "grok":
        if token:
            settings["xai_token"] = token
            await ctx.send(f"✅ Switched to **Grok** with new token.")
        else:
            await ctx.send(f"✅ Switched to **Grok** (using default token).")
    else:
        await ctx.send(f"✅ Switched to **Gemini**.")

@bot.command(name='status')
async def bot_status(ctx):
    """Check which AI provider is active"""
    current = settings["provider"]
    gemini_ok = "✅" if gemini_model else "❌"
    grok_ok = "✅" if settings["xai_token"] else "❌"
    
    await ctx.send(
        f"**PolyMind Status**\n"
        f"Current Provider: **{current.upper()}**\n"
        f"Gemini: {gemini_ok}\n"
        f"Grok: {grok_ok}"
    )

# --- EVENTS ---
@bot.event
async def on_ready():
    print(f'✅ SUCCESS: PolyMind logged in as {bot.user}')
    print(f"[BOT] Default provider: {settings['provider']}")
    print("Bot is ready to receive messages.")

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

        print(f"[MESSAGE] Received from {message.author}: {content[:50]}...")
        async with message.channel.typing():
            try:
                response = await get_ai_response(content)
                await message.reply(response)
            except Exception as e:
                error_msg = f"Oops, I crashed! Error: {str(e)[:100]}"
                print(f"[ERROR] {error_msg}")
                traceback.print_exc()
                await message.reply(error_msg)
    else:
        await bot.process_commands(message)

# --- START BOT IN BACKGROUND ---
def run_bot():
    """Start the Discord bot in a separate thread"""
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        print("❌ CRITICAL ERROR: DISCORD_BOT_TOKEN not found.")
        return
        
    try:
        print("🚀 Attempting to connect to Discord...")
        bot.run(token)
    except Exception as e:
        print(f"❌ CRITICAL ERROR during bot startup: {e}")
        traceback.print_exc()

# Start the bot automatically when gunicorn loads this file
bot_thread = Thread(target=run_bot, daemon=True)
bot_thread.start()
print("✅ Discord bot thread started in background.")

# This is only for local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
