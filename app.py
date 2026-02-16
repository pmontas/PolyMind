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
if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini AI configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
else:
    print("WARNING: GOOGLE_API_KEY not found in environment variables.")

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

# --- STATE MANAGEMENT ---
settings = {
    "provider": "gemini", 
    "xai_token": os.getenv("XAI_API_KEY", "xai-XgUOgSxJIyOJhJY5njHg7k7X4YXk9MxLxF9hOw2NrNFomvDBvv8VF76WaJDlnTDDkp6Z7B83nj6CS8Yj")
}

# --- BOT SETUP ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

# --- AI INTEGRATIONS ---

async def get_gemini_response(prompt):
    """Connects to Google Gemini API (Free Tier)"""
    if not GEMINI_KEY:
        return "❌ Gemini API Key is missing in environment variables!"
    
    try:
        response = await asyncio.to_thread(gemini_model.generate_content, f"{AI_SYSTEM_PROMPT}\n\nUser: {prompt}")
        
        if response and response.text:
            return response.text
        else:
            return "Gemini returned an empty response. This might be due to a safety filter."
    except Exception as e:
        error_msg = str(e)
        print(f"CRITICAL Gemini Error: {error_msg}")
        if "API_KEY_INVALID" in error_msg:
            return "❌ Your Google API Key is invalid. Please check it in Azure settings."
        elif "User location is not supported" in error_msg:
            return "❌ Gemini is not available in the region where this bot is hosted."
        return f"Gemini is having a moment. Please try again in a few seconds."

async def get_grok_response(prompt):
    """Connects to xAI Grok API"""
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
        "stream": False
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(XAI_API_URL, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    return f"Grok error (Status {response.status})."
    except Exception as e:
        print(f"Grok error: {e}")
        return "I can't reach the Grok servers right now!"

async def get_ai_response(prompt):
    """Main AI router"""
    clean_prompt = prompt.lower()
    
    # Safety check
    for keyword in BANNED_KEYWORDS:
        if keyword in clean_prompt:
            return "Whoa, chill out with those commands. I'm just here to chat, not break things. 😅"

    if settings["provider"] == "grok":
        return await get_grok_response(prompt)
    else:
        return await get_gemini_response(prompt)

# --- COMMANDS ---

@bot.command(name='assistant')
@commands.has_permissions(administrator=True)
async def set_assistant(ctx, provider: str, token: str = None):
    """Switch between providers."""
    provider = provider.lower()
    if provider not in ["grok", "gemini"]:
        await ctx.send("❌ Invalid provider. Use `grok` or `gemini`.")
        return

    settings["provider"] = provider
    if provider == "grok" and token:
        settings["xai_token"] = token
        await ctx.send(f"✅ Switched to **Grok** with new token.")
    else:
        await ctx.send(f"✅ Switched to **{provider.capitalize()}**.")

# --- EVENTS ---
@bot.event
async def on_ready():
    print(f'SUCCESS: PolyMind logged in as {bot.user}')
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

        async with message.channel.typing():
            try:
                response = await get_ai_response(content)
                await message.reply(response)
            except Exception as e:
                print(f"Error processing message: {e}")
                traceback.print_exc()
                await message.reply("Oops, I ran into an error. Please try again!")
    else:
        await bot.process_commands(message)

# --- START BOT IN BACKGROUND ---
def run_bot():
    """Start the Discord bot in a separate thread"""
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        print("CRITICAL ERROR: DISCORD_BOT_TOKEN not found.")
        return
        
    try:
        print("Attempting to connect to Discord...")
        bot.run(token)
    except Exception as e:
        print(f"CRITICAL ERROR during bot startup: {e}")
        traceback.print_exc()

# Start the bot automatically when gunicorn loads this file
bot_thread = Thread(target=run_bot, daemon=True)
bot_thread.start()
print("Discord bot thread started in background.")

# This is only for local development (won't run on gunicorn/Azure)
if __name__ == "__main__":
    # For local testing, also start the Flask server
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
