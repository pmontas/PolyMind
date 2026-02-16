import discord
from discord.ext import commands
import os
import aiohttp
import asyncio
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import datetime
import google.generativeai as genai
from flask import Flask
from threading import Thread
import traceback

# Load environment variables
load_dotenv()

# --- WEB SERVER FOR RENDER (KEEP-ALIVE) ---
app = Flask('')

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
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"
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
    "You are PolyMind, a cool, knowledgeable assistant with live web access. "
    "You're like a smart friend who knows a lot about everything. "
    "Talk naturally and casually, but be very helpful. "
    "When the user asks for news or latest info, use the search results provided to answer. "
    "If no search results are provided, use your internal knowledge. "
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

# --- WEB SEARCH HELPER ---
async def search_web(query):
    """Searches the web using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                search_text = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
                return f"\n\nLATEST WEB SEARCH RESULTS:\n{search_text}"
    except Exception as e:
        print(f"Search error: {e}")
    return ""

# --- AI INTEGRATIONS ---

async def get_gemini_response(prompt, search_context):
    """Connects to Google Gemini API (Free Tier)"""
    if not GEMINI_KEY:
        return "❌ Gemini API Key is missing in Render environment variables!"
    
    try:
        full_prompt = f"{AI_SYSTEM_PROMPT}\n\nCONTEXT FROM WEB:\n{search_context}\n\nUser: {prompt}"
        # Use asyncio.to_thread for the blocking Gemini call
        response = await asyncio.to_thread(gemini_model.generate_content, full_prompt)
        
        if response and response.text:
            return response.text
        else:
            return "Gemini returned an empty response. It might be a safety filter block."
    except Exception as e:
        error_msg = str(e)
        print(f"CRITICAL Gemini Error: {error_msg}")
        if "API_KEY_INVALID" in error_msg:
            return "❌ Your Google API Key is invalid. Please check it in Render settings."
        elif "User location is not supported" in error_msg:
            return "❌ Gemini is not available in the region where this bot is hosted (Render)."
        return f"Gemini is having a moment: {error_msg[:100]}..."

async def get_grok_response(prompt, search_context):
    """Connects to xAI Grok API"""
    headers = {
        "Authorization": f"Bearer {settings['xai_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "grok-beta",
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": f"{search_context}\n\nUser says: {prompt}"}
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

    # Decide if we should search the web
    search_triggers = ["news", "latest", "today", "current", "who is", "weather", "stock", "crypto", "post", "tweet"]
    is_question = any(clean_prompt.startswith(w) for w in ["who", "what", "where", "when", "why", "how"])
    needs_search = any(trigger in clean_prompt for trigger in search_triggers) or (is_question and len(clean_prompt.split()) > 3)

    search_context = ""
    if needs_search:
        search_context = await search_web(prompt)

    if settings["provider"] == "grok":
        return await get_grok_response(prompt, search_context)
    elif settings["provider"] == "gemini":
        return await get_gemini_response(prompt, search_context)
    else:
        # Fallback to Gemini if Ollama is requested but we are in the cloud
        return await get_gemini_response(prompt, search_context)

# --- COMMANDS ---

@bot.command(name='assistant')
@commands.has_permissions(administrator=True)
async def set_assistant(ctx, provider: str, token: str = None):
    """Switch between providers."""
    provider = provider.lower()
    if provider not in ["grok", "gemini"]:
        await ctx.send("❌ Invalid provider for cloud hosting. Use `grok` or `gemini`.")
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
                await message.reply("Oops, I ran into an error processing that. Check my logs!")
    else:
        await bot.process_commands(message)

# --- START BOT ---
def main():
    # Start the keep-alive web server
    keep_alive()
    
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        print("CRITICAL ERROR: DISCORD_BOT_TOKEN not found in environment variables.")
        return
        
    try:
        print("Attempting to connect to Discord...")
        bot.run(token)
    except discord.errors.LoginFailure:
        print("CRITICAL ERROR: Improper Discord token. Please check DISCORD_BOT_TOKEN in Render.")
    except Exception as e:
        print(f"CRITICAL ERROR during bot startup: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
