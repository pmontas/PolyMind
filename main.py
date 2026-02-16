import discord
from discord.ext import commands
import os
import aiohttp
import asyncio
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import datetime
import json
import google.generativeai as genai
from flask import Flask
from threading import Thread

# Load environment variables
load_dotenv()

# --- WEB SERVER FOR RENDER (KEEP-ALIVE) ---
app = Flask('')

@app.route('/')
def home():
    return "AI Bot is alive and running!"

def run_web_server():
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run_web_server)
    t.start()

# --- CONFIGURATION ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# AI SYSTEM PROMPT
AI_SYSTEM_PROMPT = (
    "You are a cool, knowledgeable assistant with live web access. "
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
    "provider": "gemini", # Default to free cloud AI for hosting
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
    try:
        full_prompt = f"{AI_SYSTEM_PROMPT}\n\nCONTEXT FROM WEB:\n{search_context}\n\nUser: {prompt}"
        response = await asyncio.to_thread(gemini_model.generate_content, full_prompt)
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return "My Gemini brain is having a moment. Try again! 🧠💨"

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
                    return f"Grok is acting up (Error {response.status})."
    except Exception as e:
        print(f"Grok error: {e}")
        return "I can't reach the Grok servers right now!"

async def get_ollama_response(prompt, search_context):
    """Connects to local Ollama"""
    try:
        async with aiohttp.ClientSession() as session:
            full_prompt = f"{AI_SYSTEM_PROMPT}{search_context}\n\nUser says: {prompt}\nYour response:"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False
            }
            async with session.post(OLLAMA_API_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', "I'm thinking, but I can't find the words!")
                else:
                    return "My local brain is a bit foggy (Ollama error)."
    except Exception as e:
        print(f"Ollama error: {e}")
        return "I can't connect to my local brain right now!"

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
        return await get_ollama_response(prompt, search_context)

# --- COMMANDS ---

@bot.command(name='assistant')
@commands.has_permissions(administrator=True)
async def set_assistant(ctx, provider: str, token: str = None):
    """Switch between ollama, grok, and gemini."""
    provider = provider.lower()
    if provider not in ["ollama", "grok", "gemini"]:
        await ctx.send("❌ Invalid provider. Use `ollama`, `grok`, or `gemini`.")
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
    print(f'AI Chat Bot logged in as {bot.user}')

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
            response = await get_ai_response(content)
            await message.reply(response)
    else:
        await bot.process_commands(message)

# --- START BOT ---
def main():
    # Start the keep-alive web server
    keep_alive()
    
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        print("ERROR: DISCORD_BOT_TOKEN not found.")
        return
    bot.run(token)

if __name__ == "__main__":
    main()
