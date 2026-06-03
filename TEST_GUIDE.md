# PolyMind – Test Guide

Use this guide to verify slash commands and channel-memory behavior after deployment.

---

## Prerequisites

- **Bot in a server**  
  PolyMind must be invited with:
  - Scopes: `bot` and `applications.commands`
  - Bot permission: **Read Message History** (required for `/ask_channel`)

- **Environment**  
  - `DISCORD_BOT_TOKEN` set
  - `GOOGLE_API_KEY` set (for Gemini) if you use the default brain

- **After code changes**  
  Restart the bot (and redeploy on Azure if applicable). Slash commands sync on `on_ready`.

---

## 1. Slash command: `/ask`

**What it does:** General AI question; same behavior as DM / @mention.

| Step | Action | Expected result |
|------|--------|------------------|
| 1.1 | In any channel, type `/ask` and choose the command. | Autocomplete shows **ask** with option **question**. |
| 1.2 | Enter a question (e.g. `What is 2+2?`) and send. | Bot shows “thinking” then replies with an answer (e.g. 4). |
| 1.3 | Send `/ask` with a long question or one that triggers a long answer. | Reply is split into multiple messages if over ~2000 characters. |
| 1.4 | Send `/ask` many times in a short period (over your rate limit). | Bot replies with the rate-limit message (ephemeral if via slash). |
| 1.5 | Use `/ask` in a DM. | Works; bot answers in the DM. |

---

## 2. Slash command: `/ask_channel`

**What it does:** Answers questions about **this channel’s** recent messages (e.g. “What did Carolyn say Friday?”). Only works in a **server text channel** where the bot can read history.

| Step | Action | Expected result |
|------|--------|------------------|
| 2.1 | In a **server text channel**, type `/ask_channel`. | Autocomplete shows **ask_channel** with **question** and **days_back** (optional). |
| 2.2 | Run `/ask_channel` in a **DM**. | Bot replies (ephemeral): *“Use this command in a server text channel so I can read recent messages.”* |
| 2.3 | In a channel with **no** recent (non-bot) messages, run `/ask_channel` with any question. | Bot replies (ephemeral): *“No messages found in this channel for that time range.”* |
| 2.4 | Have someone (e.g. Carolyn) send a few normal messages in the channel. Then run `/ask_channel` with **question**: e.g. `What did Carolyn say?` and **days_back**: e.g. `Last 7 days`. | Bot replies with an answer based on those messages (e.g. summarizes or quotes what Carolyn said). |
| 2.5 | Ask a question that **cannot** be answered from the transcript (e.g. `What did Bob say about the moon?` when Bob never wrote that). | Bot says the answer isn’t in the transcript (or similar). |
| 2.6 | Use **days_back** “Last 1 day” when there are messages from today. | Bot uses only the last day of history. |
| 2.7 | Remove the bot’s **Read Message History** permission in that channel (or use a channel the bot can’t read). Run `/ask_channel` there. | Bot replies (ephemeral): *“I don’t have permission to read this channel’s message history.”* |

---

## 3. Rate limits and premium

| Step | Action | Expected result |
|------|--------|------------------|
| 3.1 | Send `/ask` or `/ask_channel` repeatedly until you hit the limit. | After the limit, you get the cooldown message (ephemeral for slash). |
| 3.2 | (If you have premium) Use `/ask` / `/ask_channel` at the higher premium limit. | Higher limit applies; no false rate-limit. |

---

## 4. Quick checklist

- [ ] `/ask` appears in slash list and answers a simple question.
- [ ] `/ask_channel` appears in slash list.
- [ ] `/ask_channel` in DM returns “Use this command in a server text channel…”.
- [ ] `/ask_channel` in a server channel with no history returns “No messages found…”.
- [ ] After posting messages in the channel, `/ask_channel` with a question about them returns an answer based on that history.
- [ ] Rate limit message appears when exceeding the limit.
- [ ] Without Read Message History in a channel, `/ask_channel` returns the permission error.

---

## 5. Optional: Console checks

When the bot starts you should see something like:

- `✅ SUCCESS: PolyMind logged in as ...`
- `✅ Slash commands synced: 2 command(s)` (or more if you add others)

If sync fails you’ll see: `❌ Failed to sync slash commands: ...`

---

## 6. Invite URL reminder

To see slash commands in a server, the invite must include **`applications.commands`**:

```
https://discord.com/api/oauth2/authorize?client_id=YOUR_APP_ID&permissions=...&scope=bot%20applications.commands
```

Replace `YOUR_APP_ID` with your application ID and adjust `permissions` as needed. Include **Read Message History** (e.g. `2048` or a role that has it) for `/ask_channel` to work in those channels.

---

## 7. Scheduled tasks (`/task`) — BETA

**What it does:** Schedule daily automated posts to a channel — RSS digests, AI prompts, channel digests, or live web research.

### Prerequisites

- Bot needs **Send Messages** in the target channel.
- For **digest** tasks: **Read Message History** in that channel.
- Production scheduling requires Azure **Always On** (Basic tier+).
- Env: `SCHEDULED_TASKS_ENABLED=true` (default). Set `false` to disable the feature entirely.

### Create and manage

| Step | Action | Expected result |
|------|--------|------------------|
| 7.1 | Run `/task help` | Ephemeral embed explains types, limits (5/10), and examples. |
| 7.2 | `/task create` type **ai**, time `09:00`, channel #general, prompt `Say hello` | Task created with ID; shows active count (e.g. 1/5). |
| 7.3 | `/task list` | Shows task #id, type, time, channel, active/paused status. |
| 7.4 | `/task pause <id>` then `/task list` | Task shows paused; active count decreases. |
| 7.5 | `/task resume <id>` | Task active again (blocked if at cap). |
| 7.6 | `/task delete <id>` | Task removed; slot freed. |

### Task types

| Step | Action | Expected result |
|------|--------|------------------|
| 7.7 | **rss** with valid feed URLs | At scheduled time, bot posts RSS summary to channel. |
| 7.8 | **ai** with a custom prompt | At scheduled time, bot posts AI response. |
| 7.9 | **digest** in a channel with messages | At scheduled time, bot posts channel recap. |
| 7.10 | **research** with a news prompt | Uses Gemini + Google Search; may fail on free tier (clear error). |

### Limits and isolation

| Step | Action | Expected result |
|------|--------|------------------|
| 7.11 | Create tasks until server hits **5/5** active (or 10 with Premium) | Create/resume blocked with upgrade message. |
| 7.12 | Set `SCHEDULED_TASKS_ENABLED=false`, restart | Bot starts; `/task` absent; `/ask` still works. |
| 7.13 | Invalid RSS URL in task | After 3 failures, task auto-pauses; other tasks unaffected. |

### Console checks

On startup with tasks enabled:

- `ScheduledTask table ready.`
- `Registered /task command group.`
- `Scheduled tasks tick scheduler started (every 60s).`

### Known limitations

- Once per day per task (timezone-aware via `timezone` option or `DEFAULT_TZ` env).
- **research** tasks may require paid Gemini billing for Google Search grounding.
- Task tick runs every 60s; exact minute match required (Azure Always On required in production).
