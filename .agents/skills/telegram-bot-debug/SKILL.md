---
name: telegram-bot-debug
description: Debugging patterns and best practices for the python-telegram-bot handlers in this stock scanner. Use when modifying handle_message(), cmd_analyze(), adding new commands, or diagnosing message routing issues.
---

# Telegram Bot Debug Skill

## Bot Architecture Overview
This bot uses `python-telegram-bot` (v20+ async API).

```
main()
  └── Application
        ├── CommandHandler("start",   cmd_start)
        ├── CommandHandler("scan",    cmd_scan)
        ├── CommandHandler("analyze", cmd_analyze)
        ├── MessageHandler(TEXT & ~COMMAND, handle_message)   ← free-text handler
        └── JobQueue → scheduled_scan() every Saturday 21:00 Israel time
```

## Chat ID Security Guard
**Every handler** must have this as its first line:
```python
if str(update.effective_chat.id) != TELEGRAM_CHAT_ID:
    return
```
Never remove this. It prevents unauthorized users from triggering scans.
`cmd_start` is the only exception (intentionally public for future use).

## handle_message() Logic Flow
```
incoming text
    │
    ├─ contains weekly trigger? → run_scan()
    │   triggers: ['סריקה שבועית', 'weekly scan', 'scan all', 'full scan', '/scan']
    │
    ├─ contains ASCII ticker (2–5 chars, not a skip_word)? → analyze_single_stock()
    │   key rule: word.isascii() AND word.isalpha() — excludes Hebrew
    │
    └─ else → show help message
```

## Ticker Detection Rules (as of latest fix)
```python
clean = word.strip('.,!?()[]')
if 1 < len(clean) <= 5 and clean.isascii() and clean.isalpha() and clean not in skip_words:
    ticker = clean
    break
```
- **`isascii()`** — critical: prevents Hebrew words being detected as tickers
- Scans left-to-right, picks the **first** matching word
- `skip_words` contains common English command words (RUN, ON, FOR, etc.) and Hebrew command words

### When to add to `skip_words`
Add a word to `skip_words` if it's a 2–5 letter ASCII word that users commonly write in commands and could be mistaken for a ticker. Example: if you add a command word "TEST", add `'TEST'` to skip_words.

### When NOT to add to `skip_words`
Do not add real ticker symbols to skip_words. If a ticker collides with a command word, handle it via `/analyze TICKER` command instead.

## Message Length Limits
Telegram has a **4096 character** limit per message. The `tg_send()` helper handles this:
```python
async def tg_send(text):
    for i in range(0, len(text), 4096):
        await bot.send_message(..., text=text[i:i+4096], ...)
```
Photo captions are limited to **1024 characters** (handled in `tg_photo()`).
Never send raw long strings directly — always use `tg_send()`.

## parse_mode='Markdown' Gotchas
The bot uses `parse_mode='Markdown'` (Markdown V1, not V2).
- Supported: `*bold*`, `_italic_`, `` `code` ``
- **Unsupported in V1**: nested formatting, `>` blockquotes, `__underline__`
- Special chars that break V1: `[`, `]`, `(`, `)` if not forming a link
- If a message fails to send, try removing Markdown formatting or switching to `parse_mode=None`

## Adding a New Command
1. Define the async handler:
   ```python
   async def cmd_mycommand(update: Update, context: ContextTypes.DEFAULT_TYPE):
       if str(update.effective_chat.id) != TELEGRAM_CHAT_ID:
           return
       # ... logic ...
   ```
2. Register in `main()`:
   ```python
   app.add_handler(CommandHandler("mycommand", cmd_mycommand))
   ```
3. Add to the help text in `cmd_start()` and the else-branch of `handle_message()`
4. Add to the `/auto-commit` workflow after changes

## Scheduled Job (Saturday 21:00 Israel)
```python
app.job_queue.run_daily(
    scheduled_scan,
    time=time(hour=21, minute=0, tzinfo=ISRAEL_TZ),
    days=(5,),   # 0=Monday … 5=Saturday, 6=Sunday
    name="weekly_scan"
)
```
- `days=(5,)` = Saturday
- Timezone: `pytz.timezone('Asia/Jerusalem')` — handles DST automatically
- On Railway: ensure `APScheduler` is installed (comes with `python-telegram-bot[job-queue]`)

## Common Errors

### `telegram.error.Forbidden: bot was blocked by the user`
Bot was blocked. Nothing to fix in code — user must unblock.

### `telegram.error.BadRequest: Message is too long`
Message exceeds 4096 chars. Use `tg_send()` — never `bot.send_message()` directly.

### `JobQueue not found` / scheduled scan not firing
Install the job-queue extra:
```
pip install "python-telegram-bot[job-queue]"
```

### Handler not receiving messages
Check handler registration order in `main()`. `MessageHandler` with `TEXT & ~COMMAND` must come **after** all `CommandHandler`s, which is already the case.

### `update.message` is None
Can happen with edited messages or channel posts. Guard with:
```python
if not update.message:
    return
```
