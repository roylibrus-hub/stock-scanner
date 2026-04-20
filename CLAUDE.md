# Stock Scanner Bot — Project Memory

## Project Overview
Python Telegram bot. Scans US stocks for chart patterns, sends alerts.
Runs auto every Saturday 21:00 Israel time on Railway.

## Architecture
- **scanner.py** — single file, all logic
- **Finviz** — stage 1 broad filter (market cap, beta, P/B, volume, country)
- **yfinance** — stage 2 precise filter + historical OHLCV
- **Pattern detection** — algorithmic, 8 chart patterns
- **Claude Vision (claude-opus-4-5)** — visual confirm via 3-timeframe chart
- **Telegram** — alerts, charts, reports to private chat

## Fundamental Filters (Stage 2, precise)
- Market cap: $200M–$400M
- Beta: 1.5–2.0
- P/B: > 1.0
- Dollar volume: > $200K/day
- Country: USA

## Chart Patterns (priority order)
1. Cup and Handle
2. Inverse Head and Shoulders
3. Double Bottom
4. Triple Bottom
5. Head and Shoulders
6. Double Top
7. Triple Top
8. Triangle (Symmetrical / Ascending / Descending)

## Timeframes
- Daily: 2y, 1d
- Weekly: 5y, 1wk
- Monthly: 10y, 1mo

## Key Functions
- `get_stocks_from_finviz()` — two-stage stock filter
- `run_pattern_scan(df, min_cup_days)` — all detectors, priority order
- `create_multi_timeframe_chart()` — dark 3-panel chart, /tmp (Linux) or C:\stock_scanner\ (Windows)
- `confirm_with_claude()` — chart + indicators → Claude Vision → JSON
- `analyze_single_stock(symbol, update)` — full single-stock pipeline
- `run_scan()` — full weekly scan
- `handle_message()` — routes free-text to action

## Telegram Commands
- `/scan` — full weekly scan
- `/analyze <TICKER>` — single stock
- `/start` — help
- Free text with ticker → auto-analyze
- "סריקה שבועית" / "weekly scan" → full scan

## Ticker Detection (handle_message)
Words uppercased, then checked:
- `word.isascii()` — ASCII only (blocks Hebrew)
- `word.isalpha()` — letters only
- len 2–5
- Not in `skip_words`

## Environment Variables
- `ANTHROPIC_API_KEY`
- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID` — security guard on every handler

## Deployment
- Platform: Railway
- Schedule: Saturday 21:00 `Asia/Jerusalem` (pytz, DST-safe)
- `days=(5,)` = Saturday (0=Monday)

## Tech Stack
python-telegram-bot (v20+ async), yfinance, finvizfinance, anthropic, matplotlib (Agg), pandas, numpy, ta, pytz

## Rules
- Chat ID guard on every handler
- Long messages → `tg_send()` (4096 char limit)
- Chart path → OS conditional (`os.name != 'nt'`)
- yfinance: `.get()` with fallback, never direct key access on `info`
- `beta` and `pb_ratio` → `or 0` pattern (can be None)
- `isascii()` on ticker detection — do not remove
