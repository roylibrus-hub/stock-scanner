---
name: yfinance-patterns
description: Best practices, common pitfalls, and patterns for yfinance usage in this stock scanner bot. Use when fetching historical data, reading ticker info, or debugging data issues in scanner.py.
---

# yfinance Patterns Skill

## How yfinance Is Used in This Project
All yfinance calls are in `scanner.py`. Two main usage patterns:

### 1. Ticker Info (fundamentals)
```python
ticker = yf.Ticker(sym)
info = ticker.info
px   = info.get('currentPrice', info.get('regularMarketPrice', 0))
mc   = info.get('marketCap', 0)
beta = info.get('beta', 0)
pb   = info.get('priceToBook', 0)
```

### 2. Historical OHLCV (for charting and pattern detection)
```python
daily   = ticker.history(period="2y",  interval="1d")
weekly  = ticker.history(period="5y",  interval="1wk")
monthly = ticker.history(period="10y", interval="1mo")
```
Returns a DataFrame with columns: `Open, High, Low, Close, Volume, Dividends, Stock Splits`.

## Critical Pitfalls & Fixes

### `info` dict keys are inconsistent across tickers
Some tickers return `'currentPrice'`, others only `'regularMarketPrice'`. Always use:
```python
px = info.get('currentPrice', info.get('regularMarketPrice', 0))
```
Never use `info['currentPrice']` — it will raise `KeyError` on some tickers.

### `history()` returns empty DataFrame for delisted/bad tickers
Always guard with:
```python
if daily is None or len(daily) < 30:
    # skip or return early
```
The codebase uses `< 30` for single-stock and `< 60` for the weekly scan. Keep these thresholds.

### MultiIndex columns after `history()` (yfinance ≥ 0.2.x)
If you ever use `yf.download()` for multiple tickers at once, columns become a MultiIndex.
**Avoid `yf.download()` in this project** — always use `yf.Ticker(sym).history()` to keep single-level columns compatible with pattern detection functions.

### Timezone-aware index
`history()` returns a DatetimeIndex that may be timezone-aware. If you do date arithmetic:
```python
import pandas as pd
# Safe comparison:
df.index = df.index.tz_localize(None)  # strip tz if needed
```

### `beta` can be `None`
```python
beta = info.get('beta', 0) or 0  # the `or 0` handles None
pb   = info.get('priceToBook', 0) or 0
```
The `or 0` pattern (already in codebase) is essential — `None` will break numeric comparisons.

## Data Periods Used by Feature
| Feature | Period | Interval | Min rows needed |
|---------|--------|----------|----------------|
| Daily chart | `2y` | `1d` | 30 (single), 60 (scan) |
| Weekly chart | `5y` | `1wk` | — |
| Monthly chart | `10y` | `1mo` | — |
| Cup & Handle detection | `2y` | `1d` | 126 days |
| Weekly pattern scan | `5y` | `1wk` | 26 bars |
| Monthly pattern scan | `10y` | `1mo` | 6 bars |

## Rate Limiting
yfinance uses the Yahoo Finance API. For the weekly full scan (many tickers):
- Add `time.sleep(0.5)` between tickers if you see `429 Too Many Requests`
- yfinance caches within a session, but each `yf.Ticker()` instantiation is fresh

## Column Name Consistency
Pattern detection functions (`detect_*`) rely on these exact column names:
- `df['High']`, `df['Low']`, `df['Close']`, `df['Volume']`

These come from `.history()` unchanged. Never rename them. If you add a derived column, give it a new name (e.g., `df['MA20']`).

## Adding a New Data Field
To add a new fundamental field (e.g., debt-to-equity):
1. Look up the yfinance `info` key: e.g., `'debtToEquity'`
2. Add to the `stock` dict in both `get_stocks_from_finviz()` and `analyze_single_stock()`:
   ```python
   'de_ratio': info.get('debtToEquity', 'N/A'),
   ```
3. Add to `fmt_report()` for display
4. Add to the Claude prompt in `confirm_with_claude()` if relevant for analysis

## Chart Save Path
The codebase handles Windows vs Linux:
```python
path = f'/tmp/{symbol}_chart.png' if os.name != 'nt' else f'C:\\stock_scanner\\{symbol}_chart.png'
```
Never hardcode a path. Always use this conditional pattern for cross-platform compatibility (local dev on Windows, Railway on Linux).
