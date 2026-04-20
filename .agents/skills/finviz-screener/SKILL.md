---
name: finviz-screener
description: Guidelines for working with the Finviz screener in this stock scanner bot. Use when modifying get_stocks_from_finviz(), adjusting fundamental filters, or debugging Finviz-related issues.
---

# Finviz Screener Skill

## Location in Codebase
All Finviz logic lives in `get_stocks_from_finviz()` in `scanner.py` (lines ~433–477).
Uses the `finvizfinance` library: `from finvizfinance.screener.overview import Overview`.

## Current Filter Configuration
```python
foverview.set_filter(filters_dict={
    'Market Cap.': 'Small ($300mln to $2bln)',  # Finviz broad bucket
    'Beta': '1 to 2',
    'P/B': 'Over 1',
    'Average Volume': 'Over 200K',
    'Country': 'USA',
})
```
**Important**: Finviz filter buckets are coarse. The precise filter (Market Cap $200–400M, Beta 1.5–2.0) is applied **after** via yfinance in the same function. Do not rely on Finviz alone for exact ranges.

## Two-Stage Filtering Pattern
This project uses a deliberate two-stage design:
1. **Stage 1 — Finviz broad net**: Fast, returns many candidates cheaply.
2. **Stage 2 — yfinance precise filter**: Exact numeric checks against live data.

When adjusting filters, decide which stage is appropriate:
- Categorical / coarse → Finviz `filters_dict`
- Exact numeric ranges → yfinance `info` dict in the for-loop

## Current Precise Filters (Stage 2)
```python
200e6 <= mc <= 400e6    # Market Cap $200M–$400M
vol_usd >= 200_000      # Dollar volume > $200K/day
1.5 <= beta <= 2.0      # Beta 1.5–2.0
pb >= 1.0               # P/B > 1
```

## Valid finvizfinance Filter Keys (commonly used)
| Key | Example Values |
|-----|---------------|
| `'Market Cap.'` | `'Small ($300mln to $2bln)'`, `'Micro (under $300mln)'` |
| `'Beta'` | `'1 to 2'`, `'Over 1.5'` |
| `'P/B'` | `'Over 1'`, `'Under 3'` |
| `'Average Volume'` | `'Over 200K'`, `'Over 500K'` |
| `'Country'` | `'USA'` |
| `'Sector'` | `'Technology'`, `'Healthcare'` |
| `'P/E'` | `'Under 20'`, `'Profitable (>0)'` |
| `'EPS growththis year'` | `'Over 10%'` |

## Common Errors & Fixes

### `AttributeError: 'NoneType' object` from `screener_view()`
Finviz returned no results. Check:
- Filter combination is not too restrictive
- finvizfinance version compatibility (`pip show finvizfinance`)
- Finviz website not rate-limiting (add `time.sleep(2)` before the call)

### Empty DataFrame / 0 candidates
```python
if df is None or len(df) == 0:
    return [], []
```
Already handled. If persistently empty, relax a Finviz filter and rely on Stage 2.

### Rate limiting / `403` errors
Finviz has rate limits. Add a delay if running multiple scans:
```python
import time
time.sleep(3)  # before foverview.screener_view()
```

### `KeyError: 'Ticker'`
Use `.get('Ticker', '')` — already done in codebase. Never use `row['Ticker']` directly.

## Adding a New Finviz Filter
1. Find the valid string value from the Finviz website UI
2. Add it to `filters_dict` in `get_stocks_from_finviz()`
3. If the bucket is coarse, add a precise check in the Stage 2 for-loop using yfinance `info`
4. Test with a broad filter first, then tighten

## Performance Notes
- Finviz fetch is a single HTTP request (fast ~1–2s)
- The bottleneck is the Stage 2 yfinance loop (~2–5s per ticker)
- For large candidate lists (>100), consider adding `time.sleep(0.5)` between yfinance calls to avoid rate limiting
- The current chunked Telegram send (50 tickers per message) is the right pattern — keep it
