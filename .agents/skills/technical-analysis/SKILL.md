---
name: technical-analysis
description: Patterns, logic, and debugging guidance for the algorithmic chart pattern detection system in this stock scanner. Use when modifying detect_* functions, run_pattern_scan(), or the Claude Vision confirm_with_claude() prompt.
---

# Technical Analysis Skill

## Pattern Detection Architecture

### Entry Point
```python
run_pattern_scan(df, min_cup_days=126)
```
Runs detectors in **priority order** — returns the first pattern found:
1. Cup and Handle ← highest priority (most actionable bullish setup)
2. Inverse Head and Shoulders
3. Double Bottom
4. Triple Bottom
5. Head and Shoulders
6. Double Top
7. Triple Top
8. Ascending / Descending / Symmetrical Triangle ← lowest priority

### Priority Selection (multi-timeframe)
In both `analyze_single_stock()` and `run_scan()`, the best pattern across 3 timeframes is chosen:
```python
priority = ['Cup and Handle', 'Inverse Head and Shoulders', 'Double Bottom', ...]
best = {'pattern': 'None'}
for res in [algo_daily, algo_weekly, algo_monthly]:
    if priority.index(res.get('pattern','None')) < priority.index(best.get('pattern','None')):
        best = res
```

## Core Helper: `get_pivots(series, bars=6)`
Returns local highs AND lows from a price series.
```python
# A point is a pivot if it equals the max OR min of its surrounding window
if val == window.max() or val == window.min():
    pivots.iloc[i] = val
```
- `bars=6` for H&S, Inverse H&S (needs clean, significant pivots)
- `bars=5` for Double/Triple Top/Bottom
- `bars=4` for Triangle (needs more pivots, less strict)

**When to adjust `bars`**: Increase for cleaner signals on noisy tickers; decrease to detect patterns on shorter histories.

## Pattern Return Schema
Every detector returns either `None` (not found) or a dict:
```python
{
    'pattern': 'Cup and Handle',        # string name
    'support_levels': [12.50, 11.00],   # list of float
    'resistance_levels': [15.00],       # list of float
    # pattern-specific keys:
    'neckline': 13.25,                  # H&S, Inverse H&S only
    'cup_start_idx': 45,                # Cup & Handle only
    'cup_end_idx': 120,                 # Cup & Handle only
}
```
The chart renderer (`create_multi_timeframe_chart`) uses `support_levels`, `resistance_levels`, and `neckline` from this dict. Always include them.

## Pattern-Specific Notes

### Cup and Handle
- Most complex detector — 5 nested conditions on price structure + volume
- `min_days=126` on daily (6 months), `26` on weekly (6 months in weeks), `6` on monthly
- Cup depth: low must be 65–85% of the cup high (`0.65 * A <= B <= 0.85 * A`)
- Handle must be a pullback that doesn't exceed the cup right rim
- Volume during cup must be below the average (accumulation pattern)

### Head and Shoulders / Inverse H&S
- Requires 5 pivot points: Left Shoulder, Neckline-Left, Head, Neckline-Right, Right Shoulder
- Neckline symmetry: `abs(B-D) < avg_bar`
- For H&S: current price must be below the right shoulder (`F < E`)
- For Inverse H&S: current price must be above the right shoulder (`F > E`)

### Double Top / Bottom
- Volume confirmation: for Double Top, second peak must have LOWER volume than first (`vol_C < vol_A`)
- Price proximity: the two peaks/troughs must be within `avg_bar` of each other
- No volume check on Double Bottom (standard TA — volume less critical on bottoms)

### Triangles
- Uses `bars=4` for more frequent pivots
- Symmetrical: converging highs and lows
- Ascending: flat highs, rising lows
- Descending: falling highs, flat lows

## Adding a New Pattern Detector
1. Create `detect_mypattern(df) -> dict | None`
2. Return the standard schema (include `support_levels`, `resistance_levels`)
3. Add to `run_pattern_scan()` detectors list (position = priority)
4. Add the pattern name string to the `priority` list in both `analyze_single_stock()` and `run_scan()`
5. Add name to Claude's prompt in `confirm_with_claude()` (the pipe-separated enum string)

## Claude Vision Integration

### What Claude receives
- The 3-timeframe chart image (base64 PNG)
- The algo-detected pattern name
- Fundamental data (sector, beta, P/B, P/E, EPS)
- Technical indicators (RSI, MACD, Bollinger, ADX, Stoch, Vol ratio)

### Claude's output is the final signal
Claude can **override** the algo pattern (`"algo_confirmed": false`). The `fmt_report()` shows `⚠️ Algo overridden` in this case.

### Prompt modification guidelines
- The JSON schema in the prompt is strict — if you add a field, add it to `fmt_report()` too
- `watchlist: true` requires: clear pattern + high/medium confidence + ≥2 timeframe alignment
- Model: `claude-opus-4-5` — do not change without testing cost impact

## Technical Indicators (`calc_indicators`)
Uses the `ta` library. Indicators fed to Claude:
| Indicator | Library call | Used for |
|-----------|-------------|----------|
| RSI(14) | `ta.momentum.RSIIndicator` | Overbought/oversold |
| MACD | `ta.trend.MACD` | Trend momentum |
| Bollinger Bands(20) | `ta.volatility.BollingerBands` | Volatility |
| ADX | `ta.trend.ADXIndicator` | Trend strength |
| Stochastic | `ta.momentum.StochasticOscillator` | Momentum |
| Volume ratio | rolling 20-bar average | Unusual volume |

If `ta` is not installed: `pip install ta`. The function already handles `ImportError` gracefully with a try/except.

## Chart Visual Design
Dark theme (`#0d1117` background). Key colors:
- Price line: `#58a6ff` (blue)
- MA20: `#ffd700` (gold)
- MA50: `#ff9500` (orange)
- MA200: `#ff4757` (red)
- Support lines: `#2ed573` (green dashed)
- Resistance lines: `#ff4757` (red dashed)
- Neckline: `#ff9500` (orange solid, thicker)

When modifying chart appearance, keep these colors consistent — Claude is calibrated to read this specific color scheme in its system prompt context.
