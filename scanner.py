import os
import yfinance as yf
import anthropic
import asyncio
import base64
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from datetime import datetime, time
import pytz

# ── Load .env ─────────────────────────────────────────────────────────────────
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ISRAEL_TZ        = pytz.timezone('Asia/Jerusalem')


# ══════════════════════════════════════════════════════════════════════════════
#  PATTERN DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_pivots(series, bars=6):
    pivots = pd.Series(index=series.index, dtype=float)
    n = len(series)
    for i in range(bars, n - bars):
        window = series.iloc[i - bars: i + bars + 1]
        val = series.iloc[i]
        if val == window.max() or val == window.min():
            pivots.iloc[i] = val
    return pivots.dropna()

def avg_bar_range(high, low):
    return (high - low).mean()

# ── Cup and Handle ─────────────────────────────────────────────────────────────
def detect_cup_and_handle(df, min_days=126):
    """Cup depth 30-50%, right rim ≥75% of left, handle 5-15% pullback in upper third."""
    if len(df) < min_days:
        return None
    close = df['Close']
    vol   = df['Volume']
    n     = len(df)
    for start in range(0, n - min_days, 5):
        seg = close.iloc[start: start + min_days]
        if len(seg) < 50:
            continue
        K     = seg.iloc[0]
        A_idx = seg.idxmax()
        A     = seg[A_idx]
        A_pos = seg.index.get_loc(A_idx)
        if A_pos < 20:
            continue
        after_A = seg.iloc[A_pos:]
        if len(after_A) < 10:
            continue
        B_idx = after_A.idxmin()
        B     = after_A[B_idx]
        B_pos = after_A.index.get_loc(B_idx)
        # Cup depth: 30-50% correction (bottom is 50-70% of left high)
        if not (0.50 * A <= B <= 0.70 * A):
            continue
        if not (20 <= A_pos <= 80):
            continue
        after_B = after_A.iloc[B_pos:]
        if len(after_B) < 5:
            continue
        C_idx = after_B.idxmax()
        C     = after_B[C_idx]
        C_pos = after_B.index.get_loc(C_idx)
        # Right rim must reach ≥75% of left high (U-shape, not V)
        if not (0.75 * A <= C <= 1.02 * A):
            continue
        if not (5 <= C_pos <= 40):
            continue
        after_C = after_B.iloc[C_pos:]
        if len(after_C) < 3:
            continue
        D = after_C.min()  # handle low
        handle_drop = (C - D) / C
        # Handle: 5-15% pullback from right rim
        if not (0.05 <= handle_drop <= 0.15):
            continue
        # Handle stays in upper third of cup
        upper_third_floor = B + (A - B) * (2 / 3)
        if D < upper_third_floor:
            continue
        # Volume dries up in cup bottom
        cup_vol = vol.iloc[start + A_pos: start + A_pos + B_pos].mean()
        if cup_vol >= vol.mean():
            continue
        return {
            'pattern':        'Cup and Handle',
            'cup_start_price': round(K, 2),
            'cup_high':        round(A, 2),
            'cup_low':         round(B, 2),
            'cup_right':       round(C, 2),
            'handle_low':      round(D, 2),
            'breakout_level':  round(C, 2),
            'support_levels':  [round(B, 2), round(D, 2)],
            'resistance_levels': [round(A, 2)],
            'cup_start_idx':   start,
            'cup_bottom_idx':  start + A_pos + B_pos,
            'cup_end_idx':     start + A_pos + B_pos + C_pos,
        }
    return None

# ── Head and Shoulders ─────────────────────────────────────────────────────────
def detect_head_and_shoulders(df, min_bars=63):
    """LS/RS within 15%, neckline slope ≤10%, volume RS < LS, price must be near/below neckline."""
    if len(df) < min_bars:
        return None
    highs = get_pivots(df['High'], bars=6)
    lows  = get_pivots(df['Low'],  bars=6)
    if len(highs) < 3 or len(lows) < 2:
        return None
    vol      = df['Volume']
    F        = df['Close'].iloc[-1]
    pivots   = pd.concat([highs, lows]).sort_index()
    pv       = list(pivots.items())  # (index, value)
    for i in range(len(pv) - 4):
        try:
            idxA, A = pv[i];   idxB, B = pv[i+1]; idxC, C = pv[i+2]
            idxD, D = pv[i+3]; idxE, E = pv[i+4]
            # Head > both shoulders
            if not (C > max(A, E)):
                continue
            # Neckline troughs below shoulders
            if not (B < min(A, E) and D < min(A, E)):
                continue
            # Shoulder symmetry: RS within 15% of LS
            if abs(A - E) / max(A, E) > 0.15:
                continue
            # Neckline relatively flat (slope ≤10% difference)
            neckline = (B + D) / 2
            if abs(B - D) / max(B, D) > 0.10:
                continue
            # Price near or below right shoulder (pattern forming/broken)
            if F > E * 1.05:
                continue
            # Volume: RS period volume < LS period volume
            vol_ls = vol[idxA:idxB].mean() if idxA < idxB else 0
            vol_rs = vol[idxD:idxE].mean() if idxD < idxE else 0
            if vol_rs >= vol_ls and vol_ls > 0:
                continue
            price_target = round(neckline - (C - neckline), 2)
            return {
                'pattern':          'Head and Shoulders',
                'neckline':          round(neckline, 2),
                'head':              round(C, 2),
                'left_shoulder':     round(A, 2),
                'right_shoulder':    round(E, 2),
                'price_target':      price_target,
                'support_levels':    [round(neckline, 2)],
                'resistance_levels': [round(C, 2)],
                'ls_idx': idxA, 'head_idx': idxC, 'rs_idx': idxE,
                'nl_left_idx': idxB, 'nl_right_idx': idxD,
            }
        except Exception:
            continue
    return None

# ── Inverse Head and Shoulders ─────────────────────────────────────────────────
def detect_inverse_hns(df, min_bars=63):
    """Mirror of H&S: head lowest, RS within 15% of LS, volume RS < LS."""
    if len(df) < min_bars:
        return None
    lows  = get_pivots(df['Low'],  bars=6)
    highs = get_pivots(df['High'], bars=6)
    if len(lows) < 3 or len(highs) < 2:
        return None
    vol    = df['Volume']
    F      = df['Close'].iloc[-1]
    pivots = pd.concat([highs, lows]).sort_index()
    pv     = list(pivots.items())
    for i in range(len(pv) - 4):
        try:
            idxA, A = pv[i];   idxB, B = pv[i+1]; idxC, C = pv[i+2]
            idxD, D = pv[i+3]; idxE, E = pv[i+4]
            if not (C < min(A, E)):
                continue
            if not (B > max(A, E) and D > max(A, E)):
                continue
            if abs(A - E) / max(A, E) > 0.15:
                continue
            neckline = (B + D) / 2
            if abs(B - D) / max(B, D) > 0.10:
                continue
            if F < E * 0.95:
                continue
            vol_ls = vol[idxA:idxB].mean() if idxA < idxB else 0
            vol_rs = vol[idxD:idxE].mean() if idxD < idxE else 0
            if vol_rs >= vol_ls and vol_ls > 0:
                continue
            price_target = round(neckline + (neckline - C), 2)
            return {
                'pattern':          'Inverse Head and Shoulders',
                'neckline':          round(neckline, 2),
                'head':              round(C, 2),
                'left_shoulder':     round(A, 2),
                'right_shoulder':    round(E, 2),
                'price_target':      price_target,
                'support_levels':    [round(C, 2)],
                'resistance_levels': [round(neckline, 2)],
                'ls_idx': idxA, 'head_idx': idxC, 'rs_idx': idxE,
                'nl_left_idx': idxB, 'nl_right_idx': idxD,
            }
        except Exception:
            continue
    return None

# ── Double Top ─────────────────────────────────────────────────────────────────
def detect_double_top(df, min_bars=63):
    """Two peaks within 3%, volume lower on 2nd, price below middle trough."""
    if len(df) < min_bars:
        return None
    highs = get_pivots(df['High'], bars=5)
    lows  = get_pivots(df['Low'],  bars=5)
    if len(highs) < 2 or len(lows) < 1:
        return None
    vol = df['Volume']
    cur = df['Close'].iloc[-1]
    hv  = list(highs.items())
    lv  = lows.values
    for i in range(len(hv) - 1):
        idxA, A = hv[i]; idxC, C = hv[i+1]
        # Within 3%
        if abs(A - C) / max(A, C) > 0.03:
            continue
        # Volume lower on 2nd peak
        vol_A = vol.get(idxA, 0) if hasattr(vol, 'get') else vol.loc[idxA] if idxA in vol.index else 0
        vol_C = vol.get(idxC, 0) if hasattr(vol, 'get') else vol.loc[idxC] if idxC in vol.index else 0
        if vol_C >= vol_A:
            continue
        # Middle trough between the two peaks
        trough_candidates = [v for v in lv if v < min(A, C)]
        if not trough_candidates:
            continue
        B = max(trough_candidates)
        # Price near or below middle trough
        if cur > B * 1.03:
            continue
        target = round(B - (max(A, C) - B), 2)
        return {
            'pattern':          'Double Top',
            'top1':              round(A, 2),
            'top2':              round(C, 2),
            'middle_trough':     round(B, 2),
            'price_target':      target,
            'support_levels':    [round(B, 2)],
            'resistance_levels': [round(max(A, C), 2)],
            'top1_idx': idxA, 'top2_idx': idxC,
        }
    return None

# ── Double Bottom ───────────────────────────────────────────────────────────────
def detect_double_bottom(df, min_bars=63):
    """Two troughs within 3%, middle peak above both, price near/above middle peak."""
    if len(df) < min_bars:
        return None
    lows  = get_pivots(df['Low'],  bars=5)
    highs = get_pivots(df['High'], bars=5)
    if len(lows) < 2 or len(highs) < 1:
        return None
    cur = df['Close'].iloc[-1]
    lv  = list(lows.items())
    hv  = highs.values
    for i in range(len(lv) - 1):
        idxA, A = lv[i]; idxC, C = lv[i+1]
        if abs(A - C) / max(A, C) > 0.03:
            continue
        peak_candidates = [v for v in hv if v > max(A, C)]
        if not peak_candidates:
            continue
        B = min(peak_candidates)
        if cur < B * 0.97:
            continue
        target = round(B + (B - min(A, C)), 2)
        return {
            'pattern':          'Double Bottom',
            'bottom1':           round(A, 2),
            'bottom2':           round(C, 2),
            'middle_peak':       round(B, 2),
            'price_target':      target,
            'support_levels':    [round(min(A, C), 2)],
            'resistance_levels': [round(B, 2)],
            'bot1_idx': idxA, 'bot2_idx': idxC,
        }
    return None

# ── Triple Top ─────────────────────────────────────────────────────────────────
def detect_triple_top(df, min_bars=63):
    """Three peaks within 3% of each other."""
    if len(df) < min_bars:
        return None
    highs = get_pivots(df['High'], bars=5)
    if len(highs) < 3:
        return None
    hv = list(highs.items())
    for i in range(len(hv) - 2):
        idxA, A = hv[i]; idxB, B = hv[i+1]; idxC, C = hv[i+2]
        avg = (A + B + C) / 3
        if max(abs(A-avg), abs(B-avg), abs(C-avg)) / avg > 0.03:
            continue
        return {
            'pattern':          'Triple Top',
            'resistance_levels': [round(avg, 2)],
            'support_levels':   [],
            't1_idx': idxA, 't2_idx': idxB, 't3_idx': idxC,
        }
    return None

# ── Triple Bottom ───────────────────────────────────────────────────────────────
def detect_triple_bottom(df, min_bars=63):
    """Three troughs within 3% of each other."""
    if len(df) < min_bars:
        return None
    lows = get_pivots(df['Low'], bars=5)
    if len(lows) < 3:
        return None
    lv = list(lows.items())
    for i in range(len(lv) - 2):
        idxA, A = lv[i]; idxB, B = lv[i+1]; idxC, C = lv[i+2]
        avg = (A + B + C) / 3
        if max(abs(A-avg), abs(B-avg), abs(C-avg)) / avg > 0.03:
            continue
        return {
            'pattern':        'Triple Bottom',
            'support_levels': [round(avg, 2)],
            'resistance_levels': [],
            'b1_idx': idxA, 'b2_idx': idxB, 'b3_idx': idxC,
        }
    return None

# ── Triangles ──────────────────────────────────────────────────────────────────
def detect_triangle(df, min_bars=42):
    """≥2 touches each trendline, volume contracting, breakout within 2/3 of apex."""
    if len(df) < min_bars:
        return None
    highs = get_pivots(df['High'], bars=4)
    lows  = get_pivots(df['Low'],  bars=4)
    if len(highs) < 2 or len(lows) < 2:
        return None
    vol = df['Volume']
    F   = df['Close'].iloc[-1]
    hi  = list(highs.items())
    lo  = list(lows.items())
    # Volume contraction: last half avg < first half avg
    mid = len(df) // 2
    vol_early = vol.iloc[:mid].mean()
    vol_late  = vol.iloc[mid:].mean()
    vol_contracting = vol_late < vol_early
    hv = highs.values; hi_idx = list(highs.index)
    lv = lows.values;  lo_idx = list(lows.index)
    for i in range(len(hv) - 1):
        A, C = hv[i], hv[i+1]   # two resistance touches
        for j in range(len(lv) - 1):
            B, D = lv[j], lv[j+1]   # two support touches
            # Symmetrical: converging highs + lows
            if A > C and B < D:
                # Apex approximation via linear projection
                resist_slope = (C - A)
                support_slope = (D - B)
                within_two_thirds = (F > D) and (F < C)
                if within_two_thirds and vol_contracting:
                    return {
                        'pattern': 'Symmetrical Triangle',
                        'resist_hi': [round(A,2), round(C,2)],
                        'support_lo': [round(B,2), round(D,2)],
                        'support_levels':    [round(D, 2)],
                        'resistance_levels': [round(C, 2)],
                        'hi_touches': [hi_idx[i], hi_idx[i+1]],
                        'lo_touches': [lo_idx[j], lo_idx[j+1]],
                    }
            # Ascending: flat resistance, rising support (≥2 touches each)
            if abs(A - C) / max(A, C) <= 0.02 and B < D and D < F < A:
                return {
                    'pattern': 'Ascending Triangle',
                    'resist_level': round((A+C)/2, 2),
                    'support_levels':    [round(D, 2)],
                    'resistance_levels': [round((A+C)/2, 2)],
                    'hi_touches': [hi_idx[i], hi_idx[i+1]],
                    'lo_touches': [lo_idx[j], lo_idx[j+1]],
                }
            # Descending: falling resistance, flat support (≥2 touches each)
            if abs(B - D) / max(B, D) <= 0.02 and A > C and F < C:
                return {
                    'pattern': 'Descending Triangle',
                    'support_level': round((B+D)/2, 2),
                    'support_levels':    [round((B+D)/2, 2)],
                    'resistance_levels': [round(C, 2)],
                    'hi_touches': [hi_idx[i], hi_idx[i+1]],
                    'lo_touches': [lo_idx[j], lo_idx[j+1]],
                }
    return None

def run_pattern_scan(df, min_cup_days=126):
    detectors = [
        lambda d: detect_cup_and_handle(d, min_cup_days),
        detect_inverse_hns, detect_double_bottom, detect_triple_bottom,
        detect_head_and_shoulders, detect_double_top, detect_triple_top, detect_triangle,
    ]
    for fn in detectors:
        try:
            result = fn(df)
            if result:
                return result
        except Exception:
            continue
    return {'pattern': 'None'}


# ══════════════════════════════════════════════════════════════════════════════
#  CHART
# ══════════════════════════════════════════════════════════════════════════════
def _draw_candles(ax, df_tf):
    """Draw OHLC candlestick bars using Rectangle patches."""
    opens  = df_tf['Open'].values
    highs  = df_tf['High'].values
    lows   = df_tf['Low'].values
    closes = df_tf['Close'].values
    n = len(df_tf)
    for i in range(n):
        bull = closes[i] >= opens[i]
        col  = '#26a641' if bull else '#f85149'
        body_lo = min(opens[i], closes[i])
        body_hi = max(opens[i], closes[i])
        height  = max(body_hi - body_lo, (highs[i] - lows[i]) * 0.01)
        ax.add_patch(plt.Rectangle((i - 0.35, body_lo), 0.7, height,
                                    color=col, zorder=3))
        ax.plot([i, i], [lows[i], body_lo],  color=col, lw=0.7, zorder=2)
        ax.plot([i, i], [body_hi, highs[i]], color=col, lw=0.7, zorder=2)
    ax.set_xlim(-1, n)
    ax.set_ylim(lows.min() * 0.97, highs.max() * 1.03)


def _draw_volume(ax, df_tf):
    """Volume bars with 20-bar MA overlay."""
    vol    = df_tf['Volume'].values
    closes = df_tf['Close'].values
    opens  = df_tf['Open'].values
    n      = len(df_tf)
    for i in range(n):
        col = '#26a641' if closes[i] >= opens[i] else '#f85149'
        ax.bar(i, vol[i], color=col, alpha=0.6, width=0.85, zorder=2)
    vol_ma = pd.Series(vol).rolling(20).mean().values
    ax.plot(np.arange(n), vol_ma, color='#ffd700', lw=1.0, label='Vol MA20')
    ax.set_xlim(-1, n)
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    ax.grid(color='#21262d', ls='--', lw=.3)
    for sp in ax.spines.values(): sp.set_color('#30363d')


def _draw_rsi(ax, df_tf):
    """RSI(14) sub-panel with overbought/oversold zones."""
    try:
        import ta as ta_lib
        rsi = ta_lib.momentum.RSIIndicator(df_tf['Close'], window=14).rsi().values
    except Exception:
        close = df_tf['Close'].values
        delta = np.diff(close, prepend=close[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_g = pd.Series(gain).ewm(com=13, adjust=False).mean().values
        avg_l = pd.Series(loss).ewm(com=13, adjust=False).mean().values
        rs    = np.where(avg_l == 0, 100, avg_g / avg_l)
        rsi   = 100 - (100 / (1 + rs))
    n = len(rsi)
    ax.plot(np.arange(n), rsi, color='#a78bfa', lw=1.0, zorder=3)
    ax.axhline(70, color='#f85149', lw=0.7, ls='--', alpha=0.7)
    ax.axhline(30, color='#26a641', lw=0.7, ls='--', alpha=0.7)
    ax.fill_between(np.arange(n), rsi, 70, where=(rsi >= 70), alpha=0.15, color='#f85149')
    ax.fill_between(np.arange(n), rsi, 30, where=(rsi <= 30), alpha=0.15, color='#26a641')
    ax.set_ylim(0, 100)
    ax.set_xlim(-1, n)
    ax.set_yticks([30, 50, 70])
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=6)
    ax.set_ylabel('RSI', color='#8b949e', fontsize=6)
    ax.grid(color='#21262d', ls='--', lw=.3)
    for sp in ax.spines.values(): sp.set_color('#30363d')


def _draw_mas(ax, df_tf):
    """MA20 (gold), MA50 (orange), MA150 (purple), MA200 (red)."""
    close = df_tf['Close'].values
    n     = len(close)
    xs    = np.arange(n)
    mas   = [
        (20,  '#ffd700', 'MA20',  0.85),
        (50,  '#ff9500', 'MA50',  0.85),
        (150, '#c084fc', 'MA150', 0.80),
        (200, '#ff4757', 'MA200', 0.90),
    ]
    for period, color, label, alpha in mas:
        vals = pd.Series(close).rolling(period).mean().values
        valid = ~np.isnan(vals)
        if valid.any():
            ax.plot(xs[valid], vals[valid], color=color, lw=0.9,
                    label=label, alpha=alpha, zorder=4)


def _draw_pattern_annotations(ax, df_tf, pr, n):
    """Draw TradingView-style pattern markings on price panel."""
    pat = pr.get('pattern', 'None')
    close = df_tf['Close'].values
    highs = df_tf['High'].values
    lows  = df_tf['Low'].values

    def _x_for_idx(date_idx):
        """Convert DatetimeIndex label to integer position."""
        try:
            return df_tf.index.get_loc(date_idx)
        except Exception:
            return None

    # ── Cup and Handle ────────────────────────────────────
    if pat == 'Cup and Handle':
        cs = pr.get('cup_start_idx', 0)
        cb = pr.get('cup_bottom_idx', n // 2)
        ce = min(pr.get('cup_end_idx', n - 1), n - 1)
        cs = max(0, min(cs, n - 1))
        cb = max(0, min(cb, n - 1))
        if ce > cs:
            xs_cup = np.arange(cs, ce + 1)
            ax.fill_between(xs_cup, close[cs:ce+1],
                            max(close[cs:ce+1]), alpha=0.07, color='#58a6ff')
        bl = pr.get('breakout_level')
        if bl:
            ax.axhline(bl, color='#26a641', lw=1.5, ls='--', alpha=0.9)
            ax.text(n * 0.98, bl, '  BUY ZONE', color='#26a641',
                    fontsize=7, va='bottom', ha='right', fontweight='bold')
        for lbl, val, color in [
            ('Cup High', pr.get('cup_high'), '#ffd700'),
            ('Cup Low',  pr.get('cup_low'),  '#58a6ff'),
            ('Handle',   pr.get('handle_low'), '#ff9500'),
        ]:
            if val:
                ax.axhline(val, color=color, lw=0.8, ls=':', alpha=0.6)
                ax.text(n * 0.01, val, f' {lbl} ${val}', color=color, fontsize=6, va='bottom')

    # ── Head and Shoulders ────────────────────────────────
    elif pat in ('Head and Shoulders', 'Inverse Head and Shoulders'):
        nk = pr.get('neckline')
        if nk:
            ax.axhline(nk, color='#ff9500', lw=1.8, ls='-', alpha=0.9)
            ax.text(n * 0.5, nk, f' Neckline ${nk:.2f}',
                    color='#ff9500', fontsize=7, fontweight='bold', va='bottom')
        pt = pr.get('price_target')
        if pt:
            ax.axhline(pt, color='#f85149', lw=1.0, ls='-.', alpha=0.8)
            ax.text(n * 0.98, pt, f'  Target ${pt}', color='#f85149',
                    fontsize=6, va='top', ha='right')
        for key, label, color in [
            ('ls_idx', 'LS', '#a78bfa'),
            ('head_idx', 'H', '#f85149'),
            ('rs_idx', 'RS', '#a78bfa'),
        ]:
            xi = _x_for_idx(pr.get(key))
            if xi is not None and 0 <= xi < n:
                y = highs[xi] if 'Head and Shoulders' == pat else lows[xi]
                ax.plot(xi, y, 'o', color=color, ms=6, zorder=5)
                ax.text(xi, y, f'  {label}', color=color, fontsize=6,
                        va='top' if 'Head and Shoulders' == pat else 'bottom')
                if nk:
                    ax.plot([xi, xi], [min(y, nk), max(y, nk)],
                            color=color, lw=0.6, ls='--', alpha=0.5)

    # ── Double / Triple Bottom ────────────────────────────
    elif pat in ('Double Bottom', 'Triple Bottom'):
        res = pr.get('resistance_levels', [])
        if res:
            ax.axhline(res[0], color='#26a641', lw=1.4, ls='--', alpha=0.9)
            ax.axhline(res[0], color='#26a641', lw=0, alpha=0.0)
            ax.fill_between([0, n], res[0], res[0] * 1.05,
                            color='#26a641', alpha=0.08)
            ax.text(n * 0.98, res[0], '  Breakout zone', color='#26a641',
                    fontsize=6, va='bottom', ha='right')
        pt = pr.get('price_target')
        if pt:
            ax.axhline(pt, color='#58a6ff', lw=1.0, ls='-.', alpha=0.8)
            ax.text(n * 0.01, pt, f' Target ${pt}', color='#58a6ff', fontsize=6)
        for key, lbl in [('bot1_idx','B1'),('bot2_idx','B2'),('b1_idx','B1'),('b2_idx','B2'),('b3_idx','B3')]:
            xi = _x_for_idx(pr.get(key))
            if xi is not None and 0 <= xi < n:
                ax.plot(xi, lows[xi], 'v', color='#26a641', ms=7, zorder=5)
                ax.text(xi, lows[xi], f'  {lbl}', color='#26a641', fontsize=6, va='top')

    # ── Double / Triple Top ───────────────────────────────
    elif pat in ('Double Top', 'Triple Top'):
        sup = pr.get('support_levels', [])
        if sup:
            ax.axhline(sup[0], color='#f85149', lw=1.4, ls='--', alpha=0.9)
        pt = pr.get('price_target')
        if pt:
            ax.axhline(pt, color='#f85149', lw=1.0, ls='-.', alpha=0.8)
            ax.text(n * 0.01, pt, f' Target ${pt}', color='#f85149', fontsize=6)
        for key, lbl in [('top1_idx','T1'),('top2_idx','T2'),('t1_idx','T1'),('t2_idx','T2'),('t3_idx','T3')]:
            xi = _x_for_idx(pr.get(key))
            if xi is not None and 0 <= xi < n:
                ax.plot(xi, highs[xi], '^', color='#f85149', ms=7, zorder=5)
                ax.text(xi, highs[xi], f'  {lbl}', color='#f85149', fontsize=6, va='bottom')

    # ── Triangles ─────────────────────────────────────────
    elif 'Triangle' in pat:
        hi_t = pr.get('hi_touches', [])
        lo_t = pr.get('lo_touches', [])
        for dt in hi_t:
            xi = _x_for_idx(dt)
            if xi is not None and 0 <= xi < n:
                ax.plot(xi, highs[xi], 'o', color='#ff9500', ms=5, zorder=5)
        for dt in lo_t:
            xi = _x_for_idx(dt)
            if xi is not None and 0 <= xi < n:
                ax.plot(xi, lows[xi], 'o', color='#ff9500', ms=5, zorder=5)
        res = pr.get('resistance_levels', [])
        sup = pr.get('support_levels', [])
        if res:
            ax.axhline(res[0], color='#ff9500', lw=1.0, ls='--', alpha=0.7)
        if sup:
            ax.axhline(sup[0], color='#ff9500', lw=1.0, ls='--', alpha=0.7)
        if res and sup:
            ax.fill_between([0, n], sup[0], res[0], color='#ff9500', alpha=0.05)

    # ── Generic support / resistance ──────────────────────
    for lvl in pr.get('support_levels', []):
        ax.axhline(lvl, color='#2ed573', lw=1.0, ls='--', alpha=0.7)
        ax.text(n * 0.01, lvl, f' S ${lvl}', color='#2ed573', fontsize=6, va='bottom')
    for lvl in pr.get('resistance_levels', []):
        ax.axhline(lvl, color='#ff4757', lw=1.0, ls='--', alpha=0.7)
        ax.text(n * 0.01, lvl, f' R ${lvl}', color='#ff4757', fontsize=6, va='top')


def create_multi_timeframe_chart(symbol, daily, weekly, monthly, pattern_result, claude_result=None):
    """Professional 9-panel chart: price (candlestick) + volume + RSI for each timeframe."""
    try:
        from matplotlib.gridspec import GridSpec
        ROWS = 9
        heights = [3.5, 1, 1.2] * 3
        fig = plt.figure(figsize=(18, 24))
        fig.patch.set_facecolor('#0d1117')
        gs  = GridSpec(ROWS, 1, figure=fig, hspace=0.08,
                       height_ratios=heights)
        fig.suptitle(
            f'{symbol} — Multi-Timeframe Analysis  |  {datetime.now().strftime("%Y-%m-%d")}',
            color='white', fontsize=15, fontweight='bold', y=0.995
        )
        timeframes = [
            (daily.tail(180),  'Daily (6 months)',   0),
            (weekly.tail(104), 'Weekly (2 years)',    3),
            (monthly.tail(60), 'Monthly (5 years)',   6),
        ]
        pr = pattern_result or {'pattern': 'None'}
        pat = pr.get('pattern', 'None')

        for df_tf, label, row_start in timeframes:
            df_tf = df_tf.copy()
            n = len(df_tf)

            # ── Price panel ────────────────────────────────
            ax_p = fig.add_subplot(gs[row_start])
            ax_p.set_facecolor('#0d1117')
            _draw_candles(ax_p, df_tf)
            _draw_mas(ax_p, df_tf)
            _draw_pattern_annotations(ax_p, df_tf, pr, n)

            # Trade levels from Claude result
            if claude_result:
                ent = claude_result.get('entry_zone')
                stp = claude_result.get('stop_loss')
                t1  = claude_result.get('target_1')
                t2  = claude_result.get('target_2')
                def _try_float(v):
                    try: return float(str(v).replace('$','').split()[0])
                    except: return None
                for val, color, ls, lbl in [
                    (_try_float(ent), '#26a641', '-',  'Entry'),
                    (_try_float(stp), '#f85149', '-',  'Stop'),
                    (_try_float(t1),  '#58a6ff', '--', 'T1'),
                    (_try_float(t2),  '#58a6ff', ':',  'T2'),
                ]:
                    if val:
                        ax_p.axhline(val, color=color, lw=1.1, ls=ls, alpha=0.85)
                        ax_p.text(n * 0.99, val, f' {lbl} ${val}',
                                  color=color, fontsize=6, va='bottom', ha='right')

            # Pattern badge top-left
            if pat and pat != 'None':
                tf_detected = ''
                ax_p.text(0.01, 0.98, f'▶ {pat}{tf_detected}',
                          transform=ax_p.transAxes, color='white', fontsize=8,
                          fontweight='bold', va='top',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', alpha=0.9))

            ax_p.set_title(label, color='#8b949e', fontsize=10, pad=4, loc='left')
            ax_p.legend(loc='upper right', facecolor='#161b22',
                        labelcolor='white', fontsize=6, framealpha=0.8, ncol=4)
            ax_p.tick_params(colors='#8b949e', labelsize=7)
            ax_p.grid(color='#21262d', ls='--', lw=0.4)
            for sp in ax_p.spines.values(): sp.set_color('#30363d')
            ax_p.set_xticklabels([])

            # ── Volume panel ───────────────────────────────
            ax_v = fig.add_subplot(gs[row_start + 1])
            _draw_volume(ax_v, df_tf)
            ax_v.set_xticklabels([])

            # ── RSI panel ──────────────────────────────────
            ax_r = fig.add_subplot(gs[row_start + 2])
            _draw_rsi(ax_r, df_tf)

        plt.subplots_adjust(left=0.05, right=0.97, top=0.985, bottom=0.02)
        path = (f'/tmp/{symbol}_chart.png' if os.name != 'nt'
                else f'C:\\stock_scanner\\{symbol}_chart.png')
        plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        return path
    except Exception as e:
        print(f'  Chart error {symbol}: {e}')
        import traceback; traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
def calc_indicators(hist):
    try:
        import ta as ta_lib
        close = hist['Close']
        high  = hist['High']
        low   = hist['Low']
        vol   = hist['Volume']
        rsi    = ta_lib.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        macd_i = ta_lib.trend.MACD(close)
        macd   = macd_i.macd().iloc[-1]
        macd_s = macd_i.macd_signal().iloc[-1]
        macd_h = macd_i.macd_diff().iloc[-1]
        bb     = ta_lib.volatility.BollingerBands(close, window=20)
        bb_up  = bb.bollinger_hband().iloc[-1]
        bb_lo  = bb.bollinger_lband().iloc[-1]
        bb_mid = bb.bollinger_mavg().iloc[-1]
        adx    = ta_lib.trend.ADXIndicator(high, low, close).adx().iloc[-1]
        stoch  = ta_lib.momentum.StochasticOscillator(high, low, close).stoch().iloc[-1]
        vol_ma = vol.rolling(20).mean().iloc[-1]
        vol_ratio = vol.iloc[-1] / vol_ma if vol_ma > 0 else 1.0
        return (
            f"Technical Indicators:\n"
            f"- RSI(14): {rsi:.1f} {'[Overbought]' if rsi>70 else '[Oversold]' if rsi<30 else '[Neutral]'}\n"
            f"- MACD: {macd:.3f} | Signal: {macd_s:.3f} | Hist: {macd_h:.3f} {'[Bullish]' if macd_h>0 else '[Bearish]'}\n"
            f"- Bollinger: U={bb_up:.2f} M={bb_mid:.2f} L={bb_lo:.2f}\n"
            f"- ADX: {adx:.1f} {'[Strong]' if adx>25 else '[Weak]'}\n"
            f"- Stoch: {stoch:.1f} {'[Overbought]' if stoch>80 else '[Oversold]' if stoch<20 else '[Neutral]'}\n"
            f"- Vol ratio: {vol_ratio:.2f}x {'[High]' if vol_ratio>1.5 else ''}"
        )
    except Exception as e:
        return f"(Indicators unavailable: {e})"


# ══════════════════════════════════════════════════════════════════════════════
#  CLAUDE VISION
# ══════════════════════════════════════════════════════════════════════════════
def confirm_with_claude(symbol, chart_path, algo_result, stock, hist):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    indicators_text = calc_indicators(hist)
    with open(chart_path, 'rb') as f:
        img_b64 = base64.standard_b64encode(f.read()).decode()
    algo_pattern = algo_result.get('pattern', 'None')
    prompt = f"""You are a senior technical analyst reviewing a multi-timeframe chart for {symbol}.

Algorithmic scanner detected: **{algo_pattern}**

Stock data:
- Sector: {stock['sector']} | Beta: {stock['beta']} | P/B: {stock['pb_ratio']}
- P/E: {stock['pe_ratio']} | EPS TTM: {stock['eps_ttm']}

{indicators_text}

The chart shows THREE timeframes (top=Daily 6mo, middle=Weekly 2yr, bottom=Monthly 5yr).

Respond ONLY with JSON (no markdown):
{{
  "algo_confirmed": true,
  "pattern": "Cup and Handle|Double Bottom|Double Top|Triple Bottom|Triple Top|Head and Shoulders|Inverse Head and Shoulders|Ascending Triangle|Descending Triangle|Symmetrical Triangle|None",
  "pattern_confidence": "high|medium|low",
  "pattern_detail": "Detailed description with timeframe and key price levels",
  "primary_trend": "uptrend|downtrend|sideways",
  "current_trend": "uptrend|downtrend|sideways",
  "primary_trend_detail": "Evidence from monthly/weekly chart",
  "current_trend_detail": "Evidence from daily/weekly chart",
  "support_levels": [10.00, 8.50],
  "resistance_levels": [14.00, 16.50],
  "support_detail": "Why these levels matter",
  "resistance_detail": "Why these levels matter",
  "gaps_detail": "Any significant gaps visible",
  "timeframe_alignment": "Do all 3 timeframes agree?",
  "sector_direction": "bullish|bearish|neutral",
  "sector_strength": "strong|moderate|weak",
  "entry_zone": "Price range for entry",
  "stop_loss": "Stop loss level",
  "target_1": "First target",
  "target_2": "Second target",
  "watchlist": true,
  "reasoning": "4-5 sentence full analysis"
}}

Only set watchlist=true if pattern is CLEAR with high or medium confidence AND at least 2 timeframes align."""

    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1400,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    try:
        raw = msg.content[0].text.strip()
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"  JSON parse error: {e}")
        return {'watchlist': False, 'pattern': algo_pattern, 'reasoning': 'Parse error'}


# ══════════════════════════════════════════════════════════════════════════════
#  FINVIZ
# ══════════════════════════════════════════════════════════════════════════════
def get_stocks_from_finviz():
    from finvizfinance.screener.overview import Overview
    print("\nScanning US market via Finviz...")
    try:
        foverview = Overview()
        foverview.set_filter(filters_dict={
            'Market Cap.': 'Small ($300mln to $2bln)',
            'Beta': '1 to 2', 'P/B': 'Over 1',
            'Average Volume': 'Over 200K', 'Country': 'USA',
        })
        df = foverview.screener_view()
        if df is None or len(df) == 0:
            return [], []
        all_tickers = [str(r.get('Ticker','')) for _,r in df.iterrows() if r.get('Ticker','')]
        print(f"Finviz returned {len(all_tickers)} candidates")
        stocks = []
        for _, row in df.iterrows():
            try:
                sym  = row.get('Ticker','')
                if not sym: continue
                info = yf.Ticker(sym).info
                mc   = info.get('marketCap', 0)
                av   = info.get('averageVolume', 0)
                px   = info.get('currentPrice', info.get('regularMarketPrice', 0))
                beta = info.get('beta', 0)
                pb   = info.get('priceToBook', 0)
                if not all([px, mc, beta, pb]): continue
                vol_usd = av * px
                if not (200e6 <= mc <= 400e6): continue
                if vol_usd < 200_000: continue
                if not (1.5 <= beta <= 2.0): continue
                if pb < 1.0: continue
                stocks.append({
                    'symbol': sym, 'market_cap': mc, 'volume_usd': vol_usd,
                    'beta': beta, 'pb_ratio': pb, 'price': px,
                    'sector': info.get('sector','N/A'), 'pe_ratio': info.get('trailingPE','N/A'),
                    'eps_ttm': info.get('trailingEps','N/A'), 'name': info.get('longName', sym),
                })
                print(f"  ✓ {sym}")
            except Exception as e:
                print(f"  skip {row.get('Ticker','?')}: {e}")
        return all_tickers, stocks
    except Exception as e:
        print(f"Finviz error: {e}")
        return [], []


# ══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM HELPERS
# ══════════════════════════════════════════════════════════════════════════════
async def tg_send(text):
    bot = Bot(token=TELEGRAM_TOKEN)
    for i in range(0, len(text), 4096):
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text[i:i+4096], parse_mode='Markdown')

async def tg_photo(path, caption=""):
    bot = Bot(token=TELEGRAM_TOKEN)
    with open(path, 'rb') as f:
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=f, caption=caption[:1024])

def fmt_report(stock, res):
    mc_m  = stock['market_cap'] / 1e6 if stock['market_cap'] else 0
    vol_k = stock['volume_usd'] / 1e3 if stock['volume_usd'] else 0
    confirmed = "✅ Algo confirmed" if res.get('algo_confirmed') else "⚠️ Algo overridden"
    return (
        f"📊 *{stock['symbol']} — {stock['name']}*\n\n"
        f"💰 *Fundamentals:*\n"
        f"• Market Cap: ${mc_m:.0f}M  |  Volume: ${vol_k:.0f}K/day\n"
        f"• Beta: {stock['beta']:.2f}  |  P/B: {stock['pb_ratio']:.2f}\n"
        f"• P/E: {stock['pe_ratio']}  |  EPS TTM: {stock['eps_ttm']}\n"
        f"• Sector: {stock['sector']}\n\n"
        f"📈 *Trend (Multi-Timeframe):*\n"
        f"• Primary: *{res.get('primary_trend','?')}*\n_{res.get('primary_trend_detail','')}_\n"
        f"• Current: *{res.get('current_trend','?')}*\n_{res.get('current_trend_detail','')}_\n"
        f"• Alignment: _{res.get('timeframe_alignment','')}_\n\n"
        f"🔷 *Pattern: {res.get('pattern','?')}* ({res.get('pattern_confidence','?')}) {confirmed}\n"
        f"_{res.get('pattern_detail','')}_\n\n"
        f"🟢 *Support:* {res.get('support_levels',[])} — _{res.get('support_detail','')}_\n\n"
        f"🔴 *Resistance:* {res.get('resistance_levels',[])} — _{res.get('resistance_detail','')}_\n\n"
        f"⬜ *Gaps:* _{res.get('gaps_detail','None')}_\n\n"
        f"🎯 *Trade Setup:*\n"
        f"• Entry: {res.get('entry_zone','?')}\n"
        f"• Stop: {res.get('stop_loss','?')}\n"
        f"• T1: {res.get('target_1','?')}  |  T2: {res.get('target_2','?')}\n\n"
        f"🌍 *Sector:* {res.get('sector_direction','?')} / {res.get('sector_strength','?')}\n\n"
        f"💬 *Analysis:*\n_{res.get('reasoning','')}_\n"
        f"-------------------"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
async def analyze_single_stock(symbol, update):
    sym = symbol.upper().strip()
    try:
        await update.message.reply_text(f"⏳ מושך נתונים עבור *{sym}*...", parse_mode='Markdown')
        ticker = yf.Ticker(sym)
        info   = ticker.info
        px     = info.get('currentPrice', info.get('regularMarketPrice', 0))
        if not px:
            await update.message.reply_text(f"❌ לא נמצאו נתונים עבור {sym}. בדוק שהטיקר נכון.")
            return
        daily   = ticker.history(period="2y",  interval="1d")
        weekly  = ticker.history(period="5y",  interval="1wk")
        monthly = ticker.history(period="10y", interval="1mo")
        if daily is None or len(daily) < 30:
            await update.message.reply_text(f"❌ אין מספיק נתונים היסטוריים עבור {sym}.")
            return
        stock = {
            'symbol': sym, 'market_cap': info.get('marketCap', 0),
            'volume_usd': info.get('averageVolume', 0) * px,
            'beta': info.get('beta', 0) or 0, 'pb_ratio': info.get('priceToBook', 0) or 0,
            'price': px, 'sector': info.get('sector', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'), 'eps_ttm': info.get('trailingEps', 'N/A'),
            'name': info.get('longName', sym),
        }
        await update.message.reply_text(f"🔎 מריץ ניתוח אלגוריתמי על *{sym}*...", parse_mode='Markdown')
        algo_daily   = run_pattern_scan(daily,            min_cup_days=126)
        algo_weekly  = run_pattern_scan(weekly,           min_cup_days=26)
        algo_monthly = run_pattern_scan(monthly.tail(60), min_cup_days=6)
        priority = ['Cup and Handle','Inverse Head and Shoulders','Double Bottom','Triple Bottom',
                    'Head and Shoulders','Double Top','Triple Top','Ascending Triangle',
                    'Descending Triangle','Symmetrical Triangle','None']
        best = {'pattern': 'None'}
        for res in [algo_daily, algo_weekly, algo_monthly]:
            if priority.index(res.get('pattern','None')) < priority.index(best.get('pattern','None')):
                best = res
        algo_tf = "יומי" if best == algo_daily else "שבועי" if best == algo_weekly else "חודשי"
        await update.message.reply_text(
            f"📐 תבנית אלגוריתמית: *{best['pattern']}* (גרף {algo_tf})", parse_mode='Markdown')
        await update.message.reply_text(f"🎨 בונה גרף 3 טיימפריימים...")
        chart_path = create_multi_timeframe_chart(sym, daily, weekly, monthly, best)
        if not chart_path:
            await update.message.reply_text("❌ שגיאה ביצירת הגרף.")
            return
        await update.message.reply_text(f"🤖 שולח ל-Claude לניתוח ויזואלי...")
        result = confirm_with_claude(sym, chart_path, best, stock, daily)
        if result is None:
            await update.message.reply_text("❌ שגיאה בניתוח Claude.")
            return
        await tg_photo(chart_path,
                       caption=f"{sym} — {result.get('pattern','')} ({result.get('pattern_confidence','')})")
        await tg_send(fmt_report(stock, result))
        if os.path.exists(chart_path):
            os.remove(chart_path)
    except Exception as e:
        await update.message.reply_text(f"❌ שגיאה בניתוח {sym}: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#  WEEKLY SCAN
# ══════════════════════════════════════════════════════════════════════════════
async def run_scan():
    await tg_send("🔍 *Weekly scan started…*")
    all_tickers, stocks = get_stocks_from_finviz()
    if not all_tickers:
        await tg_send("⚠️ No stocks found by Finviz.")
        return
    chunks = [all_tickers[i:i+50] for i in range(0, len(all_tickers), 50)]
    first = True
    for chunk in chunks:
        header = f"📋 *Finviz found {len(all_tickers)} candidates:*\n\n" if first else ""
        first = False
        await tg_send(header + "  ".join([f"`{t}`" for t in chunk]))
    if not stocks:
        await tg_send("⚠️ No stocks passed precise fundamental filter.")
        return
    await tg_send(
        f"✅ *{len(stocks)} stocks passed precise filter:*\n\n"
        + "  ".join([f"`{s['symbol']}`" for s in stocks])
        + "\n\n_Running 3-timeframe technical analysis…_"
    )
    watchlist = []
    for stock in stocks:
        sym = stock['symbol']
        try:
            print(f"\nAnalyzing {sym}...")
            ticker  = yf.Ticker(sym)
            daily   = ticker.history(period="2y",  interval="1d")
            weekly  = ticker.history(period="5y",  interval="1wk")
            monthly = ticker.history(period="10y", interval="1mo")
            if daily is None or len(daily) < 60:
                continue
            algo_daily   = run_pattern_scan(daily,            min_cup_days=126)
            algo_weekly  = run_pattern_scan(weekly,           min_cup_days=26)
            algo_monthly = run_pattern_scan(monthly.tail(60), min_cup_days=6)
            priority = ['Cup and Handle','Inverse Head and Shoulders','Double Bottom','Triple Bottom',
                        'Head and Shoulders','Double Top','Triple Top','Ascending Triangle',
                        'Descending Triangle','Symmetrical Triangle','None']
            best = {'pattern': 'None'}
            for res in [algo_daily, algo_weekly, algo_monthly]:
                if priority.index(res.get('pattern','None')) < priority.index(best.get('pattern','None')):
                    best = res
            print(f"  Algo: {best['pattern']}")
            chart_path = create_multi_timeframe_chart(sym, daily, weekly, monthly, best)
            if not chart_path:
                continue
            result = confirm_with_claude(sym, chart_path, best, stock, daily)
            rec = result.get('watchlist', False)
            print(f"  Claude: {'YES' if rec else 'NO'} | {result.get('pattern')}")
            if rec:
                watchlist.append(sym)
                await tg_send(fmt_report(stock, result))
                await tg_photo(chart_path, caption=f"{sym} — {result.get('pattern','')} ({result.get('pattern_confidence','')})")
            if os.path.exists(chart_path):
                os.remove(chart_path)
        except Exception as e:
            print(f"  Error {sym}: {e}")
    await tg_send(
        f"\n✅ *Scan complete!*\nWatchlist: *{len(watchlist)} stocks*\n"
        + (', '.join(f"`{s}`" for s in watchlist) if watchlist else '_No clear setups_')
    )
    print("✅ Done.")


# ══════════════════════════════════════════════════════════════════════════════
#  BOT HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != TELEGRAM_CHAT_ID:
        return
    text = update.message.text.strip()
    text_lower = text.lower()

    # מילות סריקה שבועית — רק ביטויים מפורשים
    weekly_triggers = ['סריקה שבועית', 'weekly scan', 'scan all', 'full scan', '/scan']

    # מילים שאינן טיקרים
    skip_words = {
        'על', 'את', 'של', 'עם', 'לי', 'RUN', 'ON', 'THE', 'FOR', 'ME',
        'תריץ', 'תנתח', 'בדוק', 'נתח', 'ANALYZE', 'CHECK', 'TEST',
        'הרץ', 'תהרץ', 'בצע', 'תבצע', 'תסרוק', 'סרוק'
    }

    # זיהוי טיקר בהודעה — רק אותיות לטיניות ASCII (לא עברית)
    words = text.upper().split()
    ticker = None
    for word in words:
        clean = word.strip('.,!?()[]')
        if 1 < len(clean) <= 5 and clean.isascii() and clean.isalpha() and clean not in skip_words:
            ticker = clean
            break

    # אם יש טיקר — נתח אותו תמיד
    if ticker:
        await update.message.reply_text(f"🔍 מנתח את *{ticker}*...", parse_mode='Markdown')
        await analyze_single_stock(ticker, update)

    # סריקה שבועית — רק ביטוי מפורש
    elif any(t in text_lower for t in weekly_triggers):
        await update.message.reply_text("🚀 מתחיל סריקה שבועית...")
        await run_scan()

    # עזרה
    else:
        await update.message.reply_text(
            "👋 *שלום! הנה מה שאני יכול לעשות:*\n\n"
            "📊 *ניתוח מנייה ספציפית:*\n"
            "• `SOFI`\n"
            "• `תנתח TSLA`\n"
            "• `תריץ בדיקה על NVDA`\n"
            "• `/analyze MSFT`\n\n"
            "🔍 *סריקה שבועית מלאה:*\n"
            "• `סריקה שבועית`\n"
            "• `/scan`\n\n"
            "⏰ סריקה אוטומטית: כל שבת 21:00",
            parse_mode='Markdown'
        )

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != TELEGRAM_CHAT_ID:
        return
    await update.message.reply_text("🚀 Starting scan…")
    await run_scan()

async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != TELEGRAM_CHAT_ID:
        return
    args = context.args
    if not args:
        await update.message.reply_text("📊 שלח טיקר: `/analyze AAPL`", parse_mode='Markdown')
        return
    await update.message.reply_text(f"🔍 מנתח את *{args[0].upper()}*...", parse_mode='Markdown')
    await analyze_single_stock(args[0], update)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Stock Scanner Bot*\n\n"
        "📊 לניתוח מנייה: `SOFI` או `תנתח TSLA`\n"
        "🔍 לסריקה שבועית: `סריקה שבועית` או `/scan`\n"
        "⏰ סריקה אוטומטית: כל שבת 21:00",
        parse_mode='Markdown'
    )

async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    print(f"⏰ Scheduled scan at {datetime.now()}")
    await run_scan()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("🤖 Stock Scanner Bot starting…")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("scan",    cmd_scan))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.job_queue.run_daily(
        scheduled_scan,
        time=time(hour=21, minute=0, tzinfo=ISRAEL_TZ),
        days=(5,), name="weekly_scan"
    )
    print("✅ Bot running.")
    app.run_polling()

if __name__ == "__main__":
    main()