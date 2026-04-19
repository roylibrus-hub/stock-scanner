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

def detect_head_and_shoulders(df):
    highs = get_pivots(df['High'], bars=6)
    lows  = get_pivots(df['Low'],  bars=6)
    if len(highs) < 3 or len(lows) < 2:
        return None
    avg_bar = avg_bar_range(df['High'], df['Low'])
    F = df['Close'].iloc[-1]
    pivot_vals = pd.concat([highs, lows]).sort_index()
    for i in range(len(pivot_vals) - 4):
        try:
            sub = pivot_vals.iloc[i:i+5]
            A, B, C, D, E = sub.values
            if C > max(A,E) and max(B,D) < min(A,E) and F < E and abs(B-D) < avg_bar:
                return {'pattern': 'Head and Shoulders', 'neckline': (B+D)/2,
                        'head': C, 'left_shoulder': A, 'right_shoulder': E,
                        'support_levels': [round((B+D)/2,2)], 'resistance_levels': [round(C,2)]}
        except:
            continue
    return None

def detect_inverse_hns(df):
    lows  = get_pivots(df['Low'],  bars=6)
    highs = get_pivots(df['High'], bars=6)
    if len(lows) < 3 or len(highs) < 2:
        return None
    avg_bar = avg_bar_range(df['High'], df['Low'])
    F = df['Close'].iloc[-1]
    pivot_vals = pd.concat([highs, lows]).sort_index()
    for i in range(len(pivot_vals) - 4):
        try:
            sub = pivot_vals.iloc[i:i+5]
            A, B, C, D, E = sub.values
            if C < min(A,E) and min(B,D) > max(A,E) and F > E and abs(B-D) < avg_bar:
                return {'pattern': 'Inverse Head and Shoulders', 'neckline': (B+D)/2,
                        'head': C, 'left_shoulder': A, 'right_shoulder': E,
                        'support_levels': [round(C,2)], 'resistance_levels': [round((B+D)/2,2)]}
        except:
            continue
    return None

def detect_triangle(df):
    highs = get_pivots(df['High'], bars=4)
    lows  = get_pivots(df['Low'],  bars=4)
    if len(highs) < 3 or len(lows) < 2:
        return None
    avg_bar = avg_bar_range(df['High'], df['Low'])
    F = df['Close'].iloc[-1]
    hv = highs.values
    lv = lows.values
    for i in range(len(hv) - 2):
        A, C, E = hv[i], hv[i+1], hv[i+2]
        for j in range(len(lv) - 1):
            B, D = lv[j], lv[j+1]
            if A > C > E and B < D and E > F > D:
                return {'pattern': 'Symmetrical Triangle',
                        'support_levels': [round(D,2)], 'resistance_levels': [round(E,2)]}
            if abs(A-C) <= avg_bar and abs(C-E) <= avg_bar and B < D < F < E:
                return {'pattern': 'Ascending Triangle',
                        'support_levels': [round(D,2)], 'resistance_levels': [round(A,2)]}
            if abs(B-D) <= avg_bar and A > C > E > F > D:
                return {'pattern': 'Descending Triangle',
                        'support_levels': [round(B,2)], 'resistance_levels': [round(C,2)]}
    return None

def detect_double_top(df):
    highs = get_pivots(df['High'], bars=5)
    lows  = get_pivots(df['Low'],  bars=5)
    if len(highs) < 2 or len(lows) < 1:
        return None
    avg_bar = avg_bar_range(df['High'], df['Low'])
    D = df['Close'].iloc[-1]
    hv = highs.values
    hi = highs.index
    lv = lows.values
    for i in range(len(hv) - 1):
        A, C = hv[i], hv[i+1]
        if abs(A-C) > avg_bar:
            continue
        vol_A = df.loc[hi[i],   'Volume'] if hi[i]   in df.index else 0
        vol_C = df.loc[hi[i+1], 'Volume'] if hi[i+1] in df.index else 0
        if vol_C >= vol_A:
            continue
        B_candidates = [v for v in lv if v < min(A,C)]
        if not B_candidates:
            continue
        B = max(B_candidates)
        if B < D < C:
            return {'pattern': 'Double Top',
                    'support_levels': [round(B,2)], 'resistance_levels': [round(A,2)]}
    return None

def detect_double_bottom(df):
    lows  = get_pivots(df['Low'],  bars=5)
    highs = get_pivots(df['High'], bars=5)
    if len(lows) < 2 or len(highs) < 1:
        return None
    avg_bar = avg_bar_range(df['High'], df['Low'])
    D = df['Close'].iloc[-1]
    lv = lows.values
    hv = highs.values
    for i in range(len(lv) - 1):
        A, C = lv[i], lv[i+1]
        if abs(A-C) > avg_bar:
            continue
        B_candidates = [v for v in hv if v > max(A,C)]
        if not B_candidates:
            continue
        B = min(B_candidates)
        if C < D < B:
            return {'pattern': 'Double Bottom',
                    'support_levels': [round(A,2)], 'resistance_levels': [round(B,2)]}
    return None

def detect_triple_top(df):
    highs = get_pivots(df['High'], bars=5)
    if len(highs) < 3:
        return None
    avg_bar = avg_bar_range(df['High'], df['Low'])
    hv = highs.values
    for i in range(len(hv) - 2):
        A, B, C = hv[i], hv[i+1], hv[i+2]
        if abs(A-B) <= avg_bar and abs(B-C) <= avg_bar:
            return {'pattern': 'Triple Top', 'resistance_levels': [round(A,2)]}
    return None

def detect_triple_bottom(df):
    lows = get_pivots(df['Low'], bars=5)
    if len(lows) < 3:
        return None
    avg_bar = avg_bar_range(df['High'], df['Low'])
    lv = lows.values
    for i in range(len(lv) - 2):
        A, B, C = lv[i], lv[i+1], lv[i+2]
        if abs(A-B) <= avg_bar and abs(B-C) <= avg_bar:
            return {'pattern': 'Triple Bottom', 'support_levels': [round(A,2)]}
    return None

def detect_cup_and_handle(df, min_days=126):
    if len(df) < min_days:
        return None
    close = df['Close']
    vol   = df['Volume']
    n     = len(df)
    for start in range(0, n - min_days, 5):
        seg = close.iloc[start: start + min_days]
        if len(seg) < 50:
            continue
        K = seg.iloc[0]
        A_idx = seg.idxmax()
        A = seg[A_idx]
        A_pos = seg.index.get_loc(A_idx)
        if A_pos < 20:
            continue
        after_A = seg.iloc[A_pos:]
        if len(after_A) < 10:
            continue
        B_idx = after_A.idxmin()
        B = after_A[B_idx]
        B_pos = after_A.index.get_loc(B_idx)
        if not (0.65 * A <= B <= 0.85 * A):
            continue
        if not (20 <= A_pos <= 60):
            continue
        after_B = after_A.iloc[B_pos:]
        if len(after_B) < 3:
            continue
        C_idx = after_B.idxmax()
        C = after_B[C_idx]
        C_pos = after_B.index.get_loc(C_idx)
        if not (0.60 * A <= C <= A):
            continue
        if not (3 <= C_pos <= 30):
            continue
        after_C = after_B.iloc[C_pos:]
        if len(after_C) < 3:
            continue
        D = after_C.iloc[-1]
        if D > C:
            continue
        cup_vol = vol.iloc[start + A_pos: start + A_pos + B_pos].mean()
        if cup_vol >= vol.mean():
            continue
        return {
            'pattern': 'Cup and Handle',
            'cup_start_price': round(K,2), 'cup_high': round(A,2),
            'cup_low': round(B,2), 'cup_right': round(C,2), 'handle_low': round(D,2),
            'support_levels': [round(B,2), round(D,2)],
            'resistance_levels': [round(A,2)],
            'cup_start_idx': start, 'cup_end_idx': start + A_pos + B_pos + C_pos,
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
        except:
            continue
    return {'pattern': 'None'}


# ══════════════════════════════════════════════════════════════════════════════
#  CHART
# ══════════════════════════════════════════════════════════════════════════════
def create_multi_timeframe_chart(symbol, daily, weekly, monthly, pattern_result):
    try:
        fig, axes = plt.subplots(3, 1, figsize=(16, 18))
        fig.patch.set_facecolor('#0d1117')
        fig.suptitle(f'{symbol} — Multi-Timeframe Analysis  |  {datetime.now().strftime("%Y-%m-%d")}',
                     color='white', fontsize=14, fontweight='bold', y=0.98)
        timeframes = [
            (daily.tail(180),  'Daily (6 months)',  axes[0]),
            (weekly.tail(104), 'Weekly (2 years)',  axes[1]),
            (monthly.tail(60), 'Monthly (5 years)', axes[2]),
        ]
        for df_tf, label, ax in timeframes:
            ax.set_facecolor('#0d1117')
            n  = len(df_tf)
            xs = np.arange(n)
            close = df_tf['Close'].values
            ma20  = pd.Series(close).rolling(20).mean().values
            ma50  = pd.Series(close).rolling(50).mean().values
            ma200 = pd.Series(close).rolling(200).mean().values
            ax.plot(xs, close, color='#58a6ff', lw=1.6, label='Price', zorder=3)
            ax.plot(xs, ma20,  color='#ffd700', lw=0.9, label='MA20',  alpha=.85)
            ax.plot(xs, ma50,  color='#ff9500', lw=0.9, label='MA50',  alpha=.85)
            valid_ma200 = ma200[~np.isnan(ma200)]
            if len(valid_ma200):
                ax.plot(xs[-len(valid_ma200):], valid_ma200, color='#ff4757', lw=1.3, label='MA200', alpha=.9)
            for lvl in pattern_result.get('support_levels', []):
                ax.axhline(lvl, color='#2ed573', lw=1.2, ls='--', alpha=.8)
                ax.text(n*0.01, lvl, f' S ${lvl}', color='#2ed573', fontsize=7, va='bottom')
            for lvl in pattern_result.get('resistance_levels', []):
                ax.axhline(lvl, color='#ff4757', lw=1.2, ls='--', alpha=.8)
                ax.text(n*0.01, lvl, f' R ${lvl}', color='#ff4757', fontsize=7, va='top')
            nk = pattern_result.get('neckline')
            if nk:
                ax.axhline(nk, color='#ff9500', lw=1.8, ls='-', alpha=.9)
                ax.text(n*0.5, nk, f' Neckline ${nk:.2f}', color='#ff9500', fontsize=8, fontweight='bold')
            if 'cup_start_idx' in pattern_result and label.startswith('Daily'):
                cs = pattern_result['cup_start_idx']
                ce = min(pattern_result['cup_end_idx'], n-1)
                if ce > cs:
                    ax.fill_between(xs[cs:ce], close[cs:ce], max(close[cs:ce]), alpha=.08, color='#58a6ff')
            pat = pattern_result.get('pattern', 'None')
            if pat and pat != 'None':
                ax.text(0.5, 0.97, f'Pattern: {pat}', transform=ax.transAxes,
                        color='white', fontsize=9, fontweight='bold', ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='#21262d', alpha=.85))
            ax.set_title(label, color='#8b949e', fontsize=10, pad=4)
            ax.legend(loc='upper left', facecolor='#161b22', labelcolor='white', fontsize=7, framealpha=.8)
            ax.tick_params(colors='#8b949e', labelsize=7)
            ax.grid(color='#21262d', ls='--', lw=.4)
            for spine in ax.spines.values():
                spine.set_color('#30363d')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = f'/tmp/{symbol}_chart.png' if os.name != 'nt' else f'C:\\stock_scanner\\{symbol}_chart.png'
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        return path
    except Exception as e:
        print(f"  Chart error {symbol}: {e}")
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

    # זיהוי טיקר בהודעה
    words = text.upper().split()
    ticker = None
    for word in words:
        clean = word.strip('.,!?')
        if 1 < len(clean) <= 5 and clean.isalpha() and clean not in skip_words:
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