import os
import yfinance as yf
import anthropic

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# נבדוק מנייה אחת שעברה את הסינון
symbol = "ABAT"

ticker = yf.Ticker(symbol)
info = ticker.info
hist = ticker.history(period="2y")

price_summary = f"""
- Current Price: ${hist['Close'].iloc[-1]:.2f}
- 52-week High: ${hist['High'].max():.2f}
- 52-week Low: ${hist['Low'].min():.2f}
- 200-day MA: ${hist['Close'].tail(200).mean():.2f}
- 50-day MA: ${hist['Close'].tail(50).mean():.2f}
- 20-day MA: ${hist['Close'].tail(20).mean():.2f}
- Recent prices (last 10 days): {hist['Close'].tail(10).round(2).tolist()}
"""

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1500,
    messages=[{"role": "user", "content": f"""Analyze {symbol} technically and give me your full analysis including recommendation YES or NO.

{price_summary}
"""}]
)

print("="*50)
print(f"CLAUDE'S ANALYSIS OF {symbol}:")
print("="*50)
print(message.content[0].text)