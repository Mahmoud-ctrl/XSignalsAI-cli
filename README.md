# XSignals AI - Professional Crypto Analysis Toolkit
⭐ Star this repo if you find it useful!

## Quick Start (2 Minutes)

### Windows Users
1. Double-click `install.bat` to auto-install everything
2. Edit `.env` file with your API keys (see below)
3. Double-click `run.bat` to launch
4. Copy and run example commands shown

### Mac/Linux Users
1. Open Terminal in this folder
2. Run `chmod +x install.sh && ./install.sh`
3. Edit `.env` file with your API keys (see below)
4. Run `./run.sh` to launch
5. Copy and run example commands shown

---

## Getting API Keys (Required)

### Binance API (Free & Required)
1. Go to https://www.binance.com/en/my/settings/api-management
2. Create a new API key
3. **IMPORTANT:** Only enable "Enable Reading" - NO trading permissions needed
4. Copy API Key and Secret Key
5. Paste into `.env` file

### OpenRouter API (Optional, for AI Analysis)
1. Go to https://openrouter.ai/keys
2. Sign up (free tier available)
3. Create API key
4. Paste into `.env` file

**Without OpenRouter:** Tool still works, but you won't get AI-powered insights

---

## Usage Examples

### Analyze a Single Coin
```bash
python main.py --symbol BTCUSDT
```

### Scan Multiple Pairs for Opportunities
```bash
python main.py --scan
```

### Multi-Timeframe Analysis
```bash
python main.py --symbol ETHUSDT --timeframes 15m 1h 4h 1d
```

### Analyze Specific Pairs Only
```bash
python main.py --scan --pairs BTCUSDT ETHUSDT SOLUSDT BNBUSDT
```

### Different Timeframe Combinations
```bash
# Scalping (short timeframes)
python main.py --symbol BTCUSDT --timeframes 1m 5m 15m

# Swing trading (longer timeframes)
python main.py --symbol ETHUSDT --timeframes 4h 1d

# Day trading sweet spot
python main.py --symbol SOLUSDT --timeframes 15m 1h 4h
```

---

## Understanding the Output

### Timeframe Signals
- **STRONG BUY** - Very bullish conditions (confidence >80%)
- **BUY** - Bullish conditions (confidence 60-80%)
- **WAIT** - Neutral or unclear (confidence <60%)
- **SELL** - Bearish conditions
- **STRONG SELL** - Very bearish conditions

### Confidence Levels
- **90%+** - Extremely high confidence (rare)
- **75-89%** - High confidence
- **60-74%** - Moderate confidence
- **Below 60%** - Low confidence, wait for better setup

### Key Indicators Explained

**RSI (Relative Strength Index)**
- Above 70: Overbought (potential pullback)
- Below 30: Oversold (potential bounce)
- 50: Neutral

**MACD (Moving Average Convergence Divergence)**
- Positive crossover: Bullish
- Negative crossover: Bearish
- Histogram expanding: Trend strengthening

**Bollinger Bands**
- Price near upper band: Strong uptrend or overbought
- Price near lower band: Strong downtrend or oversold
- Narrow bands: Low volatility (breakout coming)

**ADX (Average Directional Index)**
- Above 25: Strong trend
- Below 20: Weak trend/ranging market

---

## Configuration

### Edit `.env` File
```
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_secret_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

### Modify Trading Pairs
Edit `main.py` line 15-35 to add/remove trading pairs:
```python
self.trading_pairs = [
    "BTCUSDT", "ETHUSDT",  # Add your favorites
    # ... add more here
]
```

### Change Default Timeframes
Edit `main.py` line 36-43 or specify via command line

---

## Troubleshooting

### "Python is not installed"
- Download from https://python.org
- Make sure to check "Add Python to PATH" during installation

### "Module not found" errors
- Run `install.bat` (Windows) or `./install.sh` (Mac/Linux) again
- Or manually: `pip install -r requirements.txt`

### "API Error" or "Invalid API Key"
- Double-check your `.env` file has correct keys
- Make sure there are no extra spaces
- Verify keys are active on Binance

### "Rate limit exceeded"
- Wait a few seconds between requests
- Binance has rate limits to prevent abuse

### No data returned for a symbol
- Symbol might be delisted or have low volume
- Check spelling (must be uppercase, e.g., "BTCUSDT")
- Try a different symbol

---

## Advanced Usage

### Custom Indicator Analysis
The `indicators.py` file contains 60+ indicators. You can:
- Modify indicator parameters (periods, thresholds)
- Add new custom indicators
- Change analysis logic

See `INDICATOR_REFERENCE.md` for detailed documentation.

### API Integration
Want to integrate this into your own app? The code is modular:
```python
from main import AdvancedTradingBot

bot = AdvancedTradingBot()
result = bot.run_comprehensive_analysis("BTCUSDT")
# Use result data in your application
```

---

## License & Legal

### License
This software is licensed for **personal use only**. You may:
- Use for your own trading analysis
- Modify for personal use
- Run on multiple personal computers

You may NOT:
- Resell or redistribute
- Use commercially without permission
- Claim as your own work

### Disclaimer
**TRADING RISK WARNING:** Trading cryptocurrencies carries substantial risk of loss. This software is for educational purposes only and is NOT financial advice. Past performance does not guarantee future results. You are solely responsible for your trading decisions. The creators are not liable for any losses.

---

## Pro Tips

1. **Start with longer timeframes** (4h, 1d) for more reliable signals
2. **Wait for alignment** across multiple timeframes for best trades
3. **Use stop losses** - the tool suggests levels, use them!
4. **Don't overtrade** - quality over quantity
5. **Paper trade first** - test strategies without real money
6. **Customize** - adjust parameters to match your trading style

---


**Happy Trading! Remember: The goal is not to predict the future, but to make more informed decisions. 📈**
