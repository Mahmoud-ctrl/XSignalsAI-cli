import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from binance.client import Client
from indicators import compute_indicators
import argparse
import time

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("MODEL_NAME")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

class AdvancedTradingBot:
    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.trading_pairs = [
            # Top market cap coins
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
            "LTCUSDT", "AVAXUSDT", "XLMUSDT", "TRXUSDT", "FILUSDT",
            "BNBUSDT", "XRPUSDT", "SOLUSDT", "MATICUSDT", "BCHUSDT",

            # Popular & trending coins
            "DOGEUSDT", "ATOMUSDT", "ETCUSDT", "NEARUSDT", "SANDUSDT",
            "AAVEUSDT", "APEUSDT", "FTMUSDT", "GALAUSDT", "ALGOUSDT",

            # More DeFi & Layer 1/Layer 2 tokens
            "CAKEUSDT", "CRVUSDT", "COMPUSDT", "KAVAUSDT", "RUNEUSDT",
            "INJUSDT", "LDOUSDT", "GMXUSDT", "SNXUSDT", "ZILUSDT",

            # Metaverse / Gaming tokens
            "MANAUSDT", "ENJUSDT", "CHZUSDT", "AXSUSDT", "ILVUSDT",

            # AI & Web3 tokens
            "FETUSDT", "AGIXUSDT", "OCEANUSDT", "RNDRUSDT", "NKNUSDT",

            # Emerging & hot tokens
            "ARBUSDT", "OPUSDT", "SKLUSDT", "RDNTUSDT", "HOOKUSDT",
            "DYDXUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT", "JASMYUSDT"
        ]
        self.timeframes = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "30m": Client.KLINE_INTERVAL_30MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY
        }
    
    def fetch_historical_data(self, symbol="ETHUSDT", interval="1h", limit=500, days_back=None):
        """Fetch extensive historical data"""
        try:
            if days_back:
                # Calculate start time for custom date range
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days_back)
                start_str = str(int(start_time.timestamp() * 1000))
                end_str = str(int(end_time.timestamp() * 1000))
                
                klines = self.client.get_historical_klines(
                    symbol, interval, start_str, end_str
                )
            else:
                klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
            if not klines:
                print(f"❌ No data received for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Clean and prepare data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
            for col in price_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ Fetched {len(df)} candles for {symbol} ({interval})")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def get_market_overview(self):
        """Get market overview for multiple symbols"""
        try:
            ticker_24hr = self.client.get_ticker()
            
            # Filter for our trading pairs
            relevant_tickers = [
                t for t in ticker_24hr 
                if t['symbol'] in self.trading_pairs
            ]
            
            # Sort by 24hr volume
            relevant_tickers.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            
            return relevant_tickers
            
        except Exception as e:
            print(f"❌ Error getting market overview: {e}")
            return None
    
    def analyze_multiple_timeframes(self, symbol, timeframes=['15m', '1h', '4h', '1d']):
        """Analyze multiple timeframes for comprehensive view"""
        results = {}
        
        for tf in timeframes:
            print(f"📊 Analyzing {symbol} on {tf} timeframe...")
            
            # Fetch data for this timeframe
            if tf == '1d':
                df = self.fetch_historical_data(symbol, tf, limit=200)
            else:
                df = self.fetch_historical_data(symbol, tf, limit=300)
            
            if df is None:
                continue
            
            # Compute indicators
            latest, analysis, full_df = compute_indicators(df)
            
            if latest is not None and analysis is not None:
                results[tf] = {
                    'latest': latest,
                    'analysis': analysis,
                    'data_points': len(full_df)
                }
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        return results
    
    
    def query_ai_analysis(self, symbol, multi_tf_analysis, market_context=None):
        """Enhanced AI analysis with comprehensive data"""
        try:
            # Prepare comprehensive prompt
            prompt = self.format_advanced_prompt(symbol, multi_tf_analysis, market_context)
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "http://localhost",
                "Content-Type": "application/json"
            }

            data = {
                "model": model_name, 
                "messages": [
                    {
                        "role": "system", 
                        "content": """You are an elite crypto trading analyst with expertise in technical analysis, 
                        multi-timeframe analysis, and market psychology. Provide detailed analysis with specific 
                        entry/exit points, risk management, and confidence levels. Always consider multiple 
                        timeframe alignment and market context."""
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 2000
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions", 
                headers=headers, 
                json=data,
                timeout=30
            )

            result = response.json()
            if "choices" not in result:
                print("❌ OpenRouter API Error:")
                print(result)
                return "Error: Could not get a valid response from OpenRouter."
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"❌ Exception during AI analysis: {e}")
            return "Error: Exception occurred during AI analysis."
    

    def format_advanced_prompt(self, symbol, multi_tf_analysis, market_context):
        """Format comprehensive crypto analysis prompt with strict signal discipline"""
        prompt = f"""
    🚀 XSignals AI ANALYSIS REQUEST 🚀

    SYMBOL: {symbol}
    TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

    ═══════════════════════════════════════════════════════════════════

    📊 MULTI-TIMEFRAME TECHNICAL ANALYSIS:
    """
        for tf, data in multi_tf_analysis.items():
            if 'analysis' not in data:
                continue

            analysis = data['analysis']
            latest = data['latest']

            # Safe fallback for candlestick patterns
            candle_patterns = ', '.join(analysis['pattern_analysis']['patterns']) if analysis['pattern_analysis']['patterns'] else "No significant pattern"

            prompt += f"""
    ┌─ {tf.upper()} TIMEFRAME ANALYSIS ─┐
    │ Current Price: ${latest['close']:.4f}
    │ Overall Signal: {analysis['overall_signal']['signal']} (Confidence: {analysis['overall_signal']['confidence']:.1f}%)
    │
    │ 🎯 TREND ANALYSIS:
    │   Strength: {analysis['trend_analysis']['strength']}
    │   Score: {analysis['trend_analysis']['score']}/7
    │   Key Signals: {', '.join(analysis['trend_analysis']['signals'][:3])}
    │
    │ ⚡ MOMENTUM ANALYSIS:
    │   Strength: {analysis['momentum_analysis']['strength']}
    │   Score: {analysis['momentum_analysis']['score']}/6
    │   RSI(14): {latest['rsi_14']:.1f}
    │   MACD: {latest['macd']:.5f}
    │   Stochastic: {latest['stoch_k']:.1f}
    │
    │ 📈 VOLATILITY & VOLUME:
    │   BB Position: {analysis['volatility_analysis']['bb_position']:.2f}
    │   ATR: {analysis['volatility_analysis']['atr']:.4f}
    │   Volume Activity: {latest['volume_momentum']:.1f}x average
    │
    │ 🕯️ CANDLESTICK PATTERNS:
    │   {candle_patterns}
    │
    │ 🎯 SUPPORT/RESISTANCE:
    │   Support: ${analysis['support_resistance']['support']:.4f} ({analysis['support_resistance']['distance_to_support']:.1f}% away)
    │   Resistance: ${analysis['support_resistance']['resistance']:.4f} ({analysis['support_resistance']['distance_to_resistance']:.1f}% away)
    └────────────────────────────────────────┘
    """

        if market_context:
            prompt += f"""
    🌍 MARKET CONTEXT:
    Top performers today: {', '.join([f"{t['symbol']}: {float(t['priceChangePercent']):.1f}%" for t in market_context[:5]])}
    """

        prompt += """
    ═══════════════════════════════════════════════════════════════════

    🎯 TRADING ANALYSIS REQUIREMENTS:

    ⚠️ SIGNAL DISCIPLINE RULES:
    - Only generate a trade signal when there is HIGH CONFIDENCE (6+/10) across multiple timeframes
    - If timeframes conflict, momentum is weak, or setup is unclear → NO SIGNAL
    - Quality over quantity: It's better to wait than force a mediocre trade
    - A proper signal requires: aligned trend + strong momentum + clear entry zone + defined risk

    ═══════════════════════════════════════════════════════════════════

    1. 📊 MULTI-TIMEFRAME ALIGNMENT:
    - Assess trend and momentum alignment across timeframes
    - Identify the dominant timeframe for trade execution
    - Note any conflicting signals or divergences
    - Determine if alignment is strong enough to trade

    2. 🎯 TRADING SIGNAL DECISION:

    **OPTION A: GENERATE SIGNAL** (Only if confidence ≥ 6/10)
    
    Format:
    ```
    SIGNAL: [BUY/SELL]
    CONFIDENCE: [X]/10
    
    ENTRY ZONE: $[price1] - $[price2]
    STOP LOSS: $[price] ([X]% risk)
    
    TAKE PROFIT TARGETS:
    TP1: $[price] ([X]% gain) - Probability: [High/Medium]
    TP2: $[price] ([X]% gain) - Probability: [Medium]
    TP3: $[price] ([X]% gain) - Probability: [Low/Medium]
    
    POSITION SIZE: Risk [1-2]% of capital
    TIMEFRAME: Best executed on [timeframe]
    ```

    **OPTION B: NO SIGNAL** (If confidence < 6/10)
    
    Format:
    ```
    NO SIGNAL GENERATED
    
    Reason: [Choose most relevant]
    - Conflicting timeframe signals (e.g., 1h bullish but 4h bearish)
    - Weak momentum across multiple indicators
    - Price in consolidation/indecision zone
    - Low volume / lack of conviction
    - Too close to major support/resistance (unclear direction)
    - Overextended price action (RSI >70 or <30 without confirmation)
    
    What to watch: [Specific price levels or conditions that would create a valid setup]
    ```

    3. ⚠️ RISK ASSESSMENT:
    - Key risks if trading this setup
    - Probability of false breakout/breakdown
    - Current volatility considerations
    - Market structure concerns

    4. 📈 MARKET PSYCHOLOGY & CONTEXT:
    - Current sentiment indicators (fear/greed)
    - Volume profile and institutional activity
    - Key psychological price levels nearby

    5. ⏰ TIMING & DURATION:
    - If signal generated: Expected hold time (scalp/swing/position)
    - If no signal: What needs to change for a valid setup

    6. 🔄 SCENARIO PLANNING:
    - BULL CASE: Key resistance levels and breakout targets
    - BEAR CASE: Key support levels and breakdown targets  
    - INVALIDATION: Price level that proves current analysis wrong

    ═══════════════════════════════════════════════════════════════════

    📝 RESPONSE GUIDELINES:
    - Be honest about uncertainty - "NO SIGNAL" is a valid and professional response
    - When generating signals, be precise with price levels
    - Use realistic take profit probabilities
    - Focus on interpretation, not data repetition
    - Keep analysis concise and actionable
    """

        return prompt

    
    def display_analysis_results(self, symbol, multi_tf_analysis, ai_analysis):
        """Display comprehensive analysis results"""
        print(f"\n{'='*80}")
        print(f"🚀 XSignals AI ANALYSIS COMPLETE: {symbol}")
        print(f"{'='*80}")
        
        # Quick summary
        signals = []
        confidences = []
        
        for tf, data in multi_tf_analysis.items():
            if 'analysis' in data:
                signal = data['analysis']['overall_signal']['signal']
                confidence = data['analysis']['overall_signal']['confidence']
                signals.append(f"{tf}: {signal}")
                confidences.append(confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        print(f"\n📊 TIMEFRAME SIGNALS:")
        for signal in signals:
            print(f"   {signal}")
        
        print(f"\n🎯 AVERAGE CONFIDENCE: {avg_confidence:.1f}%")
        
        # Detailed AI analysis
        print(f"\n🤖 AI TRADING ANALYSIS:")
        print("─" * 80)
        print(ai_analysis)
        print("─" * 80)
        
        # Key levels summary
        print(f"\n📈 KEY LEVELS SUMMARY:")
        if '1h' in multi_tf_analysis and 'analysis' in multi_tf_analysis['1h']:
            sr = multi_tf_analysis['1h']['analysis']['support_resistance']
            current = sr['current_price']
            support = sr['support']
            resistance = sr['resistance']
            
            print(f"   Current Price: ${current:.4f}")
            print(f"   Support: ${support:.4f} ({sr['distance_to_support']:.1f}% away)")
            print(f"   Resistance: ${resistance:.4f} ({sr['distance_to_resistance']:.1f}% away)")
    
    def run_comprehensive_analysis(self, symbol="ETHUSDT", timeframes=['15m', '1h', '4h', '1d']):
        """Run complete comprehensive analysis"""
        print(f"🚀 Starting XSignals AI Analysis for {symbol}")
        print(f"📊 Analyzing timeframes: {', '.join(timeframes)}")
        
        # Get market context
        print("📈 Fetching market overview...")
        market_context = self.get_market_overview()
        
        # Multi-timeframe analysis
        multi_tf_analysis = self.analyze_multiple_timeframes(symbol, timeframes)
        
        if not multi_tf_analysis:
            print("❌ No analysis data available")
            return
        
        # AI analysis
        print("🤖 Generating AI analysis...")
        ai_analysis = self.query_ai_analysis(symbol, multi_tf_analysis, market_context)
        
        # Display results
        self.display_analysis_results(symbol, multi_tf_analysis, ai_analysis)
        
        return {
            'symbol': symbol,
            'multi_tf_analysis': multi_tf_analysis,
            'ai_analysis': ai_analysis,
            'market_context': market_context
        }
    
    def scan_multiple_pairs(self, pairs=None, timeframe='15m'):
        """Scan multiple trading pairs for opportunities"""
        if pairs is None:
            pairs = self.trading_pairs[:20]  # Top 20 pairs
        
        print(f"🔍 Scanning {len(pairs)} pairs for opportunities...")
        opportunities = []
        
        for symbol in pairs:
            print(f"   Scanning {symbol}...")
            
            df = self.fetch_historical_data(symbol, timeframe, limit=200)
            if df is None:
                continue
            
            latest, analysis, _ = compute_indicators(df)
            if latest is None or analysis is None:
                continue
            
            signal = analysis['overall_signal']['signal']
            confidence = analysis['overall_signal']['confidence']
            
            if signal in ['BUY', 'STRONG BUY'] and confidence > 60:
                opportunities.append({
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'price': latest['close'],
                    'rsi': latest['rsi_14'],
                    'trend_score': analysis['trend_analysis']['score']
                })
            
            time.sleep(0.1)  # Rate limiting
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\n🎯 TOP OPPORTUNITIES FOUND:")
        print("─" * 60)
        for opp in opportunities[:5]:
            print(f"{opp['symbol']:10} | {opp['signal']:12} | Conf: {opp['confidence']:5.1f}% | "
                  f"Price: ${opp['price']:8.4f} | RSI: {opp['rsi']:5.1f}")
        
        return opportunities

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='XSignals AI Crypto Trading Bot')
    parser.add_argument('--symbol', '-s', default='ETHUSDT', help='Trading symbol')
    parser.add_argument('--timeframes', '-t', nargs='+', default=['15m', '1h', '4h', '1d'], 
                       help='Timeframes to analyze')
    parser.add_argument('--scan', action='store_true', help='Scan multiple pairs')
    parser.add_argument('--pairs', '-p', nargs='+', help='Specific pairs to analyze')
    
    args = parser.parse_args()
    
    bot = AdvancedTradingBot()
    
    if args.scan:
        # Multi-pair scanning mode
        pairs = args.pairs if args.pairs else None
        opportunities = bot.scan_multiple_pairs(pairs)
        
        if opportunities:
            print(f"\n🚀 Running detailed analysis on top opportunity: {opportunities[0]['symbol']}")
            bot.run_comprehensive_analysis(opportunities[0]['symbol'], args.timeframes)
    else:
        # Single symbol comprehensive analysis
        bot.run_comprehensive_analysis(args.symbol, args.timeframes)

if __name__ == "__main__":
    main()