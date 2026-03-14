"""
MarketMind - Data Processing Engine
Handles data fetching, technical indicators, risk metrics,
and fundamental valuation metrics. 
"""

import json
import re
import os
import datetime
import sys
from typing import Tuple, Optional, Dict, Any

import numpy as np
import yfinance as yf
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objects as go

# -- Firebase Initialization --
# Priority order:
#   1. FIREBASE_KEY_JSON env var  — production (Render / any cloud host)
#      Set this on your hosting platform to the full JSON contents of
#      your firebase-key.json service-account file.
#   2. firebase-key.json on disk  — local dev (production key file)
#   3. MM.json on disk            — local dev (legacy fallback)
# The app runs without Firebase if none of the above are found;
# analysis still works, only signal logging is disabled.
try:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        _base_dir = getattr(sys, '_MEIPASS')
    else:
        _base_dir = os.path.dirname(os.path.abspath(__file__))

    _key_primary  = os.path.join(_base_dir, "firebase-key.json")
    _key_fallback = os.path.join(_base_dir, "MM.json")

    _cred = None

    # -- Option 1: environment variable (production / Render) --
    _key_env = os.environ.get("FIREBASE_KEY_JSON", "").strip()
    if _key_env:
        try:
            _key_dict = json.loads(_key_env)
            _cred = credentials.Certificate(_key_dict)
            print("[Firebase] Initialized from FIREBASE_KEY_JSON env var (production).")
        except Exception as _env_err:
            print(f"[Firebase] FIREBASE_KEY_JSON parse error: {_env_err}")
            _cred = None

    # -- Option 2: firebase-key.json on disk (local dev) --
    if _cred is None and os.path.exists(_key_primary):
        _cred = credentials.Certificate(_key_primary)
        print(f"[Firebase] Initialized from {os.path.basename(_key_primary)} (local).")

    # -- Option 3: MM.json on disk (legacy local fallback) --
    if _cred is None and os.path.exists(_key_fallback):
        _cred = credentials.Certificate(_key_fallback)
        print(f"[Firebase] Initialized from {os.path.basename(_key_fallback)} (legacy local).")

    if _cred is not None:
        firebase_admin.initialize_app(_cred)
        db = firestore.client()
    else:
        db = None
        print("[Firebase] No credentials found. Signal logging disabled.")

except Exception as e:
    db = None
    print(f"[Firebase] Initialization failed: {e}")


# -- Ticker Resolution --

_LOOKS_LIKE_TICKER = re.compile(r'^[A-Z0-9]{1,10}(\.[A-Z]{2,3})?(-[A-Z]{2,3})?$')

# -- Sector Classification --
# IT / Technology sectors warrant a higher Graham premium (capital-light, high ROIC)
_IT_TECH_SECTORS = {
    "technology", "information technology", "software", "semiconductors",
    "it services", "computer services", "electronic technology",
    "technology services", "it"
}

def _get_graham_multiplier(ticker: str) -> float:
    """
    Return the sector-appropriate Graham Number ceiling multiplier.
    IT/Tech stocks: 2.5x  |  All others (Manufacturing, Energy, etc.): 1.5x
    Falls back to 1.5x if sector data is unavailable.
    """
    try:
        sector = (yf.Ticker(ticker).info.get("sector", "") or "").lower()
        industry = (yf.Ticker(ticker).info.get("industry", "") or "").lower()
        combined = sector + " " + industry
        if any(kw in combined for kw in _IT_TECH_SECTORS):
            return 2.5
    except Exception:
        pass
    return 1.5

def resolve_ticker(query: str) -> Tuple[str, str]:
    """Map a search query to the best yfinance ticker symbol."""
    raw = query.strip()
    upper = raw.upper()

    # Step 1: Direct Ticker Check
    if ("." in upper or "-" in upper) and _LOOKS_LIKE_TICKER.match(upper):
        return upper, upper

    # Step 2: Search by Name
    try:
        search = yf.Search(raw, max_results=10)
        candidates = [
            q for q in (search.quotes or [])
            if q.get("quoteType", "").upper() == "EQUITY"
        ]
    except Exception as exc:
        raise ValueError(f"Ticker search error: {exc}")

    if not candidates:
        if _LOOKS_LIKE_TICKER.match(upper):
            return upper, upper
        raise ValueError(f"No stock found for '{raw}'.")

    # Step 3: Volume Ranking
    def _get_vol(symbol: str) -> int:
        try:
            # Use fast_info to avoid heavy downloads
            return int(yf.Ticker(symbol).fast_info.get("three_month_average_volume", 0) or 0)
        except: return 0

    ranked = []
    for cand in candidates[:5]:
        sym = cand.get("symbol", "")
        vol = _get_vol(sym)
        ranked.append((vol, sym, cand))

    ranked.sort(key=lambda x: x[0], reverse=True)
    best_vol, best_sym, best_cand = ranked[0]

    name = best_cand.get("longname") or best_cand.get("shortname") or best_sym
    return best_sym, f"{name} ({best_sym})"


def search_tickers(query: str, max_results: int = 5) -> list:
    """Lightweight search for autocomplete."""
    if not query or len(query) < 2: return []
    try:
        search = yf.Search(query, max_results=max_results)
        return [
            {"symbol": q["symbol"], "name": q.get("shortname") or q.get("longname") or q["symbol"]}
            for q in (search.quotes or []) if q.get("quoteType") == "EQUITY"
        ]
    except: return []


# -- Data & Indicators --

def fetch_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical OHLCV data."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}.")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Close"], inplace=True)
    return df

def compute_ma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add Simple Moving Average."""
    df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()
    return df


# -- Oracle Metrics --

def compute_oracle_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Sharpe, Vol Trend, and Drawdown."""
    try:
        prices = df["Close"]
        returns = prices.pct_change().dropna()
        
        total_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        ann_vol = float(returns.std() * np.sqrt(252) * 100)
        
        # Vol Trend
        recent_std = returns.tail(20).std()
        base_std = returns.head(180).std() if len(returns) >= 180 else returns.std()
        vol_trend = "decreasing" if recent_std < base_std else "increasing"
        
        # Drawdown
        dd = (prices - prices.cummax()) / prices.cummax()
        max_dd = float(dd.min() * 100)
        
        # Sharpe (5% RF)
        ann_ret = total_ret / (len(prices) / 252.0)
        sharpe = round((ann_ret - 5.0) / ann_vol, 3) if ann_vol > 0 else 0.0
        
        return {
            "last_close": round(float(prices.iloc[-1]), 2),
            "total_return_pct": round(total_ret, 2),
            "annualised_vol_pct": round(ann_vol, 2),
            "baseline_vol_pct": round(float(base_std * 100), 4),
            "vol_trend": vol_trend,
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_approx": sharpe
        }
    except Exception as e:
        print(f"Metric error: {e}")
        return {}


def get_recommendation(
    df: pd.DataFrame,
    m: Dict[str, Any],
    f: Dict[str, Any],
    ticker: str = ""
) -> Dict[str, Any]:
    """
    Timing signals based on strict fundamental and technical criteria.

    Scoring rubric (100 pts total):
      Technical  (40 pts)
        +25  Price above 200-SMA
        +15  Volatility is decreasing
      Fundamental (60 pts)
        +20  Altman Z-Score > 1.8
        +10  (bonus) Z-Score > 4.0  → Hyper-Safe
        +15  Piotroski F-Score >= 5
        +15  Graham valuation within sector limit

    Tier labels
        80+ pts  → EXECUTIVE BUY          (Emerald Green  #2ECC71)
        60-79    → STABLE ACCUMULATE       (Sage Green     #8FBC8F)
        < 60     → AVOID / FORENSIC RED FLAG (Neon Red    #FF1744)
    """
    last   = m.get("last_close", 0)
    ma200  = df["MA_200"].iloc[-1] if "MA_200" in df.columns else None

    dist     = ((last - ma200) / ma200 * 100) if ma200 else None
    vol_down = m.get("vol_trend") == "decreasing"

    # Fundamental Metrics
    z_score    = f.get("z_score", 0.0)
    f_score    = f.get("f_score", 0)
    graham_num = f.get("graham_num", 0)

    # -- Safe Haven Exception --
    # A stock with Z-Score > 4.0 is Hyper-Safe; waive a mildly negative 200-SMA
    # (dist between -5% and 0%) rather than hard-blocking it.
    hyper_safe       = z_score > 4.0
    sma_slightly_neg = dist is not None and -5.0 <= dist < 0.0
    sma_safe_haven   = hyper_safe and sma_slightly_neg  # exception applies

    # -- Sector-Appropriate Graham Ceiling --
    graham_mult = _get_graham_multiplier(ticker) if ticker else 1.5

    # ── Point Scoring ────────────────────────────────────────────────
    score = 0
    score_breakdown = []

    # Technical: 200-SMA (25 pts)
    if dist is not None and dist > 0:
        score += 25
        score_breakdown.append("200-SMA ✓ (+25)")
    elif sma_safe_haven:
        score += 15   # partial credit: Hyper-Safe exception
        score_breakdown.append(f"200-SMA waived (Safe Haven Z={z_score:.1f}) (+15)")
    else:
        score_breakdown.append("200-SMA ✗ (+0)")

    # Technical: Volatility Trend (15 pts)
    if vol_down:
        score += 15
        score_breakdown.append("Vol decreasing ✓ (+15)")
    else:
        score_breakdown.append("Vol decreasing ✗ (+0)")

    # Fundamental: Z-Score base (20 pts)
    if z_score > 1.8:
        score += 20
        score_breakdown.append(f"Z-Score {z_score:.2f} > 1.8 ✓ (+20)")
    else:
        score_breakdown.append(f"Z-Score {z_score:.2f} ✗ (+0)")

    # Fundamental: Z-Score bonus for Hyper-Safe (10 pts)
    if hyper_safe:
        score += 10
        score_breakdown.append(f"Hyper-Safe Z > 4.0 ✓ (+10)")

    # Fundamental: F-Score (15 pts)
    if f_score >= 5:
        score += 15
        score_breakdown.append(f"F-Score {f_score} >= 5 ✓ (+15)")
    else:
        score_breakdown.append(f"F-Score {f_score} < 5 ✗ (+0)")

    # Fundamental: Graham Valuation (15 pts)
    graham_ok = graham_num <= 0 or last < graham_mult * graham_num
    if graham_ok:
        score += 15
        ceiling_label = f"{graham_mult}x Graham" if graham_num > 0 else "N/A"
        score_breakdown.append(f"Valuation within {ceiling_label} ✓ (+15)")
    else:
        score_breakdown.append(
            f"Valuation > {graham_mult}x Graham (₹{last:.0f} vs ₹{graham_mult * graham_num:.0f}) ✗ (+0)"
        )

    # ── Rejection reason strings (for detail text) ─────────────────
    tech_reasons = []
    fund_reasons = []

    if not (dist is not None and dist > 0) and not sma_safe_haven:
        tech_reasons.append("Price is below 200-SMA.")
    if not vol_down:
        tech_reasons.append("Volatility is not decreasing.")
    if z_score <= 1.8:
        fund_reasons.append(f"Z-Score too low ({z_score:.2f} <= 1.8).")
    if f_score < 5:
        fund_reasons.append(f"F-Score too low ({f_score} < 5).")
    if not graham_ok:
        fund_reasons.append(
            f"Valuation too high (Price >= {graham_mult}x Graham Number)."
        )

    all_reasons   = tech_reasons + fund_reasons
    total_return  = m.get("total_return_pct", 0)

    # ── Tier Labels ───────────────────────────────────────────────────
    if score >= 80:
        lbl   = "EXECUTIVE BUY"
        color = "#2ECC71"   # Emerald Green
        det   = (
            f"Score {score}/100. All critical Oracle parameters met. "
            f"Strong fundamentals, positive momentum, valuation within {graham_mult}x Graham ceiling. "
            + (f"Safe Haven Z-Score {z_score:.1f} active. " if hyper_safe else "")
            + "Breakdown: " + " | ".join(score_breakdown)
        )
    elif score >= 60:
        lbl   = "STABLE ACCUMULATE"
        color = "#8FBC8F"   # Sage Green
        det   = (
            f"Score {score}/100. Solid foundation with minor gaps. "
            + (f"Safe Haven exception applied (Z={z_score:.1f}). " if sma_safe_haven else "")
            + (f"Weak points: {' | '.join(all_reasons)} " if all_reasons else "")
            + "Breakdown: " + " | ".join(score_breakdown)
        )
    else:
        lbl   = "AVOID / FORENSIC RED FLAG"
        color = "#FF1744"   # Neon Red
        det   = (
            f"Score {score}/100. Multiple critical checks failed. "
            f"Rejected: {' | '.join(all_reasons) if all_reasons else 'Score below threshold.'} "
            + "Breakdown: " + " | ".join(score_breakdown)
        )

    # ── "Near Miss" Logic / Premium Quality Watchlist ─────────────────
    # If fundamentally strong (Z > 3.0, F > 6) but fails on Price (200-SMA or Graham)
    is_fundamentally_strong = z_score > 3.0 and f_score >= 6
    fails_price = not (dist is not None and dist > 0) or not graham_ok
    
    if is_fundamentally_strong and fails_price:
        lbl   = "WATCHLIST: PREMIUM QUALITY"
        color = "#FFB830" # Amber
        det   = (
            f"Score {score}/100. Fundamentally elite (Z={z_score:.1f}, F={f_score}), "
            "but current price indicates poor timing or overvaluation. "
            "Wait for a pullback or trend reversal above the 200-SMA. "
            "Breakdown: " + " | ".join(score_breakdown)
        )

    buy_cond  = score >= 80
    risk_cond = 60 <= score < 80
    trap_cond = score < 60

    return {
        "label": lbl,
        "label_color": color,
        "score": score,
        "score_breakdown": score_breakdown,
        "detail": det,
        "graham_multiplier": graham_mult,
        "safe_haven_active": bool(sma_safe_haven),
        "ma_200_value": round(ma200, 2) if ma200 else 0,
        "distance_from_mean_pct": round(dist, 2) if dist is not None else 0,
        "vol_trend": m.get("vol_trend", "stable"),
        "condition_buy": bool(buy_cond),
        "condition_risk": bool(risk_cond),
        "condition_trap": bool(trap_cond),
    }


# -- Fundamental Forensic --

def compute_fundamentals(ticker: str, price: float) -> Dict[str, Any]:
    """F-Score, Z-Score, Graham Number."""
    try:
        t = yf.Ticker(ticker)
        bs, cf, fs = t.balance_sheet, t.cashflow, t.financials
        if bs.empty or cf.empty: return {}
        
        cols = bs.columns
        if len(cols) < 2: return {}
        L, P = cols[0], cols[1] # Latest, Previous
        
        # 1. Piotroski F-Score (9 points total, simplified to 4 here for data availability)
        f_score = 0
        try:
            ni = fs.loc['Net Income', L]
            if ni > 0: f_score += 1
            cfo = cf.loc['Cash Flow From Continuing Operating Activities', L]
            if cfo > 0: f_score += 1
            if cfo > ni: f_score += 1
            roa = ni / bs.loc['Total Assets', L]
            if roa > 0: f_score += 1
        except: pass
        
        # 2. Altman Z-Score
        z_score = 0.0
        try:
            ta = bs.loc['Total Assets', L]
            wc = bs.loc['Working Capital', L]
            re = bs.loc['Retained Earnings', L]
            ebit = fs.loc['EBIT', L]
            tl = bs.loc['Total Liabilities Net Minority Interest', L]
            sales = fs.loc['Total Revenue', L]
            mc = t.info.get('marketCap', 0)
            
            x1 = wc / ta
            x2 = re / ta
            x3 = ebit / ta
            x4 = mc / tl
            x5 = sales / ta
            z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
        except: pass

        # 3. Graham Number
        gn = 0
        try:
            eps = t.info.get('trailingEps', 0)
            bvps = t.info.get('bookValue', 0)
            if eps > 0 and bvps > 0: gn = np.sqrt(22.5 * eps * bvps)
        except: pass

        return {
            "f_score": f_score, 
            "f_label": "Fortress" if f_score >= 3 else "Neutral" if f_score >= 1 else "Weak",
            "z_score": round(z_score, 2), 
            "z_label": "HYPER-SAFE" if z_score > 4.0 else "SAFE" if z_score > 3.0 else "DISTRESS" if z_score < 1.8 else "GREY ZONE",
            "graham_num": round(gn, 2), 
            "undervalued_pct": round((gn - price) / price * 100, 2) if gn and price else 0
        }
    except: return {}


# -- Pipeline --

def generate_chart(df: pd.DataFrame, ticker: str, label: str, rec_label: str):
    """Generate a high-fidelity interactive Plotly Candlestick chart."""
    try:
        # Last 6 months (~126 trading days)
        df_6m = df.tail(126)
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_6m.index,
            open=df_6m['Open'],
            high=df_6m['High'],
            low=df_6m['Low'],
            close=df_6m['Close'],
            name='Price'
        )])
        
        # 200-SMA Overlay (Gold)
        if 'MA_200' in df_6m.columns:
            fig.add_trace(go.Scatter(
                x=df_6m.index, 
                y=df_6m['MA_200'], 
                mode='lines',
                line=dict(color='#FFD700', width=2),
                name='200-Day SMA'
            ))
            
        # Signal Marker
        last_date = df_6m.index[-1]
        last_price = df_6m['Close'].iloc[-1]
        
        # Signal Marker – colours mirror the three-tier label palette
        if "EXECUTIVE BUY" in rec_label:
            marker_color = "#2ECC71"   # Emerald Green
        elif "STABLE ACCUMULATE" in rec_label:
            marker_color = "#8FBC8F"   # Sage Green
        elif "WATCHLIST" in rec_label:
            marker_color = "#FFB830"   # Amber
        else:
            marker_color = "#FF1744"   # Neon Red (AVOID / FORENSIC RED FLAG)
        
        fig.add_annotation(
            x=last_date,
            y=last_price,
            text=f"SIGNAL: {rec_label.upper()}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=marker_color,
            arrowsize=1.5,
            arrowwidth=2,
            font=dict(color=marker_color, size=12, family="Segoe UI", weight="bold"),
            ax=0,
            ay=-50,
            bgcolor="#0B2B26",
            bordercolor=marker_color,
            borderwidth=1,
            borderpad=4
        )
        
        fig.update_layout(
            template='plotly_dark',
            title=f"{label} - 6 Month Analysis",
            yaxis_title="Price (Rs.)",
            xaxis_title="",
            xaxis_rangeslider_visible=False,
            plot_bgcolor="#051F20",
            paper_bgcolor="#051F20",
            font=dict(color='#FFFFFF', family="Segoe UI"),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart.html")
        fig.write_html(path, include_plotlyjs="cdn")
    except Exception as e:
        print(f"Chart generation error: {e}")

def run_analysis(query: str, period: str = "1y") -> Dict[str, Any]:
    """Main execution flow."""
    sym, lbl = resolve_ticker(query)
    df = fetch_data(sym, period)
    df = compute_ma(df, 20)
    df = compute_ma(df, 200)
    
    m = compute_oracle_metrics(df)
    f = compute_fundamentals(sym, m["last_close"] if "last_close" in m else 0.0)
    r = get_recommendation(df, m, f, ticker=sym)
    
    # Export for JS (ensure JSON serializable)
    def _j(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, (np.floating, float)): return float(o)
        return o

    m_ser = {k: _j(v) for k, v in m.items()}
    r_ser = {k: _j(v) for k, v in r.items()}
    f_ser = {k: _j(v) for k, v in f.items()}

    export = {
        "ticker": sym, "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "close": df["Close"].round(2).fillna(0).tolist(),
        "ma_20": df["MA_20"].round(2).fillna(0).tolist(),
        "ma_200": df["MA_200"].round(2).fillna(0).tolist() if "MA_200" in df.columns else [],
        "metrics": m_ser, "rec": r_ser, "fundamental": f_ser
    }
    
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_data.json")
    with open(path, "w") as fh: json.dump(export, fh)
    
    # Generate interactive chart
    generate_chart(df, sym, lbl, r["label"])
    
    # Firebase Log
    if db:
        try:
            db.collection("signals").add({
                "ticker": sym, "signal": r["label"], "timestamp": datetime.datetime.now(),
                "price": m["last_close"]
            })
        except: pass

    return {
        "df": df, "metrics": m, "rec": r, "fundamental": f, 
        "resolved_ticker": sym, "search_label": lbl, "export": export
    }

def get_market_movers() -> list:
    """Fetch top gainers from a curated list of global and Nifty leaders."""
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "LTIM.NS", "MARUTI.NS",
        "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "META", "AMZN", "NFLX"
    ]
    try:
        data = yf.download(tickers, period="2d", interval="1d", group_by='ticker', auto_adjust=True, progress=False)
        results = []
        for ticker in tickers:
            try:
                if ticker not in data.columns.levels[0]: continue
                t_data = data[ticker].dropna(subset=['Close'])
                if len(t_data) < 2: continue
                
                last_price = float(t_data['Close'].iloc[-1])
                prev_price = float(t_data['Close'].iloc[-2])
                change = ((last_price - prev_price) / prev_price) * 100
                
                results.append({
                    "symbol": ticker,
                    "price": round(last_price, 2),
                    "change": round(change, 2)
                })
            except: continue
        
        # Sort by % change descending
        results.sort(key=lambda x: x['change'], reverse=True)
        return results[:10]
    except Exception as e:
        print(f"Movers error: {e}")
        return []

if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    print(run_analysis(t)["rec"]["label"])
