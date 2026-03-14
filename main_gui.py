# coding: utf-8
"""
MarketMind Oracle -- Launcher
Starts the Flask server and opens the web UI in the browser automatically.
Hit Run in your IDE and the dashboard opens at http://localhost:5000
"""

import sys
import time
import threading
import webbrowser
from pathlib import Path

# -- Backend import --
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, send_file
from data_engine import run_analysis, search_tickers, get_market_movers

# Chart builder — generates a rich Plotly chart on every analysis run.
# Safe to import even if plotly is missing; errors are caught at call-time.
try:
    from chart_builder import build_chart as _build_chart
    _HAS_CHART_BUILDER = True
except ImportError:
    _HAS_CHART_BUILDER = False

# -- Paths --
BASE_DIR   = Path(__file__).parent
INDEX_HTML = BASE_DIR / "index.html"
CHART_HTML = BASE_DIR / "chart.html"
PORT       = 5000

# -- Flask app --
app = Flask(__name__, static_folder=str(BASE_DIR))


@app.route("/")
def index():
    if INDEX_HTML.exists():
        return send_file(str(INDEX_HTML))
    return "<h1>index.html not found in project folder.</h1>", 404


@app.route("/chart.html")
def chart():
    if CHART_HTML.exists():
        response = send_file(str(CHART_HTML))
        # Force browser to always re-fetch — never serve a stale chart.
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"]        = "no-cache"
        response.headers["Expires"]       = "0"
        return response
    return (
        "<html><body style='background:#051F20;color:#4a7a60;"
        "font-family:monospace;padding:40px'>"
        "<h2>No chart generated yet.</h2>"
        "<p>Run an analysis first, then open this page.</p></body></html>"
    ), 404


@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    try:
        return jsonify(search_tickers(q))
    except Exception:
        return jsonify([])


@app.route("/analyze")
def analyze():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "No ticker provided."}), 400
    try:
        result = run_analysis(q)
        result["resolved_ticker"] = q.upper()

        # ── Regenerate chart.html with enhanced builder ───────────────────────
        if _HAS_CHART_BUILDER:
            df = result.pop("df", None)          # keep df for chart, remove before JSON
            try:
                if df is not None and not df.empty:
                    _build_chart(df, q, result, CHART_HTML)
                else:
                    result.pop("df", None)
            except Exception as chart_err:
                print(f"[chart_builder] Warning: {chart_err}")
                result.pop("df", None)
        else:
            result.pop("df", None)               # df is never JSON-serialisable

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/movers")
def movers():
    try:
        return jsonify(get_market_movers())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -- Auto-open browser after server starts --
def open_browser():
    time.sleep(1.2)  # wait for Flask to be ready
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    print(f"[MarketMind] Starting server at http://localhost:{PORT}")
    print(f"[MarketMind] Opening browser...")

    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Flask (blocks here until you stop it)
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False,
    )
