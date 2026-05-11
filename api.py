#!/usr/bin/env python3
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.expanduser("~"))
from load_env import load_dev_vars as _lenv; _lenv()
del _sys, _os, _lenv

"""
REST API for the web scraper.

Start:  python3 api.py
Then:   curl http://localhost:8000/v1/search?q=laptop -H "Authorization: Bearer sk_scrape_..."

New parameters for /v1/search:
  ?smart=1        — enable LLM-guided link scoring + extraction fallback
  ?clean=1        — run LLM text-normalisation on extracted titles
  ?no_cache=1     — bypass disk page cache
  ?js=1           — use JS rendering (playwright/requests_html) for the first URL only

New endpoint:
  POST /v1/llm/extract   — call LLM extraction directly on provided HTML/text
  POST /v1/llm/clean     — call LLM text cleaning on provided text
  GET  /v1/llm/status    — check if Ollama is reachable
"""

import hashlib
import json
import secrets
import time
import os
import sqlite3
import threading
from dataclasses import asdict
from functools import wraps
from flask import Flask, request, jsonify, g
from scraper import scrape_all, SITES, Listing, fetch_js
from user_profile import get_profile
from tool_registry import TOOLS, execute_tool

app = Flask(__name__)
from stats_logger import attach_stats_middleware

_DATA_DIR = os.environ.get("CARD_GRADER_DATA_DIR", os.path.dirname(__file__))
DB_PATH = os.path.join(_DATA_DIR, "scraper_api.db")
_START_TIME = time.time()
CACHE_TTL = 600          # 10 minutes
CACHE_MAX_ENTRIES = 100

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at REAL NOT NULL,
            requests_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS search_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL,
            query TEXT NOT NULL,
            sites TEXT NOT NULL,
            results_count INTEGER NOT NULL,
            elapsed REAL NOT NULL,
            smart INTEGER DEFAULT 0,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS result_cache (
            cache_key TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
    """)
    db.commit()
    db.close()


def _cache_key(query: str, sites: list, pages: int, smart: bool) -> str:
    raw = f"{query}|{'|'.join(sorted(sites))}|{pages}|{'smart' if smart else ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_cached(cache_key: str):
    db = get_db()
    row = db.execute(
        "SELECT result_json, created_at FROM result_cache WHERE cache_key=?", (cache_key,)
    ).fetchone()
    db.close()
    if row and (time.time() - row["created_at"]) < CACHE_TTL:
        return json.loads(row["result_json"])
    return None


def _set_cached(cache_key: str, data: dict):
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO result_cache (cache_key, result_json, created_at) VALUES (?,?,?)",
        (cache_key, json.dumps(data), time.time()),
    )
    count = db.execute("SELECT COUNT(*) as c FROM result_cache").fetchone()["c"]
    if count > CACHE_MAX_ENTRIES:
        db.execute(
            "DELETE FROM result_cache WHERE cache_key IN "
            "(SELECT cache_key FROM result_cache ORDER BY created_at ASC LIMIT ?)",
            (count - CACHE_MAX_ENTRIES,),
        )
    db.commit()
    db.close()


def gen_key():
    return f"sk_scrape_{secrets.token_hex(24)}"


# ---------------------------------------------------------------------------
# CORS + error handling
# ---------------------------------------------------------------------------

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    return response


@app.errorhandler(Exception)
def _handle_unhandled(e):
    import traceback
    tb = traceback.format_exc()
    print(f"[Unhandled Error] {e}\n{tb}")
    return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def require_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing API key. Use: Authorization: Bearer sk_scrape_..."}), 401
        key = auth[7:]
        db = get_db()
        row = db.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
        if not row:
            db.close()
            return jsonify({"error": "Invalid API key"}), 401
        db.execute("UPDATE api_keys SET requests_count = requests_count + 1 WHERE key=?", (key,))
        db.commit()
        g.api_key = key
        g.key_name = row["name"]
        db.close()
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    db = get_db()
    total_scrapes = db.execute("SELECT COUNT(*) as c FROM search_log").fetchone()["c"]
    db.close()
    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    # Check Ollama
    try:
        from ollama_client import is_ollama_available, _resolve_model
        ollama_ok = is_ollama_available()
        ollama_model = _resolve_model() if ollama_ok else None
    except ImportError:
        ollama_ok = False
        ollama_model = None

    return jsonify({
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "total_scrapes_run": total_scrapes,
        "db_size_bytes": db_size,
        "ollama": {"available": ollama_ok, "model": ollama_model},
    })


@app.route("/")
def index():
    return jsonify({
        "name": "Scraper API",
        "version": "2.0",
        "endpoints": {
            "GET /v1/search?q=QUERY": "Search listings (requires API key)",
            "GET /v1/sites": "List available sites (requires API key)",
            "GET /v1/usage": "View your usage stats (requires API key)",
            "POST /v1/keys": "Generate a new API key (no auth needed)",
            "GET /v1/llm/status": "Check Ollama availability (requires API key)",
            "POST /v1/llm/extract": "LLM extraction from HTML/text (requires API key)",
            "POST /v1/llm/clean": "LLM text cleaning (requires API key)",
            "GET /health": "Health check (no auth required)",
        },
        "auth": "Authorization: Bearer sk_scrape_...",
        "example": "GET /v1/search?q=laptop&sites=ebay,craigslist&pages=1&limit=50&smart=1",
        "smart_mode": {
            "?smart=1": "LLM link scoring + extraction fallback when CSS parser returns 0 results",
            "?clean=1": "LLM text-normalisation pass on all extracted titles",
            "?no_cache=1": "Bypass disk page cache (always fetch fresh HTML)",
            "?js=1": "Attempt JS rendering for page fetches",
        },
    })


@app.route("/v1/search", methods=["GET"])
@require_key
def search():
    """
    GET /v1/search?q=laptop&sites=ebay,craigslist&pages=1&limit=100&smart=1&clean=1
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing required parameter: q", "code": 400}), 400
    if len(query) > 200:
        return jsonify({"error": "Parameter 'q' must not exceed 200 characters", "code": 400}), 400

    sites_param = request.args.get("sites", "ebay,craigslist,bonanza")
    _seen_sites: set[str] = set()
    sites: list[str] = []
    for s in sites_param.split(","):
        s = s.strip().lower()
        if s and s not in _seen_sites:
            _seen_sites.add(s)
            sites.append(s)
    invalid = [s for s in sites if s not in SITES]
    if invalid:
        return jsonify({"error": f"Unknown sites: {', '.join(invalid)}", "available": list(SITES.keys()), "code": 400}), 400

    try:
        pages = max(1, min(5, int(request.args.get("pages", 1))))
    except ValueError:
        return jsonify({"error": "Parameter 'pages' must be an integer", "code": 400}), 400

    limit = min(int(request.args.get("limit", 200)), 1000)
    force_refresh = request.args.get("force", "0") in ("1", "true", "yes")
    smart = request.args.get("smart", "0") in ("1", "true", "yes")
    clean = request.args.get("clean", "0") in ("1", "true", "yes")
    no_cache = request.args.get("no_cache", "0") in ("1", "true", "yes")

    ck = _cache_key(query, sites, pages, smart)
    if not force_refresh and not no_cache:
        cached = _get_cached(ck)
        if cached:
            cached["cached"] = True
            return jsonify(cached)

    t0 = time.time()
    listings = scrape_all(
        query, sites, pages,
        max_workers=30,
        smart=smart,
        clean_text_pass=clean,
        use_cache=not no_cache,
    )
    elapsed = time.time() - t0

    listings = listings[:limit]
    site_counts: dict[str, int] = {}
    for l in listings:
        site_counts[l.site] = site_counts.get(l.site, 0) + 1

    llm_count = sum(1 for l in listings if l.llm_extracted)

    result = {
        "query": query,
        "sites": sites,
        "total": len(listings),
        "elapsed_seconds": round(elapsed, 2),
        "site_counts": site_counts,
        "llm_extracted_count": llm_count,
        "smart_mode": smart,
        "cached": False,
        "data": [asdict(l) for l in listings],
    }

    _set_cached(ck, result)

    get_profile().record_interaction(query, service="web-scraper", topics=["search"])

    db = get_db()
    db.execute(
        "INSERT INTO search_log (api_key, query, sites, results_count, elapsed, smart, created_at) VALUES (?,?,?,?,?,?,?)",
        (g.api_key, query, sites_param, len(listings), elapsed, int(smart), time.time()),
    )
    db.commit()
    db.close()

    return jsonify(result)


@app.route("/v1/sites", methods=["GET"])
@require_key
def list_sites():
    return jsonify({"sites": list(SITES.keys())})


@app.route("/v1/usage", methods=["GET"])
@require_key
def usage():
    db = get_db()
    key_row = db.execute("SELECT * FROM api_keys WHERE key=?", (g.api_key,)).fetchone()
    recent = db.execute(
        "SELECT query, sites, results_count, elapsed, smart, created_at "
        "FROM search_log WHERE api_key=? ORDER BY created_at DESC LIMIT 20",
        (g.api_key,),
    ).fetchall()
    db.close()
    return jsonify({
        "name": key_row["name"],
        "total_requests": key_row["requests_count"],
        "recent_searches": [dict(r) for r in recent],
    })


# ---------------------------------------------------------------------------
# LLM endpoints
# ---------------------------------------------------------------------------

@app.route("/v1/llm/status", methods=["GET"])
@require_key
def llm_status():
    """Check Ollama availability and which model is active."""
    try:
        from ollama_client import is_ollama_available, _resolve_model, OLLAMA_BASE
        available = is_ollama_available()
        model = _resolve_model() if available else None
        return jsonify({
            "available": available,
            "base_url": OLLAMA_BASE,
            "active_model": model,
        })
    except ImportError:
        return jsonify({"available": False, "error": "ollama_client module not found"}), 503


@app.route("/v1/llm/extract", methods=["POST"])
@require_key
def llm_extract():
    """
    POST /v1/llm/extract
    Body: {"html": "...", "query": "laptop", "site": "ebay"}

    Asks the LLM to extract structured product listings from the provided HTML/text.
    """
    data = request.get_json(force=True, silent=True) or {}
    html = data.get("html", "").strip()
    query = data.get("query", "").strip()
    site = data.get("site", "unknown")

    if not html:
        return jsonify({"error": "Missing 'html' field"}), 400
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400
    if len(html) > 50_000:
        return jsonify({"error": "'html' exceeds 50,000 character limit"}), 400

    try:
        from ollama_client import extract_listings, is_ollama_available
        if not is_ollama_available():
            return jsonify({"error": "Ollama is not reachable at localhost:11434"}), 503
        items = extract_listings(html, query, site)
        return jsonify({"query": query, "site": site, "count": len(items), "listings": items})
    except ImportError:
        return jsonify({"error": "ollama_client module not found"}), 503


@app.route("/v1/llm/clean", methods=["POST"])
@require_key
def llm_clean():
    """
    POST /v1/llm/clean
    Body: {"text": "...", "hint": "optional context hint"}

    Ask the LLM to clean/normalise raw scraped text.
    """
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    hint = data.get("hint", "")

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    if len(text) > 10_000:
        return jsonify({"error": "'text' exceeds 10,000 character limit"}), 400

    try:
        from ollama_client import clean_text, is_ollama_available
        if not is_ollama_available():
            return jsonify({"error": "Ollama is not reachable at localhost:11434"}), 503
        cleaned = clean_text(text, hint=hint)
        return jsonify({"original_length": len(text), "cleaned_length": len(cleaned), "text": cleaned})
    except ImportError:
        return jsonify({"error": "ollama_client module not found"}), 503


# ---------------------------------------------------------------------------
# Profile endpoint
# ---------------------------------------------------------------------------

@app.route("/v1/profile", methods=["GET"])
@require_key
def profile():
    """GET /v1/profile — return user profile stats and top search topics."""
    p = get_profile()
    stats = p.get_stats()
    top_topics = p.get_top_topics(10)
    return jsonify({
        "stats": stats,
        "top_search_topics": [{"topic": t, "weight": round(w, 3)} for t, w in top_topics],
        "service": "web-scraper",
    })


# ---------------------------------------------------------------------------
# MCP tool server endpoints
# ---------------------------------------------------------------------------

@app.route("/tools", methods=["GET"])
def list_tools():
    """GET /tools — MCP protocol: list available tools."""
    return jsonify({"tools": TOOLS})


@app.route("/v1/tools/call", methods=["POST"])
@require_key
def tools_call():
    """POST /v1/tools/call — execute any tool from tool_registry (MCP tool server)."""
    data = request.get_json(force=True, silent=True) or {}
    tool_name = data.get("name", "").strip()
    tool_args = data.get("args") or data.get("arguments") or {}
    if not tool_name:
        return jsonify({"error": "Missing 'name' field"}), 400
    result = execute_tool(tool_name, tool_args)
    return jsonify({"tool": tool_name, "result": result})


# ---------------------------------------------------------------------------
# Fetch endpoint (used by tool_registry._tool_fetch_url)
# ---------------------------------------------------------------------------

@app.route("/v1/fetch", methods=["POST"])
@require_key
def fetch_url_endpoint():
    """POST /v1/fetch — fetch a URL and return extracted content."""
    import re
    from bs4 import BeautifulSoup
    import requests as _req

    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url", "").strip()
    extract = data.get("extract", "text")

    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            )
        }
        r = _req.get(url, headers=headers, timeout=15, allow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        if extract == "links":
            links = [a.get("href", "") for a in soup.find_all("a", href=True)]
            content = "\n".join(l for l in links if l)[:4000]
        elif extract == "tables":
            tables = []
            for tbl in soup.find_all("table"):
                tables.append(tbl.get_text(separator="\t", strip=True))
            content = "\n\n".join(tables)[:4000]
        else:
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            content = re.sub(r"\n{3,}", "\n\n", text)[:4000]

        return jsonify({"url": url, "extract": extract, "content": content})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

@app.route("/v1/keys", methods=["POST"])
def create_key():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "default")
    key = gen_key()
    db = get_db()
    db.execute("INSERT INTO api_keys (key, name, created_at) VALUES (?,?,?)", (key, name, time.time()))
    db.commit()
    db.close()
    return jsonify({"api_key": key, "name": name}), 201


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
attach_stats_middleware(app, 'web-scraper')

if __name__ == "__main__":
    init_db()

    db = get_db()
    existing = db.execute("SELECT key FROM api_keys LIMIT 1").fetchone()
    if not existing:
        key = gen_key()
        db.execute("INSERT INTO api_keys (key, name, created_at) VALUES (?,?,?)", (key, "default", time.time()))
        db.commit()
        print(f"\n  Your API key: {key}\n")
    else:
        keys = db.execute("SELECT key, name FROM api_keys").fetchall()
        print(f"\n  Existing API keys:")
        for k in keys:
            print(f"    {k['key']}  ({k['name']})")
        print()
    db.close()

    # Print Ollama status
    try:
        from ollama_client import is_ollama_available, _resolve_model
        if is_ollama_available():
            print(f"  Ollama online — active model: {_resolve_model()}")
        else:
            print("  Ollama not reachable (smart mode will be disabled at runtime)")
    except ImportError:
        print("  ollama_client not available")

    print("  Scraper API v2.0 running on http://localhost:8000")
    print("  Docs: GET /v1/search?q=QUERY&sites=ebay,craigslist&pages=1&smart=1")
    print()
    app.run(port=8000, debug=True)
