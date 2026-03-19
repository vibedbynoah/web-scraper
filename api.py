#!/usr/bin/env python3
"""
REST API for the web scraper.

Start:  python3 api.py
Then:   curl http://localhost:8000/v1/search?q=laptop -H "Authorization: Bearer sk_scrape_..."
"""

import hashlib
import json
import secrets
import time
import os
import sqlite3
from functools import wraps
from dataclasses import asdict
from flask import Flask, request, jsonify, g
from scraper import scrape_all, SITES, Listing

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "scraper_api.db")
_START_TIME = time.time()
CACHE_TTL = 600        # 10 minutes (increased from 5 for expensive multi-site scrapes)
CACHE_MAX_ENTRIES = 100  # prune oldest after this many rows

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


def _cache_key(query: str, sites: list, pages: int) -> str:
    raw = f"{query}|{'|'.join(sorted(sites))}|{pages}"
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
        (cache_key, json.dumps(data), time.time())
    )
    # Prune oldest entries beyond max
    count = db.execute("SELECT COUNT(*) as c FROM result_cache").fetchone()["c"]
    if count > CACHE_MAX_ENTRIES:
        db.execute(
            "DELETE FROM result_cache WHERE cache_key IN "
            "(SELECT cache_key FROM result_cache ORDER BY created_at ASC LIMIT ?)",
            (count - CACHE_MAX_ENTRIES,)
        )
    db.commit()
    db.close()


def gen_key():
    return f"sk_scrape_{secrets.token_hex(24)}"


# ---------------------------------------------------------------------------
# CORS
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
    return jsonify({
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "total_scrapes_run": total_scrapes,
        "db_size_bytes": db_size,
    })


@app.route("/")
def index():
    return jsonify({
        "name": "Scraper API",
        "version": "1.0",
        "endpoints": {
            "GET /v1/search?q=QUERY": "Search listings (requires API key)",
            "GET /v1/sites": "List available sites (requires API key)",
            "GET /v1/usage": "View your usage stats (requires API key)",
            "POST /v1/keys": "Generate a new API key (no auth needed)",
            "GET /health": "Health check (no auth required)",
        },
        "auth": "Authorization: Bearer sk_scrape_...",
        "example": "GET /v1/search?q=laptop&sites=ebay,craigslist&pages=1&limit=50",
    })


@app.route("/v1/search", methods=["GET"])
@require_key
def search():
    """
    GET /v1/search?q=laptop&sites=ebay,craigslist&pages=1&limit=100
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing required parameter: q", "code": 400}), 400
    if len(query) > 200:
        return jsonify({"error": "Parameter 'q' must not exceed 200 characters", "code": 400}), 400

    sites_param = request.args.get("sites", "ebay,craigslist,bonanza")
    # Deduplicate silently while preserving order
    _seen_sites = set()
    sites = []
    for s in sites_param.split(","):
        s = s.strip().lower()
        if s and s not in _seen_sites:
            _seen_sites.add(s)
            sites.append(s)
    invalid = [s for s in sites if s not in SITES]
    if invalid:
        return jsonify({"error": f"Unknown sites: {', '.join(invalid)}", "available": list(SITES.keys()), "code": 400}), 400

    try:
        pages_raw = int(request.args.get("pages", 1))
    except ValueError:
        return jsonify({"error": "Parameter 'pages' must be an integer", "code": 400}), 400
    if not (1 <= pages_raw <= 5):
        return jsonify({"error": "Parameter 'pages' must be between 1 and 5", "code": 400}), 400
    pages = pages_raw

    limit = min(int(request.args.get("limit", 200)), 1000)
    force_refresh = request.args.get("force", "0") in ("1", "true", "yes")

    # Return cached result if fresh (bypass with ?force=1)
    ck = _cache_key(query, sites, pages)
    if not force_refresh:
        cached = _get_cached(ck)
        if cached:
            cached["cached"] = True
            return jsonify(cached)

    t0 = time.time()
    # scrape_all already deduplicates via deduplicate_listings
    listings = scrape_all(query, sites, pages, max_workers=30)
    elapsed = time.time() - t0

    listings = listings[:limit]

    # Site breakdown
    site_counts = {}
    for l in listings:
        site_counts[l.site] = site_counts.get(l.site, 0) + 1

    result = {
        "query": query,
        "sites": sites,
        "total": len(listings),
        "elapsed_seconds": round(elapsed, 2),
        "site_counts": site_counts,
        "cached": False,
        "data": [asdict(l) for l in listings],
    }

    # Store in cache
    _set_cached(ck, result)

    # Log
    db = get_db()
    db.execute(
        "INSERT INTO search_log (api_key, query, sites, results_count, elapsed, created_at) VALUES (?,?,?,?,?,?)",
        (g.api_key, query, sites_param, len(listings), elapsed, time.time()),
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
        "SELECT query, sites, results_count, elapsed, created_at FROM search_log WHERE api_key=? ORDER BY created_at DESC LIMIT 20",
        (g.api_key,),
    ).fetchall()
    db.close()
    return jsonify({
        "name": key_row["name"],
        "total_requests": key_row["requests_count"],
        "recent_searches": [dict(r) for r in recent],
    })


# ---------------------------------------------------------------------------
# Key management (no auth needed)
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

if __name__ == "__main__":
    init_db()

    # Create a default key if none exist
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

    print("  Scraper API running on http://localhost:8000")
    print("  Docs: GET /v1/search?q=QUERY&sites=ebay,craigslist&pages=1&limit=100")
    print()
    app.run(port=8000, debug=True)
