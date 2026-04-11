#!/usr/bin/env python3
"""
High-performance async web scraper for product listings across multiple sites.

Usage:
    python3 scraper.py "search query"
    python3 scraper.py "laptop" --sites ebay,craigslist,bonanza --pages 3
    python3 scraper.py "guitar" --out results.json
    python3 scraper.py "iphone" --out results.csv
    python3 scraper.py "iphone" --smart            # LLM-guided extraction + link scoring
    python3 scraper.py "iphone" --clean-text        # LLM text normalisation pass

Supported sites: ebay, craigslist, bonanza, amazon, etsy, walmart, aliexpress
"""

import json
import csv
import sys
import os
import time
import random
import argparse
import re
import hashlib
import heapq
import concurrent.futures
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Optional, Callable
from urllib.parse import quote_plus, urlparse

sys.path.insert(0, os.path.expanduser("~"))
try:
    from user_profile import get_profile as _get_profile
    _PROFILE_ENABLED = True
except Exception:
    _get_profile = None
    _PROFILE_ENABLED = False

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scraper")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Listing:
    title: str
    price: str
    url: str
    image: Optional[str]
    site: str
    condition: Optional[str] = None
    shipping: Optional[str] = None
    seller: Optional[str] = None
    location: Optional[str] = None
    # LLM-enriched fields
    llm_extracted: bool = False
    relevance_score: Optional[float] = None

# ---------------------------------------------------------------------------
# Rotating user-agents
# ---------------------------------------------------------------------------

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
]

EBAY_DOMAINS = [
    "www.ebay.com",
    "www.ebay.co.uk",
    "www.ebay.com.au",
    "www.ebay.de",
    "www.ebay.ca",
]

# ---------------------------------------------------------------------------
# Page cache — keyed by URL sha256, stored on disk as plain JSON files
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".page_cache")
CACHE_TTL_SECONDS = 3600  # 1 hour


def _cache_path(url: str) -> str:
    key = hashlib.sha256(url.encode()).hexdigest()
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.json")


def cache_get(url: str) -> Optional[str]:
    """Return cached HTML for *url* if fresh, else None."""
    path = _cache_path(url)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        if time.time() - entry.get("ts", 0) < CACHE_TTL_SECONDS:
            return entry.get("html")
    except Exception:
        pass
    return None


def cache_set(url: str, html: str) -> None:
    """Persist *html* for *url* to disk cache."""
    path = _cache_path(url)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "html": html}, f)
    except Exception as exc:
        logger.debug("cache_set failed for %s: %s", url, exc)


# ---------------------------------------------------------------------------
# Adaptive rate-limiter per host
# ---------------------------------------------------------------------------

import threading

_rate_lock = threading.Lock()
_host_delay: dict[str, float] = {}   # host -> current minimum delay (seconds)
_host_last_req: dict[str, float] = {}  # host -> epoch of last request

_BASE_DELAY = 0.1        # aggressive default; backs off automatically on 429
_MAX_DELAY  = 45.0       # cap after many back-offs
_BACKOFF_FACTOR = 2.0    # multiply delay on 429/503


def _host_of(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return url


def _wait_for_host(host: str) -> None:
    """Honour per-host rate limit before making a request."""
    with _rate_lock:
        delay = _host_delay.get(host, _BASE_DELAY)
        last  = _host_last_req.get(host, 0.0)
        now   = time.time()
        wait  = delay - (now - last)
        if wait > 0:
            time.sleep(wait)
        _host_last_req[host] = time.time()


def _backoff_host(host: str) -> None:
    """Increase delay for *host* after a 429/503 response."""
    with _rate_lock:
        current = _host_delay.get(host, _BASE_DELAY)
        _host_delay[host] = min(current * _BACKOFF_FACTOR, _MAX_DELAY)
        logger.info("Rate-limited on %s — new delay %.1fs", host, _host_delay[host])


def _reset_host(host: str) -> None:
    """Gradually recover host delay after a successful response."""
    with _rate_lock:
        current = _host_delay.get(host, _BASE_DELAY)
        _host_delay[host] = max(_BASE_DELAY, current * 0.8)

# ---------------------------------------------------------------------------
# HTTP session factory
# ---------------------------------------------------------------------------

def get_session() -> requests.Session:
    """Return a session with connection pooling and retry adapter."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET", "HEAD"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=20,
        pool_maxsize=50,
        pool_block=False,
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    })
    return s


# Thread-local storage — each worker thread gets its own session
_tls = threading.local()

def _thread_session() -> requests.Session:
    """Get or create a session for the current thread."""
    if not hasattr(_tls, "session"):
        _tls.session = get_session()
    return _tls.session


# ---------------------------------------------------------------------------
# Robust fetch — caching + adaptive rate limit + exponential backoff
# ---------------------------------------------------------------------------

def fetch(
    session: requests.Session,
    url: str,
    *,
    use_cache: bool = True,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Fetch *url* and return HTML text.

    - Checks disk cache first (respects CACHE_TTL_SECONDS).
    - Rotates User-Agent on each attempt.
    - Backs off on 429/503 and retries with exponential delay.
    - Handles encoding issues by falling back to apparent charset detection.
    - Returns None on permanent failure.
    """
    if use_cache:
        cached = cache_get(url)
        if cached is not None:
            return cached

    host = _host_of(url)
    retryable = {429, 503, 502, 504}
    base_wait = 1.0

    for attempt in range(max_retries):
        try:
            session.headers["User-Agent"] = random.choice(USER_AGENTS)
            _wait_for_host(host)

            r = session.get(url, timeout=(8, 25), allow_redirects=True)

            if r.status_code == 200:
                # Gracefully handle encoding issues
                try:
                    html = r.text
                except UnicodeDecodeError:
                    r.encoding = r.apparent_encoding or "utf-8"
                    html = r.content.decode(r.encoding, errors="replace")

                # Skip challenge/captcha pages
                if len(html) < 30_000 and re.search(r"\bChallenge\b|\bCAPTCHA\b|\bRobot\b", html):
                    logger.debug("Captcha detected at %s", url)
                    return None

                _reset_host(host)
                if use_cache:
                    cache_set(url, html)
                return html

            if r.status_code in retryable:
                _backoff_host(host)
                wait = base_wait * (2 ** attempt) + random.uniform(0, 1)
                logger.info("HTTP %s from %s — retrying in %.1fs", r.status_code, url, wait)
                time.sleep(wait)
                continue

            # 4xx (not 429) — don't retry
            logger.debug("HTTP %s from %s — giving up", r.status_code, url)
            return None

        except requests.Timeout:
            wait = base_wait * (2 ** attempt)
            logger.warning("Timeout fetching %s — retrying in %.1fs", url, wait)
            time.sleep(wait)

        except requests.ConnectionError as exc:
            wait = base_wait * (2 ** attempt)
            logger.warning("ConnectionError %s: %s — retrying in %.1fs", url, exc, wait)
            time.sleep(wait)

        except Exception as exc:
            logger.error("Unexpected error fetching %s: %s", url, exc)
            return None

    return None


# ---------------------------------------------------------------------------
# JS rendering fallback  (playwright -> requests_html -> plain requests)
# ---------------------------------------------------------------------------

def fetch_js(url: str, timeout: int = 30) -> Optional[str]:
    """
    Attempt to render JavaScript for *url*.

    Priority:
      1. playwright (headless chromium) — best fidelity
      2. requests_html (pyppeteer) — lighter
      3. Plain requests.get() — fallback (no JS)
    """
    # 1) playwright
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=random.choice(USER_AGENTS),
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            html = page.content()
            browser.close()
        return html
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("playwright failed for %s: %s", url, exc)

    # 2) requests_html
    try:
        from requests_html import HTMLSession  # type: ignore
        s = HTMLSession()
        r = s.get(url, timeout=timeout, headers={"User-Agent": random.choice(USER_AGENTS)})
        r.html.render(timeout=timeout, sleep=1)
        return r.html.html
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("requests_html failed for %s: %s", url, exc)

    # 3) Plain requests
    try:
        s = get_session()
        return fetch(s, url)
    except Exception as exc:
        logger.warning("plain fetch failed for %s: %s", url, exc)

    return None


# ---------------------------------------------------------------------------
# Priority queue for URLs
# ---------------------------------------------------------------------------

class PriorityURLQueue:
    """
    Min-heap priority queue for crawl URLs.

    Lower *priority* value = fetched first (0 = highest priority).
    Ties are broken by insertion order.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, str]] = []
        self._counter = 0
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def push(self, url: str, priority: float = 0.5) -> None:
        with self._lock:
            if url in self._seen:
                return
            self._seen.add(url)
            # Invert priority so higher-relevance URLs bubble to the top
            heapq.heappush(self._heap, (1.0 - priority, self._counter, url))
            self._counter += 1

    def pop(self) -> Optional[tuple[str, float]]:
        """Return (url, original_priority) or None if empty."""
        with self._lock:
            if not self._heap:
                return None
            inv_prio, _, url = heapq.heappop(self._heap)
            return url, 1.0 - inv_prio

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def deduplicate_listings(listings: list[Listing]) -> list[Listing]:
    seen: set[str] = set()
    unique: list[Listing] = []
    for listing in listings:
        key = listing.url if listing.url else listing.title
        if key not in seen:
            seen.add(key)
            unique.append(listing)
    return unique


def parse_price(raw: str) -> Optional[float]:
    if not raw:
        return None
    raw = raw.strip()
    if re.match(r"(?i)^free$", raw):
        return None
    cleaned = re.sub(r"[^\d.,]", "", raw.replace(",", ""))
    m = re.search(r"(\d+\.?\d*)", cleaned)
    if not m:
        return None
    try:
        val = float(m.group(1))
        return None if val == 0.0 else val
    except ValueError:
        return None


def is_valid_listing_url(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _safe_parse(html: str, parser: str = "lxml") -> BeautifulSoup:
    """Parse HTML with graceful fallback from lxml to html.parser."""
    try:
        return BeautifulSoup(html, parser)
    except Exception:
        return BeautifulSoup(html, "html.parser")


# ---------------------------------------------------------------------------
# Site parsers
# ---------------------------------------------------------------------------

# ---- eBay ----

def ebay_urls(query: str, pages: int) -> list[str]:
    urls = []
    for domain in EBAY_DOMAINS:
        for p in range(1, pages + 1):
            urls.append(f"https://{domain}/sch/i.html?_nkw={quote_plus(query)}&_pgn={p}")
    return urls


def parse_ebay(html: str) -> list[Listing]:
    soup = _safe_parse(html)
    listings: list[Listing] = []
    seen_ids: set[str] = set()

    # New eBay layout (2025+)
    for card in soup.select(".s-card"):
        link = card.select_one('a.s-card__link[href*="/itm/"]')
        if not link:
            continue
        href = link.get("href", "")
        m = re.search(r"/itm/(\d+)", href)
        if not m or m.group(1) == "123456":
            continue
        item_id = m.group(1)
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        title_el = card.select_one(".s-card__title")
        title = title_el.get_text(strip=True) if title_el else None
        if not title:
            img = card.select_one("img[alt]")
            title = img.get("alt", "") if img else ""
        if not title or title.lower() == "shop on ebay":
            continue

        price_el = card.select_one(".s-card__price")
        price = price_el.get_text(strip=True) if price_el else "N/A"
        img_el = card.select_one('img[src*="ebayimg"]')
        image = img_el.get("src") if img_el else None
        subtitle_el = card.select_one(".s-card__subtitle")
        condition = subtitle_el.get_text(strip=True) if subtitle_el else None
        full_text = card.get_text(" ", strip=True)
        ship_match = re.search(
            r"(Free (?:delivery|postage|shipping)|[+£$€]\d+[\.\d]* (?:delivery|postage|shipping))",
            full_text, re.IGNORECASE,
        )
        shipping = ship_match.group(0) if ship_match else None
        url = href.split("?")[0]

        listings.append(Listing(
            title=title, price=price, url=url, image=image,
            site="ebay", condition=condition, shipping=shipping,
        ))

    # Legacy eBay layout fallback
    if not listings:
        for item in soup.select("li.s-item"):
            title_el = item.select_one(".s-item__title span, .s-item__title")
            price_el = item.select_one(".s-item__price")
            link_el = item.select_one("a.s-item__link")
            img_el = item.select_one("img.s-item__image-img")
            cond_el = item.select_one(".SECONDARY_INFO")
            ship_el = item.select_one(".s-item__shipping, .s-item__freeXDays")

            title = title_el.get_text(strip=True) if title_el else None
            if not title or title.lower() == "shop on ebay":
                continue

            price = price_el.get_text(strip=True) if price_el else "N/A"
            url = link_el["href"].split("?")[0] if link_el and link_el.has_attr("href") else ""
            image = (img_el.get("src") or img_el.get("data-src")) if img_el else None
            condition = cond_el.get_text(strip=True) if cond_el else None
            shipping = ship_el.get_text(strip=True) if ship_el else None

            listings.append(Listing(
                title=title, price=price, url=url, image=image,
                site="ebay", condition=condition, shipping=shipping,
            ))

    return listings


# ---- Craigslist ----

CRAIGSLIST_REGIONS = [
    "sfbay", "newyork", "losangeles", "chicago", "seattle",
    "denver", "austin", "atlanta", "boston", "portland",
]


def craigslist_urls(query: str, pages: int) -> list[str]:
    urls = []
    for region in CRAIGSLIST_REGIONS:
        for p in range(pages):
            offset = p * 120
            url = f"https://{region}.craigslist.org/search/sss?query={quote_plus(query)}"
            if offset > 0:
                url += f"&s={offset}"
            urls.append(url)
    return urls


def parse_craigslist(html: str) -> list[Listing]:
    soup = _safe_parse(html)
    listings: list[Listing] = []
    for item in soup.select("li.cl-static-search-result"):
        title_el = item.select_one(".title")
        price_el = item.select_one(".price")
        link_el = item.select_one("a")
        loc_el = item.select_one(".location")

        title = title_el.get_text(strip=True) if title_el else None
        if not title:
            continue

        listings.append(Listing(
            title=title,
            price=price_el.get_text(strip=True) if price_el else "N/A",
            url=link_el["href"] if link_el and link_el.has_attr("href") else "",
            image=None,
            site="craigslist",
            location=loc_el.get_text(strip=True) if loc_el else None,
        ))
    return listings


# ---- Bonanza ----

def bonanza_urls(query: str, pages: int) -> list[str]:
    urls = []
    for p in range(1, pages + 1):
        url = f"https://www.bonanza.com/items/search?q%5Bsearch_term%5D={quote_plus(query)}"
        if p > 1:
            url += f"&page={p}"
        urls.append(url)
    return urls


def parse_bonanza(html: str) -> list[Listing]:
    soup = _safe_parse(html)
    listings: list[Listing] = []
    for item in soup.select(".search_result_item"):
        title_el = item.select_one(".item_title a")
        price_el = item.select_one(".item_price")
        img_el = item.select_one(".item_image img[alt]:not(.special_designation_icon)")

        title = title_el.get_text(strip=True) if title_el else None
        if not title:
            continue

        link = ""
        if title_el and title_el.has_attr("href"):
            href = title_el["href"]
            link = f"https://www.bonanza.com{href}" if href.startswith("/") else href

        listings.append(Listing(
            title=title,
            price=price_el.get_text(strip=True) if price_el else "N/A",
            url=link,
            image=img_el.get("src") if img_el else None,
            site="bonanza",
        ))
    return listings


# ---- Amazon ----

def amazon_urls(query: str, pages: int) -> list[str]:
    return [f"https://www.amazon.com/s?k={quote_plus(query)}&page={p}" for p in range(1, pages + 1)]


def parse_amazon(html: str) -> list[Listing]:
    soup = _safe_parse(html)
    listings: list[Listing] = []
    for item in soup.select('[data-component-type="s-search-result"]'):
        title_el = item.select_one("h2 a span, h2 span")
        price_whole = item.select_one(".a-price-whole")
        price_frac = item.select_one(".a-price-fraction")
        link_el = item.select_one("h2 a")
        img_el = item.select_one("img.s-image")

        title = title_el.get_text(strip=True) if title_el else None
        if not title:
            continue

        if price_whole:
            price = "$" + price_whole.get_text(strip=True).rstrip(".")
            if price_frac:
                price += "." + price_frac.get_text(strip=True)
        else:
            price = "N/A"

        url = "https://www.amazon.com" + link_el["href"] if link_el and link_el.has_attr("href") else ""
        image = img_el.get("src") if img_el else None

        listings.append(Listing(title=title, price=price, url=url, image=image, site="amazon"))
    return listings


# ---- Etsy ----

def etsy_urls(query: str, pages: int) -> list[str]:
    return [f"https://www.etsy.com/search?q={quote_plus(query)}&page={p}" for p in range(1, pages + 1)]


def parse_etsy(html: str) -> list[Listing]:
    soup = _safe_parse(html)
    listings: list[Listing] = []

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, dict) and data.get("@type") == "ItemList":
                for li in data.get("itemListElement", []):
                    item = li.get("item", li)
                    price_val = item.get("offers", {}).get("price", "N/A")
                    listings.append(Listing(
                        title=item.get("name", ""),
                        price=f"${price_val}" if price_val != "N/A" else "N/A",
                        url=item.get("url", ""),
                        image=item.get("image", None),
                        site="etsy",
                    ))
        except (json.JSONDecodeError, TypeError):
            pass

    if not listings:
        for item in soup.select(".v2-listing-card, [data-listing-id]"):
            link_el = item.select_one("a.listing-link, a[data-listing-id], a")
            title_el = item.select_one("h3, .v2-listing-card__title")
            price_el = item.select_one(".currency-value, .lc-price span")

            title = title_el.get_text(strip=True) if title_el else None
            if not title:
                continue
            listings.append(Listing(
                title=title,
                price=price_el.get_text(strip=True) if price_el else "N/A",
                url=link_el["href"] if link_el and link_el.has_attr("href") else "",
                image=None,
                site="etsy",
            ))

    return listings


# ---- Walmart ----

def walmart_urls(query: str, pages: int) -> list[str]:
    return [f"https://www.walmart.com/search?q={quote_plus(query)}&page={p}" for p in range(1, pages + 1)]


def parse_walmart(html: str) -> list[Listing]:
    soup = _safe_parse(html)
    listings: list[Listing] = []

    script = soup.select_one("script#__NEXT_DATA__")
    if script:
        try:
            data = json.loads(script.string or "")
            stacks = (data.get("props", {}).get("pageProps", {})
                      .get("initialData", {}).get("searchResult", {})
                      .get("itemStacks", []))
            for stack in stacks:
                for item in stack.get("items", []):
                    if not item.get("name"):
                        continue
                    price_info = item.get("priceInfo", {}).get("currentPrice", {})
                    price = f"${price_info['price']}" if price_info.get("price") else "N/A"
                    listings.append(Listing(
                        title=item["name"],
                        price=price,
                        url=f"https://www.walmart.com{item.get('canonicalUrl', '')}",
                        image=item.get("image", ""),
                        site="walmart",
                        seller=item.get("sellerName"),
                    ))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    return listings


# ---- AliExpress ----

def aliexpress_urls(query: str, pages: int) -> list[str]:
    return [f"https://www.aliexpress.com/wholesale?SearchText={quote_plus(query)}&page={p}" for p in range(1, pages + 1)]


def parse_aliexpress(html: str) -> list[Listing]:
    soup = _safe_parse(html)
    listings: list[Listing] = []

    for script in soup.find_all("script"):
        text = script.string or ""
        if '"title"' not in text or '"minPrice"' not in text:
            continue
        titles = re.findall(r'"title"\s*:\s*"([^"]+)"', text)
        prices = re.findall(r'"minPrice"\s*:\s*"([^"]+)"', text)
        urls_found = re.findall(r'"productDetailUrl"\s*:\s*"([^"]+)"', text)
        images = re.findall(r'"imgUrl"\s*:\s*"([^"]+)"', text)
        for i in range(min(len(titles), len(prices))):
            listings.append(Listing(
                title=titles[i],
                price=f"${prices[i]}",
                url=urls_found[i] if i < len(urls_found) else "",
                image=images[i] if i < len(images) else None,
                site="aliexpress",
            ))
        if listings:
            break

    return listings


# ---------------------------------------------------------------------------
# Site registry
# ---------------------------------------------------------------------------

SITES: dict[str, tuple[Callable, Callable]] = {
    "ebay":        (ebay_urls, parse_ebay),
    "craigslist":  (craigslist_urls, parse_craigslist),
    "bonanza":     (bonanza_urls, parse_bonanza),
    "amazon":      (amazon_urls, parse_amazon),
    "etsy":        (etsy_urls, parse_etsy),
    "walmart":     (walmart_urls, parse_walmart),
    "aliexpress":  (aliexpress_urls, parse_aliexpress),
}

# ---------------------------------------------------------------------------
# LLM enrichment helpers (lazy import — no hard dep on ollama_client)
# ---------------------------------------------------------------------------

def _llm_available() -> bool:
    try:
        from ollama_client import is_ollama_available
        return is_ollama_available()
    except ImportError:
        return False


def _llm_extract(html: str, query: str, site: str) -> list[Listing]:
    """Call LLM extraction and convert dicts to Listing objects."""
    try:
        from ollama_client import extract_listings
        raw_items = extract_listings(html, query, site)
        results: list[Listing] = []
        for item in raw_items:
            url = item.get("url") or ""
            if not is_valid_listing_url(url):
                url = ""
            results.append(Listing(
                title=str(item.get("title", "")).strip(),
                price=str(item.get("price", "N/A")).strip(),
                url=url,
                image=None,
                site=site,
                condition=item.get("condition"),
                shipping=item.get("shipping"),
                seller=item.get("seller"),
                location=item.get("location"),
                llm_extracted=True,
            ))
        return results
    except Exception as exc:
        logger.warning("LLM extraction failed for %s: %s", site, exc)
        return []


def _llm_score_links(urls: list[str], query: str, page_text: str = "") -> list[float]:
    """Return relevance scores for URLs via LLM."""
    try:
        from ollama_client import score_links
        return score_links(urls, query, context=page_text[:500])
    except Exception as exc:
        logger.warning("LLM link scoring failed: %s", exc)
        return [0.5] * len(urls)


def _llm_clean(text: str, hint: str = "") -> str:
    """LLM text cleaning pass."""
    try:
        from ollama_client import clean_text
        return clean_text(text, hint=hint)
    except Exception as exc:
        logger.warning("LLM clean_text failed: %s", exc)
        return text


# ---------------------------------------------------------------------------
# Core scrape-URL worker
# ---------------------------------------------------------------------------

def scrape_url(args: tuple) -> tuple[list[Listing], Optional[str]]:
    """
    Scrape a single URL. Called from thread pool.

    Returns (listings, html) — html is returned for LLM post-processing.
    """
    session, url, parse_fn, site_name, use_cache = args
    session = _thread_session()  # always use thread-local session
    # Rotate UA per request to reduce fingerprinting
    session.headers["User-Agent"] = random.choice(USER_AGENTS)
    html = fetch(session, url, use_cache=use_cache)
    if not html:
        return [], None
    try:
        return parse_fn(html), html
    except Exception as exc:
        logger.error("[%s] parse error on %s: %s", site_name, url, exc)
        return [], html


# ---------------------------------------------------------------------------
# Priority-queue driven scrape engine
# ---------------------------------------------------------------------------

def scrape_all(
    query: str,
    sites: list[str],
    pages: int,
    max_workers: int = 30,
    *,
    smart: bool = False,
    clean_text_pass: bool = False,
    use_cache: bool = True,
) -> list[Listing]:
    """
    Scrape all sites concurrently using a priority-queue backed thread pool.

    Parameters
    ----------
    smart       : If True and Ollama is reachable, use the LLM to:
                    - Score link relevance before following them
                    - Fall back to LLM extraction when CSS parsers return 0 results
    clean_text_pass : Run LLM text-normalisation on extracted titles/descriptions.
    use_cache   : Read from / write to the disk page cache.
    """
    # Build a priority queue seeded with equal-priority seed URLs
    pq = PriorityURLQueue()
    url_to_meta: dict[str, tuple[Callable, str]] = {}  # url -> (parse_fn, site_name)

    for site_name in sites:
        url_fn, parse_fn = SITES[site_name]
        seed_urls = url_fn(query, pages)

        if smart and _llm_available() and len(seed_urls) > 1:
            logger.info("LLM scoring %d seed URLs for '%s' on %s", len(seed_urls), query, site_name)
            scores = _llm_score_links(seed_urls, query)
        else:
            scores = [0.5] * len(seed_urls)

        for url, score in zip(seed_urls, scores):
            pq.push(url, priority=score)
            url_to_meta[url] = (parse_fn, site_name)

    effective_workers = min(max_workers, max(len(sites) * 4, 16))
    logger.info(
        "Fetching %d URLs across %d sites with %d threads (smart=%s, cache=%s)",
        len(pq), len(sites), effective_workers, smart, use_cache,
    )

    # Drain priority queue into a list (respects priority ordering)
    # Pass None as the session placeholder — scrape_url always uses _thread_session()
    work_items: list[tuple] = []
    while True:
        entry = pq.pop()
        if entry is None:
            break
        url, _ = entry
        parse_fn, site_name = url_to_meta[url]
        work_items.append((None, url, parse_fn, site_name, use_cache))

    all_listings: list[Listing] = []
    html_by_site: dict[str, list[str]] = {s: [] for s in sites}

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(scrape_url, item): item for item in work_items}
        done, not_done = concurrent.futures.wait(futures, timeout=60)
        for f in not_done:
            f.cancel()
        for f in done:
            try:
                listings, html = f.result()
                all_listings.extend(listings)
                # Collect HTML for LLM fallback
                if smart and html:
                    item = futures[f]
                    site_name = item[3]
                    html_by_site[site_name].append(html)
            except Exception:
                pass

    # LLM extraction fallback: if a site returned no listings, try LLM
    if smart and _llm_available():
        sites_with_results = {l.site for l in all_listings}
        for site_name in sites:
            if site_name not in sites_with_results and html_by_site.get(site_name):
                logger.info("CSS parser found 0 results for %s — trying LLM extraction", site_name)
                for html in html_by_site[site_name][:2]:   # try first 2 pages
                    llm_listings = _llm_extract(html, query, site_name)
                    if llm_listings:
                        logger.info("LLM extracted %d listings from %s", len(llm_listings), site_name)
                        all_listings.extend(llm_listings)
                        break

    # LLM text-cleaning pass on titles
    if clean_text_pass and _llm_available() and all_listings:
        logger.info("Running LLM text-cleaning pass on %d listing titles...", len(all_listings))
        for listing in all_listings:
            if listing.title and len(listing.title) > 30:
                listing.title = _llm_clean(listing.title, hint=query)

    # Filter malformed URLs then deduplicate
    all_listings = [l for l in all_listings if not l.url or is_valid_listing_url(l.url)]
    deduped = deduplicate_listings(all_listings)

    # Record interaction for personalization
    if _PROFILE_ENABLED and _get_profile is not None:
        try:
            _get_profile().record_interaction(
                query,
                service="web-scraper",
                topics=sites,
            )
        except Exception:
            pass

    return deduped


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(listings: list[Listing]) -> None:
    if not listings:
        print("\nNo results found.")
        return

    print(f"\n{'#':<5} {'SITE':<12} {'PRICE':<14} {'TITLE':<60} {'LOCATION':<20} {'LLM':<5}")
    print("-" * 120)
    for i, l in enumerate(listings, 1):
        title = (l.title[:57] + "...") if len(l.title) > 60 else l.title
        price = (l.price[:12]) if l.price else "N/A"
        loc = (l.location or l.condition or "")[:18]
        llm_flag = "Y" if l.llm_extracted else ""
        print(f"{i:<5} {l.site:<12} {price:<14} {title:<60} {loc:<20} {llm_flag:<5}")
        if i >= 200:
            print(f"  ... and {len(listings) - 200} more")
            break


def save_json(listings: list[Listing], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(l) for l in listings], f, indent=2, ensure_ascii=False)
    print(f"Saved {len(listings)} listings to {path}")


def save_csv(listings: list[Listing], path: str) -> None:
    if not listings:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(listings[0]).keys())
        writer.writeheader()
        for l in listings:
            writer.writerow(asdict(l))
    print(f"Saved {len(listings)} listings to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-performance LLM-guided web scraper for product listings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available sites: {', '.join(SITES.keys())}",
    )
    parser.add_argument("query", help="Search query (e.g. 'mechanical keyboard')")
    parser.add_argument(
        "--sites", default="craigslist,bonanza,ebay",
        help="Comma-separated sites (default: craigslist,bonanza,ebay)",
    )
    parser.add_argument("--pages", type=int, default=1, help="Pages per site (default: 1)")
    parser.add_argument("--out", help="Output file path (.json or .csv)")
    parser.add_argument("--workers", type=int, default=30, help="Max concurrent threads (default: 30)")
    parser.add_argument(
        "--smart", action="store_true",
        help="Enable LLM-guided link scoring and extraction fallback (requires Ollama)",
    )
    parser.add_argument(
        "--clean-text", action="store_true",
        help="Run LLM text-normalisation pass on extracted titles",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable disk page cache (always fetch fresh)",
    )
    parser.add_argument(
        "--js", action="store_true",
        help="Use JS rendering for a single URL (diagnostic mode: pass a URL as query)",
    )
    args = parser.parse_args()

    # --js mode: render a single URL and print the HTML length (diagnostic)
    if args.js:
        print(f"Rendering {args.query} with JS support...")
        html = fetch_js(args.query)
        if html:
            print(f"Got {len(html)} bytes of rendered HTML")
        else:
            print("Failed to render URL")
        return

    sites = [s.strip().lower() for s in args.sites.split(",")]
    invalid = [s for s in sites if s not in SITES]
    if invalid:
        print(f"Unknown sites: {', '.join(invalid)}")
        print(f"Available: {', '.join(SITES.keys())}")
        sys.exit(1)

    if args.smart:
        if _llm_available():
            print("  LLM (Ollama) is online — smart mode active")
        else:
            print("  WARNING: --smart requested but Ollama is unreachable at localhost:11434")
            print("  Continuing without LLM features.")

    print(f'Scraping {", ".join(sites)} for "{args.query}" ({args.pages} page(s) each)...')
    t0 = time.time()

    listings = scrape_all(
        args.query,
        sites,
        args.pages,
        args.workers,
        smart=args.smart,
        clean_text_pass=args.clean_text,
        use_cache=not args.no_cache,
    )

    elapsed = time.time() - t0

    site_counts: dict[str, int] = {}
    for l in listings:
        site_counts[l.site] = site_counts.get(l.site, 0) + 1

    print(f"\nDone in {elapsed:.2f}s - {len(listings)} unique listings")
    for site, count in sorted(site_counts.items()):
        print(f"  {site}: {count}")

    print_table(listings)

    if args.out:
        if args.out.endswith(".csv"):
            save_csv(listings, args.out)
        else:
            save_json(listings, args.out)


if __name__ == "__main__":
    main()
