#!/usr/bin/env python3
"""
High-performance async web scraper for product listings across multiple sites.

Usage:
    python3 scraper.py "search query"
    python3 scraper.py "laptop" --sites ebay,craigslist,bonanza --pages 3
    python3 scraper.py "guitar" --out results.json
    python3 scraper.py "iphone" --out results.csv

Supported sites: ebay, craigslist, bonanza, amazon, etsy, walmart, aliexpress
"""

import json
import csv
import sys
import time
import random
import argparse
import re
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup

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

# ---------------------------------------------------------------------------
# Rotating user-agents
# ---------------------------------------------------------------------------

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:132.0) Gecko/20100101 Firefox/132.0",
]


EBAY_DOMAINS = [
    "www.ebay.com",
    "www.ebay.co.uk",
    "www.ebay.com.au",
    "www.ebay.de",
    "www.ebay.ca",
]


def get_session():
    s = requests.Session()
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


def fetch(session: requests.Session, url: str) -> Optional[str]:
    """Fetch a URL with retry logic and exponential backoff for retryable errors."""
    retryable_status = {429, 503}
    delays = [1, 2]  # seconds between retries (up to 2 retries after the initial attempt)
    for attempt in range(3):
        try:
            session.headers["User-Agent"] = random.choice(USER_AGENTS)
            r = session.get(url, timeout=30, allow_redirects=True)
            if r.status_code == 200:
                # Skip challenge/captcha pages (eBay anti-bot)
                if len(r.text) < 30000 and "Challenge" in r.text:
                    return None
                return r.text
            if r.status_code in retryable_status and attempt < 2:
                time.sleep(delays[attempt])
                continue
            return None
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < 2:
                time.sleep(delays[attempt])
                continue
            return None
        except requests.RequestException:
            return None
    return None


def deduplicate_listings(listings: list) -> list:
    """Remove duplicate listings by URL (falls back to title if URL is empty)."""
    seen = set()
    unique = []
    for listing in listings:
        key = listing.url if listing.url else listing.title
        if key not in seen:
            seen.add(key)
            unique.append(listing)
    return unique


def parse_price(raw: str) -> Optional[float]:
    """Normalize a raw price string to a float, or None if unparseable/free/$0."""
    if not raw:
        return None
    raw = raw.strip()
    # Reject free / $0
    if re.match(r"(?i)^free$", raw):
        return None
    # Strip currency symbols and words, keep digits, commas, dots
    cleaned = re.sub(r"[^\d.,]", "", raw.replace(",", ""))
    # Handle "USD 1234" or "1234.56 USD" patterns
    match = re.search(r"(\d+\.?\d*)", cleaned)
    if not match:
        return None
    try:
        val = float(match.group(1))
        return None if val == 0.0 else val
    except ValueError:
        return None


def is_valid_listing_url(url: str) -> bool:
    """Return True if the URL is parseable and has http/https scheme."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Site parsers
# ---------------------------------------------------------------------------

# ---- eBay ----

def ebay_urls(query: str, pages: int) -> list[str]:
    """Generate eBay search URLs across multiple regional domains for reliability."""
    urls = []
    for domain in EBAY_DOMAINS:
        for p in range(1, pages + 1):
            urls.append(f"https://{domain}/sch/i.html?_nkw={quote_plus(query)}&_pgn={p}")
    return urls


def parse_ebay(html: str) -> list[Listing]:
    soup = BeautifulSoup(html, "lxml")
    listings = []
    seen_ids = set()

    # New eBay layout (2025+): uses .s-card with .s-card__title, .s-card__price
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

        # Title
        title_el = card.select_one(".s-card__title")
        title = title_el.get_text(strip=True) if title_el else None
        if not title:
            img = card.select_one("img[alt]")
            title = img.get("alt", "") if img else ""
        if not title or title.lower() == "shop on ebay":
            continue

        # Price
        price_el = card.select_one(".s-card__price")
        price = price_el.get_text(strip=True) if price_el else "N/A"

        # Image
        img_el = card.select_one('img[src*="ebayimg"]')
        image = img_el.get("src") if img_el else None

        # Condition
        subtitle_el = card.select_one(".s-card__subtitle")
        condition = subtitle_el.get_text(strip=True) if subtitle_el else None

        # Shipping
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

    # Legacy eBay layout fallback: li.s-item
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
    soup = BeautifulSoup(html, "lxml")
    listings = []
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
    soup = BeautifulSoup(html, "lxml")
    listings = []
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
            if href.startswith("/"):
                link = "https://www.bonanza.com" + href
            else:
                link = href

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
    soup = BeautifulSoup(html, "lxml")
    listings = []
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

        listings.append(Listing(
            title=title, price=price, url=url, image=image, site="amazon",
        ))
    return listings


# ---- Etsy ----

def etsy_urls(query: str, pages: int) -> list[str]:
    return [f"https://www.etsy.com/search?q={quote_plus(query)}&page={p}" for p in range(1, pages + 1)]


def parse_etsy(html: str) -> list[Listing]:
    soup = BeautifulSoup(html, "lxml")
    listings = []

    # Try JSON-LD first
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

    # Fallback HTML
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
    soup = BeautifulSoup(html, "lxml")
    listings = []

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
    soup = BeautifulSoup(html, "lxml")
    listings = []

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

SITES = {
    "ebay":        (ebay_urls, parse_ebay),
    "craigslist":  (craigslist_urls, parse_craigslist),
    "bonanza":     (bonanza_urls, parse_bonanza),
    "amazon":      (amazon_urls, parse_amazon),
    "etsy":        (etsy_urls, parse_etsy),
    "walmart":     (walmart_urls, parse_walmart),
    "aliexpress":  (aliexpress_urls, parse_aliexpress),
}

# ---------------------------------------------------------------------------
# Scraper engine — thread pool for max throughput
# ---------------------------------------------------------------------------

def scrape_url(args):
    """Scrape a single URL. Called from thread pool."""
    session, url, parse_fn, site_name = args
    html = fetch(session, url)
    if not html:
        return []
    try:
        return parse_fn(html)
    except Exception as e:
        print(f"  [{site_name}] parse error on {url}: {e}", file=sys.stderr)
        return []


def scrape_all(query: str, sites: list[str], pages: int, max_workers: int = 30) -> list[Listing]:
    """Scrape all sites concurrently using a thread pool."""
    # Build all (url, parser) pairs
    work = []
    session = get_session()
    for site_name in sites:
        url_fn, parse_fn = SITES[site_name]
        urls = url_fn(query, pages)
        for url in urls:
            work.append((session, url, parse_fn, site_name))

    # Dynamic sizing: scale workers to workload, cap at max_workers
    effective_workers = min(max_workers, max(1, len(sites) * 2))
    print(f"  Fetching {len(work)} URLs across {len(sites)} sites with {effective_workers} threads...")

    all_listings = []
    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(scrape_url, item): item for item in work}
        done, not_done = concurrent.futures.wait(futures, timeout=30)
        # Cancel any futures that didn't finish within the timeout
        for f in not_done:
            f.cancel()
        for f in done:
            try:
                all_listings.extend(f.result())
            except Exception:
                pass

    # Filter listings with malformed URLs before deduplication
    all_listings = [l for l in all_listings if is_valid_listing_url(l.url)]

    return deduplicate_listings(all_listings)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(listings: list[Listing]):
    if not listings:
        print("\nNo results found.")
        return

    print(f"\n{'#':<5} {'SITE':<12} {'PRICE':<14} {'TITLE':<60} {'LOCATION':<20}")
    print("-" * 113)
    for i, l in enumerate(listings, 1):
        title = (l.title[:57] + "...") if len(l.title) > 60 else l.title
        price = (l.price[:12]) if l.price else "N/A"
        loc = (l.location or l.condition or "")[:18]
        print(f"{i:<5} {l.site:<12} {price:<14} {title:<60} {loc:<20}")
        if i >= 200:
            print(f"  ... and {len(listings) - 200} more")
            break


def save_json(listings: list[Listing], path: str):
    with open(path, "w") as f:
        json.dump([asdict(l) for l in listings], f, indent=2)
    print(f"Saved {len(listings)} listings to {path}")


def save_csv(listings: list[Listing], path: str):
    if not listings:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(listings[0]).keys())
        writer.writeheader()
        for l in listings:
            writer.writerow(asdict(l))
    print(f"Saved {len(listings)} listings to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="High-performance web scraper for product listings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available sites: {', '.join(SITES.keys())}",
    )
    parser.add_argument("query", help="Search query (e.g. 'mechanical keyboard')")
    parser.add_argument("--sites", default="craigslist,bonanza,ebay",
                        help="Comma-separated sites (default: craigslist,bonanza,ebay)")
    parser.add_argument("--pages", type=int, default=1, help="Pages per site (default: 1)")
    parser.add_argument("--out", help="Output file path (.json or .csv)")
    parser.add_argument("--workers", type=int, default=30, help="Max concurrent threads (default: 30)")
    args = parser.parse_args()

    sites = [s.strip().lower() for s in args.sites.split(",")]
    invalid = [s for s in sites if s not in SITES]
    if invalid:
        print(f"Unknown sites: {', '.join(invalid)}")
        print(f"Available: {', '.join(SITES.keys())}")
        sys.exit(1)

    print(f'Scraping {", ".join(sites)} for "{args.query}" ({args.pages} page(s) each)...')
    t0 = time.time()

    listings = scrape_all(args.query, sites, args.pages, args.workers)

    elapsed = time.time() - t0

    # Stats per site
    site_counts = {}
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
