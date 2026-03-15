# extract_features.py
# Extracts all 9 features from a listing page
# by parsing embedded JSON and rendered DOM elements

import json
import re
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

NIGHTS = 5


def get_url_with_dates(url: str, start_day_offset: int = 60) -> str:
    """Add check-in/check-out dates to URL to force price display."""
    checkin  = (datetime.now() + timedelta(days=start_day_offset)).strftime("%Y-%m-%d")
    checkout = (datetime.now() + timedelta(days=start_day_offset + NIGHTS)).strftime("%Y-%m-%d")
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}check_in={checkin}&check_out={checkout}"


def try_extract_price(source: str) -> float | None:
    """Try to extract per-night price from page source."""
    # Match prices with optional comma thousands separator e.g. €&nbsp;1,234 or €&nbsp;234
    price_match = re.search(r'€&nbsp;([\d,]+)</span><spa', source)
    if price_match:
        raw_price = round(int(price_match.group(1).replace(",", "")) / NIGHTS, 2)
        if raw_price >= 10:
            return raw_price

    price_match2 = re.search(r'€&nbsp;([\d,]+)', source)
    if price_match2:
        raw_price = round(int(price_match2.group(1).replace(",", "")) / NIGHTS, 2)
        if raw_price >= 10:
            return raw_price

    return None


def extract_features(driver, url: str, region: str) -> dict:
    """
    Visits a listing URL and extracts all 9 required features.
    Returns a dict with all fields, or None if page failed to load.
    """
    result = {
        "url": url,
        "region": region,
        "price_per_night": None,
        "guests": None,
        "beds": None,
        "bedrooms": None,
        "baths": None,
        "is_superhost": False,
        "is_guest_favourite": False,
        "review_index": None,
        "num_reviews": None,
        "host_name": None,
        "characteristics": [],
        "latitude": None,
        "longitude": None,
    }

    try:
        # ── Try up to 5 month windows to find an available price ──
        source = None
        for attempt in range(5):
            # Each attempt moves 1 month further into the future
            # Attempt 0 = 60 days out, 1 = 90 days, 2 = 120 days, etc.
            start_offset = 60 + (attempt * 30)
            attempt_url  = get_url_with_dates(url, start_offset)

            driver.get(attempt_url)

            # Wait for page structure
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'script[type="application/ld+json"]')
                )
            )

            # Wait for price to render dynamically
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, '[data-testid="book-it-default"]')
                    )
                )
                time.sleep(2)
            except Exception:
                time.sleep(4)

            source = driver.page_source
            price  = try_extract_price(source)

            if price is not None:
                result["price_per_night"] = price
                break  # price found — no need to try more months

            # No price found — try next month window
            if attempt < 4:
                print(f"  No price at +{start_offset} days, trying +{start_offset + 30} days...")
                time.sleep(1)

        # ── Parse the final loaded page for all other features ──
        if source is None:
            return None

        soup = BeautifulSoup(source, "html.parser")

        # ── 1. LD+JSON — latitude, longitude, rating, reviews ──
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if data.get("@type") == "VacationRental":
                    result["latitude"]     = data.get("latitude")
                    result["longitude"]    = data.get("longitude")
                    rating = data.get("aggregateRating", {})
                    result["review_index"] = rating.get("ratingValue")
                    result["num_reviews"]  = rating.get("ratingCount")
            except Exception:
                continue

        # ── 2. Deferred JSON blob — all other features ──
        deferred = soup.find("script", id="data-deferred-state-0")
        if deferred:
            raw = json.loads(deferred.string)

            # ── Sections array ──
            try:
                sections = (
                    raw["niobeClientData"][0][1]["data"]
                    ["presentation"]["stayProductDetailPage"]
                    ["sections"]["sections"]
                )
                for section in sections:
                    s_id = section.get("sectionId", "")
                    sec  = section.get("section") or {}

                    # Guests, beds, bedrooms, baths
                    if s_id == "OVERVIEW_DEFAULT_V2":
                        for item in sec.get("overviewItems", []):
                            title = item.get("title", "").lower()
                            num   = re.search(r"\d+", title)
                            if not num:
                                continue
                            n = int(num.group())
                            if "guest"    in title: result["guests"]   = n
                            elif "bedroom" in title: result["bedrooms"] = n
                            elif "bed"    in title: result["beds"]     = n
                            elif "bath"   in title: result["baths"]    = n

                    # Guest favourite, review index, num reviews
                    if s_id == "REVIEWS_DEFAULT":
                        result["is_guest_favourite"] = sec.get("isGuestFavorite", False)
                        result["review_index"]       = sec.get("overallRating")
                        result["num_reviews"]        = sec.get("overallCount")

                    # Superhost, characteristics
                    if s_id == "HIGHLIGHTS_DEFAULT":
                        chars = []
                        for h in sec.get("highlights", []):
                            title = h.get("title", "")
                            chars.append(title)
                            if "superhost" in title.lower():
                                result["is_superhost"] = True
                        result["characteristics"] = chars

                    # Host name
                    if s_id == "MEET_YOUR_HOST":
                        result["host_name"] = sec.get("cardData", {}).get("name")

            except (KeyError, IndexError, TypeError) as e:
                print(f"  Warning: sections parse error — {e}")

            # ── sbuiData fallback for guests/beds/bedrooms/baths ──
            try:
                sbui_sections = (
                    raw["niobeClientData"][0][1]["data"]
                    ["presentation"]["stayProductDetailPage"]
                    ["sections"]["sbuiData"]
                    ["sectionConfiguration"]["root"]["sections"]
                )
                for sbui_sec in sbui_sections:
                    if sbui_sec.get("sectionId") == "OVERVIEW_DEFAULT_V2":
                        for item in sbui_sec.get("sectionData", {}).get("overviewItems", []):
                            title = item.get("title", "").lower()
                            num   = re.search(r"\d+", title)
                            if not num:
                                continue
                            n = int(num.group())
                            if "guest"    in title: result["guests"]   = n
                            elif "bedroom" in title: result["bedrooms"] = n
                            elif "bed"    in title: result["beds"]     = n
                            elif "bath"   in title: result["baths"]    = n
            except (KeyError, IndexError, TypeError):
                pass

    except Exception as e:
        print(f"  ERROR on {url}: {e}")
        return None

    return result


if __name__ == "__main__":
    import json
    from test_browser import get_driver

    with open("../json_listings/listing_urls_merged.json") as f:
        data = json.load(f)

    test_url = data["Kalamaria"][0]
    print(f"Testing on: {test_url}\n")

    driver = get_driver()
    result = extract_features(driver, test_url, "Kalamaria")

    print("── Extracted Features ──")
    for k, v in result.items():
        print(f"  {k}: {v}")

    input("\nDone. Press Enter to close...")
    driver.quit()