# main_scraper.py
# Main scraper loop — visits all listings, extracts features,
# saves results to a local JSON file incrementally

import json
import time
import random
import logging
import os
from test_browser import get_driver
from extract_features import extract_features

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("../scraper.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
URLS_FILE    = "../json_listings/listing_urls_merged.json"
OUTPUT_FILE  = "../json_listings/listings_data.json"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_urls() -> dict:
    with open(URLS_FILE) as f:
        return json.load(f)

def load_existing_data() -> dict:
    """Load already scraped listings so we can resume if interrupted."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_data(data: dict):
    """Save all scraped listings to JSON file."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def already_scraped(data: dict, url: str) -> bool:
    """Check if URL already exists in our output data."""
    for region_listings in data.values():
        for listing in region_listings:
            if listing.get("url") == url:
                return True
    return False

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("Loading listing URLs...")
    all_urls = load_urls()
    total    = sum(len(v) for v in all_urls.values())
    log.info(f"Total listings to scrape: {total}")

    # Load existing data to support resuming
    scraped_data = load_existing_data()
    for region in all_urls:
        if region not in scraped_data:
            scraped_data[region] = []

    already_done = sum(len(v) for v in scraped_data.values())
    log.info(f"Already scraped: {already_done} — resuming from there")

    log.info("Launching browser...")
    driver = get_driver()

    scraped   = 0
    skipped   = 0
    failed    = 0
    listing_n = 0

    try:
        for region, urls in all_urls.items():
            log.info(f"\n── Region: {region} ({len(urls)} listings) ──")

            for url in urls:
                listing_n += 1

                # Skip if already scraped
                if already_scraped(scraped_data, url):
                    skipped += 1
                    log.info(f"  [{listing_n}/{total}] SKIP: {url[-40:]}")
                    continue

                log.info(f"  [{listing_n}/{total}] Scraping: {url[-40:]}")

                result = extract_features(driver, url, region)

                if result is None:
                    failed += 1
                    log.warning(f"    FAILED: {url}")
                    continue

                # Append to region list and save immediately
                scraped_data[region].append(result)
                save_data(scraped_data)
                scraped += 1

                log.info(
                    f"    OK - E{result['price_per_night']}/night | "
                    f"{result['guests']} guests | "
                    f"R{result['review_index']} ({result['num_reviews']} reviews)"
                )

                # Polite delay
                time.sleep(random.uniform(2, 4))

    except KeyboardInterrupt:
        log.info("\nScraping interrupted — progress saved.")

    finally:
        driver.quit()
        log.info("\n── Scraping Complete ──")
        log.info(f"  Scraped : {scraped}")
        log.info(f"  Skipped : {skipped}")
        log.info(f"  Failed  : {failed}")
        log.info(f"  Total   : {listing_n}/{total}")

if __name__ == "__main__":
    main()