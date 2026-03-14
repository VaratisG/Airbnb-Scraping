# get_listing_urls.py
# Extracts all listing URLs from a single search results page

import re
import os
import time
import random
import json
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup

def get_listing_urls_from_page(driver) -> list[str]:
    """
    Waits for listing cards to load, then extracts
    all unique /rooms/XXXXXXX URLs from the current page.
    """

    # Wait until at least one listing card link is present
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'a[href*="/rooms/"]')
        )
    )

    # Small extra wait for lazy-loaded cards to finish rendering
    time.sleep(2)

    # Parse the fully rendered page with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Find all anchor tags that link to a /rooms/ page
    anchors = soup.find_all("a", href=re.compile(r"/rooms/\d+"))

    # Build full URLs and deduplicate
    urls = set()
    for a in anchors:
        href = a["href"]
        # Strip query params — we just want the clean room URL
        clean = "https://www.airbnb.com" + href.split("?")[0]
        urls.add(clean)

    return list(urls)

def get_all_listing_urls(driver, region_name: str, url: str) -> list[str]:
    """
    Iterates through all pages for a region and collects
    every listing URL.
    """
    all_urls = set()
    page = 1

    driver.get(url)
    time.sleep(3)

    while True:
        print(f"  [{region_name}] Scraping page {page}...")

        # Get URLs from current page
        page_urls = get_listing_urls_from_page(driver)
        before = len(all_urls)
        all_urls.update(page_urls)
        after = len(all_urls)
        print(f"    Found {len(page_urls)} listings, {after - before} new (total: {after})")

        # Look for the 'Next' pagination button
        try:
            next_btn = driver.find_element(
                By.CSS_SELECTOR, 'a[aria-label="Next"]'
            )
            next_url = next_btn.get_attribute("href")

            if not next_url:
                print("  No more pages (next button disabled).")
                break

            # Navigate to next page
            driver.get(next_url)
            time.sleep(random.uniform(2, 4))  # polite delay
            page += 1

        except NoSuchElementException:
            print("  No more pages (no next button found).")
            break

    return list(all_urls)

if __name__ == "__main__":
    from test_browser import get_driver
    from search_urls import REGIONS

    driver = get_driver()
    all_region_urls = {}

    for region_name, url in REGIONS.items():
        print(f"\n── {region_name} ──")
        urls = get_all_listing_urls(driver, region_name, url)
        all_region_urls[region_name] = urls
        print(f"  TOTAL for {region_name}: {len(urls)} listings")

    print("\n── Grand Total ──")
    total = sum(len(v) for v in all_region_urls.values())
    print(f"  {total} listings across all regions")

    # Save to json_listings folder
    os.makedirs("../json_listings", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../json_listings/listing_urls_run_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(all_region_urls, f, indent=2)

    print(f"\nURLs saved to {filename}")

    input("\nDone. Press Enter to close...")
    driver.quit()