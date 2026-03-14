import time
import json
import os
from test_browser import get_driver

# Read from json_listings folder
with open("../json_listings/listing_urls.json", "r") as f:
    data = json.load(f)

test_url = data["Kalamaria"][0]
print(f"Opening: {test_url}")

driver = get_driver()
driver.get(test_url)
time.sleep(4)

# Save HTML to Scraping/ (one level up from src/)
html_path = "../htmls/sample_listing.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(driver.page_source)

print(f"Page source saved to {html_path}")
print("Title:", driver.title)

input("Press Enter to close...")
driver.quit()