# preprocessing.py
# Loads raw scraped data, cleans it, handles nulls,
# and saves a clean JSON ready for MongoDB upload

import json
import os
import re

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH    = "../../Scraping/listings_data.json"
OUTPUT_DATA_PATH = "../data/listings_cleaned.json"

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_raw_data(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_clean_data(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Cleaning helpers ──────────────────────────────────────────────────────────
def clean_characteristics(chars: list) -> list:
    """
    Remove characteristics that are just 'X is a Superhost'
    since we already capture that in is_superhost field.
    Strip whitespace from all entries.
    """
    cleaned = []
    for c in chars:
        c = c.strip()
        if not c:
            continue
        if re.search(r"is a superhost", c, re.IGNORECASE):
            continue
        cleaned.append(c)
    return cleaned


def clean_listing(listing: dict, region_avg_price: float, overall_avg_price: float) -> dict:
    """
    Cleans a single listing dict:
    - Fills null bedrooms with 0 (studio)
    - Fills null price with region average, fallback to overall average
    - Fills null review_index with None (no reviews is valid data)
    - Fills null num_reviews with 0
    - Fills null guests/beds/baths with 1 as safe minimum
    - Cleans characteristics list
    - Ensures boolean fields are actual booleans
    - Rounds price to 2 decimals
    """
    cleaned = listing.copy()

    # ── Price ──
    if cleaned.get("price_per_night") is None:
        cleaned["price_per_night"] = round(region_avg_price, 2)
        cleaned["price_imputed"]   = True
    else:
        cleaned["price_per_night"] = round(float(cleaned["price_per_night"]), 2)
        cleaned["price_imputed"]   = False

    # ── Bedrooms — studios have 0 bedrooms ──
    if cleaned.get("bedrooms") is None:
        cleaned["bedrooms"] = 0

    # ── Guests, beds, baths — safe minimum of 1 ──
    for field in ["guests", "beds", "baths"]:
        if cleaned.get(field) is None:
            cleaned[field] = 1

    # ── Reviews ──
    if cleaned.get("num_reviews") is None:
        cleaned["num_reviews"] = 0

    # review_index stays None if no reviews — that's meaningful data

    # ── Booleans ──
    cleaned["is_superhost"]       = bool(cleaned.get("is_superhost", False))
    cleaned["is_guest_favourite"] = bool(cleaned.get("is_guest_favourite", False))

    # ── Characteristics ──
    cleaned["characteristics"] = clean_characteristics(
        cleaned.get("characteristics") or []
    )

    # ── Host name fallback ──
    if not cleaned.get("host_name"):
        cleaned["host_name"] = "Unknown"

    return cleaned


# ── Main ──────────────────────────────────────────────────────────────────────
def preprocess(raw_data: dict) -> list:
    """
    Takes the raw dict {region: [listings]},
    flattens it to a single list,
    computes averages for imputation,
    and cleans every listing.
    """
    # ── Flatten all listings into one list ──
    all_listings = []
    for region, listings in raw_data.items():
        for listing in listings:
            listing["region"] = region  # ensure region is set
            all_listings.append(listing)

    print(f"Total raw listings     : {len(all_listings)}")

    # ── Compute region average prices (excluding nulls) ──
    region_prices = {}
    for listing in all_listings:
        region = listing["region"]
        price  = listing.get("price_per_night")
        if price is not None:
            region_prices.setdefault(region, []).append(float(price))

    region_avg = {
        region: sum(prices) / len(prices)
        for region, prices in region_prices.items()
    }

    # ── Overall average price as fallback ──
    all_prices   = [p for prices in region_prices.values() for p in prices]
    overall_avg  = sum(all_prices) / len(all_prices) if all_prices else 0.0

    print(f"\n── Region average prices ──")
    for region, avg in region_avg.items():
        print(f"  {region}: €{avg:.2f}/night")
    print(f"  Overall : €{overall_avg:.2f}/night")

    # ── Clean every listing ──
    cleaned_listings = []
    for listing in all_listings:
        region      = listing["region"]
        r_avg_price = region_avg.get(region, overall_avg)
        cleaned     = clean_listing(listing, r_avg_price, overall_avg)
        cleaned_listings.append(cleaned)

    # ── Report ──
    null_prices  = sum(1 for l in all_listings if l.get("price_per_night") is None)
    null_reviews = sum(1 for l in all_listings if l.get("review_index") is None)
    null_beds    = sum(1 for l in all_listings if l.get("bedrooms") is None)

    print(f"\n── Null value summary (before cleaning) ──")
    print(f"  Null prices   : {null_prices}")
    print(f"  Null reviews  : {null_reviews}")
    print(f"  Null bedrooms : {null_beds}")

    print(f"\n── Final dataset ──")
    print(f"  Total listings : {len(cleaned_listings)}")
    for region in raw_data:
        count = sum(1 for l in cleaned_listings if l["region"] == region)
        print(f"  {region}: {count} listings")

    return cleaned_listings


if __name__ == "__main__":
    print("Loading raw data...")
    raw_data = load_raw_data(RAW_DATA_PATH)

    print("Preprocessing...\n")
    cleaned = preprocess(raw_data)

    print("\nSaving cleaned data...")
    save_clean_data(cleaned, OUTPUT_DATA_PATH)
    print(f"Saved to {OUTPUT_DATA_PATH}")