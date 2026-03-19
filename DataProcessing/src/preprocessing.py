# preprocessing.py
# Loads raw scraped data, cleans it, removes incomplete listings,
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


def is_complete(listing: dict) -> tuple[bool, str]:
    """
    Returns (True, '') if listing has all required fields,
    or (False, reason) if it should be dropped.
    """
    # Must have a price
    if listing.get("price_per_night") is None:
        return False, "missing price"

    # Must have at least 3 reviews and a review score
    if listing.get("num_reviews") is None or listing["num_reviews"] < 3:
        return False, f"too few reviews ({listing.get('num_reviews')})"

    if listing.get("review_index") is None:
        return False, "missing review score"

    # Must have core property info
    for field in ["guests", "beds", "baths", "latitude", "longitude"]:
        if listing.get(field) is None:
            return False, f"missing {field}"

    # Must have a host name
    if not listing.get("host_name"):
        return False, "missing host name"

    return True, ""


def clean_listing(listing: dict) -> dict:
    """
    Cleans a single complete listing:
    - Fills null bedrooms with 0 (studios legitimately have 0 bedrooms)
    - Cleans characteristics list
    - Ensures boolean fields are actual booleans
    - Rounds price to 2 decimals
    """
    cleaned = listing.copy()

    # ── Price ──
    cleaned["price_per_night"] = round(float(cleaned["price_per_night"]), 2)

    # ── Bedrooms — studios legitimately have 0 bedrooms ──
    if cleaned.get("bedrooms") is None:
        cleaned["bedrooms"] = 0

    # ── Booleans ──
    cleaned["is_superhost"]       = bool(cleaned.get("is_superhost", False))
    cleaned["is_guest_favourite"] = bool(cleaned.get("is_guest_favourite", False))

    # ── Characteristics ──
    cleaned["characteristics"] = clean_characteristics(
        cleaned.get("characteristics") or []
    )

    return cleaned


# ── Main ──────────────────────────────────────────────────────────────────────
def preprocess(raw_data: dict) -> list:
    """
    Takes the raw dict {region: [listings]},
    flattens to a single list,
    drops incomplete listings,
    and cleans the remaining ones.
    """
    # ── Flatten ──
    all_listings = []
    for region, listings in raw_data.items():
        for listing in listings:
            listing["region"] = region
            all_listings.append(listing)

    print(f"Total raw listings: {len(all_listings)}")

    # ── Filter incomplete listings ──
    complete   = []
    dropped    = []
    drop_reasons = {}

    for listing in all_listings:
        ok, reason = is_complete(listing)
        if ok:
            complete.append(listing)
        else:
            dropped.append(listing)
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    print(f"\n── Dropped {len(dropped)} incomplete listings ──")
    for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    print(f"\n── Kept {len(complete)} complete listings ──")

    # ── Clean the complete listings ──
    cleaned_listings = [clean_listing(l) for l in complete]

    # ── Per region summary ──
    print(f"\n── Final dataset per region ──")
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