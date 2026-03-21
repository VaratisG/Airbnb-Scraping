# preprocessing.py
# Clean, Filter, and Validate Thessaloniki AirBnB data for 3 specific regions.
# Removes any listings that fall outside defined geographic "Bounding Boxes".

import json
import os
import re

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH    = "../../Scraping/listings_data.json"
OUTPUT_DATA_PATH = "../data/listings_cleaned.json"

# ── Geographic Constraints (Strict Geofencing) ────────────────────────────────
# Each list follows: [min_latitude, max_latitude, min_longitude, max_longitude]
# These coordinates define the "Allowed Zones" for your project.
STRICT_BOUNDARIES = {
    "Kalamaria":      [40.570, 40.610, 22.920, 22.970],
    "Panorama":       [40.575, 40.610, 22.990, 23.050],
    "Neapoli-Sikies": [40.645, 40.665, 22.935, 22.970],
}

def get_verified_region(lat, lon):
    """
    Checks if a listing's coordinates fall inside one of the three target regions.
    If it's outside all three (e.g., in the City Center), it returns None.
    """
    for region_name, box in STRICT_BOUNDARIES.items():
        if box[0] <= lat <= box[1] and box[2] <= lon <= box[3]:
            return region_name
    return None

# ── Cleaning Helpers ──────────────────────────────────────────────────────────
def clean_characteristics(chars: list) -> list:
    """Standardizes characteristics by removing whitespace and junk tags."""
    cleaned = []
    for c in chars:
        c = c.strip()
        if not c or re.search(r"is a superhost", c, re.IGNORECASE):
            continue
        cleaned.append(c)
    return cleaned

def is_complete(listing: dict) -> bool:
    """Verifies that the listing has all necessary data for analysis/ML."""
    required = ["price_per_night", "review_index", "num_reviews", 
                "guests", "beds", "baths", "latitude", "longitude"]
    
    # Ensure all required keys exist and are not None
    if not all(listing.get(f) is not None for f in required):
        return False
    
    # Filter out 'New' listings or those with very low review counts for quality
    if listing["num_reviews"] < 3:
        return False
        
    return True

def clean_listing(listing: dict) -> dict:
    """Performs type conversion and final field formatting."""
    cleaned = listing.copy()
    
    # Convert and round prices
    cleaned["price_per_night"] = round(float(cleaned["price_per_night"]), 2)
    
    # Handle studios (0 bedrooms)
    if cleaned.get("bedrooms") is None:
        cleaned["bedrooms"] = 0

    # Ensure Booleans are correct types
    cleaned["is_superhost"]       = bool(cleaned.get("is_superhost", False))
    cleaned["is_guest_favourite"] = bool(cleaned.get("is_guest_favourite", False))
    
    # Clean the tags list
    cleaned["characteristics"]    = clean_characteristics(cleaned.get("characteristics") or [])
    
    return cleaned

# ── Main Processing Pipeline ──────────────────────────────────────────────────
def preprocess(raw_data: dict) -> list:
    # 1. Flatten the raw dictionary {Region: [Listings]} into a single list
    all_raw = []
    for region_key, listings in raw_data.items():
        for item in listings:
            all_raw.append(item)

    print(f"Total raw listings extracted from scraper: {len(all_raw)}")

    cleaned_final = []
    dropped_incomplete = 0
    dropped_outside_bounds = 0

    # 2. Iterate and apply filters
    for item in all_raw:
        # Step A: Check for data completeness
        if not is_complete(item):
            dropped_incomplete += 1
            continue
        
        # Step B: Check for physical location (The Spatial Filter)
        # This removes all "City Center" results that leaked into your search
        actual_region = get_verified_region(item["latitude"], item["longitude"])
        
        if actual_region:
            # Step C: Clean the data and overwrite the region with the verified one
            clean_item = clean_listing(item)
            clean_item["region"] = actual_region 
            cleaned_final.append(clean_item)
        else:
            dropped_outside_bounds += 1

    # ── Console Summary Report ──
    print(f"\n── Preprocessing Summary ──")
    print(f"  ❌ Dropped (Incomplete/Low Reviews): {dropped_incomplete}")
    print(f"  ❌ Dropped (Outside Target Boundaries): {dropped_outside_bounds}")
    print(f"  ✅ Final Cleaned Listings Kept: {len(cleaned_final)}")

    print(f"\n── Final Region Distribution ──")
    for reg in STRICT_BOUNDARIES.keys():
        count = sum(1 for l in cleaned_final if l["region"] == reg)
        print(f"  📍 {reg}: {count} listings")

    return cleaned_final

# ── File I/O ──────────────────────────────────────────────────────────────────
def load_raw_data(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_clean_data(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        print("Starting data transformation...")
        raw = load_raw_data(RAW_DATA_PATH)
        
        processed_data = preprocess(raw)
        
        save_clean_data(processed_data, OUTPUT_DATA_PATH)
        print(f"\n🎉 Success! Cleaned file saved to: {OUTPUT_DATA_PATH}")
        
    except FileNotFoundError:
        print(f"Error: Could not find raw file at {RAW_DATA_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")