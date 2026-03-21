# mongo_upload.py
# Uploads the cleaned listings data to MongoDB, enforcing a final region check.

import json
from pymongo import MongoClient, errors

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_DATA_PATH = "../data/listings_cleaned.json"

# ── MongoDB Config ────────────────────────────────────────────────────────────
MONGO_URI   = "mongodb://eu:Gm8WQhwE@db.csd.auth.gr:27117/?authSource=admin"
DB_NAME     = "eu"
COLLECTION  = "219_229_220_collection"

# ── Allowed Regions ───────────────────────────────────────────────────────────
ALLOWED_REGIONS = {"Kalamaria", "Panorama", "Neapoli-Sikies"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return client[DB_NAME][COLLECTION]

def load_clean_data(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ── Main ──────────────────────────────────────────────────────────────────────
def upload(listings: list, collection):
    """
    Upserts listings into MongoDB only if they belong to the 3 target regions.
    """
    inserted   = 0
    updated    = 0
    failed     = 0
    skipped    = 0

    for listing in listings:
        # ── Region Filter (Hard Check) ──
        # If the region key is missing or not in our allowed set, skip it entirely.
        region = listing.get("region")
        if region not in ALLOWED_REGIONS:
            skipped += 1
            continue

        try:
            # Match on 'url' as the unique identifier
            result = collection.update_one(
                {"url": listing["url"]},   
                {"$set": listing},         
                upsert=True                
            )
            
            if result.upserted_id:
                inserted += 1
            else:
                updated += 1
                
        except errors.PyMongoError as e:
            print(f"  ERROR on {listing.get('url', 'Unknown URL')}: {e}")
            failed += 1

    return inserted, updated, failed, skipped


def main():
    print("Loading cleaned data...")
    try:
        listings = load_clean_data(CLEAN_DATA_PATH)
        print(f"  Loaded {len(listings)} listings from JSON.")
    except FileNotFoundError:
        print(f"  Error: {CLEAN_DATA_PATH} not found. Run preprocessing first.")
        return

    print("\nConnecting to MongoDB...")
    try:
        collection = get_collection()
        # Trigger server_info to confirm connection is active
        collection.database.client.server_info()
        print(f"  Connected to {DB_NAME}.{COLLECTION} ✓")
    except errors.PyMongoError as e:
        print(f"  Connection failed: {e}")
        return

    print(f"\nProcessing upload for {len(listings)} candidates...")
    inserted, updated, failed, skipped = upload(listings, collection)

    print("\n── Upload Summary ──")
    print(f"  ✅ Inserted : {inserted}")
    print(f"  🔄 Updated  : {updated}")
    print(f"  ⚠️  Skipped  : {skipped} (Wrong region/Filtered)")
    print(f"  ❌ Failed   : {failed}")
    
    # ── Final Count ──
    total_in_db = collection.count_documents({})
    print(f"\n  Total documents now in MongoDB: {total_in_db}")


if __name__ == "__main__":
    main()