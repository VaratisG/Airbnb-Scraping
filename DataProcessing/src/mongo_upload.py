# mongo_upload.py
# Uploads the cleaned listings data to MongoDB

import json
from pymongo import MongoClient, errors

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_DATA_PATH = "../data/listings_cleaned.json"

# ── MongoDB Config ────────────────────────────────────────────────────────────
MONGO_URI   = "mongodb://eu:Gm8WQhwE@db.csd.auth.gr:27117/?authSource=admin"
DB_NAME     = "eu"
COLLECTION  = "EGD_Collection"

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
    Upserts all listings into MongoDB using URL as unique key.
    Safe to run multiple times — won't create duplicates.
    """
    inserted  = 0
    updated   = 0
    failed    = 0

    for listing in listings:
        try:
            result = collection.update_one(
                {"url": listing["url"]},   # match on URL
                {"$set": listing},         # update all fields
                upsert=True                # insert if not exists
            )
            if result.upserted_id:
                inserted += 1
            else:
                updated += 1
        except errors.PyMongoError as e:
            print(f"  ERROR on {listing['url']}: {e}")
            failed += 1

    return inserted, updated, failed


def main():
    print("Loading cleaned data...")
    listings = load_clean_data(CLEAN_DATA_PATH)
    print(f"  Loaded {len(listings)} listings")

    print("\nConnecting to MongoDB...")
    try:
        collection = get_collection()
        collection.database.client.server_info()
        print(f"  Connected to {DB_NAME}.{COLLECTION} ✓")
    except errors.ServerSelectionTimeoutError as e:
        print(f"  Connection failed: {e}")
        return

    print(f"\nUploading {len(listings)} listings...")
    inserted, updated, failed = upload(listings, collection)

    print("\n── Upload Complete ──")
    print(f"  Inserted : {inserted}")
    print(f"  Updated  : {updated}")
    print(f"  Failed   : {failed}")
    print(f"  Total    : {inserted + updated + failed}")

    # ── Verify ──
    count = collection.count_documents({})
    print(f"\n  Documents in MongoDB: {count}")


if __name__ == "__main__":
    main()