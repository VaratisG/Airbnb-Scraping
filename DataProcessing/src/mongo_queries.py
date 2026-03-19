# mongo_queries.py
# Runs required MongoDB queries against the teamB collection

from pymongo import MongoClient
import json

# ── MongoDB Config ────────────────────────────────────────────────────────────
MONGO_URI  = "mongodb://eu:Gm8WQhwE@db.csd.auth.gr:27117/?authSource=admin"
DB_NAME    = "eu"
COLLECTION = "219_229_220_collection"

# ── Connection ────────────────────────────────────────────────────────────────
def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return client[DB_NAME][COLLECTION]

# ── Pretty printer ────────────────────────────────────────────────────────────
def print_results(title: str, results: list):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    if not results:
        print("  No results found.")
        return
    for r in results:
        # Remove MongoDB _id for clean printing
        r.pop("_id", None)
        print(json.dumps(r, indent=2, ensure_ascii=False))

# ── Query 1: Top 10 rated listings per region ─────────────────────────────────
def top_rated_per_region(col):
    results = []
    for region in ["Kalamaria", "Panorama", "Neapoli-Sikies"]:
        top = list(col.find(
            {
                "region": region,
                "review_index": {"$ne": None},
                "num_reviews":  {"$gte": 3}
            },
            {
                "url": 1, "region": 1, "host_name": 1,
                "review_index": 1, "num_reviews": 1,
                "price_per_night": 1, "_id": 0
            }
        ).sort([
            ("review_index", -1),
            ("num_reviews",  -1)
        ]).limit(5))
        results.extend(top)
    print_results("Top 5 Rated Listings Per Region", results)


# ── Query 2: Average price per region ────────────────────────────────────────
def average_price_per_region(col):
    pipeline = [
        {"$match": {"price_per_night": {"$ne": None}}},
        {"$group": {
            "_id":           "$region",
            "avg_price":     {"$avg": "$price_per_night"},
            "min_price":     {"$min": "$price_per_night"},
            "max_price":     {"$max": "$price_per_night"},
            "total_listings": {"$sum": 1}
        }},
        {"$sort": {"avg_price": -1}}
    ]
    results = list(col.aggregate(pipeline))
    print(f"\n{'─' * 60}")
    print(f"  Average Price Per Region")
    print(f"{'─' * 60}")
    for r in results:
        print(
            f"  {r['_id']:<20} "
            f"Avg: €{r['avg_price']:.2f}  "
            f"Min: €{r['min_price']:.2f}  "
            f"Max: €{r['max_price']:.2f}  "
            f"({r['total_listings']} listings)"
        )


# ── Query 3: Superhost listings ───────────────────────────────────────────────
def superhost_listings(col):
    pipeline = [
        {"$match": {"is_superhost": True}},
        {"$group": {
            "_id":   "$region",
            "count": {"$sum": 1},
            "avg_price": {"$avg": "$price_per_night"},
            "avg_rating": {"$avg": "$review_index"}
        }},
        {"$sort": {"count": -1}}
    ]
    results = list(col.aggregate(pipeline))
    print(f"\n{'─' * 60}")
    print(f"  Superhost Listings Per Region")
    print(f"{'─' * 60}")
    for r in results:
        print(
            f"  {r['_id']:<20} "
            f"Count: {r['count']}  "
            f"Avg Price: €{r['avg_price']:.2f}  "
            f"Avg Rating: {r['avg_rating']:.2f}"
        )


# ── Query 4: Guest favourite listings ────────────────────────────────────────
def guest_favourite_listings(col):
    pipeline = [
        {"$match": {"is_guest_favourite": True}},
        {"$group": {
            "_id":        "$region",
            "count":      {"$sum": 1},
            "avg_price":  {"$avg": "$price_per_night"},
            "avg_rating": {"$avg": "$review_index"}
        }},
        {"$sort": {"count": -1}}
    ]
    results = list(col.aggregate(pipeline))
    print(f"\n{'─' * 60}")
    print(f"  Guest Favourite Listings Per Region")
    print(f"{'─' * 60}")
    for r in results:
        print(
            f"  {r['_id']:<20} "
            f"Count: {r['count']}  "
            f"Avg Price: €{r['avg_price']:.2f}  "
            f"Avg Rating: {r['avg_rating']:.2f}"
        )


# ── Query 5: Listings by price range ─────────────────────────────────────────
def listings_by_price_range(col):
    pipeline = [
        {"$match": {"price_per_night": {"$ne": None}}},
        {"$bucket": {
            "groupBy": "$price_per_night",
            "boundaries": [0, 50, 100, 150, 200, 300, 500, 99999],
            "default": "Other",
            "output": {
                "count":      {"$sum": 1},
                "avg_rating": {"$avg": "$review_index"},
                "listings":   {"$push": "$url"}
            }
        }}
    ]
    results = list(col.aggregate(pipeline))
    print(f"\n{'─' * 60}")
    print(f"  Listings by Price Range")
    print(f"{'─' * 60}")
    labels = {
        0: "€0-50", 50: "€50-100", 100: "€100-150",
        150: "€150-200", 200: "€200-300", 300: "€300-500",
        500: "€500+"
    }
    for r in results:
        label = labels.get(r["_id"], str(r["_id"]))
        avg_r = r["avg_rating"] or 0
        print(
            f"  {label:<12} "
            f"Count: {r['count']:<5} "
            f"Avg Rating: {avg_r:.2f}"
        )


# ── Query 6: Most common characteristics ─────────────────────────────────────
def most_common_characteristics(col):
    pipeline = [
        {"$unwind": "$characteristics"},
        {"$group": {
            "_id":   "$characteristics",
            "count": {"$sum": 1}
        }},
        {"$sort":  {"count": -1}},
        {"$limit": 15}
    ]
    results = list(col.aggregate(pipeline))
    print(f"\n{'─' * 60}")
    print(f"  Top 15 Most Common Characteristics")
    print(f"{'─' * 60}")
    for r in results:
        print(f"  {r['count']:<5} {r['_id']}")


# ── Query 7: Listings with most reviews ──────────────────────────────────────
def most_reviewed_listings(col):
    results = list(col.find(
        {"num_reviews": {"$gte": 1}},
        {
            "url": 1, "region": 1, "host_name": 1,
            "num_reviews": 1, "review_index": 1,
            "price_per_night": 1, "_id": 0
        }
    ).sort("num_reviews", -1).limit(10))
    print_results("Top 10 Most Reviewed Listings", results)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Connecting to MongoDB...")
    col = get_collection()
    print(f"  Connected — {col.count_documents({})} documents in collection")

    top_rated_per_region(col)
    average_price_per_region(col)
    superhost_listings(col)
    guest_favourite_listings(col)
    listings_by_price_range(col)
    most_common_characteristics(col)
    most_reviewed_listings(col)

    print(f"\n{'─' * 60}")
    print("  All queries completed.")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()