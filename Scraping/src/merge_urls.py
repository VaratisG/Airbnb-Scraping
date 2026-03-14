# merge_urls.py
# Run get_listing_urls.py multiple times, then use this to merge all results

import json
import os

def merge_url_files(folder: str) -> dict:
    """
    Takes a list of listing_urls JSON files,
    merges them and removes duplicates per region.
    """
    merged = {}

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".json")
    ]

    if not files:
        print("No JSON files found in folder.")
        return {}

    for filepath in files:
        print(f"  Reading {filepath}...")
        with open(filepath, "r") as f:
            data = json.load(f)

        for region, urls in data.items():
            if region not in merged:
                merged[region] = set()
            merged[region].update(urls)

    # Convert sets back to lists for JSON serialization
    return {region: list(urls) for region, urls in merged.items()}

if __name__ == "__main__":
    # Add as many run files as you have
    folder = "../json_listings"

    print("Merging all JSON files from json_listings/...\n")
    merged = merge_url_files(folder)

    print("\n── Results ──")
    total = 0
    for region, urls in merged.items():
        print(f"  {region}: {len(urls)} unique listings")
        total += len(urls)
    print(f"  TOTAL: {total} listings")

    # Save merged result back into json_listings
    output_path = "../json_listings/listing_urls_merged.json"
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nSaved to {output_path}")
