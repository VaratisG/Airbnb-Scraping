# merge_urls.py
# Run get_listing_urls.py multiple times, then use this to merge all results
# Deduplicates both within regions AND across regions

import json
import os

def merge_url_files(folder: str) -> dict:
    """
    Reads all listing_urls JSON files from the folder,
    merges them, removes duplicates within regions,
    then removes duplicates across regions (first region wins).
    """
    merged = {}

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".json") and "merged" not in f and "deduped" not in f
    ]

    if not files:
        print("No JSON files found in folder.")
        return {}

    # ── Step 1: Merge all files, dedup within each region using sets ──
    for filepath in files:
        print(f"  Reading {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for region, urls in data.items():
            if region not in merged:
                merged[region] = set()
            merged[region].update(urls)

    # ── Step 2: Dedup across regions (first region encountered wins) ──
    seen        = set()
    deduped     = {}
    total_in    = sum(len(v) for v in merged.values())
    removed     = 0

    for region, urls in merged.items():
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
            else:
                removed += 1
        deduped[region] = unique_urls

    total_out = sum(len(v) for v in deduped.values())

    return deduped, total_in, total_out, removed


if __name__ == "__main__":
    folder = "../json_listings"

    print("Merging all JSON files from json_listings/...\n")
    deduped, total_in, total_out, removed = merge_url_files(folder)

    print("\n── Results ──")
    for region, urls in deduped.items():
        print(f"  {region}: {len(urls)} unique listings")

    print(f"\n  Before cross-region dedup : {total_in}")
    print(f"  After cross-region dedup  : {total_out}")
    print(f"  Duplicates removed        : {removed}")

    output_path = "../json_listings/listing_urls_merged.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")