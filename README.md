# AirBnB Thessaloniki — Web Data Mining & Analytics Platform

[![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Selenium](https://img.shields.io/badge/Selenium_4-43B02A?style=for-the-badge&logo=selenium&logoColor=white)](https://selenium.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

> **Course:** Web Data Mining · School of Informatics, Aristotle University of Thessaloniki · AY 2025–2026  
> **Team:** Team B  
> **Study Area:** Kalamaria · Panorama · Neapoli-Sikies (Thessaloniki, Greece)

---

## Abstract

This project presents a complete, automated data-acquisition and analytics pipeline targeting short-term rental listings on the Airbnb platform across three neighbourhoods of Thessaloniki, Greece. The system encompasses four tightly coupled modules: a resilient Selenium-based web scraper that bypasses dynamic JavaScript rendering and bot-detection mechanisms; a data-processing layer implementing geofence validation, quality filters, and MongoDB persistence; a supervised machine learning pipeline that benchmarks three regression algorithms for nightly-price prediction; and an interactive Streamlit dashboard that synthesises all findings into six analytical views. Approximately **450 unique listings** were collected, cleaned, and analysed, offering quantitative insights into pricing dynamics, host quality signals, and amenity distributions across the target regions.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Repository Structure](#2-repository-structure)
3. [Module 1 — Data Acquisition (Scraping)](#3-module-1--data-acquisition-scraping)
4. [Module 2 — Data Processing & Storage](#4-module-2--data-processing--storage)
5. [Module 3 — Machine Learning Price Predictor](#5-module-3--machine-learning-price-predictor)
6. [Module 4 — Visualisation Dashboard](#6-module-4--visualisation-dashboard)
7. [Technology Stack](#7-technology-stack)
8. [Setup & Execution](#8-setup--execution)

---

## 1. System Architecture

The platform is structured as a sequential four-stage pipeline. Each stage produces artefacts consumed by the next, enabling independent development, testing, and re-execution of any stage without affecting upstream data.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    END-TO-END DATA PIPELINE OVERVIEW                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────┐     JSON     ┌──────────────────┐    JSON / MongoDB
  │   STAGE 1        │ ──────────►  │   STAGE 2        │ ──────────────────►
  │  Data Acquisition│              │  Data Processing │
  │                  │              │                  │
  │  Selenium Chrome │              │  Validation      │
  │  BeautifulSoup   │              │  Geofencing      │
  │  JSON Parsing    │              │  Cleaning        │
  │  Multi-session   │              │  MongoDB Upsert  │
  └──────────────────┘              └──────────────────┘
           │                                  │
           │  ~450 raw listings               │  ~450 clean listings
           ▼                                  ▼
  listing_urls_merged.json         listings_cleaned.json
  listings_data.json               MongoDB collection


  ┌──────────────────┐    model.pkl ┌──────────────────┐
  │   STAGE 3        │ ──────────►  │   STAGE 4        │
  │  ML Training     │              │  Dashboard       │
  │                  │              │                  │
  │  Random Forest   │              │  6 Analysis Views│
  │  Gradient Boost  │              │  Interactive Maps│
  │  Linear Regress. │              │  Price Predictor │
  │  5-fold CV MAE   │              │  Dark Theme UI   │
  └──────────────────┘              └──────────────────┘
```

### Data-Flow Diagram

```
  Airbnb.com
      │
      │  HTTP (Selenium + ChromeDriver)
      ▼
 ┌─────────────────────────────────────────────────────┐
 │  get_listing_urls.py  ──►  merge_urls.py            │
 │  (per-region pagination)    (cross-session dedup)   │   SCRAPING
 │                                    │                │    MODULE
 │              listing_urls_merged.json               │
 │                                    │                │
 │  extract_features.py  ◄────────────┘                │
 │  (JSON-blob parsing)                                │
 │         │  main_scraper.py (orchestrator)           │
 └─────────┼───────────────────────────────────────────┘
           │  listings_data.json  (raw, ~450 entries)
           ▼
 ┌─────────────────────────────────────────────────────┐
 │  preprocessing.py                                   │      DATA
 │  · completeness filter        · type casting        │   PROCESSING
 │  · geofence validation        · price sanity        │     MODULE
 │  · studio imputation          · superhost dedup     │
 │         │  listings_cleaned.json                    │
 │         ▼                                           │
 │  mongo_upload.py  ──►  MongoDB (upsert on URL)      │
 └─────────────────────────────────────────────────────┘
           │  listings_cleaned.json
           ▼
 ┌─────────────────────────────────────────────────────┐
 │  train_model.py                                     │     ML
 │  · feature engineering (region_enc, char_* flags)   │   MODULE
 │  · 5-fold cross-validation (MAE)                    │
 │  · best-model selection → model.pkl                 │
 └─────────────────────────────────────────────────────┘
           │  model.pkl  +  listings_cleaned.json
           ▼
 ┌─────────────────────────────────────────────────────┐
 │  app.py  (Streamlit)                                │   DASHBOARD
 │  Overview │ Prices │ Ratings │ Chars │ Map │ ML     │    MODULE
 └─────────────────────────────────────────────────────┘
```

---

## 2. Repository Structure

```
Airbnb-Scraping/
│
├── Scraping/
│   ├── src/
│   │   ├── test_browser.py         ← Chrome WebDriver factory (anti-bot config)
│   │   ├── search_urls.py          ← Regional Airbnb search URL registry
│   │   ├── get_listing_urls.py     ← Paginated URL harvesting
│   │   ├── merge_urls.py           ← Multi-session deduplication
│   │   ├── extract_features.py     ← Per-listing JSON-blob feature extractor
│   │   ├── inspect_listing.py      ← Debug utility (single listing dump)
│   │   └── main_scraper.py         ← Orchestration loop & state manager
│   └── json_listings/
│       ├── listing_urls_run_<ts>.json   ← Per-run URL captures (timestamped)
│       ├── listing_urls_merged.json     ← Deduplicated master URL list (~450)
│       └── listings_data.json           ← Raw feature dataset
│
├── DataProcessing/
│   ├── src/
│   │   ├── preprocessing.py        ← Validation, geofencing, cleaning pipeline
│   │   ├── mongo_upload.py         ← Idempotent MongoDB upsert
│   │   └── mongo_queries.py        ← 7 aggregation/analysis queries
│   └── data/
│       └── listings_cleaned.json   ← Final analysis-ready dataset
│
├── PricePredictor/
│   ├── src/
│   │   └── train_model.py          ← Model training, evaluation, serialisation
│   └── model/
│       └── model.pkl               ← Serialised model payload (pickle)
│
├── Vizualization/
│   └── src/
│       └── app.py                  ← 6-page Streamlit analytics dashboard
│
├── requirements.txt                ← Full dependency manifest (69 packages)
├── .gitignore
└── README.md
```

---

## 3. Module 1 — Data Acquisition (Scraping)

### 3.1 Overview & Motivation

Airbnb does not provide a public API. The platform renders listing content dynamically via JavaScript, hides pricing behind mandatory date selection, and employs bot-detection heuristics. The scraping module addresses all three constraints through a combination of browser automation, JSON-blob parsing, and session management.

### 3.2 Target Regions

Three neighbourhoods in Thessaloniki were selected based on their distinct socioeconomic profiles and geographic separation, enabling meaningful cross-regional comparative analysis.

| Region | Google Place ID | Approx. Bounding Box |
|---|---|---|
| **Kalamaria** | `ChIJoyaC8xo1qBQRTNQiEzMuaQ0` | 40.570°–40.610°N, 22.920°–22.970°E |
| **Panorama** | *(Thessaloniki search URL)* | 40.575°–40.610°N, 22.990°–23.050°E |
| **Neapoli-Sikies** | *(Thessaloniki search URL)* | 40.645°–40.665°N, 22.935°–22.970°E |

### 3.3 Anti-Bot Evasion

The `test_browser.py` module encapsulates the WebDriver factory with the following hardened configuration applied via ChromeOptions and the Chrome DevTools Protocol (CDP):

```
┌─────────────────────────────────────────────────────────────────┐
│                  Anti-Detection Measures                        │
├──────────────────────────────┬──────────────────────────────────┤
│  Measure                     │  Purpose                         │
├──────────────────────────────┼──────────────────────────────────┤
│  Custom User-Agent string    │  Mimics a real desktop browser   │
│  --disable-blink-features    │  Masks AutomationControlled flag │
│  CDP: navigator.webdriver=0  │  Removes JS-detectable property  │
│  Window size: 1920×1080      │  Matches typical viewport        │
│  Non-headless mode           │  Required — Airbnb detects       │
│                              │  headless browsers               │
└──────────────────────────────┴──────────────────────────────────┘
```

### 3.4 URL Collection Strategy

Airbnb hard-caps search result pages at **15 pages × ~18 listings ≈ 270 listings per region**. To exceed this limit, the URL collection script (`get_listing_urls.py`) was executed across multiple independent sessions on different days. Each run saves its output to a timestamped JSON file.

```
  Session 1 ──► listing_urls_run_20260315_143022.json  ┐
  Session 2 ──► listing_urls_run_20260318_091500.json  ├──► merge_urls.py ──► listing_urls_merged.json
  Session 3 ──► listing_urls_run_20260322_184711.json  ┘       (~450 unique listings)
```

`merge_urls.py` performs **two-stage deduplication**:
1. **Within-region deduplication** — Collapses duplicate URLs within the same region using Python sets.
2. **Cross-region deduplication** — For URLs appearing under multiple region labels (e.g., boundary listings), the first-encountered region assignment is retained.

### 3.5 Feature Extraction

Rather than relying on brittle CSS selectors (which Airbnb changes frequently), `extract_features.py` parses two embedded JSON blobs that are always present in the page source:

| JSON Source | Fields Extracted |
|---|---|
| `<script type="application/ld+json">` | `latitude`, `longitude`, `review_index` (aggregateRating) |
| `<script id="data-deferred-state-0">` | `guests`, `beds`, `bedrooms`, `baths`, `is_superhost`, `is_guest_favourite`, `host_name`, `characteristics` |
| `sbuiData` (fallback) | Same property fields — used when primary path is absent |

**Complete Feature Schema:**

| Feature | Type | Description |
|---|---|---|
| `url` | `str` | Canonical listing URL (unique identifier) |
| `region` | `str` | Assigned neighbourhood label |
| `price_per_night` | `float` | Total price ÷ 5 nights (€) |
| `guests` | `int` | Maximum guest capacity |
| `beds` | `int` | Number of beds |
| `bedrooms` | `int` | Number of bedrooms (0 for studios) |
| `baths` | `float` | Number of bathrooms |
| `is_superhost` | `bool` | Host holds Superhost status |
| `is_guest_favourite` | `bool` | Listed as Guest Favourite |
| `review_index` | `float` | Aggregate rating score (1.0–5.0) |
| `num_reviews` | `int` | Total number of guest reviews |
| `host_name` | `str` | Display name of the host |
| `characteristics` | `list[str]` | Amenity/property descriptors |
| `latitude` | `float` | WGS-84 latitude coordinate |
| `longitude` | `float` | WGS-84 longitude coordinate |

### 3.6 Nightly Price Resolution

Airbnb withholds pricing unless a specific stay window is provided. The extractor appends a **5-night stay** starting `N` days from the scrape date as URL query parameters. If no price is parsed (listing unavailable for that period), the window is shifted forward in five attempts:

```
  Attempt 1: check_in = today + 60 days  ──►  price found?  ──► ✓ use it
  Attempt 2: check_in = today + 90 days  ──►  price found?  ──► ✓ use it
  Attempt 3: check_in = today + 120 days ──►  price found?  ──► ✓ use it
  Attempt 4: check_in = today + 150 days ──►  price found?  ──► ✓ use it
  Attempt 5: check_in = today + 180 days ──►  price found?  ──► ✓ use it
                                                   │ (all failed)
                                                   └──► return None (listing dropped)
```

A regex targeting the `€ N,NNN` pattern in the raw page source extracts the total price. Prices below **€10 per night** are discarded as parsing artefacts.

### 3.7 Orchestration & Resilience

`main_scraper.py` manages the full scraping loop with the following operational properties:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Scraper State Machine                         │
│                                                                  │
│  START ──► Load merged URLs ──► Load existing data (resume)      │
│               │                                                  │
│               ▼                                                  │
│  ┌─── For each region / listing URL ──────────────────────────┐  │
│  │                                                            │  │
│  │   already_scraped? ──► YES ──► skip (counter++)            │  │
│  │         │                                                  │  │
│  │         NO                                                 │  │
│  │         ▼                                                  │  │
│  │   extract_features()                                       │  │
│  │         │                                                  │  │
│  │      success? ──► save_data()  ──► scraped counter++       │  │
│  │         │                                                  │  │
│  │      failure? ──► log warning  ──► failed counter++        │  │
│  │         │                                                  │  │
│  │   sleep(random 2–4 s)   ← polite crawl delay               │  │
│  └────────────────────────────────────────────────────────────┘  │
│               │                                                  │
│   Ctrl+C ──► graceful shutdown + final log summary               │
└──────────────────────────────────────────────────────────────────┘
```

- **Resumable:** State is persisted to `listings_data.json` after every listing.
- **Polite crawling:** Random 2–4 second delays between requests.
- **Dual logging:** Writes to both `scraper.log` and stdout.
- **Estimated runtime:** 2–4 hours for a full scan of ~450 listings.

---

## 4. Module 2 — Data Processing & Storage

### 4.1 Preprocessing Pipeline

Raw data from the scraper undergoes a multi-stage cleaning pipeline in `preprocessing.py` before any analysis or storage:

```
  Raw listings_data.json
          │
          ▼
  ┌───────────────────┐
  │  1. Flatten       │  {region: [listings]}  →  [{listing, region}, ...]
  └────────┬──────────┘
           ▼
  ┌───────────────────┐
  │  2. Completeness  │  Drop if: price | review_index | guests | beds |
  │     Filter        │           baths | lat | lon is null
  │                   │  Drop if: num_reviews < 3
  └────────┬──────────┘
           ▼
  ┌───────────────────┐
  │  3. Geofence      │  Validate (lat, lon) against strict bounding
  │     Validation    │  boxes per region. Drops city-centre leakage.
  └────────┬──────────┘
           ▼
  ┌───────────────────┐
  │                   │  · Round prices to 2 d.p.
  │ 4. Data Cleaning  │  · Cast to int/float/bool as appropriate
  │                   │  · Studio imputation: bedrooms = null → 0
  │                   │  · Strip "is a superhost" from characteristics
  └────────┬──────────┘
           ▼
  listings_cleaned.json  (~450 validated entries)
```

### 4.2 Geofencing

A common data-quality issue in region-tagged Airbnb data is **geographic leakage** — listings assigned to one neighbourhood that are physically located in another (often the city centre). Strict bounding boxes enforce spatial integrity:

| Region | Latitude Range | Longitude Range |
|---|---|---|
| **Kalamaria** | 40.570° – 40.610° N | 22.920° – 22.970° E |
| **Panorama** | 40.575° – 40.610° N | 22.990° – 23.050° E |
| **Neapoli-Sikies** | 40.645° – 40.665° N | 22.935° – 22.970° E |

Any listing whose coordinates fall outside all three bounding boxes is discarded entirely, regardless of the region label assigned by the scraper.

### 4.3 Quality Filter Summary

| Filter | Criterion | Rationale |
|---|---|---|
| Completeness | All core fields non-null | Prevents NaN propagation in analysis |
| Review threshold | ≥ 3 reviews | Ensures statistical relevance of the rating |
| Price sanity | ≥ €10 / night | Eliminates regex false-positive extractions |
| Geofence | Within strict bounding box | Removes city-centre and mis-labelled listings |
| Studio imputation | `bedrooms = null → 0` | Studios have no separate bedroom; 0 is semantically correct |
| Superhost dedup | Strip from `characteristics` | Captured as `is_superhost` boolean; duplicate information |

### 4.4 MongoDB Architecture

Cleaned data is stored in a MongoDB instance hosted on the Aristotle University infrastructure.

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    MongoDB Configuration                    │
  ├────────────────────────┬────────────────────────────────────┤
  │  Host                  │  db.csd.auth.gr                    │
  │  Port                  │  27117                             │
  │  Auth Source           │  admin                             │
  │  Database              │  eu                                │
  │  Collection            │  219_229_220_collection            │
  │  Upsert Key            │  url  (unique listing identifier)  │
  │  Upload Mode           │  update_one(..., upsert=True)      │
  └────────────────────────┴────────────────────────────────────┘
```

> **Network Requirement:** Access from outside the campus network requires an active **AUTH University VPN** connection.

The upsert strategy (`$set` on the `url` key) makes `mongo_upload.py` **fully idempotent** — re-running the script after a data refresh updates existing documents without creating duplicates.

### 4.5 MongoDB Analysis Queries

`mongo_queries.py` provides seven pre-built queries for exploratory analysis directly against the database:

| # | Query | Aggregation Technique |
|---|---|---|
| 1 | Top 5 rated listings per region | `$sort` + `$limit` per region |
| 2 | Average / min / max price per region | `$group` + `$avg`, `$min`, `$max` |
| 3 | Superhost count & avg price/rating by region | `$match` + `$group` |
| 4 | Guest Favourite count & avg price/rating by region | `$match` + `$group` |
| 5 | Listings per price bracket | `$bucket` (7 brackets: €0–500+) |
| 6 | Top 15 most common characteristics | `$unwind` + `$group` + `$sort` |
| 7 | Top 10 most-reviewed listings | `$sort` on `num_reviews` + `$limit` |

---

## 5. Module 3 — Machine Learning Price Predictor

### 5.1 Feature Engineering

The raw dataset is enriched with two categories of engineered features before model training:

**Categorical Encoding:**
- `region` → `region_enc` (integer via `sklearn.LabelEncoder`)

**Characteristic Flags:**
The top 10 most frequently occurring characteristics across all listings are identified. For each listing, a binary flag `char_<name>` is created indicating presence or absence of each characteristic. This converts an unstructured list into a sparse but informative feature space.

**Final Feature Vector:**

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    Feature Set (F)                          │
  ├─────────────────────────────────────────────────────────────┤
  │  Numeric      │  guests, beds, bedrooms, baths              │
  │               │  review_index, num_reviews                  │
  ├───────────────┼─────────────────────────────────────────────┤
  │  Encoded Cat. │  region_enc                                 │
  ├───────────────┼─────────────────────────────────────────────┤
  │  Boolean      │  is_superhost, is_guest_favourite           │
  ├───────────────┼─────────────────────────────────────────────┤
  │  Char. Flags  │  char_Wifi, char_Kitchen, ...  (top 10)     │
  └─────────────────────────────────────────────────────────────┘
  Target:  price_per_night  (continuous, €)
```

### 5.2 Model Comparison & Selection

Three regression algorithms are trained on an 80/20 train-test split and evaluated using **5-fold cross-validation MAE** as the primary selection criterion. All experiments use `random_state=42` for reproducibility.

| Model | Hyperparameters | Strengths |
|---|---|---|
| **Random Forest** | `n_estimators=200`, `n_jobs=-1` | Handles non-linearities, robust to outliers |
| **Gradient Boosting** | `n_estimators=200`, `lr=0.05` | Sequential error correction, high accuracy |
| **Linear Regression** | Default (OLS) | Interpretable baseline, fast training |

```
  Training Pipeline
  ─────────────────
  listings_cleaned.json
          │
          ▼
  Feature Engineering (encoding + char flags)
          │
          ├──► Random Forest     ──► 5-fold CV MAE  ─┐
          ├──► Gradient Boosting ──► 5-fold CV MAE  ─┼──► argmin(CV MAE) ──► model.pkl
          └──► Linear Regression ──► 5-fold CV MAE  ─┘
```

### 5.3 Saved Model Payload

The serialised `model.pkl` contains everything the dashboard needs for inference, with no dependency on the training data at runtime:

```python
{
  "model":          <best sklearn estimator>,
  "model_name":     "Random Forest | Gradient Boosting | Linear Regression",
  "feature_cols":   [...],          # ordered feature names
  "region_classes": [...],          # LabelEncoder class list
  "top_chars":      [...],          # top-10 characteristic strings
  "test_mae":       float,          # held-out MAE (€)
  "test_r2":        float,          # held-out R²
  "cv_mae":         float,          # mean 5-fold CV MAE
  "cv_std":         float,          # std of 5-fold CV MAE
  "all_results":    [...]           # metrics for all 3 candidate models
}
```

---

## 6. Module 4 — Visualisation Dashboard

### 6.1 Architecture

The Streamlit application (`app.py`) supports **two data sources** switchable from the sidebar, enabling analysis of both locally-stored data and the live MongoDB collection:

```
  Sidebar ──► Data Source Selector
                  │
          ┌───────┴───────┐
          │               │
    JSON file         MongoDB URI
    (default)         (live data)
          │               │
          └───────┬───────┘
                  │
            load_data()  ← @st.cache_data
                  │
            Region Filter (All | Kalamaria | Panorama | Neapoli-Sikies)
                  │
            ┌─────┴──────────────────────────────────────────────┐
              📊 Overview  │ 💰 Prices │ ⭐ Ratings │ 🏷️ Chars  
              🗺️ Map       │ 🤖 ML Predictor                   
            └────────────────────────────────────────────────────┘
```

### 6.2 Dashboard Pages

#### Page 1 — Overview (`📊`)
Provides a high-level summary of the dataset with KPI metrics and distributional charts.

| Component | Description |
|---|---|
| KPI Cards (×5) | Total listings · Median price/night · Average rating · Superhost count · Region count |
| Region Statistics Table | Per-region: listing count, median price, avg rating, superhost count, guest favourite count |
| Bar Charts | Listings per region; Median price per region |
| Price Histogram | 40-bin distribution of `price_per_night` across all listings |
| Box Plot | `price_per_night` stratified by region |

#### Page 2 — Price Analysis (`💰`)
Quantitative investigation of pricing determinants.

| Component | Description |
|---|---|
| Pearson Correlation Matrix | Heatmap across: price, guests, beds, bedrooms, baths, review_index, num_reviews |
| Scatter Plot (interactive) | Price vs. user-selected feature (guests/beds/bedrooms/baths) with OLS trendline |
| Feature Box Plots | Distribution of each numeric property feature |
| Superhost Premium | Side-by-side box plots: Superhost vs. non-Superhost pricing |
| Guest Favourite Premium | Side-by-side box plots: Guest Favourite vs. standard listings |

#### Page 3 — Ratings & Rankings (`⭐`)
Ranking methodology uses a **Bayesian weighted score** to avoid ranking bias toward low-volume listings:

```
        v           m
  S = ───── × R + ───── × C
      v + m       v + m

  where:
    v = number of reviews for the listing
    R = listing's mean review score
    m = minimum vote threshold  (= 10)
    C = global mean rating       (= 4.5)
```

| Component | Description |
|---|---|
| Top 10 Listings | Horizontal bar chart, coloured by Bayesian score |
| Bottom 10 Listings | Horizontal bar chart, identifying underperformers |
| Review Score Histogram | 20-bin distribution of `review_index` |
| Reviews vs. Rating Scatter | Point size = price; colour = region |

#### Page 4 — Characteristics (`🏷️`)
Amenity-level analysis across the dataset.

| Component | Description |
|---|---|
| Top 20 Amenities | Frequency bar chart of most common `characteristics` |
| Characteristic–Rating Correlation | Bar chart of Pearson r (±1 scale, diverging colour) |
| Comparative Rating Chart | Avg rating: *has* characteristic vs. *does not have* (grouped bars) |

#### Page 5 — Map (`🗺️`)
Geospatial view of all listings using Mapbox (carto-darkmatter style).

| Component | Description |
|---|---|
| Scatter Map | Latitude/longitude; colour = region; size = price |
| Hover Info | `host_name` · `€price/night` · `★rating` |
| Below-Map Metrics | Filtered count · Avg price · Avg rating |

#### Page 6 — ML Price Predictor (`🤖`)
End-user interface for on-demand price estimation using the trained model.

**Input Form (3-column layout):**

| Input | Range / Options | Default |
|---|---|---|
| Guests | 1 – 10 | 2 |
| Beds | 1 – 10 | 1 |
| Bedrooms | 0 – 10 | 1 |
| Baths | 1 – 7 | 1 |
| Superhost | True / False | False |
| Guest Favourite | True / False | False |
| Review Index | 1.0 – 5.0 (slider) | 4.5 |
| Num Reviews | 0 – 2000 | 20 |
| Region | Dropdown (from training data) | — |
| Characteristics | Multiselect (top 10) | — |

**Output:** Predicted `€ price/night` with a contextual price-bucket badge:

```
  Predicted Price          Price Bucket
  ═══════════════          ══════════════════════════════════
  €XX.XX / night  ──────►  💚  Budget      (≤ 33rd percentile)
                           🟡  Mid-range   (33rd – 66th percentile)
                           🔴  Luxury      (> 66th percentile)
```

---

## 7. Technology Stack

| Category | Library / Tool | Version | Role |
|---|---|---|---|
| Language | Python | 3.10+ | All modules |
| Browser Automation | Selenium | 4.x | Dynamic page rendering |
| HTML Parsing | BeautifulSoup4 | 4.x | URL extraction from search pages |
| Browser Driver | webdriver-manager | — | Auto-managed ChromeDriver |
| Data Manipulation | Pandas | 2.x | DataFrame operations in ML & dashboard |
| Numerical Computing | NumPy | 1.x | Feature array operations |
| Machine Learning | scikit-learn | 1.x | Model training, encoding, CV evaluation |
| Model Serialisation | joblib / pickle | — | `model.pkl` persistence |
| Database | MongoDB | 6.x | Listing storage & aggregation queries |
| DB Driver | pymongo | 4.x | Python ↔ MongoDB interface |
| Dashboard | Streamlit | 1.x | Interactive web application |
| Visualisation | Plotly | 5.x | Interactive charts and maps |
| Statistical Analysis | SciPy / statsmodels | — | OLS trendlines, correlation |
| Environment | python-dotenv | — | Credential management |

---

## 8. Setup & Execution

> **Note:** All scripts use relative paths and **must be run from within their respective `src/` directory**.

### 8.1 Environment Setup

```bash
git clone <repository-url>
cd Airbnb-Scraping

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 8.2 Execution Order

Run each stage sequentially. Each stage depends on the artefacts produced by the previous one.

```
Step 1 ──► Collect listing URLs (repeat across sessions for full coverage)
Step 2 ──► Scrape features from all listings
Step 3 ──► Clean and validate raw data
Step 4 ──► Upload to MongoDB  (requires AUTH VPN)
Step 5 ──► Train the ML model
Step 6 ──► Launch the dashboard
```

```bash
# Step 1 — URL Collection
cd Scraping/src/
python get_listing_urls.py       # run multiple times across sessions
python merge_urls.py             # merge all runs
cd ../..

# Step 2 — Feature Scraping  (~2–4 hours, resumable)
cd Scraping/src/
python main_scraper.py
cd ../..

# Step 3 — Data Preprocessing
cd DataProcessing/src/
python preprocessing.py
cd ../..

# Step 4 — MongoDB Upload  (AUTH VPN required)
cd DataProcessing/src/
python mongo_upload.py
cd ../..

# Step 5 — Train the ML Model
cd PricePredictor/src/
python train_model.py
cd ../..

# Step 6 — Launch Dashboard
cd Vizualization/src/
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your default browser.

---

<div align="center">

*Web Data Mining · Aristotle University of Thessaloniki · Team B · 2025–2026*

</div>
