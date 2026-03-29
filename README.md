# AirBnB Thessaloniki — Web Data Mining & Analytics Platform

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white)](https://selenium.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

**Course:** Web Data Mining, SS 2025-2026  
**Team:** Team B  
**Regions studied:** Kalamaria, Panorama, Neapoli-Sikies (Thessaloniki, Greece)

## 📌 Project Overview

This project implements a full end-to-end data pipeline that scrapes AirBnB listings from three neighborhoods in Thessaloniki, stores them in MongoDB, preprocesses and cleans the data, and presents it through an interactive Streamlit analytics dashboard with a built-in Machine Learning price predictor.

## 🛠️ Technology Stack

* **Scraping:** Python, Selenium, BeautifulSoup, ChromeDriver (via `webdriver-manager`)
* **Storage:** MongoDB (hosted at AUTH university server, db: `eu`, collection: `teamB`)
* **Data Processing:** Python, Pandas, JSON
* **Machine Learning:** scikit-learn (Random Forest, Gradient Boosting, Linear Regression)
* **Visualization:** Streamlit, Plotly
* **Other Tools:** `pymongo`, `python-dotenv`

## 🗂️ Folder Structure

```text
Airbnb-Scraping/
├── Scraping/
│   ├── src/
│   │   ├── test_browser.py         # Reusable Chrome WebDriver factory
│   │   ├── search_urls.py          # Region search URL definitions
│   │   ├── get_listing_urls.py     # Paginates search results and collects listing URLs
│   │   ├── merge_urls.py           # Merges and deduplicates all URL run files
│   │   ├── inspect_listing.py      # Debug tool for inspecting a single listing
│   │   ├── extract_features.py     # Core feature extractor for a single listing
│   │   └── main_scraper.py         # Full scraper loop over all listings
│   └── json_listings/
│       ├── listing_urls_run_*.json # Per-run URL collection files
│       ├── listing_urls_merged.json# Merged deduplicated URLs (~450 unique)
│       └── listings_data.json      # Raw scraped features for all listings
├── DataProcessing/
│   ├── src/
│   │   ├── preprocessing.py        # Cleans raw data and saves listings_cleaned.json
│   │   ├── mongo_upload.py         # Uploads cleaned data to MongoDB
│   │   └── mongo_queries.py        # Runs analysis queries against MongoDB
│   └── data/
│       └── listings_cleaned.json   # Final cleaned dataset
├── PricingPredictor/
│   ├── src/
│   │   └── train_model.py          # Trains 3 ML models and saves best to model.pkl
│   └── model/
│       └── model.pkl               # Saved best model payload
├── Vizualization/
│   └── src/
│       └── app.py                  # Streamlit dashboard
├── .gitignore
├── requirements.txt
└── README.md
```

## 🕸️ Scraping Strategy

AirBnB hard-caps search results at 15 pages × 18 listings (≈ 270 listings per region). To overcome this, `get_listing_urls.py` was run multiple times across sessions, saving results to timestamped JSON files. The script `merge_urls.py` merges all runs and deduplicates URLs both within and across regions, yielding approximately 450 unique listings.

**Nightly Price Extraction:**
Since AirBnB requires check-in/check-out dates to render pricing, we appended a 5-night stay starting 60 days from the scrape date to each listing URL. If no price was found, up to 5 retries were attempted, shifting the window forward by 30 days (60, 90, 120, 150, 180 days out) to accommodate minimum stay requirements and blocked periods. The total displayed price was divided by 5 to obtain the per-night rate.

**Feature Extraction & Resilience:**
To avoid fragile CSS selectors, the scraper parses embedded JSON blobs directly:
* `application/ld+json` schema block for coordinates and ratings.
* `data-deferred-state-0` script block for guests, beds, bedrooms, baths, superhost status, guest favourite, characteristics, and host name.
* An `sbuiData` fallback is implemented for listings missing the primary data path. 

The scraper is resumable: it saves state after every listing and skips already-scraped URLs upon restart. Requires a non-headless Chrome browser to evade AirBnB bot detection. The full scrape of ~450 listings takes approximately 2-4 hours.

## 🧹 Preprocessing Decisions

* **Quality over Quantity:** Listings missing price, review score, or core property fields are dropped.
* **Review Threshold:** A minimum of 3 reviews is required for a listing to be included.
* **Logical Imputation:** Null bedrooms are filled with `0` (e.g., studios have no separate bedroom).
* **Superhost Handling:** Mentions are stripped from the characteristics list as they are captured as a boolean field.
* **Sanity Checks:** Extracted prices below €10/night are discarded as anomalous scraping artifacts.
* **Flattening:** Raw data is converted from a nested dictionary `{region: [listings]}` to a flat list containing `region` as a discrete field.
* **Typing:** Boolean fields are explicitly cast to Python `bool` to prevent JSON type inconsistencies.

## 🗄️ MongoDB Architecture

* **Host:** `db.csd.auth.gr`
* **Port:** `27117`
* **Auth:** Username `eu`, AuthSource `admin`
* **Database / Collection:** `eu` / `teamB`

Upload employs an `upsert` operation using the listing URL as the unique key, making the script safe to run multiple times without introducing duplicates. 
> [!IMPORTANT]
> Connecting to the database from outside the campus network requires an active AUTH university VPN connection.

## 📊 Streamlit Dashboard

The interactive analytics dashboard supports both local JSON files and the MongoDB database as data sources, featuring a dark/light theme toggle. Key views include:

* **Overview:** KPI cards, region-specific statistics, and price distribution charts.
* **Price Analysis:** Pearson correlation matrices, price vs. property feature comparisons, and analysis of superhost/guest favourite premiums.
* **Ratings & Rankings:** Bayesian-weighted top-10 and bottom-10 listings, alongside rating distributions.
* **Characteristics:** Most common amenities, correlation between characteristics and review scores, and comparative analysis (with vs. without specific amenities).
* **Map:** An interactive mapbox scatter plot of all listings, color-coded by region and sized by price.
* **ML Price Predictor:** An interactive suite displaying the results of the model training and allowing users to predict prices.

## 🤖 ML Price Predictor

The `train_model.py` pipeline engineers features such as guests, beds, bedrooms, baths, review index, number of reviews, region (label encoded), and the top-10 characteristics as binary flags. 

Three models (Random Forest, Gradient Boosting, and Linear Regression) are trained and evaluated using 5-fold cross-validation Mean Absolute Error (MAE). The best-performing model is saved to `model.pkl` along with feature columns, region classes, top characteristics, and evaluation metrics, which are then loaded by the Streamlit application for on-the-fly inference.

## 🚀 Setup & Execution Instructions

> [!NOTE]
> All scripts utilize relative paths and must be executed from within their respective `src/` directories. Windows users may encounter UTF-8 encoding issues; ensure `encoding="utf-8"` is explicitly used in file operations if you modify the code.

### 1. Environment Setup

Clone the repository and set up your Python virtual environment:

```bash
git clone <repository-url>
cd Airbnb-Scraping
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Scraper

```bash
cd Scraping/src/
python main_scraper.py
cd ../..
```

### 3. Data Preprocessing

```bash
cd DataProcessing/src/
python preprocessing.py
cd ../..
```

### 4. Upload to MongoDB

> Ensure you are connected to the AUTH University VPN.

```bash
cd DataProcessing/src/
python mongo_upload.py
cd ../..
```

### 5. Train the Machine Learning Model

```bash
cd PricingPredictor/src/
python train_model.py
cd ../..
```

### 6. Launch the Dashboard

```bash
cd Vizualization/src/
streamlit run app.py
```
