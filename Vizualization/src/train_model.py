"""
train_model.py
--------------
Run this script ONCE to train all models and save the best one.

Usage:
    cd Vizualization/src
    python train_model.py
"""

import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "..", "..", "DataProcessing", "data", "listings_cleaned.json")
MODEL_OUT = os.path.join(BASE_DIR, "model.pkl")

# ── Load data ─────────────────────────────────────────────────────────────────
print("📂 Loading data...")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

for col in ["price_per_night", "review_index", "latitude", "longitude"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
for col in ["guests", "beds", "bedrooms", "baths", "num_reviews"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
for col in ["is_superhost", "is_guest_favourite"]:
    if col in df.columns:
        df[col] = df[col].astype(bool)
if "characteristics" in df.columns:
    df["characteristics"] = df["characteristics"].apply(
        lambda x: x if isinstance(x, list) else []
    )

print(f"✅ Loaded {len(df)} listings.")

# ── Feature engineering ───────────────────────────────────────────────────────
print("⚙️  Engineering features...")

df["is_superhost"] = df["is_superhost"].astype(int)
df["is_guest_favourite"] = df["is_guest_favourite"].astype(int)

# Encode region
le = LabelEncoder()
if "region" in df.columns:
    df["region_enc"] = le.fit_transform(df["region"].fillna("unknown"))
    region_classes = list(le.classes_)
else:
    df["region_enc"] = 0
    region_classes = ["unknown"]

# Top-10 characteristics as binary features
all_chars = [c for lst in df["characteristics"] for c in lst]
top_chars = [c for c, _ in Counter(all_chars).most_common(10)]
for c in top_chars:
    df[f"char_{c}"] = df["characteristics"].apply(lambda x: 1 if c in x else 0)

feature_cols = (
    ["guests", "beds", "bedrooms", "baths",
     "is_superhost", "is_guest_favourite",
     "review_index", "num_reviews", "region_enc"]
    + [f"char_{c}" for c in top_chars]
)
feature_cols = [c for c in feature_cols if c in df.columns]

df_clean = df[feature_cols + ["price_per_night"]].dropna()
print(f"✅ Clean samples for training: {len(df_clean)}")

if len(df_clean) < 20:
    raise ValueError("Not enough clean data to train (need ≥ 20 rows).")

X = df_clean[feature_cols]
y = df_clean["price_per_night"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train & evaluate all 3 models ─────────────────────────────────────────────
print("\n🏋️  Training models...\n")

candidates = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    "Linear Regression": LinearRegression(),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in candidates.items():
    # Cross-validation MAE
    cv_scores = cross_val_score(model, X_train, y_train,
                                scoring="neg_mean_absolute_error", cv=kf)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()

    # Fit on full train set, evaluate on test set
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds)
    test_r2 = r2_score(y_test, preds)

    results.append({
        "name": name,
        "model": model,
        "cv_mae": cv_mae,
        "cv_std": cv_std,
        "test_mae": test_mae,
        "test_r2": test_r2,
    })

    print(f"  {name}")
    print(f"    CV MAE:   €{cv_mae:.2f} ± €{cv_std:.2f}")
    print(f"    Test MAE: €{test_mae:.2f}  |  R²: {test_r2:.3f}\n")

# ── Pick best model (lowest test MAE) ────────────────────────────────────────
best = min(results, key=lambda r: r["test_mae"])
print(f"🏆 Best model: {best['name']}  (Test MAE €{best['test_mae']:.2f}, R² {best['test_r2']:.3f})")

# ── Save everything to model.pkl ─────────────────────────────────────────────
payload = {
    "model": best["model"],
    "model_name": best["name"],
    "feature_cols": feature_cols,
    "region_classes": region_classes,
    "top_chars": top_chars,
    "test_mae": best["test_mae"],
    "test_r2": best["test_r2"],
    "cv_mae": best["cv_mae"],
    "cv_std": best["cv_std"],
    "all_results": [
        {k: v for k, v in r.items() if k != "model"}
        for r in results
    ],
}

with open(MODEL_OUT, "wb") as f:
    pickle.dump(payload, f)

print(f"\n✅ Model saved to: {MODEL_OUT}")
print("   → Now run:  streamlit run app.py")