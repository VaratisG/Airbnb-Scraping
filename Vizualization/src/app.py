import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirBnB Thessaloniki Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 400;
    letter-spacing: 0.01em;
    line-height: 1.6;
}
h1, h2, h3 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.main { background: #0f0f13; }
[data-testid="stAppViewContainer"] { background: #0f0f13; }
[data-testid="stSidebar"] { background: #16161d; border-right: 1px solid #2a2a3a; }

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ff385c, #ff6b8a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.metric-label {
    color: #888;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 6px;
    font-weight: 500;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.section-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0eee8;
    border-left: 3px solid #ff385c;
    padding-left: 14px;
    margin: 28px 0 14px 0;
    letter-spacing: -0.02em;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.badge-red { background: #ff385c22; color: #ff385c; border: 1px solid #ff385c44; }
.badge-gold { background: #f5a62322; color: #f5a623; border: 1px solid #f5a62344; }
.badge-green { background: #00c85122; color: #00c851; border: 1px solid #00c85144; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(source="json", path="listings_cleaned.json", mongo_uri=None, db=None, col=None):
    if source == "mongodb" and mongo_uri:
        try:
            from pymongo import MongoClient
            client = MongoClient(mongo_uri)
            records = list(client[db][col].find({}, {"_id": 0}))
            df = pd.DataFrame(records)
        except Exception as e:
            st.error(f"MongoDB error: {e}")
            return pd.DataFrame()
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    # Normalise types
    for col_name in ["price_per_night", "review_index", "latitude", "longitude"]:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    for col_name in ["guests", "beds", "bedrooms", "baths", "num_reviews"]:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0).astype(int)
    for col_name in ["is_superhost", "is_guest_favourite"]:
        if col_name in df.columns:
            df[col_name] = df[col_name].astype(bool)
    if "characteristics" in df.columns:
        df["characteristics"] = df["characteristics"].apply(
            lambda x: x if isinstance(x, list) else []
        )
    return df

# ── Helpers ───────────────────────────────────────────────────────────────────
def weighted_score(row):
    """Bayesian-style score combining rating and review count."""
    m = 10  # minimum reviews threshold
    C = 4.5  # global mean prior
    v = row["num_reviews"]
    R = row["review_index"]
    return (v / (v + m)) * R + (m / (v + m)) * C

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#ff385c"
COLORS = ["#ff385c", "#f5a623", "#00c851", "#00aaff", "#b44fff", "#ff6b8a"]

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏠 AirBnB Analytics")
    st.markdown("**Thessaloniki** · SS 2025-26")
    st.divider()

    data_source = st.radio("Data source", ["JSON file", "MongoDB"])
    if data_source == "JSON file":
        json_path = st.text_input("JSON path", value="../../DataProcessing/data/listings_cleaned.json")
        df = load_data(source="json", path=json_path)
        st.session_state["df"] = df 
    else:
        uri = st.text_input("MongoDB URI",
            value="mongodb://eu:Gm8WQhwE@db.csd.auth.gr:27117/?authSource=admin")
        db_name = st.text_input("Database", value="eu")
        col_name = st.text_input("Collection", value="219_229_220_collection")
        if st.button("Connect"):
            with st.spinner("Connecting..."):
                loaded = load_data(source="mongodb", mongo_uri=uri, db=db_name, col=col_name)
            if not loaded.empty:
                st.session_state["df"] = loaded
                st.success(f"✅ Connected — {len(loaded)} listings loaded")
            else:
                st.error("Connection failed or empty collection.")

        if "df" in st.session_state:
            if st.button("🔌 Disconnect"):
                del st.session_state["df"]
                st.rerun()

        df = st.session_state.get("df", pd.DataFrame())

    st.divider()
    if not df.empty and "region" in df.columns:
        regions = ["All"] + sorted(df["region"].unique().tolist())
        selected_region = st.selectbox("Filter by region", regions)
        if selected_region != "All":
            df = df[df["region"] == selected_region]

    st.divider()
    page = st.radio("Navigate", [
        "📊 Overview",
        "💰 Price Analysis",
        "⭐ Ratings & Rankings",
        "🏷️ Characteristics",
        "🗺️ Map",
        "🤖 ML Price Predictor",
    ])

# ── Guard ─────────────────────────────────────────────────────────────────────
if df.empty:
    st.warning("⚠️ No data loaded. Check your JSON path or MongoDB connection.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 Dataset Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, len(df), "Listings"),
        (c2, f"€{df['price_per_night'].median():.0f}", "Median Price/Night"),
        (c3, f"{df['review_index'].mean():.2f} ★", "Avg Rating"),
        (c4, f"{df['is_superhost'].sum()}", "Superhosts"),
        (c5, df["region"].nunique() if "region" in df.columns else "—", "Regions"),
    ]
    for col_w, val, label in metrics:
        with col_w:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Stats per Location</div>', unsafe_allow_html=True)

    if "region" in df.columns:
        stats = df.groupby("region").agg(
            listings=("price_per_night", "count"),
            median_price=("price_per_night", "median"),
            avg_rating=("review_index", "mean"),
            superhosts=("is_superhost", "sum"),
            guest_fav=("is_guest_favourite", "sum"),
        ).reset_index()
        stats.columns = ["Region", "Listings", "Median Price (€)", "Avg Rating", "Superhosts", "Guest Favourites"]
        stats["Median Price (€)"] = stats["Median Price (€)"].round(1)
        stats["Avg Rating"] = stats["Avg Rating"].round(2)
        st.dataframe(stats, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(stats, x="Region", y="Listings", color="Region",
                         color_discrete_sequence=COLORS, template=PLOTLY_TEMPLATE,
                         title="Listings per Region")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.bar(stats, x="Region", y="Median Price (€)", color="Region",
                         color_discrete_sequence=COLORS, template=PLOTLY_TEMPLATE,
                         title="Median Price per Region")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Price Distribution</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x="price_per_night", nbins=40,
                           color_discrete_sequence=[ACCENT],
                           template=PLOTLY_TEMPLATE, title="Price per Night Distribution")
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.box(df, x="region" if "region" in df.columns else None,
                     y="price_per_night", color="region" if "region" in df.columns else None,
                     color_discrete_sequence=COLORS, template=PLOTLY_TEMPLATE,
                     title="Price Box Plot per Region")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRICE ANALYSIS  (Correlation matrix Q1)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Price Analysis":
    st.markdown("# 💰 Price Analysis")

    st.markdown('<div class="section-header">Q1 · Correlation Matrix — Property Type vs Price</div>', unsafe_allow_html=True)
    st.caption("Property 'type' is defined by beds, baths, bedrooms, guests.")

    num_cols = ["price_per_night", "guests", "beds", "bedrooms", "baths",
                "review_index", "num_reviews"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale=["#0d1b2a", "#1b3a5c", ACCENT],
        template=PLOTLY_TEMPLATE,
        title="Pearson Correlation Matrix",
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Price vs Features</div>', unsafe_allow_html=True)
    feat = st.selectbox("X-axis feature", ["guests", "beds", "bedrooms", "baths"])
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.scatter(df, x=feat, y="price_per_night",
                         color="region" if "region" in df.columns else None,
                         color_discrete_sequence=COLORS, opacity=0.7,
                         trendline="ols", template=PLOTLY_TEMPLATE,
                         title=f"Price vs {feat.title()}")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.box(df, x=feat, y="price_per_night",
                     color_discrete_sequence=[ACCENT],
                     template=PLOTLY_TEMPLATE, title=f"Price Distribution by {feat.title()}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Superhost & Guest Favourite Premium</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.box(df, x="is_superhost", y="price_per_night",
                     color="is_superhost", color_discrete_map={True: ACCENT, False: "#444"},
                     template=PLOTLY_TEMPLATE, title="Price: Superhost vs Non-Superhost")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.box(df, x="is_guest_favourite", y="price_per_night",
                     color="is_guest_favourite", color_discrete_map={True: "#f5a623", False: "#444"},
                     template=PLOTLY_TEMPLATE, title="Price: Guest Favourite vs Not")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RATINGS & RANKINGS  (Q2)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⭐ Ratings & Rankings":
    st.markdown("# ⭐ Ratings & Rankings")
    st.markdown('<div class="section-header">Q2 · Top-10 & Bottom-10 Stays (Bayesian Score)</div>', unsafe_allow_html=True)
    st.caption("Score = weighted combination of rating index and number of reviews (Bayesian average).")

    df_scored = df.copy()
    df_scored["score"] = df_scored.apply(weighted_score, axis=1)

    id_col = "url" if "url" in df_scored.columns else df_scored.index.astype(str)
    label_col = "url" if "url" in df_scored.columns else "index"
    df_scored["label"] = df_scored[label_col].astype(str).str[-10:] if label_col == "url" else df_scored.index.astype(str)
    if "host_name" in df_scored.columns:
        df_scored["label"] = df_scored["host_name"] + " · " + df_scored["region"].fillna("")

    top10 = df_scored.nlargest(10, "score")
    bot10 = df_scored.nsmallest(10, "score")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🥇 Top 10")
        fig = px.bar(top10, x="score", y="label", orientation="h",
                     color="score", color_continuous_scale=["#ff6b8a", ACCENT],
                     template=PLOTLY_TEMPLATE,
                     hover_data=["review_index", "num_reviews", "price_per_night"])
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown("### 📉 Bottom 10")
        fig = px.bar(bot10, x="score", y="label", orientation="h",
                     color="score", color_continuous_scale=["#333", "#666"],
                     template=PLOTLY_TEMPLATE,
                     hover_data=["review_index", "num_reviews", "price_per_night"])
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x="review_index", nbins=20,
                           color_discrete_sequence=[ACCENT],
                           template=PLOTLY_TEMPLATE, title="Review Index Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.scatter(df, x="num_reviews", y="review_index",
                         color="region" if "region" in df.columns else None,
                         size="price_per_night", opacity=0.7,
                         color_discrete_sequence=COLORS,
                         template=PLOTLY_TEMPLATE, title="Reviews Count vs Rating")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CHARACTERISTICS  (Q3)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏷️ Characteristics":
    st.markdown("# 🏷️ Characteristics Analysis")
    st.markdown('<div class="section-header">Q3 · Most Common Characteristics & Correlation with Rating</div>', unsafe_allow_html=True)

    all_chars = [c for lst in df["characteristics"] for c in lst]
    char_counts = Counter(all_chars)

    if not char_counts:
        st.info("No characteristics found in the dataset.")
    else:
        char_df = pd.DataFrame(char_counts.most_common(20), columns=["Characteristic", "Count"])

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(char_df, x="Count", y="Characteristic", orientation="h",
                         color="Count", color_continuous_scale=["#1b3a5c", ACCENT],
                         template=PLOTLY_TEMPLATE, title="Top 20 Most Common Characteristics")
            fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Correlation of each characteristic with rating
        top_chars = [c for c, _ in char_counts.most_common(15)]
        corr_data = []
        for char in top_chars:
            df[f"has_{char}"] = df["characteristics"].apply(lambda x: 1 if char in x else 0)
            c = df[[f"has_{char}", "review_index"]].corr().iloc[0, 1]
            avg_r = df[df[f"has_{char}"] == 1]["review_index"].mean()
            avg_r_no = df[df[f"has_{char}"] == 0]["review_index"].mean()
            corr_data.append({"Characteristic": char, "Correlation": round(c, 3),
                               "Avg Rating (has)": round(avg_r, 3),
                               "Avg Rating (no)": round(avg_r_no, 3)})

        corr_df = pd.DataFrame(corr_data).sort_values("Correlation", ascending=False)

        with col_b:
            fig = px.bar(corr_df, x="Correlation", y="Characteristic", orientation="h",
                         color="Correlation",
                         color_continuous_scale=["#cc2244", "#333", "#00aa55"],
                         template=PLOTLY_TEMPLATE,
                         title="Characteristic → Rating Correlation")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Avg Rating With vs Without Each Characteristic</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Has characteristic", x=corr_df["Characteristic"],
                              y=corr_df["Avg Rating (has)"], marker_color=ACCENT))
        fig.add_trace(go.Bar(name="Does not have", x=corr_df["Characteristic"],
                              y=corr_df["Avg Rating (no)"], marker_color="#444"))
        fig.update_layout(barmode="group", template=PLOTLY_TEMPLATE,
                          title="Average Rating: Has vs Doesn't Have Characteristic",
                          xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MAP  (Q5)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Map":
    st.markdown("# 🗺️ Property Map")
    st.markdown('<div class="section-header">Q5 · Properties per Location</div>', unsafe_allow_html=True)

    map_df = df.dropna(subset=["latitude", "longitude"])
    if map_df.empty:
        st.warning("No valid coordinates found.")
    else:
        map_df["info"] = (
            map_df.get("host_name", pd.Series("N/A", index=map_df.index)).fillna("N/A")
            + " | €" + map_df["price_per_night"].round(0).astype(str)
            + " | ★ " + map_df["review_index"].round(2).astype(str)
        )
        fig = px.scatter_mapbox(
            map_df,
            lat="latitude", lon="longitude",
            color="region" if "region" in map_df.columns else None,
            size="price_per_night",
            hover_name="info",
            hover_data={
                "price_per_night": True,
                "review_index": True,
                "num_reviews": True,
                "is_superhost": True,
                "latitude": False, "longitude": False,
            },
            color_discrete_sequence=COLORS,
            zoom=12, height=600,
            mapbox_style="carto-darkmatter",
            title="AirBnB Listings Map",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total plotted", len(map_df))
        with col_b:
            st.metric("Avg Price", f"€{map_df['price_per_night'].mean():.1f}")
        with col_c:
            st.metric("Avg Rating", f"{map_df['review_index'].mean():.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ML PRICE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Price Predictor":
    st.markdown("# 🤖 ML Price Predictor")

    # ── Load model.pkl ────────────────────────────────────────────────────────
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../PricePredictor/model/model.pkl")
    
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ No trained model found. Run `train_model.py` first:")
        st.code("cd Vizualization/src\npython train_model.py", language="bash")
        st.stop()

    @st.cache_resource
    def load_model(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    payload     = load_model(MODEL_PATH)
    model       = payload["model"]
    model_name  = payload["model_name"]
    feat_cols   = payload["feature_cols"]
    region_classes = payload["region_classes"]
    top_chars   = payload["top_chars"]
    mae         = payload["test_mae"]
    r2          = payload["test_r2"]
    cv_mae      = payload["cv_mae"]
    cv_std      = payload["cv_std"]
    all_results = payload["all_results"]


    # ── Prediction form ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔮 Predict a Listing\'s Price</div>', unsafe_allow_html=True)
    st.markdown("Fill in the listing details and get an estimated nightly price:")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            p_guests  = st.number_input("Guests", 1, 10, 2)
            p_beds    = st.number_input("Beds", 1, 10, 1)
            p_bedrooms = st.number_input("Bedrooms", 0, 10, 1)
        with col2:
            p_baths    = st.number_input("Baths", 1, 7, 1)
            p_superhost = st.selectbox("Superhost?", [False, True])
            p_guestfav  = st.selectbox("Guest Favourite?", [False, True])
        with col3:
            p_rating  = st.slider("Review Index", 1.0, 5.0, 4.5, 0.01)
            p_reviews = st.number_input("Num Reviews", 0, 2000, 20)
            p_region  = st.selectbox("Region", region_classes)

        p_chars = st.multiselect("Characteristics (select all that apply)", top_chars)
        submitted = st.form_submit_button("💡 Predict Price", use_container_width=True)

    if submitted:
        region_enc = region_classes.index(p_region) if p_region in region_classes else 0
        row = {
            "guests": p_guests, "beds": p_beds, "bedrooms": p_bedrooms, "baths": p_baths,
            "is_superhost": int(p_superhost), "is_guest_favourite": int(p_guestfav),
            "review_index": p_rating, "num_reviews": p_reviews, "region_enc": region_enc,
        }
        for c in top_chars:
            row[f"char_{c}"] = 1 if c in p_chars else 0

        input_df = pd.DataFrame([row])[feat_cols]
        pred_price = model.predict(input_df)[0]

        # Price bucket
        q33 = df["price_per_night"].quantile(0.33)
        q66 = df["price_per_night"].quantile(0.66)
        if pred_price <= q33:
            bucket, bucket_color = "💚 Budget", "#00c851"
        elif pred_price <= q66:
            bucket, bucket_color = "🟡 Mid-range", "#f5a623"
        else:
            bucket, bucket_color = "🔴 Luxury", "#ff385c"

        st.markdown(f"""
        <div class="metric-card" style="margin-top:16px;">
            <div class="metric-label">Estimated Price per Night</div>
            <div class="metric-value" style="font-size:3rem;">€{pred_price:.2f}</div>
            <div style="font-size:1.2rem; margin-top:8px; color:{bucket_color}; font-weight:700;">{bucket}</div>
        </div>
        """, unsafe_allow_html=True)