import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

# ─── 0) YOUR PROJECT FOLDER ─────────────────────────────────────────────────
data_dir = r"C:\Users\neera\OneDrive\Desktop\New Project"

# ─── 1) Auto-locate the tomato CSV ───────────────────────────────────────────
tomato_file = None
for fn in os.listdir(data_dir):
    if re.match(r"(?i)tomato.*\.csv", fn):
        sample = pd.read_csv(os.path.join(data_dir, fn), nrows=3)
        if any(re.search(r"date", c, re.I) for c in sample.columns):
            tomato_file = fn
            break
if tomato_file is None:
    st.error("No `tomato*.csv` with a date column found in data_dir")
    st.stop()

# ─── 2) Detect date / price / centre columns ────────────────────────────────
tmp = pd.read_csv(os.path.join(data_dir, tomato_file), nrows=5)
date_col = next(c for c in tmp.columns if re.search(r"date", c, re.I))
try:
    price_col = next(c for c in tmp.columns if re.search(r"price", c, re.I))
except StopIteration:
    numerics = tmp.select_dtypes(include=[np.number]).columns.tolist()
    price_col = numerics[0] if numerics else None
    if price_col is None:
        st.error("No numeric column found to use as Price")
        st.stop()
others = [c for c in tmp.columns if c not in {date_col, price_col}]
objcols = tmp[others].select_dtypes(include=["object"]).columns.tolist()
centre_col = objcols[0] if objcols else others[0] if others else None
if centre_col is None:
    st.error("Cannot detect a centre/market column")
    st.stop()

# ─── 3) Load & rename tomato data ────────────────────────────────────────────
@st.cache_data
def load_tomato():
    df = pd.read_csv(
        os.path.join(data_dir, tomato_file),
        usecols=[date_col, price_col, centre_col],
        parse_dates=[date_col],
    )
    return df.rename(columns={date_col: "Date", price_col: "Price", centre_col: "Center"})
tomato = load_tomato()

# ─── 4) Monthly mean price by centre ────────────────────────────────────────
tomato["Date"] = tomato["Date"].dt.to_period("M").dt.to_timestamp()
price_df = (
    tomato
    .groupby(["Center", "Date"])['Price']
    .mean()
    .reset_index()
)

# ─── 5) Load weather series (rain & temp) ───────────────────────────────────
MONTH_ABBRS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def load_monthly(fn, colname):
    df = pd.read_csv(os.path.join(data_dir, fn))
    ycol = next(c for c in df.columns if re.search(r"year", c, re.I))
    month_cols = [c for c in df.columns if c[:3].title() in MONTH_ABBRS]
    long = (
        df.melt(id_vars=[ycol], value_vars=month_cols, var_name="Mon", value_name=colname)
          .dropna(subset=[colname])
    )
    long["Mon"] = long["Mon"].str[:3].str.title()
    long = long[long["Mon"].isin(MONTH_ABBRS)]
    long["Date"] = pd.to_datetime(
        long[ycol].astype(int).astype(str) + "-" + long["Mon"],
        format="%Y-%b"
    )
    return long[["Date", colname]].sort_values("Date")

rain_df = load_monthly("rainfall.csv", "Rainfall")
temp_df = load_monthly("temperature.csv", "Temperature")

# ─── 6) Load centre distances ────────────────────────────────────────────────
dist_df = pd.read_csv(os.path.join(data_dir, "center_distances.csv"))
if any(c.lower() == "centre" for c in dist_df.columns):
    dist_df = dist_df.rename(columns={next(c for c in dist_df.columns if c.lower()=="centre"): "Center"})
elif any(c.lower() == "center" for c in dist_df.columns):
    dist_df = dist_df.rename(columns={next(c for c in dist_df.columns if c.lower()=="center"): "Center"})
else:
    st.error("No centre/center column in center_distances.csv")
    st.stop()

# ─── 7) Merge everything together ───────────────────────────────────────────
df = (
    price_df
    .merge(rain_df, on="Date", how="left")
    .merge(temp_df, on="Date", how="left")
    .merge(dist_df, on="Center", how="left")
)

# ─── 8) Feature engineering ─────────────────────────────────────────────────
df["Time_to_Market_proxy"] = df["Distance_km"] / 500.0
df = df.sort_values(["Center", "Date"]).set_index(["Center", "Date"])
df["Volatility_4m"]  = df.groupby(level=0)["Price"].rolling(4).std().reset_index(0, drop=True)
df["Volatility_12m"] = df.groupby(level=0)["Price"].rolling(12).std().reset_index(0, drop=True)
for lag in (1, 2):
    df[f"Rainfall_lag{lag}"]    = df.groupby(level=0)["Rainfall"].shift(lag)
    df[f"Temperature_lag{lag}"] = df.groupby(level=0)["Temperature"].shift(lag)

dates = df.index.get_level_values("Date")
df["Lockdown_2020Q2"]  = ((dates >= pd.Timestamp("2020-04-01")) & (dates <= pd.Timestamp("2020-06-30"))).astype(int)
df["Onion_Export_Ban"] = 0
df = df.groupby(level=0, group_keys=False).apply(lambda d: d.ffill().bfill())

# ─── 9) Export enriched CSV (run once) ───────────────────────────────────────
out = df.reset_index()
out = out.loc[:, ~out.columns.duplicated()]
out.to_csv(os.path.join(data_dir, "enriched_tomato.csv"), index=False)

# ─── 10) STREAMLIT DASHBOARD ─────────────────────────────────────────────────
st.title("Tomato Price & Weather Dashboard")

@st.cache_data
def load_enriched():
    return pd.read_csv(os.path.join(data_dir, "enriched_tomato.csv"), parse_dates=["Date"])

data = load_enriched()

# Sidebar filters
centers = st.sidebar.multiselect(
    "Select Center",
    options=data["Center"].unique(),
    default=list(data["Center"].unique())
)
min_date = data["Date"].min().date()
max_date = data["Date"].max().date()
date_min, date_max = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)
filtered = data[
    (data["Center"].isin(centers)) &
    (data["Date"].dt.date.between(date_min, date_max))
]

# Main charts
st.subheader("Price over Time")
st.line_chart(
    filtered.set_index("Date")["Price"].unstack(level=0),
    use_container_width=True
)

st.subheader("Rainfall & Temperature Trends")
weather = filtered.set_index("Date")[['Rainfall', 'Temperature']]
st.area_chart(weather, use_container_width=True)

st.subheader("Price Volatility")
st.line_chart(
    filtered.set_index("Date")["Volatility_12m"],
    use_container_width=True
)

# Show raw data
if st.checkbox("Show raw data"):
    st.write(filtered.reset_index())
