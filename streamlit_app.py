# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide", page_title="🍅 Tomato Price Dashboard")
st.title("🍅 Tomato Price Dashboard")

# 1) Upload & load data
uploaded = st.sidebar.file_uploader(
    "📥 Upload your enriched tomato CSV",
    type="csv",
)
if not uploaded:
    st.warning("Please upload your enriched CSV to proceed.")
    st.stop()

@st.cache_data
def load_data(f) -> pd.DataFrame:
    df = pd.read_csv(f, parse_dates=["Date"])
    # ensure Month & index
    df["Month"] = df["Date"].dt.month
    df.set_index(["Center", "Date"], inplace=True)
    return df

data = load_data(uploaded)
centres = data.index.get_level_values(0).unique()

# 2) Sidebar controls
centre        = st.sidebar.selectbox("🏷️ Select Centre", centres)
rain_shock    = st.sidebar.slider("🌧️ Rainfall shock (%)", -50, 50, 0)
storage_credit= st.sidebar.number_input("💰 Storage credit (₹/quintal)", min_value=0.0, value=0.0, step=10.0)
cold_units    = st.sidebar.slider("❄️ Cold-chain units", 0, 100, 0)

# 3) Prepare series for this centre
horizon = 12
dfc = data.xs(centre).copy()

# apply rainfall shock
dfc["Rain_adj"] = dfc["Rainfall"] * (1 + rain_shock/100.0)

# feature set
feat_cols = ["Rain_adj", "Temperature", "Month", "Distance_km"]

# split train/test
train = dfc.iloc[:-horizon].dropna(subset=["Price"])
test  = dfc.iloc[-horizon:].dropna(subset=["Price"])

X_train = train[feat_cols]
y_train = train["Price"]
X_test  = test[feat_cols]
y_test  = test["Price"]
dates_test = test.index

# 4) Train XGBoost
xgb = XGBRegressor(random_state=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# 5) Train Prophet
prop_df = train.reset_index()[["Date","Price","Rain_adj","Temperature"]]
prop_df.columns = ["ds","y","Rain_adj","Temperature"]
m = Prophet(yearly_seasonality=True)
m.add_regressor("Rain_adj")
m.add_regressor("Temperature")
m.fit(prop_df)

future = m.make_future_dataframe(periods=horizon, freq="M")
# align regressors
future["Rain_adj"]     = dfc["Rain_adj"].reindex(future["ds"]).values
future["Temperature"]  = dfc["Temperature"].reindex(future["ds"]).values

forecast = m.predict(future)
prophet_pred = forecast["yhat"].iloc[-horizon:].values

# 6) Ensemble
ens_pred = (xgb_pred + prophet_pred) / 2

# 7) Plot historical + forecasts
fig, ax = plt.subplots(figsize=(10,4))
# last 2 years of actual
window = horizon*2
ax.plot(dfc.index[-window:], dfc["Price"].iloc[-window:], label="Actual", color="black")
ax.plot(dates_test, xgb_pred,    label="XGB Forecast",    linestyle="--")
ax.plot(dates_test, prophet_pred,label="Prophet Forecast", linestyle="-.")
ax.plot(dates_test, ens_pred,    label="Ensemble",         linewidth=2, alpha=0.7)
ax.set_title(f"{centre}: 12-Month Forecasts under {rain_shock:+}% Rainfall Shock")
ax.set_ylabel("Price (₹)")
ax.legend()
st.pyplot(fig)

# 8) RMSE metrics
rmse_x = np.sqrt(mean_squared_error(y_test, xgb_pred))
rmse_p = np.sqrt(mean_squared_error(y_test, prophet_pred))
rmse_e = np.sqrt(mean_squared_error(y_test, ens_pred))

st.subheader("📊 Forecast Accuracy (RMSE)")
st.write(f"- XGBoost:   {rmse_x:0.2f}")
st.write(f"- Prophet:   {rmse_p:0.2f}")
st.write(f"- Ensemble:  {rmse_e:0.2f}")

# 9) Simple ROI simulation
st.subheader("💡 ROI Simulation")
annual_benefit = cold_units * storage_credit * 12  # simplistic: credit × units × months
st.write(f"- Storage credit per month: ₹{storage_credit:0.2f}")
st.write(f"- Cold-chain units: {cold_units}")
st.write(f"**→ Estimated annual benefit:** ₹{annual_benefit:,.2f}")

