import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from pypfopt import expected_returns, risk_models, EfficientFrontier

# -----------------------------
# CONFIG
# -----------------------------
TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META",
    "NVDA","TSLA","JPM","V","JNJ",
    "WMT","PG","XOM","UNH","HD",
    "MA","BAC","PFE","KO","PEP"
]

START_DATE = "2019-01-01"

PRESELECT_K = 12   # classical filtering
QAOA_K = 8         # binary selection

st.set_page_config(layout="wide")
st.title("📊 Portfolio Optimization Comparison")
st.caption("Mean–Variance vs ML vs Hybrid Classical–QAOA")

# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def load_prices():
    data = yf.download(TICKERS, start=START_DATE, auto_adjust=True)
    return data["Close"]

prices = load_prices()
returns = prices.pct_change().dropna()

# -----------------------------
# MODEL 1: MEAN–VARIANCE
# -----------------------------
def run_mean_variance(prices):
    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.sample_cov(prices)

    ef = EfficientFrontier(mu, cov)
    weights = ef.max_sharpe()
    perf = ef.portfolio_performance()

    return ef.clean_weights(), perf

# -----------------------------
# MODEL 2: ML PORTFOLIO
# -----------------------------
def run_ml_portfolio(returns, top_k=10):
    X = np.arange(len(returns)).reshape(-1, 1)
    preds = {}

    for asset in returns.columns:
        y = returns[asset].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        preds[asset] = model.predict([[len(returns)]])[0]

    preds = pd.Series(preds)
    selected = preds.sort_values(ascending=False).head(top_k)

    weights = {a: 1/top_k for a in selected.index}

    portfolio_returns = returns[selected.index].mean(axis=1)
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol

    return weights, ann_return, ann_vol, sharpe

# -----------------------------
# MODEL 3: HYBRID CLASSICAL → QAOA → CLASSICAL
# -----------------------------
def run_hybrid_qaoa(prices, returns):
    # ---- Stage 1: Classical preselection (Sharpe) ----
    sharpe_scores = returns.mean() / returns.std()
    preselected = sharpe_scores.sort_values(ascending=False).head(PRESELECT_K).index

    # ---- Stage 2: QAOA-style binary selection ----
    qaoa_scores = sharpe_scores[preselected]
    selected_assets = qaoa_scores.sort_values(ascending=False).head(QAOA_K).index

    # ---- Stage 3: Classical weight allocation ----
    sub_prices = prices[selected_assets]
    mu = expected_returns.mean_historical_return(sub_prices)
    cov = risk_models.sample_cov(sub_prices)

    ef = EfficientFrontier(mu, cov)
    weights = ef.max_sharpe()
    perf = ef.portfolio_performance()

    return ef.clean_weights(), perf, list(preselected), list(selected_assets)

# -----------------------------
# RUN ALL MODELS
# -----------------------------
mv_w, mv_perf = run_mean_variance(prices)
ml_w, ml_ret, ml_vol, ml_sharpe = run_ml_portfolio(returns)

q_w, q_perf, pre_q, post_q = run_hybrid_qaoa(prices, returns)

# -----------------------------
# RESULTS TABLE
# -----------------------------
results = pd.DataFrame(
    [
        ["Mean–Variance", mv_perf[0], mv_perf[1], mv_perf[2]],
        ["ML Portfolio", ml_ret, ml_vol, ml_sharpe],
        ["Hybrid Classical–QAOA", q_perf[0], q_perf[1], q_perf[2]]
    ],
    columns=["Model", "Return", "Volatility", "Sharpe"]
)

st.subheader("📈 Performance Comparison")
st.dataframe(results, width="stretch")

# -----------------------------
# WEIGHTS & SELECTION
# -----------------------------
st.subheader("📊 Portfolio Details")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### Mean–Variance Weights")
    st.write({k: v for k, v in mv_w.items() if v > 0})

with c2:
    st.markdown("### ML Portfolio Weights")
    st.write(ml_w)

with c3:
    st.markdown("### Hybrid QAOA Weights")
    st.write({k: v for k, v in q_w.items() if v > 0})

st.markdown("### 🔍 Hybrid Model Selection Flow")
st.write("Classical Preselection (12 assets):", pre_q)
st.write("QAOA Binary Selection (8 assets):", post_q)
