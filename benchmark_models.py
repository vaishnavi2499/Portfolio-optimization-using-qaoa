import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ==============================
# Utility Functions
# ==============================

def annualized_metrics(returns, rf=0.04):
    mean_ret = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sharpe = (mean_ret - rf) / vol if vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    years = len(returns) / 252
    cagr = cumulative.iloc[-1] ** (1 / years) - 1

    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Return": mean_ret,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd
    }


# ==============================
# 1️⃣ Mean-Variance Model
# ==============================

def mean_variance_portfolio(price_df, rf=0.04):
    returns = price_df.pct_change().dropna()

    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(mu)

    def neg_sharpe(w):
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -(port_ret - rf) / port_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints)
    weights = result.x

    portfolio_returns = returns.dot(weights)

    return weights, annualized_metrics(portfolio_returns)




# ==============================
# 3️⃣ Evaluate Your Quantum Portfolio
# ==============================

def evaluate_quantum(price_df, quantum_weights):
    returns = price_df.pct_change().dropna()
    portfolio_returns = returns.dot(quantum_weights)
    return annualized_metrics(portfolio_returns)


# ==============================
# Run Comparison
# ==============================

def compare_models(price_df, quantum_weights):
    print("Running Mean-Variance...")
    mv_w, mv_metrics = mean_variance_portfolio(price_df)


    print("Evaluating Quantum Portfolio...")
    q_metrics = evaluate_quantum(price_df, quantum_weights)

    results = pd.DataFrame({
        "Hybrid quantum algorithm": q_metrics,
        "MeanVariance": mv_metrics,
    })

    return results
