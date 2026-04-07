"""
QuantumFolio - Expert Mode with Real Quantum Optimization (Qiskit QAOA)
========================================================================

Complete implementation:
1. Classical pre-filtering → Top N assets by Sharpe ratio
2. Real Quantum QAOA → Discrete combinatorial optimization (binary selection)
3. Equal-weight allocation across QAOA-selected assets

Install:
pip install qiskit qiskit-optimization qiskit-algorithms scipy numpy pandas streamlit plotly

Run:
streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Data loaders
import sys
sys.path.append('.')
try:
    from loaders.assets_loader import load_assets
    from loaders.universe_loader import load_universe
    from loaders.price_loader import load_prices, fetch_benchmark_api
    LOADERS_AVAILABLE = True
except ImportError:
    LOADERS_AVAILABLE = False

# Qiskit imports
QISKIT_AVAILABLE = False
QISKIT_ERROR_MESSAGE = ""

try:
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
    from qiskit.algorithms.optimizers import COBYLA
    try:
        from qiskit_aer import Aer
    except ImportError:
        from qiskit import Aer
    from qiskit.utils import QuantumInstance
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_ERROR_MESSAGE = str(e)
    # Define dummy types for when Qiskit is not available
    QuadraticProgram = type('QuadraticProgram', (), {})

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Portfolio optimization",
    page_icon="⚛️",
    layout="wide"
)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f0c1d 0%, #1a1435 100%); font-family: Arial, sans-serif;}
    h1, h2, h3 {color: #00f5ff !important; text-shadow: 0 0 20px rgba(0, 245, 255, 0.5);}
    .quantum-card {
        background: rgba(26, 20, 53, 0.6);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    .metric-card {
        background: rgba(0, 245, 255, 0.1);
        border-left: 4px solid #00f5ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #00f5ff;
    }
    .metric-label {
        font-size: 0.75em;
        color: #9ca3af;
        text-transform: uppercase;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00f5ff, #8a2be2);
        color: #0f0c1d;
        font-weight: 700;
        border-radius: 8px;
        padding: 12px 24px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'step' not in st.session_state:
    st.session_state.step = 1

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_universe_data(universe_name: str, lookback_days: int = 1260):
    """Load price data from universe - matches original app.py data loading"""

    if not LOADERS_AVAILABLE:
        st.error("❌ Data loaders not available. Check that loaders/ directory exists.")
        st.info("""
        Expected structure:
        - loaders/universe_loader.py
        - loaders/price_loader.py
        - data/universes/*.csv
        - data/prices/*.csv
        """)
        return [], {}, {}

    # Step 1: Load universe tickers
    ticker_to_sector = {}
    try:
        universe_data = load_universe(universe_name)
        if isinstance(universe_data, pd.DataFrame):
            # Extract sector information if available
            if 'sector' in universe_data.columns:
                ticker_to_sector = dict(zip(universe_data['ticker'], universe_data['sector']))

            # Extract ticker column
            if 'ticker' in universe_data.columns:
                universe_tickers = universe_data['ticker'].tolist()
            else:
                universe_tickers = universe_data.index.tolist()
        else:
            universe_tickers = universe_data
    except Exception as e:
        st.error(f"Failed to load data {universe_name}: {e}")
        return [], {}, {}

    st.info(f"Loading {len(universe_tickers)} tickers from {universe_name}")

    # Step 2: Load price data for each ticker (same as original app.py)
    prices_dict = {}
    progress = st.progress(0)

    for i, ticker in enumerate(universe_tickers):
        try:
            df = load_prices(ticker)  # loader: single ticker only

            # Normalize date handling (EXACT SAME AS ORIGINAL)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

                # 🔑 CRITICAL FIX: remove timezone if present
                if df["date"].dt.tz is not None:
                    df["date"] = df["date"].dt.tz_localize(None)

                df = df.set_index("date")

            df = df.sort_index()

            # Get close prices
            if "close" in df.columns:
                prices = df["close"]
            elif "adj_close" in df.columns:
                prices = df["adj_close"]
            else:
                continue

            # Filter by lookback period
            if len(prices) > lookback_days:
                prices = prices.iloc[-lookback_days:]

            # Only keep if we have at least 1 year of data
            if len(prices) >= 252 and not prices.empty:
                prices_dict[ticker] = prices

            progress.progress((i + 1) / len(universe_tickers))

        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
            continue

    progress.empty()

    if len(prices_dict) == 0:
        st.error("❌ No valid price data loaded")
        return [], {}, {}

    st.success(f"✅ Loaded prices for {len(prices_dict)} assets with sufficient data")
    return list(prices_dict.keys()), prices_dict, ticker_to_sector


def classical_prefilter(prices_dict: Dict[str, pd.Series], n_candidates: int, rf_rate: float):
    """Step 1: Select top assets by Sharpe ratio"""
    sharpe_ratios = {}

    for ticker, prices in prices_dict.items():
        returns = prices.pct_change().dropna()
        if len(returns) < 20:
            continue

        mean_ret = returns.mean() * 252
        vol = returns.std() * np.sqrt(252)

        if vol > 0:
            sharpe_ratios[ticker] = (mean_ret - rf_rate) / vol

    sharpe_series = pd.Series(sharpe_ratios).sort_values(ascending=False)
    return sharpe_series.head(n_candidates).index.tolist(), sharpe_series


def build_qaoa_problem(
    candidates,
    prices_dict,
    n_select,
    risk_aversion
)-> Tuple[Any, pd.Series, pd.DataFrame]:
    """
    Step 2: Build QAOA optimization problem

    Formulate as QUBO:
    - Binary variables: x_i = 1 if asset i selected, 0 otherwise
    - Constraint: Σx_i = n_select
    - Objective: Maximize diversification / returns
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit not available. Install with: pip install qiskit qiskit-optimization qiskit-algorithms")

    n = len(candidates)

    # Calculate returns and covariance
    price_df = pd.DataFrame({t: prices_dict[t] for t in candidates})
    returns = price_df.pct_change().dropna()

    # Geometric annualized return
    compounded = (1 + returns).prod()
    years = len(returns) / 252
    expected_returns = compounded ** (1 / years) - 1
    # --- Shrinkage toward long-term market return ---
    market_mean = 0.08  # 8% realistic long-term equity return
    shrinkage = 0.25  # 0.5–0.8 works well

    expected_returns = (
            shrinkage * expected_returns
            + (1 - shrinkage) * market_mean
    )

    cov_matrix = returns.cov() * 252

    # Create quadratic program
    qp = QuadraticProgram('portfolio_selection')

    # Binary variables for each asset
    for i, ticker in enumerate(candidates):
        qp.binary_var(f'x_{i}')

    # -------------------------
    # QUBO with cardinality penalty
    # -------------------------
    penalty = 50.0  # ← tune between 20–200

    linear = {}
    quadratic = {}

    # Expected return term
    # --- Mean-Variance Objective ---

    for i in range(n):
        # Linear part: -Return + lambda * variance
        linear[f'x_{i}'] = (
                -expected_returns.iloc[i]
                + risk_aversion * cov_matrix.iloc[i, i]
        )

    # Add covariance interaction terms
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(f'x_{i}', f'x_{j}')] = (
                    2 * risk_aversion * cov_matrix.iloc[i, j]
            )

    # Cardinality penalty: (sum x_i - n_select)^2
    for i in range(n):
        # Linear part
        linear[f'x_{i}'] += penalty * (1 - 2 * n_select)

        # Quadratic part
        for j in range(i + 1, n):
            quadratic[(f'x_{i}', f'x_{j}')] = (
                quadratic.get((f'x_{i}', f'x_{j}'), 0)
                + 2 * penalty
            )

    # Set the objective function with penalty constraint
    qp.minimize(linear=linear, quadratic=quadratic)

    return qp, expected_returns, cov_matrix


def greedy_fallback(
    candidates: list,
    sharpe_series: pd.Series,
    n_select: int
) -> list:
    """
    Classical greedy fallback:
    Select top n_select assets by Sharpe ratio.

    Args:
        candidates: list of candidate tickers (e.g. top 50)
        sharpe_series: pd.Series with Sharpe ratios indexed by ticker
        n_select: number of assets to select (e.g. 8–12)

    Returns:
        List of selected tickers
    """
    # Filter Sharpe series to candidates only
    filtered = sharpe_series.loc[
        sharpe_series.index.intersection(candidates)
    ]

    # Select top n_select
    return filtered.sort_values(ascending=False).head(n_select).index.tolist()


def calculate_portfolio_metrics(
    selected_assets: List[str],
    prices_dict: Dict[str, pd.Series],
    rf_rate: float
) -> Dict[str, float]:
    """
    Calculate return, risk, and Sharpe ratio for a given asset selection

    Args:
        selected_assets: List of asset tickers
        prices_dict: Price history
        rf_rate: Risk-free rate

    Returns:
        Dictionary with 'return', 'risk', 'sharpe'
    """
    try:
        # Prepare data
        price_df = pd.DataFrame({t: prices_dict[t] for t in selected_assets})
        returns = price_df.pct_change().dropna()

        # Equal-weight portfolio for quick metrics
        n = len(selected_assets)
        weights = np.ones(n) / n

        # Geometric annualized return
        compounded = (1 + returns).prod()
        years = len(returns) / 252
        expected_returns = compounded ** (1 / years) - 1
        # --- Shrinkage toward long-term market return ---
        market_mean = 0.08  # 8% realistic long-term equity return
        shrinkage = 0.25  # 0.5–0.8 works well

        expected_returns = (
                shrinkage * expected_returns
                + (1 - shrinkage) * market_mean
        )

        cov_matrix = returns.cov() * 252

        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe = (port_return - rf_rate) / port_vol if port_vol > 0 else 0

        return {
            'return': port_return,
            'risk': port_vol,
            'sharpe': sharpe
        }
    except:
        return {
            'return': 0.0,
            'risk': 0.0,
            'sharpe': 0.0
        }


def solve_qaoa(
    qp,
    candidates: list,
    sharpe_series: pd.Series,
    n_select: int,
    use_quantum: bool = True,
    p: int = 1,
    prices_dict: Dict[str, pd.Series] = None,
    rf_rate: float = 0.04,
    top_k: int = 5
):
    """
    Step 2: Solve asset selection using QAOA and extract top-K portfolios.

    Args:
        qp: QuadraticProgram (QUBO)
        candidates: list of candidate assets (e.g. top 50)
        sharpe_series: Sharpe ratios for candidates
        n_select: number of assets to select (e.g. 8–12)
        use_quantum: whether to use QAOA
        p: QAOA depth
        prices_dict: Price history for calculating metrics (NEW)
        rf_rate: Risk-free rate (NEW)
        top_k: Number of alternative portfolios to extract (NEW)

    Returns:
        (results_dict, solver_type)

    results_dict contains:
        - qaoa_assets: Best portfolio from QAOA
        - exact_assets: Exact solution (if n <= 10)
        - qaoa_obj: QAOA objective value
        - exact_obj: Exact objective value
        - top_portfolios: List of top-K alternative portfolios (NEW)
    """

    # -----------------------------
    # Quantum path (QAOA)
    # -----------------------------
    if use_quantum and QISKIT_AVAILABLE:
        try:
            try:
                from qiskit_aer import Aer
            except ImportError:
                from qiskit import Aer
            from qiskit.utils import QuantumInstance

            backend = Aer.get_backend("qasm_simulator")
            quantum_instance = QuantumInstance(backend)

            optimizer = COBYLA(maxiter=100)

            qaoa = QAOA(
                optimizer=optimizer,
                reps=p,
                quantum_instance=quantum_instance
            )

            qaoa_optimizer = MinimumEigenOptimizer(qaoa)
            result = qaoa_optimizer.solve(qp)

            # Exact classical solution (for benchmarking)
            exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
            exact_result = exact_solver.solve(qp)

            exact_assets = [
                candidates[i]
                for i, val in enumerate(exact_result.x)
                if val > 0.5
            ]

            # Extract binary solution from QAOA (best solution)
            selected_assets = [
                candidates[i]
                for i, val in enumerate(result.x)
                if val > 0.5
            ]

            # NEW: Extract top-K portfolios from QAOA samples
            top_portfolios = []

            if hasattr(result, 'samples') and result.samples is not None:
                # Extract samples from QAOA result
                samples = result.samples[:top_k * 3]  # Get more samples to filter

                for sample in samples:
                    bitstring = sample.x

                    # Filter: only portfolios with correct cardinality
                    if sum(bitstring) == n_select:
                        portfolio_assets = [
                            candidates[i]
                            for i, val in enumerate(bitstring)
                            if val > 0.5
                        ]

                        # Calculate metrics for this portfolio
                        if prices_dict is not None and portfolio_assets:
                            metrics = calculate_portfolio_metrics(
                                portfolio_assets,
                                prices_dict,
                                rf_rate
                            )

                            top_portfolios.append({
                                'assets': portfolio_assets,
                                'objective': sample.fval,
                                'probability': sample.probability,
                                'expected_return': metrics['return'],
                                'risk': metrics['risk'],
                                'sharpe': metrics['sharpe']
                            })
                        else:
                            top_portfolios.append({
                                'assets': portfolio_assets,
                                'objective': sample.fval,
                                'probability': sample.probability,
                                'expected_return': None,
                                'risk': None,
                                'sharpe': None
                            })

                # Sort by objective (best first)
                top_portfolios = sorted(
                    top_portfolios,
                    key=lambda x: x['objective']
                )[:top_k]

            # Safety check (QAOA should respect constraint, but just in case)
            if len(selected_assets) != n_select:
                # Rank selected assets by Sharpe and trim
                ranked = sorted(
                    selected_assets,
                    key=lambda t: sharpe_series.get(t, 0),
                    reverse=True
                )
                selected_assets = ranked[:n_select]

            # Return both QAOA and exact results for comparison
            return {
                "qaoa_assets": selected_assets,
                "exact_assets": exact_assets,
                "qaoa_obj": result.fval,
                "exact_obj": exact_result.fval,
                "top_portfolios": top_portfolios  # NEW
            }, "QAOA"

        except Exception as e:
            print(f"⚠️ QAOA failed: {e}")
            print("➡️ Falling back to greedy classical selection")

    # -----------------------------
    # Classical SAFE fallback
    # -----------------------------
    selected_assets = greedy_fallback(
        candidates=candidates,
        sharpe_series=sharpe_series,
        n_select=n_select
    )

    return {
        "qaoa_assets": selected_assets,
        "exact_assets": selected_assets,
        "qaoa_obj": None,
        "exact_obj": None,
        "top_portfolios": []
    }, "Classical (Greedy)"



def extract_selected_assets(result: Any, candidates: List[str]) -> List[str]:
    """Extract selected assets from QAOA result"""
    selected = []

    for i, ticker in enumerate(candidates):
        if result.x[i] > 0.5:  # Binary variable threshold
            selected.append(ticker)

    return selected


def optimize_weights(
    selected_assets: List[str],
    prices_dict: Dict[str, pd.Series],
    rf_rate: float = 0.04,
    risk_aversion: float = 0.5
):
    """
    Step 3: Compute optimal portfolio weights via mean-variance optimization.

    QAOA selects *which* assets (binary combinatorial problem).
    This function then solves *how much* to allocate to each, using the
    classical mean-variance objective:

        maximize  μᵀw - (λ/2) wᵀΣw
        subject to  Σw = 1,  w_i >= 0.05  (min 5% per asset)

    Higher risk_aversion (λ) → weights shift toward lower-volatility assets.
    Lower risk_aversion → weights tilt toward higher-return assets.
    """
    from scipy.optimize import minimize

    n = len(selected_assets)
    if n == 0:
        return {}, {'expected_return': 0, 'expected_risk': 0, 'sharpe_ratio': 0, 'n_assets': 0}

    price_df = pd.DataFrame({t: prices_dict[t] for t in selected_assets})
    returns = price_df.pct_change().dropna()

    # Geometric annualized returns with shrinkage
    compounded = (1 + returns).prod()
    years = len(returns) / 252
    expected_returns = compounded ** (1 / years) - 1
    market_mean = 0.08
    shrinkage = 0.25
    expected_returns = shrinkage * expected_returns + (1 - shrinkage) * market_mean
    mu = expected_returns.values

    cov_matrix = returns.cov() * 252
    sigma = cov_matrix.values

    # Mean-variance objective: maximize μᵀw - (λ/2) wᵀΣw
    def neg_utility(w):
        port_return = np.dot(w, mu)
        port_variance = np.dot(w, np.dot(sigma, w))
        return -(port_return - (risk_aversion / 2.0) * port_variance)

    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    # Bounds: each asset gets at least 5%, at most 60%
    min_weight = max(0.05, 1.0 / (n * 3))  # adaptive floor
    bounds = [(min_weight, 0.60)] * n

    # Warm-start with equal weights
    w0 = np.ones(n) / n

    result = minimize(
        neg_utility,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    if result.success:
        weights = result.x
        # Re-normalize to handle tiny floating point drift
        weights = np.clip(weights, 0, 1)
        weights /= weights.sum()
    else:
        # Fallback to equal weights if optimizer fails
        weights = np.ones(n) / n

    allocation = {ticker: float(w) for ticker, w in zip(selected_assets, weights)}

    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
    sharpe = (port_return - rf_rate) / port_vol if port_vol > 0 else 0

    metrics = {
        'expected_return': port_return,
        'expected_risk': port_vol,
        'sharpe_ratio': sharpe,
        'n_assets': n
    }

    return allocation, metrics


def run_backtest(allocation, prices_dict, initial_capital, rf_rate: float = 0.04):

    tickers = list(allocation.keys())
    weights = np.array([allocation[t] for t in tickers])

    # Align price data
    price_df = pd.concat(
        [prices_dict[t] for t in tickers],
        axis=1,
        join='inner'
    )
    price_df.columns = tickers

    returns = price_df.pct_change().dropna()

    # ---- TRAIN / TEST SPLIT ----
    split_index = int(len(returns) * 0.7)

    test_returns = returns.iloc[split_index:]

    # Portfolio (optimized weights)
    portfolio_returns = test_returns.dot(weights)
    portfolio_curve = initial_capital * (1 + portfolio_returns).cumprod()

    # Benchmark: equal-weight across the same assets (naive baseline)
    eq_weights = np.ones(len(tickers)) / len(tickers)
    benchmark_returns = test_returns.dot(eq_weights)
    benchmark_curve = initial_capital * (1 + benchmark_returns).cumprod()

    final_value = portfolio_curve.iloc[-1]
    total_return = (final_value / initial_capital) - 1

    years = len(portfolio_returns) / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    volatility = portfolio_returns.std() * np.sqrt(252)

    sharpe = (
        (portfolio_returns.mean() * 252 - rf_rate) /
        (portfolio_returns.std() * np.sqrt(252))
        if portfolio_returns.std() > 0 else 0
    )

    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Benchmark stats
    bench_final = benchmark_curve.iloc[-1]
    bench_return = (bench_final / initial_capital) - 1
    bench_cagr = (1 + bench_return) ** (1 / years) - 1 if years > 0 else 0
    bench_vol = benchmark_returns.std() * np.sqrt(252)
    bench_sharpe = (
        (benchmark_returns.mean() * 252 - rf_rate) / bench_vol
        if bench_vol > 0 else 0
    )

    return {
        'portfolio_curve': portfolio_curve,
        'benchmark_curve': benchmark_curve,
        'final_value': final_value,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'bench_final': bench_final,
        'bench_return': bench_return,
        'bench_cagr': bench_cagr,
        'bench_vol': bench_vol,
        'bench_sharpe': bench_sharpe,
    }


# ============================================================================
# MAIN APP
# ============================================================================

st.markdown("# Hybrid quantum portfolio optimization")

if not QISKIT_AVAILABLE:
    st.error(f"""
    ❌ **Qiskit packages not found**
    
    Error: `{QISKIT_ERROR_MESSAGE}`
    
    **Install Qiskit:**
    """)

    st.code("""
# Option 1: Latest stable versions
pip install qiskit qiskit-algorithms qiskit-optimization

# Option 2: If that doesn't work, try specific versions
pip install qiskit==0.45.0 qiskit-algorithms==0.2.1 qiskit-optimization==0.5.0
    """, language="bash")

    st.warning("""
    ⚠️ **After installing:**
    1. **Stop** this Streamlit app (Ctrl+C in terminal)
    2. **Restart** the app: `streamlit run app.py`
    3. If it still doesn't work, try closing the terminal and opening a new one
    """)

    st.info("""
    💡 **Check what's installed:**
    
    Run `python check_qiskit.py` to see detailed diagnostics
    """)

    st.stop()



# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================

if st.session_state.step == 1:
    st.markdown("## ⚙️ Configuration")

    with st.form("config_form"):
        col1, col2 = st.columns(2)

        with col1:
            universe = st.selectbox(
                "Universe",
                ['sp500', 'nifty500'],
                help="Select asset universe"
            )

            investment = st.number_input(
                "Investment Amount (₹)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000
            )

            n_prefilter = st.slider(
                "Pre-filter: Top N Assets",
                min_value=20,
                max_value=100,
                value=30,
                help="Select top N assets by Sharpe ratio before quantum"
            )

        with col2:
            n_select = st.slider(
                "Quantum: Select M Assets",
                min_value=5,
                max_value=20,
                value=10,
                help="Number of assets to select using QAOA"
            )

            qaoa_depth = st.slider(
                "QAOA Depth (p)",
                min_value=1,
                max_value=3,
                value=1,
                help="QAOA circuit depth (higher = more accurate but slower)"
            )

            risk_aversion = st.slider(
                "Risk Aversion (λ)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Higher = more conservative portfolio (also affects final weight distribution)"
            )

            rf_rate = st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.25,
                help="Annual risk-free rate (e.g. 4% for Indian T-bills, 5.5% for US Fed funds)"
            ) / 100.0



        submitted = st.form_submit_button("## Start Optimization", use_container_width=True)

        if submitted:
            st.session_state.universe = universe
            st.session_state.investment = investment
            st.session_state.n_prefilter = n_prefilter
            st.session_state.n_select = n_select
            st.session_state.qaoa_depth = qaoa_depth
            st.session_state.risk_aversion = risk_aversion
            st.session_state.rf_rate = rf_rate
            st.session_state.step = 2
            st.rerun()

# ============================================================================
# STEP 2: OPTIMIZATION
# ============================================================================

elif st.session_state.step == 2:
    st.markdown("## Quantum Optimization")

    # Load data
    if not hasattr(st.session_state, 'prices_dict'):
        with st.spinner("Loading data..."):
            tickers, prices_dict, ticker_to_sector = load_universe_data(st.session_state.universe)

            if len(prices_dict) == 0:
                st.error("Failed to load data")
                st.stop()

            st.session_state.tickers = tickers
            st.session_state.prices_dict = prices_dict
            st.session_state.ticker_to_sector = ticker_to_sector  # NEW

    # Step 1: Classical Pre-filter
    if not hasattr(st.session_state, 'candidates'):
        st.markdown("## Classical Pre-filtering")

        with st.spinner("Calculating Sharpe ratios..."):
            candidates, sharpe_ratios = classical_prefilter(
                st.session_state.prices_dict,
                st.session_state.n_prefilter,
                st.session_state.rf_rate
            )

            st.session_state.candidates = candidates
            st.session_state.sharpe_ratios = sharpe_ratios

        st.success(f"✅ Selected top {len(candidates)} assets")

        # Add sector information if available
        ticker_to_sector = st.session_state.get('ticker_to_sector', {})

        # Option to expand/collapse full list
        # Always show all selected candidates (top N prefiltered)
        display_tickers = sharpe_ratios.loc[candidates]

        # Build dataframe with sector column
        if ticker_to_sector:
            df = pd.DataFrame({
                'Ticker': display_tickers.index,
                'Sector': [ticker_to_sector.get(t, 'Unknown') for t in display_tickers.index],
                'Sharpe': display_tickers.values.round(2)
            })
        else:
            df = pd.DataFrame({
                'Ticker': display_tickers.index,
                'Sharpe': display_tickers.values.round(2)
            })

        st.dataframe(df, hide_index=True, use_container_width=True)



    # Step 2: Quantum QAOA
    if not hasattr(st.session_state, 'selected_assets'):
        st.markdown("## Quantum QAOA Asset Selection")

        with st.spinner(f"Running QAOA (depth={st.session_state.qaoa_depth})..."):
            # Build problem
            qp, exp_returns, cov_matrix = build_qaoa_problem(
                st.session_state.candidates,
                st.session_state.prices_dict,
                st.session_state.n_select,
                st.session_state.risk_aversion
            )

            # Solve with QAOA
            results_dict, solver_type = solve_qaoa(
                qp=qp,
                candidates=st.session_state.candidates,
                sharpe_series=st.session_state.sharpe_ratios,
                n_select=st.session_state.n_select,
                use_quantum=True,
                p=st.session_state.qaoa_depth,
                prices_dict=st.session_state.prices_dict,
                rf_rate=st.session_state.rf_rate,
                top_k=st.session_state.get('top_k', 5)  # NEW
            )

            # Extract selected assets from QAOA
            selected_assets = results_dict["qaoa_assets"]

            st.session_state.selected_assets = selected_assets
            st.session_state.solver_type = solver_type
            st.session_state.results_dict = results_dict

        st.markdown(f"""
        <div class='quantum-card'>
        <strong>Quantum Selection Complete</strong><br>
        Solver: <strong>{st.session_state.solver_type}</strong><br>
        Selected: <strong>{len(st.session_state.selected_assets)} assets</strong>
        </div>
        """, unsafe_allow_html=True)

        # Display Quantum vs Classical Comparison
        st.markdown("### 🔬 Quantum vs Classical Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("QAOA Solution")
            st.write(f"Objective Value: {st.session_state.results_dict['qaoa_obj']:.4f}")
            st.write(f"Selected Assets: {st.session_state.results_dict['qaoa_assets']}")

        with col2:
            st.markdown("Exact Classical Solution")
            st.write(f"Objective Value: {st.session_state.results_dict['exact_obj']:.4f}")
            st.write(f"Selected Assets: {st.session_state.results_dict['exact_assets']}")

        # Display selected assets table (QAOA)
        st.markdown("### QAOA Selected Assets")
        ticker_to_sector = st.session_state.get('ticker_to_sector', {})

        if ticker_to_sector:
            selected_df = pd.DataFrame({
                'Ticker': st.session_state.selected_assets,
                'Sector': [ticker_to_sector.get(t, 'Unknown') for t in st.session_state.selected_assets],
                'Sharpe': [st.session_state.sharpe_ratios.get(t, 0) for t in st.session_state.selected_assets]
            })
        else:
            selected_df = pd.DataFrame({
                'Ticker': st.session_state.selected_assets,
                'Sharpe': [st.session_state.sharpe_ratios.get(t, 0) for t in st.session_state.selected_assets]
            })

        st.dataframe(selected_df, hide_index=True, use_container_width=True)

        # Display Top-K Alternative Portfolios (if available)
        if 'top_portfolios' in st.session_state.results_dict and st.session_state.results_dict['top_portfolios']:
            st.markdown("### 🔬 Top-K Alternative Portfolios from QAOA")

            st.info("These are alternative asset selections discovered by QAOA, ranked by their objective value.")

            top_portfolios = st.session_state.results_dict['top_portfolios']
            ticker_to_sector = st.session_state.get('ticker_to_sector', {})

            # Build display dataframe
            portfolio_data = []
            for rank, portfolio in enumerate(top_portfolios, 1):
                # Get sector distribution
                if ticker_to_sector:
                    sectors = [ticker_to_sector.get(t, 'Unknown') for t in portfolio['assets']]
                    sector_summary = ', '.join(sorted(set(sectors)))
                else:
                    sector_summary = 'N/A'

                portfolio_data.append({
                    'Rank': rank,
                    'Probability': f"{portfolio.get('probability', 0) * 100:.2f}%" if portfolio.get('probability') else 'N/A',
                    'Expected Return': f"{portfolio.get('expected_return', 0) * 100:.2f}%" if portfolio.get('expected_return') is not None else 'N/A',
                    'Risk (Vol)': f"{portfolio.get('risk', 0) * 100:.2f}%" if portfolio.get('risk') is not None else 'N/A',
                    'Sharpe Ratio': f"{portfolio.get('sharpe', 0):.2f}" if portfolio.get('sharpe') is not None else 'N/A',
                    'Objective': f"{portfolio['objective']:.4f}",
                    'Assets': ', '.join(portfolio['assets']),
                    'Sectors': sector_summary
                })

            portfolio_df = pd.DataFrame(portfolio_data)
            st.dataframe(portfolio_df, hide_index=True, use_container_width=True)

            # Allow user to select an alternative
            st.markdown("Use Alternative Portfolio")

            selected_rank = st.selectbox(
                "Select alternative portfolio to use:",
                options=[0] + list(range(1, len(top_portfolios) + 1)),
                format_func=lambda x: "Best" if x == 0 else f"Alternative Rank {x}"
            )

            if selected_rank > 0:
                # Use alternative portfolio
                alt_portfolio = top_portfolios[selected_rank - 1]
                st.session_state.selected_assets = alt_portfolio['assets']
                st.info(f"Selected alternative portfolio (Rank {selected_rank}). Proceeding to equal-weight allocation.")

                st.session_state.step = 2
                st.rerun()

    # Step 3: Mean-Variance Weight Optimization
    if not hasattr(st.session_state, 'allocation'):
        st.markdown("## ⚖️ Portfolio Weight Optimization")
        st.info(
            f"QAOA selected the asset subset. Now computing optimal weights using "
            f"mean-variance optimization (λ={st.session_state.risk_aversion:.1f}, "
            f"rf={st.session_state.rf_rate*100:.2f}%). "
            f"Higher risk aversion shifts weight toward lower-volatility assets."
        )

        with st.spinner("Optimizing portfolio weights..."):
            allocation, metrics = optimize_weights(
                st.session_state.selected_assets,
                st.session_state.prices_dict,
                rf_rate=st.session_state.rf_rate,
                risk_aversion=st.session_state.risk_aversion
            )

            st.session_state.allocation = allocation
            st.session_state.metrics = metrics

        st.success("Portfolio allocation ready!")

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{metrics['expected_return']:.2%}</div>
                <div class='metric-label'>Expected Return</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{metrics['expected_risk']:.2%}</div>
                <div class='metric-label'>Risk (Vol)</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{metrics['sharpe_ratio']:.2f}</div>
                <div class='metric-label'>Sharpe Ratio</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{metrics['n_assets']}</div>
                <div class='metric-label'>Assets</div>
            </div>
            """, unsafe_allow_html=True)

        # Portfolio allocation table
        st.markdown("### 💼 Portfolio Allocation (Optimized Weights)")
        sorted_alloc = dict(sorted(allocation.items(), key=lambda x: x[1], reverse=True))

        # Add sector column if available
        ticker_to_sector = st.session_state.get('ticker_to_sector', {})
        if ticker_to_sector:
            alloc_df = pd.DataFrame([
                {
                    'Ticker': t,
                    'Sector': ticker_to_sector.get(t, 'Unknown'),
                    'Weight': f"{w*100:.2f}%",
                    'Amount': f"₹{st.session_state.investment * w:,.2f}"
                }
                for t, w in sorted_alloc.items()
            ])
        else:
            alloc_df = pd.DataFrame([
                {'Ticker': t, 'Weight': f"{w*100:.2f}%", 'Amount': f"₹{st.session_state.investment * w:,.2f}"}
                for t, w in sorted_alloc.items()
            ])
        st.dataframe(alloc_df, hide_index=True, use_container_width=True)

        # Display sector summary
        if ticker_to_sector:
            st.markdown("Sector Distribution")

            # Calculate sector weights
            sector_weights = {}
            for ticker, weight in st.session_state.allocation.items():
                sector = ticker_to_sector.get(ticker, 'Unknown')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

            # Sort by weight
            sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)

            # Display as dataframe
            sector_df = pd.DataFrame({
                'Sector': [s[0] for s in sorted_sectors],
                'Weight': [f"{s[1]*100:.2f}%" for s in sorted_sectors],
                'Allocation': [f"₹{st.session_state.investment * s[1]:,.2f}" for s in sorted_sectors]
            })

            st.dataframe(sector_df, hide_index=True, use_container_width=True)

            # Pie chart
            if len(sorted_sectors) > 1:
                fig = go.Figure(data=[go.Pie(
                    labels=[s[0] for s in sorted_sectors],
                    values=[s[1] * 100 for s in sorted_sectors],
                    hole=0.3
                )])

                fig.update_layout(
                    title="Sector Distribution",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15,12,29,0.8)',
                    font=dict(color='#9ca3af'),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        # try:
        #     from benchmark_models import compare_models
        # except ImportError:
        #     compare_models = None
        #
        # if compare_models is not None:
        #     price_df = pd.DataFrame({
        #         t: st.session_state.prices_dict[t]
        #         for t in st.session_state.allocation.keys()
        #     })
        #
        #     quantum_weights = np.array(list(st.session_state.allocation.values()))
        #
        #     results = compare_models(price_df, quantum_weights)
        #
        #     st.dataframe(results)

        # =====================================================
        # 📈 AUTOMATIC BACKTEST (Displayed on same page)
        # =====================================================

        # st.markdown("## 📈 Backtest Results")
        #
        # backtest_results = run_backtest(
        #     st.session_state.allocation,
        #     st.session_state.prices_dict,
        #     st.session_state.investment,
        #     rf_rate=st.session_state.rf_rate
        # )
        #
        # if backtest_results is None:
        #     st.error("Backtest failed.")
        # else:
        #
        #     col1, col2, col3, col4 = st.columns(4)
        #
        #     with col1:
        #         st.metric(
        #             "Final Value",
        #             f"₹{backtest_results['final_value']:,.0f}",
        #             delta=f"vs ₹{backtest_results['bench_final']:,.0f} eq-wt"
        #         )
        #
        #     with col2:
        #         st.metric(
        #             "Total Return",
        #             f"{backtest_results['total_return']:.2%}",
        #             delta=f"{backtest_results['total_return'] - backtest_results['bench_return']:.2%} vs eq-wt"
        #         )
        #
        #     with col3:
        #         st.metric(
        #             "CAGR",
        #             f"{backtest_results['cagr']:.2%}",
        #             delta=f"{backtest_results['cagr'] - backtest_results['bench_cagr']:.2%} vs eq-wt"
        #         )
        #
        #     with col4:
        #         st.metric(
        #             "Sharpe",
        #             f"{backtest_results['sharpe']:.2f}",
        #             delta=f"{backtest_results['sharpe'] - backtest_results['bench_sharpe']:.2f} vs eq-wt"
        #         )

            # Performance chart with benchmark
            # fig = go.Figure()
            # fig.add_trace(go.Scatter(
            #     x=backtest_results['portfolio_curve'].index,
            #     y=backtest_results['portfolio_curve'].values,
            #     mode='lines',
            #     name='Optimized Portfolio',
            #     line=dict(color='#00f5ff', width=2)
            # ))
            # fig.add_trace(go.Scatter(
            #     x=backtest_results['benchmark_curve'].index,
            #     y=backtest_results['benchmark_curve'].values,
            #     mode='lines',
            #     name='Equal-Weight Benchmark',
            #     line=dict(color='#f59e0b', width=1.5, dash='dash')
            # ))
            #
            # fig.update_layout(
            #     title="Portfolio vs Equal-Weight Benchmark (test period)",
            #     paper_bgcolor='rgba(0,0,0,0)',
            #     plot_bgcolor='rgba(15,12,29,0.8)',
            #     font=dict(color='#9ca3af'),
            #     legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb')),
            #     height=420,
            #     yaxis_title="Portfolio Value (₹)",
            #     xaxis_title="Date"
            # )
            #
            # st.plotly_chart(fig, use_container_width=True)
            #
            # col1, col2, col3, col4 = st.columns(4)
            #
            # with col1:
            #     st.metric("Volatility", f"{backtest_results['volatility']:.2%}",
            #               delta=f"{backtest_results['volatility'] - backtest_results['bench_vol']:.2%} vs eq-wt",
            #               delta_color="inverse")
            #
            # with col2:
            #     st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.2%}")
            #
            # with col3:
            #     st.metric("Bench CAGR", f"{backtest_results['bench_cagr']:.2%}")
            #
            # with col4:
            #     st.metric("Bench Sharpe", f"{backtest_results['bench_sharpe']:.2f}")