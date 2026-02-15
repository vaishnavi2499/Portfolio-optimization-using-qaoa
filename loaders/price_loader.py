import os
import yfinance as yf
import pandas as pd
PRICES_DIR = "data/prices"
def fetch_benchmark_api(universe_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch benchmark index using Yahoo Finance
    """
    if universe_id == "nifty500":
        ticker = "^NSEI"     # NIFTY 50 (proxy)
    elif universe_id == "sp500":
        ticker = "^GSPC"     # S&P 500 index
    else:
        raise ValueError("Unknown universe")

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True
    )

    if df.empty:
        raise RuntimeError("Benchmark API returned no data")

    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["date", "close"]

    return df



def load_prices(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/prices/{ticker}.csv")

    # ---- REQUIRED FIX ----
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.set_index("date")

    # Remove timezone (CRITICAL for SP500)
    # Ensure datetime index
    # --- HARD DATE NORMALIZATION (MANDATORY) ---
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # Drop rows with invalid timestamps
    df = df[~df.index.isna()]

    # Convert to tz-naive AFTER normalization
    df.index = df.index.tz_convert(None)

    return df

