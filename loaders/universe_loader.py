import pandas as pd

def load_universe(universe_name: str) -> pd.DataFrame:
    universe = pd.read_csv(f"data/universes/{universe_name}.csv")
    assets = pd.read_csv("data/assets_master_all.csv")

    # Normalize column names
    universe.columns = universe.columns.str.lower().str.strip()
    assets.columns = assets.columns.str.lower().str.strip()

    if "ticker" not in universe.columns:
        raise ValueError(f"{universe_name}.csv must contain 'ticker' column")

    # Normalize tickers
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    assets["ticker"] = assets["ticker"].astype(str).str.upper().str.strip()

    # LEFT JOIN: keep all universe tickers
    df = universe.merge(
        assets[["ticker", "sector"]],
        on="ticker",
        how="left"
    )

    # Ensure sector column always exists
    df["sector"] = df["sector"].fillna("Unknown")

    if df.empty:
        raise ValueError("Universe CSV is empty")

    return df
