import yfinance as yf
import pandas as pd
from tqdm import tqdm
import os

INPUT_FILE = "../data/universes/sp500.csv"
OUTPUT_FILE = "../data/assets_master_sp500.csv"
FAILED_FILE = "../data/failed_sp500_tickers.csv"

os.makedirs("data", exist_ok=True)

tickers = pd.read_csv(INPUT_FILE)["ticker"].tolist()

rows = []
failed = []
asset_id = 1

for ticker in tqdm(tickers, desc="Validating S&P 500 stocks"):
    try:
        yf_ticker = yf.Ticker(ticker)

        # US stocks: 1 day history is enough
        hist = yf_ticker.history(period="1d")

        if hist.empty:
            failed.append(ticker)
            continue

        rows.append({
            "asset_id": asset_id,
            "ticker": ticker,
            "name": ticker,                 # we’ll enrich names later
            "country": "US",
            "exchange": "NYSE/NASDAQ",
            "sector": "Unknown",             # fill later
            "asset_class": "Equity",
            "currency": "USD"
        })

        asset_id += 1

    except Exception:
        failed.append(ticker)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

pd.DataFrame({"ticker": failed}).to_csv(
    FAILED_FILE, index=False
)

print(f"✅ assets_master_sp500.csv created with {len(df)} valid stocks")
print(f"⚠️ {len(failed)} tickers failed")
