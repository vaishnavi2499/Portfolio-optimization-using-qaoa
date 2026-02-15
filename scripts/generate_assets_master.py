import yfinance as yf
import pandas as pd
from tqdm import tqdm

tickers = pd.read_csv("../data/universes/nifty500.csv")["ticker"].tolist()

rows = []
failed = []
asset_id = 1

for ticker in tqdm(tickers, desc="Validating NSE tickers"):
    try:
        yf_ticker = yf.Ticker(ticker)

        # Try to fetch recent price data
        hist = yf_ticker.history(period="5d")

        if hist.empty:
            failed.append(ticker)
            continue

        rows.append({
            "asset_id": asset_id,
            "ticker": ticker,
            "name": ticker.replace(".NS", ""),
            "country": "IN",
            "exchange": "NSE",
            "sector": "Unknown",      # enrich later
            "asset_class": "Equity",
            "currency": "INR"
        })

        asset_id += 1

    except Exception:
        failed.append(ticker)

df = pd.DataFrame(rows)
df.to_csv("../data/assets_master.csv", index=False)

pd.DataFrame({"ticker": failed}).to_csv(
    "../data/failed_tickers.csv", index=False
)

print(f"✅ assets_master.csv created with {len(df)} valid stocks")
print(f"⚠️ {len(failed)} tickers failed")
