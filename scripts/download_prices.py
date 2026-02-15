import yfinance as yf
import pandas as pd
from tqdm import tqdm
import os
import time

ASSETS_FILE = "../data/assets_master_all.csv"
PRICES_DIR = "../data/prices"

START_DATE = "2018-01-01"   # long enough for backtests

os.makedirs(PRICES_DIR, exist_ok=True)

assets = pd.read_csv(ASSETS_FILE)
tickers = assets["ticker"].tolist()

def price_file_exists(ticker):
    return os.path.exists(f"{PRICES_DIR}/{ticker}.csv")

failed = []

for ticker in tqdm(tickers, desc="Downloading prices"):
    try:
        if price_file_exists(ticker):
            continue  # skip already downloaded

        yf_ticker = yf.Ticker(ticker)

        # NSE stocks need more days sometimes
        if ticker.endswith(".NS"):
            df = yf_ticker.history(start=START_DATE, period="max")
        else:
            df = yf_ticker.history(start=START_DATE)

        if df.empty:
            failed.append(ticker)
            continue

        price_df = df[["Close"]].reset_index()
        price_df.columns = ["date", "close"]

        price_df.to_csv(
            f"{PRICES_DIR}/{ticker}.csv",
            index=False
        )

        # gentle rate limit
        time.sleep(0.1)

    except Exception as e:
        failed.append(ticker)

pd.DataFrame({"ticker": failed}).to_csv(
    "../data/failed_price_downloads.csv", index=False
)

print(f"✅ Price download complete")
print(f"⚠️ Failed downloads: {len(failed)}")
