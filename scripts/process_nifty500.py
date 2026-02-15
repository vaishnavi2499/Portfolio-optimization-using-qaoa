import pandas as pd
import os

RAW_FILE = "../data/raw/nifty500_raw.csv"
OUT_FILE = "../data/universes/nifty500.csv"

os.makedirs("data/universes", exist_ok=True)

df = pd.read_csv(RAW_FILE)

# NSE symbol → Yahoo Finance ticker
df["ticker"] = df["Symbol"].astype(str).str.strip() + ".NS"

# Keep only ticker column
universe = df[["ticker"]].drop_duplicates().sort_values("ticker")

universe.to_csv(OUT_FILE, index=False)

print(f"✅ NIFTY 500 universe created with {len(universe)} stocks")
