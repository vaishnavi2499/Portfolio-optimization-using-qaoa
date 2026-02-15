import pandas as pd

# Input files
INDIA_FILE = "../data/assets_master.csv"
US_FILE = "../data/assets_master_sp500.csv"

OUTPUT_FILE = "../data/assets_master_all.csv"

# Load both
df_india = pd.read_csv(INDIA_FILE)
df_us = pd.read_csv(US_FILE)

# Combine
df_all = pd.concat([df_india, df_us], ignore_index=True)

# Drop duplicate tickers if any (safety)
df_all = df_all.drop_duplicates(subset=["ticker"], keep="first")

# Reassign asset_id cleanly
df_all = df_all.reset_index(drop=True)
df_all["asset_id"] = df_all.index + 1

# Sort for readability (optional)
df_all = df_all.sort_values(["country", "ticker"]).reset_index(drop=True)

# Save
df_all.to_csv(OUTPUT_FILE, index=False)

print("✅ assets_master_all.csv created")
print(f"Total assets: {len(df_all)}")
print(df_all["country"].value_counts())
