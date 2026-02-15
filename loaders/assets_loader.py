import pandas as pd

ASSETS_FILE = "data/assets_master_all.csv"

def load_assets():
    df = pd.read_csv(ASSETS_FILE)
    return df
