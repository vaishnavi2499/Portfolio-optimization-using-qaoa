import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch S&P 500 page: {response.status_code}")

    soup = BeautifulSoup(response.text, "lxml")

    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError("S&P 500 table not found on Wikipedia")

    tickers = []
    for row in table.find_all("tr")[1:]:
        ticker = row.find_all("td")[0].text.strip()
        ticker = ticker.replace(".", "-")  # Yahoo format
        tickers.append(ticker)

    return tickers


if __name__ == "__main__":
    sp500 = fetch_sp500()
    print(f"✅ Fetched {len(sp500)} S&P 500 stocks")

    pd.DataFrame({"ticker": sp500}).to_csv(
        "../data/universes/sp500.csv",
        index=False
    )

    print("✅ data/universes/sp500.csv written")
