import yfinance as yf
import pandas as pd
import re

tickers = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]
start_date = "2009-01-01"
end_date = "2023-12-31"

for ticker in tickers:
    print(f"\nDownloading data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        print(f"⚠️ No data returned for {ticker}. Skipping.")
        continue

    df.reset_index(inplace=True)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns]

    # Clean column names
    df.columns = [re.sub(rf'(^|\s){ticker}($|\s)', '', col, flags=re.IGNORECASE).strip() for col in df.columns]

    # Normalize column names
    rename_map = {col: col.title() for col in df.columns if col.lower() in ["open", "high", "low", "close", "volume"]}
    if "Adj Close" in df.columns and "Close" not in df.columns:
        rename_map["Adj Close"] = "Close"
    df.rename(columns=rename_map, inplace=True)

    # Ensure numeric and drop rows with all price columns missing
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=price_cols, how="all", inplace=True)

    # Save to CSV
    filename = f"{ticker}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {filename} with columns {df.columns.tolist()}")

print("\nDone.")