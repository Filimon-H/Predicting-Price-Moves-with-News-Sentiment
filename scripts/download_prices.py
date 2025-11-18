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

    df = df.reset_index()

    # Flatten MultiIndex columns (yfinance can sometimes return MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join([str(x) for x in col if x and str(x) != '']).strip() for col in df.columns.values]

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Remove ticker prefix/suffix like "Close AAPL" -> "Close" or "AAPL Close" -> "Close"
    def strip_ticker(colname, t):
        # remove ticker at start or end, optionally separated by space/underscore/hyphen
        pat = re.compile(rf'^(?:{re.escape(t)}[ _-]+)|(?:[ _-]+{re.escape(t)})$', flags=re.IGNORECASE)
        new = pat.sub('', str(colname)).strip()
        return new or colname

    new_cols = [strip_ticker(c, ticker) for c in df.columns]
    # if stripping creates duplicates, make them unique by appending an index
    seen = {}
    unique_cols = []
    for col in new_cols:
        key = col
        if key in seen:
            seen[key] += 1
            unique = f"{col}_{seen[key]}"
        else:
            seen[key] = 0
            unique = col
        unique_cols.append(unique)
    df.columns = unique_cols

    # Normalize common column names (handle 'Adj Close' etc.)
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    if 'adj close' in cols_lower and 'close' not in cols_lower:
        rename_map[cols_lower['adj close']] = 'Close'
    # Standardize capitalization for primary price cols if present
    for expected in ["open", "high", "low", "close", "volume"]:
        if expected in cols_lower:
            actual = cols_lower[expected]
            proper = expected.title()
            if actual != proper:
                rename_map[actual] = proper
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Only collect price columns that exist (after normalization)
    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]

    # If price columns exist, clean them
    if price_cols:
        # make numeric (coerce bad values) and drop rows with all price cols missing
        df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")
        try:
            df = df.dropna(subset=price_cols, how="all")
        except KeyError as e:
            print(f"⚠️ dropna KeyError for {ticker}: {e}. Available cols: {df.columns.tolist()}")
            # fallback: only drop using intersection
            existing = [c for c in price_cols if c in df.columns]
            if existing:
                df = df.dropna(subset=existing, how="all")

    # Save
    filename = f"{ticker}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {filename} with columns {df.columns.tolist()}")

print("\nDone.")
