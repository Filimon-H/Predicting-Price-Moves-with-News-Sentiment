# src/news_stock_alignment.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


# ---------------------------
# Loading functions
# ---------------------------

def load_price_data(
    tickers: Iterable[str],
    data_dir: str | Path,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Load daily OHLCV price data for multiple tickers from CSV files.

    Expects files like:
        data_dir / "AAPL.csv"
        data_dir / "MSFT.csv"
        ...

    Returns a single DataFrame with at least:
        ['Ticker', 'Date', 'Close', ...]
    where 'Date' is a datetime64[ns] normalized to date (no time).
    """
    data_dir = Path(data_dir)
    frames = []

    for ticker in tickers:
        csv_path = data_dir / f"{ticker}.csv"
        if not csv_path.exists():
            print(f"⚠️ Price file not found for {ticker}: {csv_path}. Skipping.")
            continue

        df = pd.read_csv(csv_path)

        # Detect/rename date column if needed
        if date_col not in df.columns:
            # assume first column is date-like
            df.rename(columns={df.columns[0]: date_col}, inplace=True)

        # Parse and normalize to date
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

        # Add ticker column
        df["Ticker"] = ticker

        frames.append(df)

    if not frames:
        raise ValueError("No price data loaded. Check tickers and data_dir.")

    prices = pd.concat(frames, ignore_index=True)

    # Keep a neat column order if possible
    cols = list(prices.columns)
    # move Ticker, Date to the front
    for c in ["Ticker", date_col]:
        cols.insert(0, cols.pop(cols.index(c)))
    prices = prices[cols]

    return prices


def load_cleaned_news(
    news_csv_path: str | Path,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
) -> pd.DataFrame:
    """
    Load the cleaned news dataset from CSV.

    Assumes:
    - The file already has parsed dates (but we will parse again to be safe).
    - There is a 'Date' column (or given via date_col) and a ticker column.

    Returns a DataFrame with:
        ['Ticker', 'Date', ...]
    where 'Date' is a datetime64[ns] normalized to date.
    """
    news_csv_path = Path(news_csv_path)
    if not news_csv_path.exists():
        raise FileNotFoundError(f"News CSV not found at {news_csv_path}")

    news = pd.read_csv(news_csv_path)

    # Ensure ticker column is named consistently
    if ticker_col not in news.columns:
        # Try common alternatives
        for alt in ["ticker", "symbol"]:
            if alt in news.columns:
                news.rename(columns={alt: ticker_col}, inplace=True)
                break

    if ticker_col not in news.columns:
        raise ValueError(
            f"Ticker column '{ticker_col}' not found in news data. "
            f"Available columns: {news.columns.tolist()}"
        )

    # Ensure date column exists
    if date_col not in news.columns:
        # Try common alternatives
        for alt in ["date", "published_at", "datetime", "time"]:
            if alt in news.columns:
                news.rename(columns={alt: date_col}, inplace=True)
                break

    if date_col not in news.columns:
        raise ValueError(
            f"Date column '{date_col}' not found in news data. "
            f"Available columns: {news.columns.tolist()}"
        )

    # Parse/normalize date
    news[date_col] = pd.to_datetime(news[date_col]).dt.normalize()

    # Reorder columns to bring Ticker & Date to front
    cols = list(news.columns)
    for c in [ticker_col, date_col]:
        cols.insert(0, cols.pop(cols.index(c)))
    news = news[cols]

    return news


# ---------------------------
# Date range & filtering
# ---------------------------

def get_overlapping_date_range(
    prices: pd.DataFrame,
    news: pd.DataFrame,
    date_col: str = "Date",
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Compute the overlapping date window between prices and news.
    """
    price_min, price_max = prices[date_col].min(), prices[date_col].max()
    news_min, news_max = news[date_col].min(), news[date_col].max()

    overlap_start = max(price_min, news_min)
    overlap_end = min(price_max, news_max)

    return overlap_start, overlap_end


def filter_to_overlap(
    prices: pd.DataFrame,
    news: pd.DataFrame,
    date_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Filter prices and news DataFrames to their overlapping date window.
    Returns (prices_filtered, news_filtered, (start, end)).
    """
    overlap_start, overlap_end = get_overlapping_date_range(prices, news, date_col)

    prices_f = prices[
        (prices[date_col] >= overlap_start) & (prices[date_col] <= overlap_end)
    ].copy()

    news_f = news[
        (news[date_col] >= overlap_start) & (news[date_col] <= overlap_end)
    ].copy()

    return prices_f, news_f, (overlap_start, overlap_end)


# ---------------------------
# Alignment / Merge
# ---------------------------

def align_news_with_prices(
    prices: pd.DataFrame,
    news: pd.DataFrame,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    price_close_col: str = "Close",
    how: str = "inner",
) -> pd.DataFrame:
    """
    Align news rows with daily price data by Ticker + Date.

    For each news row, attach the corresponding closing price for that
    ticker and date.

    Parameters
    ----------
    how : merge method ('inner', 'left', etc.).
        'inner' = keep only rows where both news and prices exist.

    Returns
    -------
    pd.DataFrame with columns:
        ['Ticker', 'Date', ..., 'ClosePrice']
    """
    if ticker_col not in prices.columns or ticker_col not in news.columns:
        raise ValueError(f"Both prices and news must contain '{ticker_col}' column.")

    if date_col not in prices.columns or date_col not in news.columns:
        raise ValueError(f"Both prices and news must contain '{date_col}' column.")

    if price_close_col not in prices.columns:
        raise ValueError(
            f"Price close column '{price_close_col}' not found in prices. "
            f"Available columns: {prices.columns.tolist()}"
        )

    # Prepare price subset for merging
    price_for_merge = prices[[ticker_col, date_col, price_close_col]].copy()
    price_for_merge = price_for_merge.rename(columns={price_close_col: "ClosePrice"})

    # Merge: attach ClosePrice to each news row
    aligned = news.merge(
        price_for_merge,
        on=[ticker_col, date_col],
        how=how,
        validate="m:1",  # many news rows to one price row
    )

    return aligned


def save_aligned_data(
    aligned_df: pd.DataFrame,
    output_path: str | Path,
    index: bool = False,
) -> None:
    """
    Save the aligned news+price data to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(output_path, index=index)
