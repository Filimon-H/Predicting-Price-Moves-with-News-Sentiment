# scripts/returns_utils.py

from __future__ import annotations

import pandas as pd


def compute_daily_returns(
    prices: pd.DataFrame,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    close_col: str = "Close",
    return_col: str = "DailyReturn",
) -> pd.DataFrame:
    """
    Compute daily percentage returns for each stock (ticker) based on the
    closing price.

    DailyReturn_t = (Close_t / Close_{t-1}) - 1

    Parameters
    ----------
    prices : DataFrame
        Must contain at least [ticker_col, date_col, close_col].
    date_col : str
        Name of the date column.
    ticker_col : str
        Name of the ticker/stock symbol column.
    close_col : str
        Name of the closing price column.
    return_col : str
        Name of the output daily return column.

    Returns
    -------
    DataFrame
        Same as `prices` but with an extra column [return_col].
        The first row per ticker will have NaN return (no previous day).
    """
    required = {ticker_col, date_col, close_col}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in prices: {missing}. "
            f"Available columns: {prices.columns.tolist()}"
        )

    df = prices.copy()

    # Ensure proper sorting: per ticker by date
    df = df.sort_values([ticker_col, date_col])

    # Group by ticker and compute pct_change on Close
    df[return_col] = (
        df.groupby(ticker_col)[close_col]
          .pct_change()
    )

    return df
