# scripts/correlation_utils.py

from __future__ import annotations

import pandas as pd


def aggregate_daily_sentiment(
    news_sent_df: pd.DataFrame,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    sentiment_col: str = "sentiment_polarity",
    agg_col_name: str = "DailySentiment",
) -> pd.DataFrame:
    """
    Aggregate article-level sentiment into daily sentiment per (ticker, date).

    By default uses mean of sentiment_col and also returns article_count.
    """
    required = {date_col, ticker_col, sentiment_col}
    missing = required - set(news_sent_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in news_sent_df: {missing}. "
            f"Available columns: {news_sent_df.columns.tolist()}"
        )

    df = news_sent_df.copy()

    # Ensure Date is datetime (or at least same type as in returns)
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    grouped = (
        df.groupby([ticker_col, date_col])[sentiment_col]
          .agg(["mean", "count"])
          .reset_index()
          .rename(columns={
              "mean": agg_col_name,
              "count": "ArticleCount",
          })
    )

    return grouped


def merge_returns_and_sentiment(
    returns_df: pd.DataFrame,
    daily_sent_df: pd.DataFrame,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    return_col: str = "DailyReturn",
    sent_col: str = "DailySentiment",
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge daily returns with daily sentiment on (ticker, date).
    """
    required_ret = {ticker_col, date_col, return_col}
    missing_ret = required_ret - set(returns_df.columns)
    if missing_ret:
        raise ValueError(
            f"Missing columns in returns_df: {missing_ret}. "
            f"Available columns: {returns_df.columns.tolist()}"
        )

    required_sent = {ticker_col, date_col, sent_col}
    missing_sent = required_sent - set(daily_sent_df.columns)
    if missing_sent:
        raise ValueError(
            f"Missing columns in daily_sent_df: {missing_sent}. "
            f"Available columns: {daily_sent_df.columns.tolist()}"
        )

    df_returns = returns_df.copy()
    df_sent = daily_sent_df.copy()

    # Ensure dates are datetime in both
    if not pd.api.types.is_datetime64_any_dtype(df_returns[date_col]):
        df_returns[date_col] = pd.to_datetime(df_returns[date_col])

    if not pd.api.types.is_datetime64_any_dtype(df_sent[date_col]):
        df_sent[date_col] = pd.to_datetime(df_sent[date_col])

    merged = df_returns.merge(
        df_sent,
        on=[ticker_col, date_col],
        how=how,
        validate="1:1",  # one return row per (ticker, date), one sentiment row
    )

    return merged


def compute_correlations(
    merged_df: pd.DataFrame,
    ticker_col: str = "Ticker",
    return_col: str = "DailyReturn",
    sent_col: str = "DailySentiment",
) -> dict:
    """
    Compute Pearson correlations between daily sentiment and daily returns.

    Returns a dict with:
        - 'overall_corr': single float correlation over all rows
        - 'per_ticker': DataFrame with [ticker_col, 'corr']
    """
    required = {ticker_col, return_col, sent_col}
    missing = required - set(merged_df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in merged_df: {missing}. "
            f"Available columns: {merged_df.columns.tolist()}"
        )

    df = merged_df.copy().dropna(subset=[return_col, sent_col])

    # Overall correlation (all tickers together)
    overall_corr = df[return_col].corr(df[sent_col])

    # Per-ticker correlations
    per_ticker_list = []
    for t, grp in df.groupby(ticker_col):
        if len(grp) < 2:
            continue  # not enough data to compute correlation
        corr = grp[return_col].corr(grp[sent_col])
        per_ticker_list.append({ticker_col: t, "corr": corr, "n": len(grp)})

    per_ticker_df = pd.DataFrame(per_ticker_list).sort_values("corr")

    return {
        "overall_corr": overall_corr,
        "per_ticker": per_ticker_df,
    }
