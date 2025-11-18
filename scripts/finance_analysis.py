# src/finance_analysis.py

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import TA-Lib; if not available, we can still do basic SMA with pandas
try:
    import talib
    HAS_TALIB = True
except ImportError:
    talib = None
    HAS_TALIB = False


# ---------------------------
# Data loading
# ---------------------------

def load_ticker_data(
    ticker: str,
    data_dir: str = "data",
    date_col: str = "Date"
) -> pd.DataFrame:
    """
    Load a single ticker's OHLCV data from CSV.

    Expects files like: data/AAPL.csv, data/MSFT.csv, etc.

    Returns a DataFrame indexed by Date with at least:
    ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    path = Path(data_dir) / f"{ticker}.csv"

    if not path.exists():
        raise FileNotFoundError(f"CSV for {ticker} not found at {path}")

    df = pd.read_csv(path)

    # Detect date column if not explicitly present
    if date_col not in df.columns:
        # assume first column is date-like
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Keep core OHLCV columns if present
    cols = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in cols if c in df.columns]

    if len(available_cols) < 4:
        raise ValueError(
            f"{ticker} CSV does not contain enough OHLCV columns. "
            f"Found: {df.columns.tolist()}"
        )

    df = df[available_cols].copy()

    # Drop rows with missing values in key fields
    df = df.dropna(subset=["Close"])

    return df


def load_all_tickers(
    tickers: List[str],
    data_dir: str = "data"
) -> Dict[str, pd.DataFrame]:
    """
    Load data for all tickers into a dict: {ticker: DataFrame}.
    """
    data = {}
    for t in tickers:
        try:
            df = load_ticker_data(t, data_dir=data_dir)
            data[t] = df
        except Exception as e:
            print(f"⚠️ Skipping {t} due to error: {e}")
    return data


# ---------------------------
# Technical indicators
# ---------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to a price DataFrame:

    - SMA_20, SMA_50
    - RSI_14
    - MACD, MACD_signal, MACD_hist (if TA-Lib is available)

    Expects a 'Close' column.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    close = df["Close"]

    # Simple moving averages
    df["SMA_20"] = close.rolling(window=20, min_periods=20).mean()
    df["SMA_50"] = close.rolling(window=50, min_periods=50).mean()

    # RSI (prefer TA-Lib if available, otherwise simple manual version)
    if HAS_TALIB:
        df["RSI_14"] = talib.RSI(close, timeperiod=14)
    else:
        # Simple RSI implementation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (only if TA-Lib is available)
    if HAS_TALIB:
        macd, macdsignal, macdhist = talib.MACD(
            close.values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )
        df["MACD"] = macd
        df["MACD_signal"] = macdsignal
        df["MACD_hist"] = macdhist
    else:
        df["MACD"] = np.nan
        df["MACD_signal"] = np.nan
        df["MACD_hist"] = np.nan

    return df


def add_indicators_to_all(
    data: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Apply technical indicators to each ticker DataFrame in the dict.
    """
    out = {}
    for ticker, df in data.items():
        out[ticker] = add_technical_indicators(df.copy())
    return out


# ---------------------------
# Plotting helpers
# ---------------------------
def plot_macd(df: pd.DataFrame, ticker: str, last_n: int = 250):
    """
    Plot MACD, MACD Signal and Histogram.
    """
    if last_n is not None:
        df = df.tail(last_n)

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["MACD"], label="MACD", linewidth=1.5)
    plt.plot(df.index, df["MACD_signal"], label="Signal", linestyle="--")
    plt.bar(df.index, df["MACD_hist"], label="Histogram", alpha=0.4)

    plt.title(f"{ticker} - MACD")
    plt.ylabel("MACD")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_price_with_sma(
    df: pd.DataFrame,
    ticker: str,
    last_n: Optional[int] = 250
) -> None:
    """
    Plot Close price with SMA_20 and SMA_50 (and optional Volume on 2nd axis).
    last_n: number of most recent rows to show (None = all).
    """
    if last_n is not None:
        df = df.tail(last_n)

    plt.figure(figsize=(12, 6))

    plt.plot(df.index, df["Close"], label="Close", linewidth=1.5)
    if "SMA_20" in df.columns:
        plt.plot(df.index, df["SMA_20"], label="SMA 20", linestyle="--")
    if "SMA_50" in df.columns:
        plt.plot(df.index, df["SMA_50"], label="SMA 50", linestyle="-.")

    plt.title(f"{ticker} - Close Price with SMA(20) & SMA(50)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rsi(df: pd.DataFrame, ticker: str, last_n: Optional[int] = 250) -> None:
    """
    Plot RSI_14 with overbought/oversold lines.
    """
    if "RSI_14" not in df.columns:
        raise ValueError("RSI_14 column not found. Did you run add_technical_indicators?")

    if last_n is not None:
        df = df.tail(last_n)

    plt.figure(figsize=(12, 3))
    plt.plot(df.index, df["RSI_14"], label="RSI 14")
    plt.axhline(70, color="red", linestyle="--", linewidth=1)
    plt.axhline(30, color="green", linestyle="--", linewidth=1)
    plt.title(f"{ticker} - RSI(14)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.tight_layout()
    plt.show()
