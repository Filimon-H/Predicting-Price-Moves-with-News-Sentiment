# scripts/date_parsing.py

from __future__ import annotations

from dateutil import parser
import pandas as pd
from typing import Optional

def parse_date_safe(x: object) -> Optional[pd.Timestamp]:
    """
    Safely parse a date-like value into a pandas Timestamp.
    Returns NaT if parsing fails.

    Tries:
    1) dateutil.parser.parse (flexible, handles many formats + timezones)
    2) '%m/%d/%Y %H:%M'  (e.g. 05/22/2020 11:23)
    3) '%m/%d/%Y'        (e.g. 05/22/2020)
    """
    x = str(x).strip()

    # 1. General parser
    try:
        return parser.parse(x)
    except Exception:
        pass

    # 2. Explicit US format with time
    try:
        return pd.to_datetime(x, format="%m/%d/%Y %H:%M")
    except Exception:
        pass

    # 3. US format without time
    try:
        return pd.to_datetime(x, format="%m/%d/%Y")
    except Exception:
        pass

    # If everything fails
    return pd.NaT


def parse_date_column(
    df: pd.DataFrame,
    column: str = "date",
    drop_timezone: bool = True,
) -> pd.DataFrame:
    """
    Apply `parse_date_safe` to a column in the DataFrame.

    - Parses each value safely.
    - Optionally removes timezone info to get naive datetimes.

    Returns a new DataFrame (original is not modified).
    """
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    out = df.copy()
    out[column] = out[column].apply(parse_date_safe)

    if drop_timezone and pd.api.types.is_datetime64tz_dtype(out[column]):
        out[column] = out[column].dt.tz_localize(None)

    return out
