# scripts/sentiment_utils.py

from __future__ import annotations

from typing import Literal

import pandas as pd
from textblob import TextBlob


SentimentLabel = Literal["positive", "negative", "neutral"]


def _label_from_polarity(p: float,
                         pos_threshold: float = 0.1,
                         neg_threshold: float = -0.1) -> SentimentLabel:
    """
    Map a continuous polarity score in [-1, 1] to a discrete label.
    """
    if p >= pos_threshold:
        return "positive"
    if p <= neg_threshold:
        return "negative"
    return "neutral"


def add_textblob_sentiment(
    df: pd.DataFrame,
    text_col: str = "headline",
    polarity_col: str = "sentiment_polarity",
    subjectivity_col: str = "sentiment_subjectivity",
    label_col: str = "sentiment_label",
) -> pd.DataFrame:
    """
    Compute TextBlob sentiment for a text column and add:

    - sentiment_polarity    (float, -1 to 1)
    - sentiment_subjectivity (float, 0 to 1)
    - sentiment_label        ('positive', 'negative', 'neutral')

    Parameters
    ----------
    df : DataFrame containing the text column.
    text_col : name of the column with news headlines (or article text).

    Returns
    -------
    A new DataFrame with additional sentiment columns.
    """
    if text_col not in df.columns:
        raise ValueError(
            f"Text column '{text_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    out = df.copy()

    # Ensure text is string
    texts = out[text_col].astype(str)

    polarities = []
    subjectivities = []

    for t in texts:
        blob = TextBlob(t)
        sent = blob.sentiment  # (polarity, subjectivity)
        polarities.append(sent.polarity)
        subjectivities.append(sent.subjectivity)

    out[polarity_col] = polarities
    out[subjectivity_col] = subjectivities
    out[label_col] = out[polarity_col].apply(_label_from_polarity)

    return out
