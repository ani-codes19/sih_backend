"""
sentiment_engine.py
--------------------
NLP layer of the ML Engine (matches the architecture: "NLP Sentiment - VADER").

Turns free-text daily journal / check-in entries into numeric sentiment
features that feed into the XGBoost risk model alongside HR data.

VADER is lightweight (no training/GPU needed) and tuned for short,
informal text -- a good fit for a 30-second daily check-in entry.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> dict:
    """
    Returns VADER polarity scores for a single piece of text.

    compound: single normalized score in [-1, 1] (-1 = very negative,
              1 = very positive) -- the main feature we'll feed the model.
    neg/neu/pos: proportions of the text falling into each category.
    """
    if not text or not isinstance(text, str) or not text.strip():
        # Neutral fallback for missing/empty check-ins
        return {"sentiment_compound": 0.0, "sentiment_neg": 0.0,
                "sentiment_neu": 1.0, "sentiment_pos": 0.0}

    s = _analyzer.polarity_scores(text)
    return {
        "sentiment_compound": s["compound"],
        "sentiment_neg": s["neg"],
        "sentiment_neu": s["neu"],
        "sentiment_pos": s["pos"],
    }


def add_sentiment_features(df: pd.DataFrame, text_col: str = "journal_text") -> pd.DataFrame:
    """Vectorized helper: appends sentiment_* columns to a dataframe in place-safe way."""
    scores = df[text_col].apply(score_text).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)


if __name__ == "__main__":
    samples = [
        "Extremely exhausted, haven't slept properly in days.",
        "Good day today, training went well and I got some rest.",
        "Routine day, nothing much to report.",
    ]
    for t in samples:
        print(t, "->", score_text(t))
