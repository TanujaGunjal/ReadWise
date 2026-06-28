"""
model/content_based.py
========================
Content-Based Filtering using TF-IDF + Cosine Similarity.

Strategy
--------
Each book is represented as a text "document" combining:
    "{Book-Title} {Book-Author} {Publisher}"

TF-IDF vectorises these documents. Cosine similarity between
any two book vectors gives a content proximity score in [0, 1].

Why TF-IDF?
-----------
TF-IDF (Term Frequency–Inverse Document Frequency) down-weights
common words (e.g. "the", "and") and up-weights discriminating
terms (e.g. author surnames, unique title words, publisher names).
This prevents very common authors from dominating every result.

Complexity
----------
- Fitting:    O(n · d)  where n = books, d = vocabulary size
- Query:      O(d)      for vectorising + O(n) for cosine sim
- Space:      O(n · d)  for TF-IDF matrix (sparse, memory-efficient)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-Based Book Recommender using TF-IDF on book metadata.

    Parameters
    ----------
    max_features : int
        Maximum number of TF-IDF vocabulary terms (default 10_000).
    ngram_range  : tuple
        Character/word n-gram range (default (1, 2) = unigrams + bigrams).

    Attributes
    ----------
    vectorizer   : TfidfVectorizer — fitted TF-IDF model
    tfidf_matrix : sparse matrix   — (n_books × vocab) representation
    books_df     : pd.DataFrame    — deduplicated book metadata
    title_index  : dict            — {title: row_index} for O(1) lookup
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.vectorizer   = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            analyzer="word",
            strip_accents="unicode",
            sublinear_tf=True,   # log(1 + tf) — dampens high-frequency terms
        )
        self.tfidf_matrix = None
        self.books_df: pd.DataFrame | None = None
        self.title_index: dict[str, int] = {}

    # ── Feature Engineering ──────────────────────────────────────────────────

    @staticmethod
    def _build_content_string(row: pd.Series) -> str:
        """
        Concatenate book metadata fields into a single text document.

        Fields used  : Book-Title, Book-Author, Publisher
        Preprocessing: lowercase, collapse whitespace, fill NaN with ''

        Example output:
            "harry potter and the sorcerer stone j. k. rowling scholastic"
        """
        title     = str(row.get("Book-Title",  "") or "").lower().strip()
        author    = str(row.get("Book-Author", "") or "").lower().strip()
        publisher = str(row.get("Publisher",   "") or "").lower().strip()

        # Repeat author 2× to give authorship more weight
        content = f"{title} {author} {author} {publisher}"
        # Collapse repeated whitespace
        content = re.sub(r"\s+", " ", content).strip()
        return content

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, books_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build the TF-IDF matrix from the books dataframe.

        Parameters
        ----------
        books_df : pd.DataFrame
            Must contain at minimum: 'Book-Title', 'Book-Author'
            Optionally: 'Publisher', 'Image-URL-M'

        Returns
        -------
        self (for method chaining)
        """
        # Deduplicate on title (keep first occurrence, which has ISBN / image)
        df = books_df.drop_duplicates(subset="Book-Title").reset_index(drop=True)

        # Fill any missing metadata
        df["Book-Author"] = df["Book-Author"].fillna("Unknown Author")
        df["Publisher"]   = df.get("Publisher", pd.Series([""] * len(df))).fillna("")

        # Build content strings
        df["_content"] = df.apply(self._build_content_string, axis=1)

        # Fit TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(df["_content"])
        self.books_df     = df
        self.title_index  = {title: idx for idx, title in enumerate(df["Book-Title"])}

        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def recommend(
        self,
        title: str,
        n: int = 4,
        exclude_same_author: bool = False,
    ) -> list[dict]:
        """
        Return top-n content-similar books for a given title.

        Algorithm
        ---------
        1. Look up the query book's TF-IDF vector.
        2. Compute cosine similarity against all other book vectors.
        3. Return top-n most similar (excluding self).

        Parameters
        ----------
        title               : str  — query book title (case-sensitive exact match)
        n                   : int  — number of results
        exclude_same_author : bool — if True, exclude books by the same author
                                     (useful to force discovery of new authors)

        Returns
        -------
        List of dicts with keys: title, author, image, similarity, publisher

        Raises
        ------
        ValueError — if the title is not found in the TF-IDF index
        """
        if self.tfidf_matrix is None or self.books_df is None:
            raise RuntimeError("Call fit() before recommend().")

        if title not in self.title_index:
            raise ValueError(
                f"Book '{title}' not found in content-based index. "
                f"Index contains {len(self.title_index):,} titles."
            )

        idx = self.title_index[title]
        query_vec = self.tfidf_matrix[idx]

        # Cosine similarity between query and all books
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Exclude self (idx where score == 1.0 at position idx)
        sim_scores[idx] = 0.0

        # Optionally exclude same-author books
        if exclude_same_author:
            query_author = str(self.books_df.iloc[idx].get("Book-Author", ""))
            for i, row in self.books_df.iterrows():
                if str(row.get("Book-Author", "")) == query_author:
                    sim_scores[i] = 0.0

        # Get top-n indices
        top_indices = np.argsort(sim_scores)[::-1][:n]

        results = []
        for i in top_indices:
            if sim_scores[i] <= 0:
                continue
            row = self.books_df.iloc[i]
            results.append({
                "title":      str(row["Book-Title"]),
                "author":     str(row.get("Book-Author", "Unknown")),
                "image":      str(row.get("Image-URL-M", "")),
                "publisher":  str(row.get("Publisher", "")),
                "similarity": round(float(sim_scores[i]), 4),
            })

        return results[:n]

    def get_vocabulary_size(self) -> int:
        """Return number of unique terms in the fitted TF-IDF vocabulary."""
        if self.vectorizer and hasattr(self.vectorizer, "vocabulary_"):
            return len(self.vectorizer.vocabulary_)
        return 0

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "model_artifacts/content_based.pkl") -> None:
        """Serialise the fitted recommender to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "model_artifacts/content_based.pkl") -> "ContentBasedRecommender":
        """Load a previously serialised recommender from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
