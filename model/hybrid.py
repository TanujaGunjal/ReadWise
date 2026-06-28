"""
model/hybrid.py
================
Hybrid Recommender — Combines three recommendation signals.

Architecture
------------
                    ┌─────────────────────────┐
     Query Title ──►│  Collaborative Filter   │──► CF Score  (weight: α)
                    └─────────────────────────┘
                    ┌─────────────────────────┐
     Query Title ──►│  Content-Based Filter   │──► CB Score  (weight: β)
                    └─────────────────────────┘
                    ┌─────────────────────────┐
     Candidate  ──►│   Popularity Score      │──► Pop Score (weight: γ)
                    └─────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Weighted Fusion   │
                    │  α + β + γ = 1.0  │
                    └─────────┬─────────┘
                              │
                         Final Score → Ranked Results

Default weights: α=0.5 (CF), β=0.3 (CB), γ=0.2 (Popularity)

Rationale
---------
CF weight is highest because user behaviour is the strongest signal
for personalised recommendations. CB weight is secondary to handle the
cold-start problem for titles with few ratings. Popularity acts as a
tie-breaker and quality floor.

Comparison Methodology
-----------------------
The compare() method evaluates all three systems by computing:
- Average similarity score (proxy for confidence)
- Coverage (number of results returned out of n requested)
- Result overlap (Jaccard similarity between result sets)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .content_based import ContentBasedRecommender
    from .clustering import BookClusterer


class HybridRecommender:
    """
    Hybrid Book Recommender merging CF, Content-Based, and Popularity signals.

    Parameters
    ----------
    cf_weight  : float — weight for collaborative filtering score (default 0.5)
    cb_weight  : float — weight for content-based score           (default 0.3)
    pop_weight : float — weight for popularity score              (default 0.2)

    Attributes
    ----------
    pt                : pd.DataFrame  — user-book pivot table (CF index)
    similarity_scores : np.ndarray    — pre-computed CF cosine similarity
    content_model     : ContentBasedRecommender
    clusterer         : BookClusterer
    popular_df        : pd.DataFrame  — popularity stats for all books
    """

    def __init__(
        self,
        cf_weight:  float = 0.50,
        cb_weight:  float = 0.30,
        pop_weight: float = 0.20,
    ) -> None:
        if not abs(cf_weight + cb_weight + pop_weight - 1.0) < 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self.cf_weight  = cf_weight
        self.cb_weight  = cb_weight
        self.pop_weight = pop_weight

        self.pt: pd.DataFrame | None                = None
        self.similarity_scores: np.ndarray | None   = None
        self.content_model                          = None
        self.clusterer                              = None
        self.popular_df: pd.DataFrame | None        = None
        self._pop_norm: dict[str, float]            = {}   # normalised popularity scores

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        pt: pd.DataFrame,
        similarity_scores: np.ndarray,
        content_model,
        popular_df: pd.DataFrame,
        clusterer=None,
    ) -> "HybridRecommender":
        """
        Attach all component models.

        Parameters
        ----------
        pt                : pivot table from collaborative filtering
        similarity_scores : pre-computed cosine similarity matrix
        content_model     : fitted ContentBasedRecommender instance
        popular_df        : popularity dataframe (Book-Title, num_ratings, avg_rating)
        clusterer         : optional fitted BookClusterer instance

        Returns
        -------
        self
        """
        self.pt                = pt
        self.similarity_scores = similarity_scores
        self.content_model     = content_model
        self.popular_df        = popular_df
        self.clusterer         = clusterer

        # Pre-compute normalised popularity score for each title
        max_ratings = float(popular_df["num_ratings"].max())
        max_rating  = float(popular_df["avg_rating"].max()) if popular_df["avg_rating"].max() > 0 else 10.0

        for _, row in popular_df.iterrows():
            # Popularity score = 0.7 × normalised_votes + 0.3 × normalised_avg_rating
            norm_votes  = float(row["num_ratings"]) / max_ratings
            norm_rating = float(row["avg_rating"])  / max_rating
            self._pop_norm[str(row["Book-Title"])] = 0.7 * norm_votes + 0.3 * norm_rating

        return self

    # ── CF Helpers ────────────────────────────────────────────────────────────

    def _cf_scores(self, title: str, n: int) -> dict[str, float]:
        """
        Return {book_title: cf_score} dict for top CF results.

        Returns empty dict if title not in CF index.
        """
        matches = np.where(self.pt.index == title)[0]
        if len(matches) == 0:
            return {}

        idx = matches[0]
        raw = list(enumerate(self.similarity_scores[idx]))
        raw_sorted = sorted(raw, key=lambda x: x[1], reverse=True)[1: n * 3 + 1]

        return {
            self.pt.index[i]: float(score)
            for i, score in raw_sorted
        }

    def _cb_scores(self, title: str, n: int) -> dict[str, float]:
        """
        Return {book_title: cb_score} dict for top content-based results.

        Returns empty dict if title not in content index.
        """
        try:
            recs = self.content_model.recommend(title, n=n * 3)
            return {r["title"]: r["similarity"] for r in recs}
        except (ValueError, RuntimeError):
            return {}

    # ── Fusion ────────────────────────────────────────────────────────────────

    def recommend(self, title: str, n: int = 4) -> list[dict]:
        """
        Return top-n hybrid recommendations for a given book title.

        Fusion Strategy
        ---------------
        1. Collect candidate titles from CF and CB pools.
        2. For each candidate, compute:
               score = α·cf_score + β·cb_score + γ·pop_score
        3. Sort by score descending, return top-n.

        Parameters
        ----------
        title : str — query book title (exact match)
        n     : int — number of results

        Returns
        -------
        List of dicts: title, author, image, cf_score, cb_score,
                       pop_score, hybrid_score, cluster_name
        """
        if self.pt is None or self.content_model is None:
            raise RuntimeError("Call setup() before recommend().")

        cf_map  = self._cf_scores(title, n)
        cb_map  = self._cb_scores(title, n)

        # Union of all candidate titles
        candidates = set(cf_map) | set(cb_map)
        if not candidates:
            raise ValueError(
                f"No results found for '{title}' in either CF or Content-Based index."
            )

        scored = []
        for t in candidates:
            if t == title:
                continue
            cf_s  = cf_map.get(t, 0.0)
            cb_s  = cb_map.get(t, 0.0)
            pop_s = self._pop_norm.get(t, 0.0)

            hybrid = (
                self.cf_weight  * cf_s +
                self.cb_weight  * cb_s +
                self.pop_weight * pop_s
            )
            scored.append((t, cf_s, cb_s, pop_s, hybrid))

        # Sort by hybrid score descending
        scored.sort(key=lambda x: x[4], reverse=True)

        # Build result dicts with book metadata
        results = []
        for t, cf_s, cb_s, pop_s, hybrid in scored[:n]:
            # Fetch metadata from content model (it has the full books df)
            meta = self._get_book_meta(t)
            cluster_info = {}
            if self.clusterer:
                cluster_info = self.clusterer.get_cluster(t) or {}

            results.append({
                "title":        t,
                "author":       meta.get("author", "Unknown"),
                "image":        meta.get("image", ""),
                "cf_score":     round(cf_s, 4),
                "cb_score":     round(cb_s, 4),
                "pop_score":    round(pop_s, 4),
                "hybrid_score": round(hybrid, 4),
                "cluster_name": cluster_info.get("cluster_name", ""),
            })

        return results

    def _get_book_meta(self, title: str) -> dict:
        """Fetch author + image from the content model's books dataframe."""
        if self.content_model is None or self.content_model.books_df is None:
            return {}
        df = self.content_model.books_df
        mask = df["Book-Title"] == title
        if not mask.any():
            return {}
        row = df[mask].iloc[0]
        return {
            "author": str(row.get("Book-Author", "Unknown")),
            "image":  str(row.get("Image-URL-M", "")),
        }

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare(self, title: str, n: int = 4) -> dict:
        """
        Run all three recommenders and compare their outputs side-by-side.

        Metrics computed
        ----------------
        - avg_score    : mean similarity/hybrid score (confidence proxy)
        - coverage     : fraction of n results returned (some may fail)
        - overlap_cf_cb: Jaccard(CF results, CB results)
        - overlap_cf_h : Jaccard(CF results, Hybrid results)
        - overlap_cb_h : Jaccard(CB results, Hybrid results)

        Parameters
        ----------
        title : str — query book title
        n     : int — results per system

        Returns
        -------
        dict with keys:
            'collaborative', 'content_based', 'hybrid', 'metrics'
        """
        results = {}

        # ── Collaborative Filtering ──
        try:
            cf_recs = []
            cf_raw = self._cf_scores(title, n)
            for t, s in sorted(cf_raw.items(), key=lambda x: x[1], reverse=True)[:n]:
                meta = self._get_book_meta(t)
                cf_recs.append({
                    "title":      t,
                    "author":     meta.get("author", "Unknown"),
                    "image":      meta.get("image", ""),
                    "similarity": round(s, 4),
                    "method":     "Collaborative Filtering",
                })
            results["collaborative"] = cf_recs
        except Exception as e:
            results["collaborative"] = []
            results["cf_error"] = str(e)

        # ── Content-Based Filtering ──
        try:
            cb_recs = self.content_model.recommend(title, n=n)
            for r in cb_recs:
                r["method"] = "Content-Based (TF-IDF)"
            results["content_based"] = cb_recs
        except Exception as e:
            results["content_based"] = []
            results["cb_error"] = str(e)

        # ── Hybrid ──
        try:
            h_recs = self.recommend(title, n=n)
            for r in h_recs:
                r["method"] = "Hybrid (CF + CB + Popularity)"
            results["hybrid"] = h_recs
        except Exception as e:
            results["hybrid"] = []
            results["h_error"] = str(e)

        # ── Comparison Metrics ──
        cf_titles = {r["title"] for r in results["collaborative"]}
        cb_titles = {r["title"] for r in results["content_based"]}
        h_titles  = {r["title"] for r in results["hybrid"]}

        def jaccard(a: set, b: set) -> float:
            if not a and not b:
                return 0.0
            return round(len(a & b) / len(a | b), 3)

        def avg_score(recs: list[dict], key: str = "similarity") -> float:
            vals = [r.get(key, r.get("hybrid_score", 0)) for r in recs]
            return round(float(np.mean(vals)), 4) if vals else 0.0

        results["metrics"] = {
            "query":             title,
            "n":                 n,
            "cf_avg_score":      avg_score(results["collaborative"]),
            "cb_avg_score":      avg_score(results["content_based"]),
            "hybrid_avg_score":  avg_score(results["hybrid"], key="hybrid_score"),
            "cf_coverage":       len(results["collaborative"]) / n,
            "cb_coverage":       len(results["content_based"])  / n,
            "hybrid_coverage":   len(results["hybrid"])          / n,
            "overlap_cf_cb":     jaccard(cf_titles, cb_titles),
            "overlap_cf_hybrid": jaccard(cf_titles, h_titles),
            "overlap_cb_hybrid": jaccard(cb_titles, h_titles),
            "unique_in_hybrid":  list(h_titles - cf_titles - cb_titles),
            "weights": {
                "cf":         self.cf_weight,
                "cb":         self.cb_weight,
                "popularity": self.pop_weight,
            },
        }

        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "model_artifacts/hybrid_recommender.pkl") -> None:
        """Serialise the fitted hybrid recommender to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "model_artifacts/hybrid_recommender.pkl") -> "HybridRecommender":
        """Load a previously serialised hybrid recommender from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
