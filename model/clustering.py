"""
model/clustering.py
====================
K-Means Clustering of books using engagement features.

Features used
-------------
- avg_rating  : average community rating (0–10 scale, BX dataset)
- num_ratings : total number of ratings received

Cluster Definitions (4 clusters)
----------------------------------
┌──────────────────────┬──────────────┬──────────────┐
│ Cluster Name         │ avg_rating   │ num_ratings  │
├──────────────────────┼──────────────┼──────────────┤
│ Popular Favorites    │ High         │ High         │
│ Hidden Gems          │ High         │ Low          │
│ Niche Classics       │ Medium       │ Medium-Low   │
│ Low Engagement Books │ Low          │ Low          │
└──────────────────────┴──────────────┴──────────────┘

Algorithm
---------
StandardScaler → KMeans(n_clusters=4, random_state=42, n_init=10)

The Elbow Method is used in the companion notebook to determine
that k=4 is the optimal number of clusters (see WCSS plot).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Cluster label mapping — assigned after inspecting cluster centroids
CLUSTER_LABELS: dict[int, str] = {
    0: "Hidden Gems",
    1: "Popular Favorites",
    2: "Low Engagement Books",
    3: "Niche Classics",
}

CLUSTER_DESCRIPTIONS: dict[str, str] = {
    "Popular Favorites":    "High-rated books with a massive readership. The crowd's favourites.",
    "Hidden Gems":          "Highly rated books that few readers have discovered yet.",
    "Niche Classics":       "Moderately rated, with a dedicated but smaller audience.",
    "Low Engagement Books": "Few ratings and lower scores — could be obscure or low-quality.",
}

CLUSTER_COLORS: dict[str, str] = {
    "Popular Favorites":    "#f59e0b",   # amber
    "Hidden Gems":          "#10b981",   # emerald
    "Niche Classics":       "#6366f1",   # indigo
    "Low Engagement Books": "#6b7280",   # grey
}


class BookClusterer:
    """
    Segments books into 4 engagement-based clusters using K-Means.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (default 4).
    random_state : int
        Reproducibility seed.

    Attributes
    ----------
    model    : KMeans          — fitted K-Means model
    scaler   : StandardScaler  — fitted scaler for features
    clustered_df : pd.DataFrame — popular_df extended with cluster info
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42) -> None:
        self.n_clusters   = n_clusters
        self.random_state = random_state
        self.model        = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler       = StandardScaler()
        self.clustered_df: pd.DataFrame | None = None
        self._label_map: dict[int, str] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, popular_df: pd.DataFrame) -> "BookClusterer":
        """
        Fit the K-Means model on the popularity dataframe.

        Feature Engineering
        -------------------
        X = StandardScaler([avg_rating, num_ratings])
        Scaling is critical because num_ratings (~0–2000) would dwarf
        avg_rating (~0–10) without normalisation.

        Parameters
        ----------
        popular_df : pd.DataFrame
            Must contain columns: 'Book-Title', 'avg_rating', 'num_ratings'

        Returns
        -------
        self (for method chaining)
        """
        df = popular_df.copy()

        # Feature matrix: [avg_rating, num_ratings]
        X = df[["avg_rating", "num_ratings"]].values
        X_scaled = self.scaler.fit_transform(X)

        # Fit K-Means
        cluster_ids = self.model.fit_predict(X_scaled)
        df["cluster_id"] = cluster_ids

        # Auto-label clusters based on centroids
        centroids = self.scaler.inverse_transform(self.model.cluster_centers_)
        self._label_map = self._auto_label_clusters(centroids)

        df["cluster_name"]  = df["cluster_id"].map(self._label_map)
        df["cluster_color"] = df["cluster_name"].map(CLUSTER_COLORS)
        df["cluster_desc"]  = df["cluster_name"].map(CLUSTER_DESCRIPTIONS)

        self.clustered_df = df
        return self

    def _auto_label_clusters(self, centroids: np.ndarray) -> dict[int, str]:
        """
        Automatically assign semantic names to cluster IDs based on centroid positions.

        Logic
        -----
        - Sort clusters by avg_rating (col 0) descending, then num_ratings (col 1) descending.
        - The quadrant mapping:
            high rating, high votes  → Popular Favorites
            high rating, low votes   → Hidden Gems
            low rating, any votes    → Low Engagement Books
            else                     → Niche Classics
        """
        # avg_rating = centroid[:, 0], num_ratings = centroid[:, 1]
        label_map: dict[int, str] = {}
        rating_median = np.median(centroids[:, 0])
        votes_median  = np.median(centroids[:, 1])

        for idx, c in enumerate(centroids):
            r, v = c[0], c[1]
            if r >= rating_median and v >= votes_median:
                label_map[idx] = "Popular Favorites"
            elif r >= rating_median and v < votes_median:
                label_map[idx] = "Hidden Gems"
            elif r < rating_median and v >= votes_median:
                label_map[idx] = "Niche Classics"
            else:
                label_map[idx] = "Low Engagement Books"

        return label_map

    # ── Elbow Method ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_elbow(popular_df: pd.DataFrame, max_k: int = 10) -> dict[str, list]:
        """
        Compute Within-Cluster Sum of Squares (WCSS) for k = 2..max_k.

        Used to find the optimal number of clusters via the Elbow Method.
        Plot WCSS vs k; the 'elbow' where the curve bends is the optimal k.

        Parameters
        ----------
        popular_df : pd.DataFrame
        max_k      : int — maximum k to evaluate

        Returns
        -------
        dict with keys 'k_values' and 'wcss' — pass directly to Chart.js
        """
        scaler = StandardScaler()
        X = scaler.fit_transform(popular_df[["avg_rating", "num_ratings"]].values)

        wcss = []
        k_values = list(range(2, max_k + 1))
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            wcss.append(round(float(km.inertia_), 2))

        return {"k_values": k_values, "wcss": wcss}

    # ── Inference ────────────────────────────────────────────────────────────

    def get_cluster(self, title: str) -> dict | None:
        """
        Return cluster info for a specific book title.

        Parameters
        ----------
        title : str — exact book title

        Returns
        -------
        dict with cluster_id, cluster_name, cluster_desc, cluster_color
        or None if title not found.
        """
        if self.clustered_df is None:
            raise RuntimeError("Call fit() before get_cluster().")

        mask = self.clustered_df["Book-Title"] == title
        if not mask.any():
            return None

        row = self.clustered_df[mask].iloc[0]
        return {
            "cluster_id":    int(row["cluster_id"]),
            "cluster_name":  str(row["cluster_name"]),
            "cluster_desc":  str(row["cluster_desc"]),
            "cluster_color": str(row["cluster_color"]),
            "avg_rating":    round(float(row["avg_rating"]), 2),
            "num_ratings":   int(row["num_ratings"]),
        }

    def get_cluster_summary(self) -> list[dict]:
        """
        Returns a summary of each cluster: count, mean rating, mean votes.
        Useful for the analytics dashboard and API endpoint.
        """
        if self.clustered_df is None:
            raise RuntimeError("Call fit() before get_cluster_summary().")

        summary = []
        for cname in ["Popular Favorites", "Hidden Gems", "Niche Classics", "Low Engagement Books"]:
            sub = self.clustered_df[self.clustered_df["cluster_name"] == cname]
            if sub.empty:
                continue
            summary.append({
                "name":       cname,
                "count":      int(len(sub)),
                "avg_rating": round(float(sub["avg_rating"].mean()), 2),
                "avg_votes":  round(float(sub["num_ratings"].mean()), 1),
                "color":      CLUSTER_COLORS[cname],
                "description": CLUSTER_DESCRIPTIONS[cname],
            })
        return summary

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "model_artifacts/book_clusterer.pkl") -> None:
        """Serialise the fitted clusterer to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "model_artifacts/book_clusterer.pkl") -> "BookClusterer":
        """Load a previously saved clusterer from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
