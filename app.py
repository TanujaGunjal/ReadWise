"""
ReadWise - Intelligent Book Recommendation System
==================================================
Flask web application with three recommendation engines:
  1. Collaborative Filtering  — cosine similarity on user-book pivot matrix
  2. Content-Based Filtering  — TF-IDF on title + author + publisher metadata
  3. Hybrid Recommender       — weighted fusion of CF + CB + popularity

Routes
------
  GET  /                      — Home: Top 50 popular books
  GET  /recommend             — Recommendation UI
  POST /recommend_books       — CF recommendations (form submit)
  GET  /autocomplete?q=       — Live search autocomplete (JSON)
  GET  /dashboard             — Analytics dashboard
  GET  /api/recommend?title=  — CF recommendations (REST JSON)
  GET  /api/recommend/content?title= — Content-Based (REST JSON)
  GET  /api/recommend/hybrid?title=  — Hybrid (REST JSON)
  GET  /api/compare?title=    — Side-by-side comparison (REST JSON)
  GET  /api/clusters          — Cluster summary (REST JSON)
  GET  /api/book/cluster?title= — Cluster for a specific book (REST JSON)

Author : ReadWise
Stack  : Python 3.10+, Flask, pandas, NumPy, scikit-learn
"""

from __future__ import annotations

import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ── Load core collaborative-filtering artifacts ────────────────────────────────
popular_df: pd.DataFrame = pd.read_pickle("popular.pkl")
pt: pd.DataFrame         = pickle.load(open("pt.pkl", "rb"))
books: pd.DataFrame      = pickle.load(open("books.pkl", "rb"))
similarity_scores        = pickle.load(open("similarity_scores.pkl", "rb"))

app = Flask(__name__)

# ── Lazily load new ML models (avoids crash if train_models.py hasn't been run) ─
_clusterer       = None
_content_model   = None
_hybrid_model    = None
_ml_models_ready = False


def _load_ml_models() -> bool:
    """
    Load K-Means, Content-Based, and Hybrid models from disk.
    Called once on first request that needs them.

    Returns True if all models loaded successfully, False otherwise.
    """
    global _clusterer, _content_model, _hybrid_model, _ml_models_ready

    if _ml_models_ready:
        return True

    try:
        from model.clustering    import BookClusterer
        from model.content_based import ContentBasedRecommender
        from model.hybrid        import HybridRecommender

        _clusterer     = BookClusterer.load("model_artifacts/book_clusterer.pkl")
        _content_model = ContentBasedRecommender.load("model_artifacts/content_based.pkl")
        _hybrid_model  = HybridRecommender.load("model_artifacts/hybrid_recommender.pkl")
        _ml_models_ready = True
        app.logger.info("✓ ML models loaded (clustering, content-based, hybrid)")
        return True

    except FileNotFoundError:
        app.logger.warning(
            "ML model artifacts not found. Run 'python train_models.py' first. "
            "Falling back to collaborative filtering only."
        )
        return False
    except Exception as e:
        app.logger.error(f"Error loading ML models: {e}")
        return False


# ── Pre-compute analytics for dashboard (once at startup) ─────────────────────
def _build_dashboard_data() -> tuple[dict, dict, list]:
    """Pre-compute KPI stats, chart data, and top-authors table for the dashboard."""
    df = popular_df.copy()

    stats = {
        "total_books":   int(len(df)),
        "avg_rating":    round(float(df["avg_rating"].mean()), 1),
        "total_ratings": int(df["num_ratings"].sum()),
        "top_author":    df.loc[df["num_ratings"].idxmax(), "Book-Author"],
    }

    top_votes = df.nlargest(10, "num_ratings")[["Book-Title", "num_ratings"]].iloc[::-1]
    top_rated = df[df["num_ratings"] >= 50].nlargest(10, "avg_rating")[
        ["Book-Title", "avg_rating"]
    ].iloc[::-1]

    bins   = [i / 2 for i in range(0, 21)]
    labels = [f"{b:.1f}–{b+0.5:.1f}" for b in bins[:-1]]
    counts, _ = np.histogram(df["avg_rating"].dropna(), bins=bins)

    sample = df.sample(min(50, len(df)), random_state=42)
    scatter = [
        {"x": int(r["num_ratings"]), "y": round(float(r["avg_rating"]), 1)}
        for _, r in sample.iterrows()
    ]

    chart_data = {
        "top_votes": {
            "labels": [t[:30] + "…" if len(t) > 30 else t
                       for t in top_votes["Book-Title"].tolist()],
            "values": top_votes["num_ratings"].tolist(),
        },
        "top_rated": {
            "labels": [t[:30] + "…" if len(t) > 30 else t
                       for t in top_rated["Book-Title"].tolist()],
            "values": [round(v, 2) for v in top_rated["avg_rating"].tolist()],
        },
        "rating_dist": {"labels": labels, "values": counts.tolist()},
        "scatter": scatter,
    }

    author_stats = (
        df.groupby("Book-Author")
        .agg(votes=("num_ratings", "sum"), avg_rating=("avg_rating", "mean"))
        .nlargest(10, "votes")
        .reset_index()
    )
    top_authors = [
        {
            "name":       row["Book-Author"],
            "votes":      int(row["votes"]),
            "avg_rating": round(float(row["avg_rating"]), 1),
        }
        for _, row in author_stats.iterrows()
    ]

    return stats, chart_data, top_authors


DASH_STATS, DASH_CHART_DATA, DASH_TOP_AUTHORS = _build_dashboard_data()
ALL_TITLES: list[str] = list(pt.index)


# ── CF Helper ─────────────────────────────────────────────────────────────────
def _cf_recommend(title: str, n: int = 4) -> list[dict]:
    """
    Collaborative Filtering recommendations via cosine similarity.

    Args:
        title : Exact book title (must exist in pivot table index)
        n     : Number of recommendations

    Returns:
        List of dicts — title, author, image, similarity

    Raises:
        ValueError if title not in index
    """
    matches = np.where(pt.index == title)[0]
    if len(matches) == 0:
        raise ValueError(f"Book '{title}' not found in the recommendation index.")

    idx = matches[0]
    similar = sorted(
        enumerate(similarity_scores[idx]),
        key=lambda x: x[1],
        reverse=True,
    )[1: n + 1]

    results = []
    for item_idx, score in similar:
        book_title = pt.index[item_idx]
        temp = books[books["Book-Title"] == book_title].drop_duplicates("Book-Title")
        if temp.empty:
            continue
        row = temp.iloc[0]
        results.append({
            "title":      str(row["Book-Title"]),
            "author":     str(row["Book-Author"]),
            "image":      str(row["Image-URL-M"]),
            "similarity": round(float(score), 4),
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Home page — shows Top 50 popular books."""
    return render_template(
        "index.html",
        book_name=list(popular_df["Book-Title"].values),
        author=list(popular_df["Book-Author"].values),
        image=list(popular_df["Image-URL-M"].values),
        votes=list(popular_df["num_ratings"].values),
        rating=list(popular_df["avg_rating"].values),
    )


@app.route("/recommend")
def recommend_ui():
    """Render the recommendation search page."""
    return render_template("recommend.html", data=None, query=None, error=None)


@app.route("/recommend_books", methods=["POST"])
def recommend():
    """
    POST /recommend_books
    
    Runs Collaborative Filtering for a user-submitted book title.
    """
    user_input = request.form.get("user_input", "").strip()
    if not user_input:
        return render_template(
            "recommend.html", data=None, query=None,
            error="Please enter a book title.",
        )

    try:
        recs = _cf_recommend(user_input, n=4)
        if not recs:
            return render_template(
                "recommend.html", data=None, query=user_input,
                error="No similar books found.",
            )
        data = [[r["title"], r["author"], r["image"], r["similarity"]] for r in recs]
        return render_template("recommend.html", data=data, query=user_input, error=None)

    except ValueError as e:
        return render_template(
            "recommend.html", data=None, query=user_input, error=str(e)
        )
    except Exception as e:
        app.logger.error(f"CF error for '{user_input}': {e}")
        return render_template(
            "recommend.html", data=None, query=user_input,
            error="Something went wrong. Please try again.",
        )


@app.route("/autocomplete")
def autocomplete():
    """
    GET /autocomplete?q=<query>
    Returns up to 8 matching book title suggestions as JSON.
    """
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify([])
    suggestions = [t for t in ALL_TITLES if q in t.lower()][:8]
    return jsonify(suggestions)


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard with Chart.js visualisations."""
    return render_template(
        "dashboard.html",
        stats=DASH_STATS,
        chart_data=DASH_CHART_DATA,
        top_authors=DASH_TOP_AUTHORS,
    )


# ── REST API — Collaborative Filtering ────────────────────────────────────────
@app.route("/api/recommend")
def api_recommend():
    """
    GET /api/recommend?title=<book_title>&n=<count>

    Returns collaborative filtering recommendations as JSON.
    
    Example: GET /api/recommend?title=Harry+Potter&n=4
    """
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "Missing required param: title"}), 400

    try:
        n = min(int(request.args.get("n", 4)), 10)
    except ValueError:
        return jsonify({"error": "Param 'n' must be an integer"}), 400

    try:
        recs = _cf_recommend(title, n=n)
        return jsonify({
            "query":           title,
            "method":          "Collaborative Filtering (Cosine Similarity)",
            "count":           len(recs),
            "recommendations": recs,
            "model_info": {
                "algorithm":    "User-Item Collaborative Filtering",
                "matrix_shape": list(pt.shape),
                "index_size":   len(ALL_TITLES),
            },
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        app.logger.error(f"API CF error for '{title}': {e}")
        return jsonify({"error": "Internal server error"}), 500


# ── REST API — Content-Based Filtering ────────────────────────────────────────
@app.route("/api/recommend/content")
def api_recommend_content():
    """
    GET /api/recommend/content?title=<book_title>&n=<count>&diverse=<bool>

    Returns TF-IDF content-based recommendations as JSON.
    
    Params:
        diverse : if 'true', excludes books by the same author (forces discovery)

    Example: GET /api/recommend/content?title=Harry+Potter&n=4&diverse=true
    """
    if not _load_ml_models():
        return jsonify({
            "error": "Content-based model not available. Run 'python train_models.py' first."
        }), 503

    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "Missing required param: title"}), 400

    try:
        n = min(int(request.args.get("n", 4)), 10)
    except ValueError:
        return jsonify({"error": "Param 'n' must be an integer"}), 400

    diverse = request.args.get("diverse", "false").lower() == "true"

    try:
        recs = _content_model.recommend(title, n=n, exclude_same_author=diverse)
        return jsonify({
            "query":           title,
            "method":          "Content-Based Filtering (TF-IDF + Cosine Similarity)",
            "diverse_mode":    diverse,
            "count":           len(recs),
            "recommendations": recs,
            "model_info": {
                "algorithm":      "TF-IDF + Cosine Similarity",
                "features":       ["Book-Title", "Book-Author", "Publisher"],
                "ngram_range":    [1, 2],
                "vocabulary_size": _content_model.get_vocabulary_size(),
            },
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        app.logger.error(f"API CB error for '{title}': {e}")
        return jsonify({"error": "Internal server error"}), 500


# ── REST API — Hybrid Recommender ─────────────────────────────────────────────
@app.route("/api/recommend/hybrid")
def api_recommend_hybrid():
    """
    GET /api/recommend/hybrid?title=<book_title>&n=<count>

    Returns hybrid (CF + Content + Popularity) recommendations as JSON.
    Each result includes per-signal scores and cluster information.

    Example: GET /api/recommend/hybrid?title=Harry+Potter&n=4
    """
    if not _load_ml_models():
        return jsonify({
            "error": "Hybrid model not available. Run 'python train_models.py' first."
        }), 503

    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "Missing required param: title"}), 400

    try:
        n = min(int(request.args.get("n", 4)), 10)
    except ValueError:
        return jsonify({"error": "Param 'n' must be an integer"}), 400

    try:
        recs = _hybrid_model.recommend(title, n=n)
        return jsonify({
            "query":           title,
            "method":          "Hybrid Recommender (CF + Content-Based + Popularity)",
            "count":           len(recs),
            "recommendations": recs,
            "model_info": {
                "algorithm": "Weighted Score Fusion",
                "weights": {
                    "collaborative_filtering": _hybrid_model.cf_weight,
                    "content_based":           _hybrid_model.cb_weight,
                    "popularity":              _hybrid_model.pop_weight,
                },
            },
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        app.logger.error(f"API Hybrid error for '{title}': {e}")
        return jsonify({"error": "Internal server error"}), 500


# ── REST API — Side-by-Side Comparison ────────────────────────────────────────
@app.route("/api/compare")
def api_compare():
    """
    GET /api/compare?title=<book_title>&n=<count>

    Runs all three recommenders and returns a detailed side-by-side comparison
    with Jaccard overlap metrics, coverage rates, and average scores.

    Example: GET /api/compare?title=Harry+Potter&n=4
    """
    if not _load_ml_models():
        return jsonify({
            "error": "ML models not available. Run 'python train_models.py' first."
        }), 503

    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "Missing required param: title"}), 400

    try:
        n = min(int(request.args.get("n", 4)), 10)
    except ValueError:
        return jsonify({"error": "Param 'n' must be an integer"}), 400

    try:
        comparison = _hybrid_model.compare(title, n=n)
        return jsonify(comparison)
    except Exception as e:
        app.logger.error(f"API Compare error for '{title}': {e}")
        return jsonify({"error": str(e)}), 500


# ── REST API — Cluster Summary ─────────────────────────────────────────────────
@app.route("/api/clusters")
def api_clusters():
    """
    GET /api/clusters

    Returns the 4 K-Means cluster summaries with count, avg rating, avg votes.
    """
    if not _load_ml_models():
        return jsonify({
            "error": "Clustering model not available. Run 'python train_models.py' first."
        }), 503

    try:
        summary = _clusterer.get_cluster_summary()
        return jsonify({
            "n_clusters": 4,
            "algorithm":  "K-Means (StandardScaler + Cosine distance)",
            "features":   ["avg_rating", "num_ratings"],
            "clusters":   summary,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── REST API — Book Cluster ────────────────────────────────────────────────────
@app.route("/api/book/cluster")
def api_book_cluster():
    """
    GET /api/book/cluster?title=<book_title>

    Returns the cluster assignment and description for a specific book.
    """
    if not _load_ml_models():
        return jsonify({
            "error": "Clustering model not available. Run 'python train_models.py' first."
        }), 503

    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "Missing required param: title"}), 400

    cluster_info = _clusterer.get_cluster(title)
    if cluster_info is None:
        return jsonify({"error": f"Book '{title}' not found in cluster index."}), 404

    return jsonify({"title": title, **cluster_info})


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)