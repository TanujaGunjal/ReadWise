"""
train_models.py
================
Full ML training pipeline for the ReadWise recommendation system.

Run this script once to:
  1. Load raw data (books.pkl, popular.pkl, pt.pkl, similarity_scores.pkl)
  2. Train K-Means clustering (4 clusters)
  3. Train TF-IDF Content-Based recommender
  4. Assemble the Hybrid Recommender
  5. Save all artifacts to model_artifacts/

Usage
-----
    python train_models.py

Output files (model_artifacts/)
---------------------------------
    book_clusterer.pkl       — fitted BookClusterer
    content_based.pkl        — fitted ContentBasedRecommender
    hybrid_recommender.pkl   — assembled HybridRecommender
    cluster_summary.csv      — human-readable cluster stats
    elbow_data.json          — WCSS curve data for the notebook

Runtime: ~30-60 seconds on a modern laptop (BX dataset, 271K books).
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

# Add project root to path so we can import model.*
sys.path.insert(0, os.path.dirname(__file__))

from model.clustering     import BookClusterer
from model.content_based  import ContentBasedRecommender
from model.hybrid         import HybridRecommender

OUTPUT_DIR = "model_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


# ── Step 1: Load existing artifacts ──────────────────────────────────────────
banner("Step 1 — Loading existing model artifacts")

popular_df        = pd.read_pickle("popular.pkl")
pt                = pickle.load(open("pt.pkl", "rb"))
books             = pickle.load(open("books.pkl", "rb"))
similarity_scores = pickle.load(open("similarity_scores.pkl", "rb"))

print(f"  popular_df : {popular_df.shape}  columns: {list(popular_df.columns)}")
print(f"  pt         : {pt.shape}  (User-ID × Book-Title pivot)")
print(f"  books      : {books.shape}")
print(f"  sim_scores : {similarity_scores.shape}")


# ── Step 2: K-Means Clustering ───────────────────────────────────────────────
banner("Step 2 — K-Means Clustering (k=4)")

t0 = time.time()

# Elbow method data (k = 2..10) — used in notebook
print("  Computing elbow curve (k=2..10)…")
elbow_data = BookClusterer.compute_elbow(popular_df, max_k=10)
with open(os.path.join(OUTPUT_DIR, "elbow_data.json"), "w") as f:
    json.dump(elbow_data, f, indent=2)
print(f"  Elbow data saved → {OUTPUT_DIR}/elbow_data.json")

# Fit clusterer
print("  Fitting K-Means (k=4)…")
clusterer = BookClusterer(n_clusters=4, random_state=42)
clusterer.fit(popular_df)

# Print cluster summary
summary = clusterer.get_cluster_summary()
print("\n  Cluster Summary:")
print(f"  {'Name':<25} {'Count':>6} {'Avg Rating':>10} {'Avg Votes':>10}")
print(f"  {'-'*55}")
for c in summary:
    print(f"  {c['name']:<25} {c['count']:>6} {c['avg_rating']:>10.2f} {c['avg_votes']:>10.1f}")

# Save cluster summary as CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv"), index=False)

# Save clusterer
clusterer.save(os.path.join(OUTPUT_DIR, "book_clusterer.pkl"))
print(f"\n  Clusterer saved → {OUTPUT_DIR}/book_clusterer.pkl  [{time.time()-t0:.1f}s]")


# ── Step 3: Content-Based Filtering ──────────────────────────────────────────
banner("Step 3 — Content-Based Filtering (TF-IDF)")

t0 = time.time()

# Check which columns are available in books
print(f"  books columns: {list(books.columns)}")
required_cols = ["Book-Title", "Book-Author"]
for col in required_cols:
    if col not in books.columns:
        raise ValueError(f"Expected column '{col}' not found in books DataFrame.")

print("  Fitting TF-IDF on Book-Title + Book-Author + Publisher…")
cb_model = ContentBasedRecommender(max_features=10_000, ngram_range=(1, 2))
cb_model.fit(books)

print(f"  Vocabulary size : {cb_model.get_vocabulary_size():,} terms")
print(f"  Books indexed   : {len(cb_model.title_index):,}")

# Quick smoke test
test_title = pt.index[0]  # First title in CF index
try:
    test_recs = cb_model.recommend(test_title, n=3)
    print(f"\n  Smoke test — Content recs for '{test_title}':")
    for r in test_recs:
        print(f"    [{r['similarity']:.3f}] {r['title']} — {r['author']}")
except ValueError as e:
    print(f"  (Smoke test skipped: {e})")

cb_model.save(os.path.join(OUTPUT_DIR, "content_based.pkl"))
print(f"\n  Content-Based model saved → {OUTPUT_DIR}/content_based.pkl  [{time.time()-t0:.1f}s]")


# ── Step 4: Hybrid Recommender ───────────────────────────────────────────────
banner("Step 4 — Hybrid Recommender (CF=0.5, CB=0.3, Popularity=0.2)")

t0 = time.time()

hybrid = HybridRecommender(cf_weight=0.50, cb_weight=0.30, pop_weight=0.20)
hybrid.setup(
    pt=pt,
    similarity_scores=similarity_scores,
    content_model=cb_model,
    popular_df=popular_df,
    clusterer=clusterer,
)

# Full comparison smoke test
print(f"\n  Running comparison for '{test_title}'…")
try:
    comparison = hybrid.compare(test_title, n=4)
    m = comparison["metrics"]
    print(f"\n  {'System':<30} {'Avg Score':>10} {'Coverage':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Collaborative Filtering':<30} {m['cf_avg_score']:>10.4f} {m['cf_coverage']:>10.0%}")
    print(f"  {'Content-Based (TF-IDF)':<30} {m['cb_avg_score']:>10.4f} {m['cb_coverage']:>10.0%}")
    print(f"  {'Hybrid (CF+CB+Pop)':<30} {m['hybrid_avg_score']:>10.4f} {m['hybrid_coverage']:>10.0%}")
    print(f"\n  CF ∩ CB overlap  : {m['overlap_cf_cb']:.3f}  (Jaccard)")
    print(f"  CF ∩ Hybrid      : {m['overlap_cf_hybrid']:.3f}")
    print(f"  CB ∩ Hybrid      : {m['overlap_cb_hybrid']:.3f}")
    if m["unique_in_hybrid"]:
        print(f"  Unique in Hybrid : {m['unique_in_hybrid']}")
except Exception as e:
    print(f"  (Comparison skipped: {e})")

hybrid.save(os.path.join(OUTPUT_DIR, "hybrid_recommender.pkl"))
print(f"\n  Hybrid Recommender saved → {OUTPUT_DIR}/hybrid_recommender.pkl  [{time.time()-t0:.1f}s]")


# ── Done ─────────────────────────────────────────────────────────────────────
banner("Training Complete ✓")
print(f"  Artifacts in ./{OUTPUT_DIR}/")
print("    book_clusterer.pkl")
print("    content_based.pkl")
print("    hybrid_recommender.pkl")
print("    cluster_summary.csv")
print("    elbow_data.json")
print("\n  Next: python app.py to start the server with all models loaded.")
