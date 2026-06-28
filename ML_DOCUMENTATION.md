# ML Documentation — ReadWise Recommendation System

## Architecture Overview

```
                        ┌─────────────────────────────────────────┐
                        │           ReadWise ML Stack              │
                        └─────────────────────────────────────────┘
                                         │
              ┌──────────────────────────┼─────────────────────────┐
              │                          │                          │
   ┌──────────▼──────────┐  ┌───────────▼───────────┐  ┌──────────▼──────────┐
   │ Collaborative        │  │  Content-Based         │  │  K-Means            │
   │ Filtering            │  │  Filtering             │  │  Clustering         │
   │                      │  │                        │  │                     │
   │ User-Book Pivot      │  │  TF-IDF Vectorizer     │  │  StandardScaler     │
   │ → Cosine Similarity  │  │  → Cosine Similarity   │  │  → KMeans (k=4)     │
   └──────────┬───────────┘  └───────────┬────────────┘  └──────────┬──────────┘
              │                          │                           │
              └──────────────────────────┼───────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Hybrid Recommender  │
                              │                      │
                              │  0.5·CF + 0.3·CB     │
                              │  + 0.2·Popularity    │
                              └──────────────────────┘
```

---

## Model 1: Collaborative Filtering (Existing)

### Algorithm
**User-Item Collaborative Filtering** using cosine similarity

### Data Pipeline
```
ratings.csv (1,149,780 rows)
    ↓ merge with books.csv on ISBN
    ↓ filter: users who rated ≥200 books
    ↓ filter: books with ≥200 ratings
    ↓ create pivot table: User-ID × Book-Title
    ↓ fit cosine_similarity(pivot_table)
    ↓ save: pt.pkl, similarity_scores.pkl
```

### Input / Output
- **Input:** Book title (must exist in pivot table index)
- **Output:** Top-N books sorted by cosine similarity score

### Cosine Similarity Formula
$$\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

Where A and B are the **book's rating vectors** across all users (sparse, most entries = 0).

### Strengths
- Captures latent reading patterns without needing content data
- Works well for books with many ratings

### Limitations
- Cold-start problem: fails for books with < threshold ratings
- Requires significant rating history per user

### Artifacts
| File | Contents |
|---|---|
| `pt.pkl` | User × Book pivot table (sparse) |
| `similarity_scores.pkl` | Pre-computed full cosine similarity matrix |
| `books.pkl` | Book metadata (title, author, image) |
| `popular.pkl` | Top-50 popularity-ranked books |

---

## Model 2: Content-Based Filtering (TF-IDF)

### Algorithm
**TF-IDF Vectorisation + Cosine Similarity**

### Feature Engineering
```python
content_string = f"{Book-Title} {Book-Author} {Book-Author} {Publisher}"
# Author repeated 2× to boost authorship signal weight
```

### TF-IDF Configuration
| Parameter | Value | Rationale |
|---|---|---|
| `max_features` | 10,000 | Caps vocabulary; prevents memory blowup |
| `ngram_range` | (1, 2) | Captures "Harry Potter" as a unit |
| `stop_words` | 'english' | Removes "the", "of", etc. |
| `sublinear_tf` | True | Uses log(1+tf) to dampen high-frequency terms |
| `strip_accents` | 'unicode' | Normalises accented characters |

### Mathematical Basis
$$\text{TF-IDF}(t, d) = \text{tf}(t, d) \times \log\frac{N+1}{df(t)+1}$$

- **tf(t,d):** frequency of term t in document d (with sublinear_tf: log(1+count))
- **df(t):** number of documents containing term t
- **N:** total number of documents (books)

### Cosine Similarity
Identical formula to CF, but applied to TF-IDF vectors instead of rating vectors.

### Strengths
- Works for any book in the catalog, regardless of rating count
- Captures author and publisher relationships naturally
- Fast at query time (sparse matrix operations)

### Limitations
- Cannot capture "readers who liked X also liked Y" patterns
- Recommends within similar content — less serendipitous

### Artifacts
| File | Contents |
|---|---|
| `model_artifacts/content_based.pkl` | Fitted TfidfVectorizer + TF-IDF matrix + title index |

---

## Model 3: K-Means Clustering

### Algorithm
**K-Means Clustering** (k=4) on standardised engagement features

### Features
| Feature | Scale | Role |
|---|---|---|
| `avg_rating` | 0–10 | Quality signal |
| `num_ratings` | 0–~2000 | Popularity/engagement signal |

### Preprocessing
```
StandardScaler → zero mean, unit variance
→ Prevents num_ratings (large range) from dominating avg_rating (small range)
```

### Elbow Method (k selection)
Compute WCSS for k = 2..10:
$$\text{WCSS}(k) = \sum_{j=1}^{k} \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$

**Result:** k=4 is optimal — the curve's inflection point.

### Cluster Definitions
| Cluster | avg_rating | num_ratings | Business Meaning |
|---|---|---|---|
| **Popular Favorites** | High (≥median) | High (≥median) | Safe bets for cold-start users |
| **Hidden Gems** | High (≥median) | Low (<median) | Opportunity: promote to increase reach |
| **Niche Classics** | Low (<median) | High (≥median) | Have audience but lower satisfaction |
| **Low Engagement Books** | Low (<median) | Low (<median) | Deprioritise in recommendations |

### Auto-Labelling Logic
```python
if   rating ≥ median AND votes ≥ median → "Popular Favorites"
elif rating ≥ median AND votes <  median → "Hidden Gems"
elif rating <  median AND votes ≥ median → "Niche Classics"
else                                     → "Low Engagement Books"
```

### K-Means Algorithm Steps
1. Randomly initialise k centroids (n_init=10 restarts, best kept)
2. **Assign:** Each point to nearest centroid (Euclidean distance in scaled space)
3. **Update:** Recompute centroid = mean of all assigned points
4. Repeat until centroids stabilise (convergence)

### Artifacts
| File | Contents |
|---|---|
| `model_artifacts/book_clusterer.pkl` | Fitted BookClusterer (KMeans + StandardScaler + label map) |
| `model_artifacts/cluster_summary.csv` | Cluster statistics table |
| `model_artifacts/elbow_data.json` | WCSS curve data (k=2..10) |
| `model_artifacts/elbow_curve.png` | Elbow curve plot |
| `model_artifacts/cluster_scatter.png` | Cluster scatter visualisation |

---

## Model 4: Hybrid Recommender

### Algorithm
**Weighted Score Fusion** of three independent signals

### Fusion Formula
```
hybrid_score(book) = 0.5 × CF_score
                   + 0.3 × CB_score
                   + 0.2 × popularity_score
```

### Popularity Score
```python
pop_score = 0.7 × (num_ratings / max_ratings)
          + 0.3 × (avg_rating  / max_avg_rating)
```

Votes weighted 70% because sheer volume of ratings is a stronger engagement signal than average score alone.

### Weight Rationale
| Signal | Weight | Justification |
|---|---|---|
| Collaborative Filtering | 0.50 | Strongest personalisation signal |
| Content-Based | 0.30 | Handles cold-start, adds diversity |
| Popularity | 0.20 | Quality floor, tie-breaker |

### Candidate Pool
The hybrid uses the **union** of CF and CB candidate books. This means it considers books that appear in either signal's top results, then re-ranks by combined score.

### Comparison Metrics
| Metric | Formula | Interpretation |
|---|---|---|
| **Avg Score** | mean(scores) | Confidence / signal strength |
| **Coverage** | results / n | Completeness (1.0 = all n returned) |
| **Jaccard Overlap** | `\|A∩B\| / \|A∪B\|` | 0=no overlap, 1=identical |

### Artifacts
| File | Contents |
|---|---|
| `model_artifacts/hybrid_recommender.pkl` | Fitted HybridRecommender with all references |
| `model_artifacts/comparison_chart.png` | Bar chart comparison from notebook |

---

## API Endpoints Reference

| Endpoint | Method | Model | Description |
|---|---|---|---|
| `/api/recommend?title=X&n=4` | GET | CF | Collaborative filtering |
| `/api/recommend/content?title=X&n=4&diverse=true` | GET | CB | TF-IDF content-based |
| `/api/recommend/hybrid?title=X&n=4` | GET | Hybrid | Weighted fusion |
| `/api/compare?title=X&n=4` | GET | All | Side-by-side comparison |
| `/api/clusters` | GET | K-Means | Cluster summary |
| `/api/book/cluster?title=X` | GET | K-Means | Cluster for specific book |
| `/autocomplete?q=harry` | GET | Index | Live search suggestions |

---

## Running the Pipeline

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train all ML models (~30-60 seconds)
python train_models.py

# Step 3: Start the server
python app.py

# Step 4: Open the notebook for analysis
jupyter notebook ml_analysis.ipynb
```

---

## File Structure

```
Book Recommender/
├── app.py                     ← Flask app (7 API routes)
├── train_models.py            ← Full ML training pipeline
├── ml_analysis.ipynb          ← Analysis notebook (K-Means, TF-IDF, Hybrid, Comparison)
├── requirements.txt           ← Dependencies
├── RESUME_BULLETS.md          ← Resume bullet points
├── ML_DOCUMENTATION.md        ← This file
│
├── model/
│   ├── __init__.py
│   ├── clustering.py          ← BookClusterer (K-Means)
│   ├── content_based.py       ← ContentBasedRecommender (TF-IDF)
│   └── hybrid.py              ← HybridRecommender (CF + CB + Popularity)
│
├── model_artifacts/           ← Generated by train_models.py
│   ├── book_clusterer.pkl
│   ├── content_based.pkl
│   ├── hybrid_recommender.pkl
│   ├── cluster_summary.csv
│   ├── elbow_data.json
│   ├── elbow_curve.png
│   ├── cluster_scatter.png
│   └── comparison_chart.png
│
├── popular.pkl                ← Pre-existing: top-50 popularity df
├── pt.pkl                     ← Pre-existing: User-Book pivot table
├── books.pkl                  ← Pre-existing: book metadata
└── similarity_scores.pkl      ← Pre-existing: CF cosine similarity matrix
```

---

## Technology Stack

| Technology | Role |
|---|---|
| **Python 3.10+** | Core language |
| **scikit-learn** | KMeans, TfidfVectorizer, cosine_similarity, StandardScaler |
| **pandas** | Data manipulation, pivot tables, groupby aggregations |
| **NumPy** | Matrix operations, similarity score sorting |
| **Flask** | REST API framework |
| **pickle** | Model serialisation / deserialisation |
| **matplotlib** | Elbow curve, cluster scatter, comparison bar charts |

---

*Documentation generated for ReadWise Recommendation System — Portfolio Project*
