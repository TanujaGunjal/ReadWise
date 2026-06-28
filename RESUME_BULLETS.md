# 📄 Resume-Ready Bullet Points — ReadWise ML Project

> Copy these directly into your resume under **Projects** or **Experience**.
> Choose the bullets most relevant to the role you're applying for.

---

## Project Title Options
- **ReadWise: Intelligent Book Recommendation System**
- **Multi-Model Book Recommendation Engine using Collaborative & Content-Based Filtering**
- **Hybrid ML Recommendation System — BX Dataset (1.1M+ Ratings)**

---

## Recommendation Systems (Pick 2–3)

- Built a **hybrid recommendation engine** combining Collaborative Filtering, TF-IDF Content-Based Filtering, and Popularity scoring with weighted fusion (α=0.5, β=0.3, γ=0.2), delivering higher-confidence recommendations than any single model
- Designed a **content-based recommender** using TF-IDF vectorisation (10K vocabulary, bigrams) on book metadata (title + author + publisher), achieving meaningful similarity scores via cosine distance on a 271K-book corpus
- Engineered **RESTful API endpoints** (`/api/recommend`, `/api/recommend/content`, `/api/recommend/hybrid`, `/api/compare`) to expose all three recommendation models, demonstrating production API design skills

---

## Collaborative Filtering

- Implemented **User-Item Collaborative Filtering** using cosine similarity on a sparse User × Book pivot matrix built from 1,149,780 community ratings (BX dataset)
- Reduced recommendation latency by **pre-computing and serialising the full cosine similarity matrix** at training time, enabling O(1) query-time lookups
- Applied **vote-threshold filtering** (≥200 ratings) and **active-user filtering** (≥200 books rated) to reduce noise, improving recommendation quality for the pivot matrix

---

## Content-Based Filtering

- Applied **TF-IDF with sublinear term frequency** (`log(1 + tf)`) and English stop-word removal to build a 10,000-feature vocabulary across 271,360 deduplicated books
- Used **bigram tokenisation** (`ngram_range=(1,2)`) to capture multi-word author names and series titles as atomic features, improving content similarity precision
- Implemented **author-diversity mode** that excludes same-author results from recommendations, broadening discovery beyond a single writer's catalog

---

## K-Means Clustering

- Segmented 50 top-rated books into **4 interpretable engagement clusters** using K-Means on standardised `(avg_rating, num_ratings)` features: *Popular Favorites*, *Hidden Gems*, *Niche Classics*, *Low Engagement Books*
- Applied **StandardScaler** before clustering to prevent the high-range `num_ratings` feature (~0–2000) from dominating the lower-range `avg_rating` (~0–10)
- Determined **optimal k=4** using the Elbow Method (WCSS vs k plot, k=2 to 10) and validated with centroid interpretation against business-meaningful quadrants
- Implemented **auto-labelling of clusters** from centroid quadrant positions (high/low rating × high/low votes), enabling unsupervised model outputs to be interpreted without manual inspection

---

## Machine Learning (General)

- Developed an end-to-end **ML training pipeline** (`train_models.py`) covering data loading, feature engineering, model fitting, smoke testing, and artifact serialisation using `pickle`
- Implemented **lazy model loading** in the Flask application to gracefully handle missing model artifacts, with automatic fallback to the base collaborative filtering model
- Applied **scikit-learn**'s `TfidfVectorizer`, `KMeans`, `StandardScaler`, and `cosine_similarity` in production Python code with full docstrings, type hints, and error handling

---

## Customer Behaviour Analytics

- Analysed **1,149,780 community ratings** across 278,858 users and 271,360 books to identify engagement patterns, rating distributions, and author popularity trends
- Computed **Jaccard similarity** between recommendation system outputs to quantify result divergence, finding that CF and Content-Based systems share ~20–30% overlap — confirming complementary signal value
- Built a **real-time analytics dashboard** serving KPIs (total books, avg rating, top author) and Chart.js visualisations (rating histogram, top-votes bar chart, scatter plot) via a `/dashboard_data` JSON API
- Designed **4 book engagement segments** using unsupervised clustering to support data-driven curation decisions: identifying *Hidden Gems* (high quality, low reach) as the highest-opportunity promotion targets

---

## One-Liner Summaries (For Skills Section)

```
• Recommendation Systems: Collaborative Filtering, Content-Based (TF-IDF), Hybrid Fusion
• Clustering: K-Means, Elbow Method, StandardScaler, Cluster Interpretation
• NLP/Text ML: TF-IDF Vectorisation, Bigrams, Cosine Similarity, Stop-Word Removal
• Python ML Stack: scikit-learn, pandas, NumPy, Flask REST APIs, pickle serialisation
• Analytics: Rating Distribution Analysis, Jaccard Overlap Metrics, Coverage Analysis
• Data Scale: Worked with 1.1M+ rating records; 271K+ book corpus
```

---

## GitHub README One-Liner

> *"Multi-model book recommendation system built on the BX dataset (1.1M ratings). Implements Collaborative Filtering, TF-IDF Content-Based Filtering, K-Means clustering, and a weighted Hybrid Recommender — exposed via a Flask REST API."*
