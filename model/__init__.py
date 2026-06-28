"""
model/__init__.py
Exposes the three recommendation engines for easy import.
"""
from .content_based import ContentBasedRecommender
from .clustering import BookClusterer
from .hybrid import HybridRecommender

__all__ = ["ContentBasedRecommender", "BookClusterer", "HybridRecommender"]
