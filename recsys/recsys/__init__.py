"""
recsys - A simple recommendation engine

Supports:
- Top-N best-rated products within a time window
- User-specific recommendations using item-based collaborative filtering
- Rating normalization and data validation
- Comprehensive evaluation metrics
- Data exploration and statistics
"""

__version__ = "0.1.0"

from .data_loader import load_ratings
from .recommended_top import get_top_n_products
from .recommend_user import recommend_for_user, get_popular_items
from .utils import (
    normalize_ratings_per_user,
    get_user_statistics,
    calculate_sparsity
)
from .data_exploration import (
    explore_dataset,
    analyze_user,
    analyze_product
)
from .evaluation import (
    split_train_test,
    calculate_rmse,
    calculate_mae,
    precision_at_k,
    recall_at_k,
    f1_score_at_k,
    evaluate_recommendations,
    evaluate_rating_prediction
)

__all__ = [
    # Data loading
    "load_ratings",
    
    # Recommendations
    "get_top_n_products",
    "recommend_for_user",
    "get_popular_items",
    
    # Utilities
    "normalize_ratings_per_user",
    "get_user_statistics",
    "calculate_sparsity",
    
    # Exploration
    "explore_dataset",
    "analyze_user",
    "analyze_product",
    
    # Evaluation
    "split_train_test",
    "calculate_rmse",
    "calculate_mae",
    "precision_at_k",
    "recall_at_k",
    "f1_score_at_k",
    "evaluate_recommendations",
    "evaluate_rating_prediction",
]
