import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend_for_user(df, user_id, n=10, min_similarity=0.1, use_normalized=False, 
                       normalization_method='mean_center', min_common_items=2):
    """
    Generate personalized recommendations using item-based collaborative filtering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: user_id, product_id, rating
    user_id : int
        The user ID to generate recommendations for
    n : int
        Number of recommendations to return
    min_similarity : float
        Minimum similarity threshold (0-1). Items with lower similarity are ignored.
    use_normalized : bool
        Whether to use normalized ratings for similarity calculation
    normalization_method : str
        Method for rating normalization ('mean_center', 'z_score', 'min_max')
    min_common_items : int
        Minimum number of common ratings needed to calculate similarity
    
    Returns:
    --------
    pd.DataFrame
        Recommended products with columns: product_id, score, estimated_rating
    
    Raises:
    -------
    ValueError
        If user_id is not found in the dataset
    """
    # Check if user exists
    if user_id not in df['user_id'].values:
        raise ValueError(f"User {user_id} not found in dataset.")
    
    # Normalize ratings if requested
    if use_normalized:
        from .utils import normalize_ratings_per_user
        df = normalize_ratings_per_user(df, method=normalization_method)
        rating_col = 'normalized_rating'
    else:
        rating_col = 'rating'
    
    # Create user-item matrix
    pivot = df.pivot_table(index="user_id", columns="product_id", values=rating_col)
    matrix = pivot.fillna(0)
    
    # Calculate item-item similarity
    similarity = cosine_similarity(matrix.T)
    sim_df = pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)
    
    # Get user's ratings
    user_ratings = matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index
    
    # Check for cold start problem
    if len(rated_items) == 0:
        print(f"⚠️  User {user_id} has no ratings (cold start problem)")
        # Fall back to top-rated items
        return get_popular_items(df, n)
    
    if len(rated_items) < min_common_items:
        print(f"⚠️  User {user_id} has very few ratings ({len(rated_items)}), recommendations may be unreliable")
    
    # Calculate weighted scores for unrated items
    scores = {}
    score_weights = {}  # Track sum of similarities for normalization
    
    for product in rated_items:
        # Get similar items
        similar_items = sim_df[product].drop(product)
        rating = user_ratings[product]
        
        for other, sim in similar_items.items():
            # Only consider unrated items with sufficient similarity
            if user_ratings[other] == 0 and sim >= min_similarity:
                scores.setdefault(other, 0)
                score_weights.setdefault(other, 0)
                
                # Weighted sum: similarity * rating
                scores[other] += sim * rating
                score_weights[other] += abs(sim)
    
    # Normalize scores by sum of similarities
    for item in scores:
        if score_weights[item] > 0:
            scores[item] = scores[item] / score_weights[item]
    
    # Convert to DataFrame
    if not scores:
        print(f"⚠️  No recommendations found for user {user_id} (try lowering min_similarity)")
        return pd.DataFrame(columns=["product_id", "score", "estimated_rating"])
    
    recs = pd.DataFrame(scores.items(), columns=["product_id", "score"])
    recs = recs.sort_values("score", ascending=False).head(n)
    
    # Denormalize if needed
    if use_normalized:
        from .utils import get_user_statistics
        user_stats = get_user_statistics(df[df['user_id'] == user_id])
        user_mean = user_stats['mean_rating'].iloc[0]
        
        if normalization_method == 'mean_center':
            recs['estimated_rating'] = recs['score'] + user_mean
        elif normalization_method == 'z_score':
            user_std = user_stats['std_rating'].iloc[0]
            recs['estimated_rating'] = (recs['score'] * user_std) + user_mean
        else:
            recs['estimated_rating'] = recs['score']
    else:
        recs['estimated_rating'] = recs['score']
    
    # Clip ratings to valid range
    recs['estimated_rating'] = recs['estimated_rating'].clip(0.5, 5.0)
    
    return recs


def get_popular_items(df, n=10, min_ratings=5):
    """
    Get most popular items as fallback for cold start users.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: user_id, product_id, rating
    n : int
        Number of items to return
    min_ratings : int
        Minimum number of ratings required
    
    Returns:
    --------
    pd.DataFrame
        Popular items with columns: product_id, score, estimated_rating
    """
    popular = (
        df.groupby('product_id')['rating']
        .agg(['mean', 'count'])
        .reset_index()
    )
    popular = popular[popular['count'] >= min_ratings]
    popular = popular.sort_values('mean', ascending=False).head(n)
    
    result = pd.DataFrame({
        'product_id': popular['product_id'],
        'score': popular['mean'],
        'estimated_rating': popular['mean']
    })
    
    return result
