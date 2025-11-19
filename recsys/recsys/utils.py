import pandas as pd
import numpy as np


def normalize_ratings_per_user(df, method='mean_center'):
    """
    Normalize ratings per user to handle rating bias.
    
    Different users have different rating scales:
    - Some users are lenient (rate everything 4-5)
    - Some users are strict (rate everything 1-3)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: user_id, product_id, rating
    method : str
        Normalization method:
        - 'mean_center': Subtract user's mean rating
        - 'z_score': Z-score normalization (mean=0, std=1)
        - 'min_max': Scale to 0-1 range per user
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional 'normalized_rating' column
    """
    df = df.copy()
    
    if method == 'mean_center':
        # Subtract each user's mean rating
        user_means = df.groupby('user_id')['rating'].transform('mean')
        df['normalized_rating'] = df['rating'] - user_means
        
    elif method == 'z_score':
        # Z-score: (rating - mean) / std
        user_means = df.groupby('user_id')['rating'].transform('mean')
        user_stds = df.groupby('user_id')['rating'].transform('std')
        # Avoid division by zero for users with constant ratings
        user_stds = user_stds.replace(0, 1)
        df['normalized_rating'] = (df['rating'] - user_means) / user_stds
        
    elif method == 'min_max':
        # Scale to 0-1 range per user
        user_min = df.groupby('user_id')['rating'].transform('min')
        user_max = df.groupby('user_id')['rating'].transform('max')
        user_range = user_max - user_min
        # Avoid division by zero
        user_range = user_range.replace(0, 1)
        df['normalized_rating'] = (df['rating'] - user_min) / user_range
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df


def denormalize_rating(normalized_rating, user_mean, user_std=1.0, method='mean_center'):
    """
    Convert normalized rating back to original scale.
    
    Parameters:
    -----------
    normalized_rating : float
        The normalized rating value
    user_mean : float
        The user's mean rating
    user_std : float
        The user's rating standard deviation (for z_score method)
    method : str
        The normalization method used
    
    Returns:
    --------
    float
        Denormalized rating
    """
    if method == 'mean_center':
        return normalized_rating + user_mean
    elif method == 'z_score':
        return (normalized_rating * user_std) + user_mean
    else:
        return normalized_rating


def get_user_statistics(df):
    """
    Calculate per-user rating statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: user_id, rating
    
    Returns:
    --------
    pd.DataFrame
        User statistics with columns: user_id, mean_rating, std_rating, 
        min_rating, max_rating, rating_count
    """
    stats = df.groupby('user_id')['rating'].agg([
        ('mean_rating', 'mean'),
        ('std_rating', 'std'),
        ('min_rating', 'min'),
        ('max_rating', 'max'),
        ('rating_count', 'count')
    ]).reset_index()
    
    return stats


def calculate_sparsity(df):
    """
    Calculate the sparsity of the user-item matrix.
    
    Sparsity = 1 - (number of ratings / (users * items))
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with user_id and product_id columns
    
    Returns:
    --------
    dict
        Dictionary with sparsity metrics
    """
    n_users = df['user_id'].nunique()
    n_items = df['product_id'].nunique()
    n_ratings = len(df)
    
    possible_ratings = n_users * n_items
    sparsity = 1 - (n_ratings / possible_ratings)
    density = n_ratings / possible_ratings
    
    return {
        'n_users': n_users,
        'n_items': n_items,
        'n_ratings': n_ratings,
        'possible_ratings': possible_ratings,
        'sparsity': sparsity,
        'density': density,
        'sparsity_percent': f"{sparsity * 100:.2f}%",
        'avg_ratings_per_user': n_ratings / n_users,
        'avg_ratings_per_item': n_ratings / n_items
    }

