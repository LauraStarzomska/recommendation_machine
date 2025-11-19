"""
Data exploration and statistics module for recommendation system.

Use this to analyze your dataset before building recommendations.
"""

import pandas as pd
import numpy as np
from .utils import calculate_sparsity, get_user_statistics


def explore_dataset(df, show_top_users=10, show_top_products=10):
    """
    Comprehensive dataset exploration and statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: user_id, product_id, rating, timestamp
    show_top_users : int
        Number of top users to display
    show_top_products : int
        Number of top products to display
    
    Returns:
    --------
    dict
        Dictionary containing all statistics
    """
    print("=" * 70)
    print("📊 DATASET EXPLORATION")
    print("=" * 70)
    
    # Basic statistics
    n_ratings = len(df)
    n_users = df['user_id'].nunique()
    n_products = df['product_id'].nunique()
    
    print(f"\n📈 Basic Statistics:")
    print(f"   Total Ratings: {n_ratings:,}")
    print(f"   Unique Users: {n_users:,}")
    print(f"   Unique Products: {n_products:,}")
    print(f"   Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Rating distribution
    print(f"\n⭐ Rating Distribution:")
    rating_dist = df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = (count / n_ratings) * 100
        bar = "█" * int(percentage / 2)
        print(f"   {rating:.1f}: {count:6,} ({percentage:5.2f}%) {bar}")
    
    print(f"\n   Mean Rating: {df['rating'].mean():.3f}")
    print(f"   Median Rating: {df['rating'].median():.3f}")
    print(f"   Std Rating: {df['rating'].std():.3f}")
    
    # Sparsity analysis
    print(f"\n🕸️  Matrix Sparsity:")
    sparsity_info = calculate_sparsity(df)
    print(f"   Matrix Size: {sparsity_info['n_users']:,} users × {sparsity_info['n_items']:,} items")
    print(f"   Possible Ratings: {sparsity_info['possible_ratings']:,}")
    print(f"   Actual Ratings: {sparsity_info['n_ratings']:,}")
    print(f"   Sparsity: {sparsity_info['sparsity_percent']} (higher = more sparse)")
    print(f"   Density: {sparsity_info['density']:.6f}")
    print(f"   Avg Ratings per User: {sparsity_info['avg_ratings_per_user']:.1f}")
    print(f"   Avg Ratings per Product: {sparsity_info['avg_ratings_per_item']:.1f}")
    
    # User statistics
    print(f"\n👥 User Activity:")
    user_ratings = df.groupby('user_id').size()
    print(f"   Min ratings per user: {user_ratings.min()}")
    print(f"   Max ratings per user: {user_ratings.max()}")
    print(f"   Mean ratings per user: {user_ratings.mean():.1f}")
    print(f"   Median ratings per user: {user_ratings.median():.1f}")
    
    # Power users
    print(f"\n   Top {show_top_users} Most Active Users:")
    top_users = user_ratings.nlargest(show_top_users)
    for i, (user_id, count) in enumerate(top_users.items(), 1):
        print(f"      {i}. User {user_id}: {count} ratings")
    
    # Product statistics
    print(f"\n📦 Product Popularity:")
    product_ratings = df.groupby('product_id').size()
    print(f"   Min ratings per product: {product_ratings.min()}")
    print(f"   Max ratings per product: {product_ratings.max()}")
    print(f"   Mean ratings per product: {product_ratings.mean():.1f}")
    print(f"   Median ratings per product: {product_ratings.median():.1f}")
    
    # Popular products
    print(f"\n   Top {show_top_products} Most Rated Products:")
    top_products = product_ratings.nlargest(show_top_products)
    for i, (product_id, count) in enumerate(top_products.items(), 1):
        avg_rating = df[df['product_id'] == product_id]['rating'].mean()
        print(f"      {i}. Product {product_id}: {count} ratings (avg: {avg_rating:.2f}⭐)")
    
    # Cold start analysis
    print(f"\n❄️  Cold Start Analysis:")
    users_with_few_ratings = (user_ratings < 5).sum()
    products_with_few_ratings = (product_ratings < 5).sum()
    print(f"   Users with < 5 ratings: {users_with_few_ratings} ({users_with_few_ratings/n_users*100:.1f}%)")
    print(f"   Products with < 5 ratings: {products_with_few_ratings} ({products_with_few_ratings/n_products*100:.1f}%)")
    
    # Rating bias per user
    print(f"\n🎯 User Rating Patterns:")
    user_stats = get_user_statistics(df)
    print(f"   Users with mean rating > 4.0 (lenient): {(user_stats['mean_rating'] > 4.0).sum()}")
    print(f"   Users with mean rating < 3.0 (strict): {(user_stats['mean_rating'] < 3.0).sum()}")
    print(f"   Overall user mean std: {user_stats['mean_rating'].std():.3f}")
    
    print("\n" + "=" * 70)
    
    return {
        'n_ratings': n_ratings,
        'n_users': n_users,
        'n_products': n_products,
        'sparsity': sparsity_info,
        'rating_distribution': rating_dist.to_dict(),
        'user_stats': user_stats
    }


def analyze_user(df, user_id):
    """
    Detailed analysis of a specific user's rating behavior.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: user_id, product_id, rating
    user_id : int
        The user ID to analyze
    """
    user_data = df[df['user_id'] == user_id]
    
    if len(user_data) == 0:
        print(f"User {user_id} not found in dataset")
        return
    
    print(f"\n👤 User {user_id} Profile:")
    print(f"   Total Ratings: {len(user_data)}")
    print(f"   Mean Rating: {user_data['rating'].mean():.2f}")
    print(f"   Std Rating: {user_data['rating'].std():.2f}")
    print(f"   Min Rating: {user_data['rating'].min()}")
    print(f"   Max Rating: {user_data['rating'].max()}")
    print(f"   First Rating: {user_data['timestamp'].min()}")
    print(f"   Last Rating: {user_data['timestamp'].max()}")
    
    print(f"\n   Rating Distribution:")
    for rating in sorted(user_data['rating'].unique()):
        count = (user_data['rating'] == rating).sum()
        print(f"      {rating}: {count} times")


def analyze_product(df, product_id):
    """
    Detailed analysis of a specific product's ratings.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: user_id, product_id, rating
    product_id : int
        The product ID to analyze
    """
    product_data = df[df['product_id'] == product_id]
    
    if len(product_data) == 0:
        print(f"Product {product_id} not found in dataset")
        return
    
    print(f"\n📦 Product {product_id} Profile:")
    print(f"   Total Ratings: {len(product_data)}")
    print(f"   Mean Rating: {product_data['rating'].mean():.2f} ⭐")
    print(f"   Std Rating: {product_data['rating'].std():.2f}")
    print(f"   Min Rating: {product_data['rating'].min()}")
    print(f"   Max Rating: {product_data['rating'].max()}")
    print(f"   First Rating: {product_data['timestamp'].min()}")
    print(f"   Last Rating: {product_data['timestamp'].max()}")
    
    print(f"\n   Rating Distribution:")
    for rating in sorted(product_data['rating'].unique()):
        count = (product_data['rating'] == rating).sum()
        percentage = (count / len(product_data)) * 100
        print(f"      {rating}: {count} times ({percentage:.1f}%)")
