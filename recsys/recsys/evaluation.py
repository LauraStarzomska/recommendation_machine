"""
Evaluation metrics for recommendation systems.

Includes:
- RMSE and MAE for rating prediction
- Precision@K and Recall@K for recommendation quality
- Train/test split utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_train_test(df, test_size=0.2, random_state=42, min_ratings_per_user=5):
    """
    Split data into train and test sets, ensuring each user has ratings in both.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: user_id, product_id, rating
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility
    min_ratings_per_user : int
        Minimum ratings a user must have to be included
    
    Returns:
    --------
    train_df, test_df : tuple of pd.DataFrame
        Training and testing dataframes
    """
    # Filter users with sufficient ratings
    user_counts = df.groupby('user_id').size()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    df_filtered = df[df['user_id'].isin(valid_users)].copy()
    
    print(f"Filtered to {len(valid_users)} users with >= {min_ratings_per_user} ratings")
    
    # Split per user to ensure each user is in both train and test
    train_dfs = []
    test_dfs = []
    
    for user_id in valid_users:
        user_data = df_filtered[df_filtered['user_id'] == user_id]
        
        if len(user_data) >= min_ratings_per_user:
            train_user, test_user = train_test_split(
                user_data, 
                test_size=test_size, 
                random_state=random_state
            )
            train_dfs.append(train_user)
            test_dfs.append(test_user)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"Train set: {len(train_df)} ratings")
    print(f"Test set: {len(test_df)} ratings")
    
    return train_df, test_df


def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error.
    
    Parameters:
    -----------
    actual : array-like
        Actual ratings
    predicted : array-like
        Predicted ratings
    
    Returns:
    --------
    float
        RMSE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error.
    
    Parameters:
    -----------
    actual : array-like
        Actual ratings
    predicted : array-like
        Predicted ratings
    
    Returns:
    --------
    float
        MAE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    mae = np.mean(np.abs(actual - predicted))
    
    return mae


def precision_at_k(recommended_items, relevant_items, k):
    """
    Calculate Precision@K.
    
    Precision@K = (# of recommended items that are relevant) / K
    
    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs (in order)
    relevant_items : set
        Set of relevant item IDs
    k : int
        Number of top recommendations to consider
    
    Returns:
    --------
    float
        Precision@K value (0-1)
    """
    if k == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    relevant_in_top_k = len([item for item in top_k if item in relevant_items])
    
    return relevant_in_top_k / k


def recall_at_k(recommended_items, relevant_items, k):
    """
    Calculate Recall@K.
    
    Recall@K = (# of recommended items that are relevant) / (total # of relevant items)
    
    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs (in order)
    relevant_items : set
        Set of relevant item IDs
    k : int
        Number of top recommendations to consider
    
    Returns:
    --------
    float
        Recall@K value (0-1)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    relevant_in_top_k = len([item for item in top_k if item in relevant_items])
    
    return relevant_in_top_k / len(relevant_items)


def f1_score_at_k(recommended_items, relevant_items, k):
    """
    Calculate F1 Score@K (harmonic mean of Precision@K and Recall@K).
    
    Parameters:
    -----------
    recommended_items : list
        List of recommended item IDs (in order)
    relevant_items : set
        Set of relevant item IDs
    k : int
        Number of top recommendations to consider
    
    Returns:
    --------
    float
        F1@K value (0-1)
    """
    precision = precision_at_k(recommended_items, relevant_items, k)
    recall = recall_at_k(recommended_items, relevant_items, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def evaluate_recommendations(train_df, test_df, recommend_func, k_values=[5, 10, 20],
                             relevance_threshold=4.0):
    """
    Comprehensive evaluation of recommendation quality.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Testing data
    recommend_func : callable
        Function that takes (train_df, user_id, n) and returns recommendations
    k_values : list
        List of K values to evaluate
    relevance_threshold : float
        Rating threshold to consider an item "relevant"
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print("\n" + "=" * 70)
    print("📊 EVALUATING RECOMMENDATION QUALITY")
    print("=" * 70)
    
    results = {k: {'precision': [], 'recall': [], 'f1': []} for k in k_values}
    
    # Get unique users in test set
    test_users = test_df['user_id'].unique()
    evaluated_users = 0
    
    for user_id in test_users:
        try:
            # Get user's test items that they rated highly (relevant items)
            user_test = test_df[test_df['user_id'] == user_id]
            relevant_items = set(
                user_test[user_test['rating'] >= relevance_threshold]['product_id']
            )
            
            if len(relevant_items) == 0:
                continue  # Skip users with no relevant items
            
            # Generate recommendations
            max_k = max(k_values)
            recommendations = recommend_func(train_df, user_id, max_k)
            
            if len(recommendations) == 0:
                continue
            
            recommended_items = recommendations['product_id'].tolist()
            
            # Calculate metrics for each K
            for k in k_values:
                precision = precision_at_k(recommended_items, relevant_items, k)
                recall = recall_at_k(recommended_items, relevant_items, k)
                f1 = f1_score_at_k(recommended_items, relevant_items, k)
                
                results[k]['precision'].append(precision)
                results[k]['recall'].append(recall)
                results[k]['f1'].append(f1)
            
            evaluated_users += 1
            
        except Exception as e:
            # Skip users that cause errors
            continue
    
    # Calculate average metrics
    print(f"\nEvaluated {evaluated_users} users")
    print(f"Relevance threshold: {relevance_threshold}⭐ and above\n")
    
    summary = {}
    for k in k_values:
        if results[k]['precision']:
            avg_precision = np.mean(results[k]['precision'])
            avg_recall = np.mean(results[k]['recall'])
            avg_f1 = np.mean(results[k]['f1'])
            
            print(f"📈 Metrics @{k}:")
            print(f"   Precision@{k}: {avg_precision:.4f}")
            print(f"   Recall@{k}:    {avg_recall:.4f}")
            print(f"   F1@{k}:        {avg_f1:.4f}")
            print()
            
            summary[k] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1
            }
    
    print("=" * 70)
    
    return summary


def evaluate_rating_prediction(train_df, test_df, predict_func):
    """
    Evaluate rating prediction accuracy using RMSE and MAE.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Testing data  
    predict_func : callable
        Function that takes (train_df, user_id, product_id) and returns predicted rating
    
    Returns:
    --------
    dict
        Dictionary with RMSE and MAE values
    """
    print("\n" + "=" * 70)
    print("📊 EVALUATING RATING PREDICTION")
    print("=" * 70)
    
    actual_ratings = []
    predicted_ratings = []
    
    for _, row in test_df.iterrows():
        try:
            predicted = predict_func(train_df, row['user_id'], row['product_id'])
            actual_ratings.append(row['rating'])
            predicted_ratings.append(predicted)
        except:
            continue
    
    if len(actual_ratings) == 0:
        print("⚠️  No predictions could be made")
        return {}
    
    rmse = calculate_rmse(actual_ratings, predicted_ratings)
    mae = calculate_mae(actual_ratings, predicted_ratings)
    
    print(f"\nPredicted {len(actual_ratings)} ratings")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print("=" * 70)
    
    return {'rmse': rmse, 'mae': mae, 'n_predictions': len(actual_ratings)}
