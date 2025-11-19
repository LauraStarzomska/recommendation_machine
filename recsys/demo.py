#!/usr/bin/env python3
"""
Demo script showcasing all features of the recommendation system.

Usage:
    python demo.py ratings.csv
"""

import sys
from recsys import (
    load_ratings,
    explore_dataset,
    get_top_n_products,
    recommend_for_user,
    split_train_test,
    evaluate_recommendations
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py <ratings.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    print("\n" + "="*70)
    print("🎬 RECOMMENDATION SYSTEM DEMO")
    print("="*70)
    
    # 1. Load and validate data
    print("\n1️⃣  Loading and validating data...")
    df = load_ratings(csv_path, validate=True)
    
    # 2. Explore dataset
    print("\n2️⃣  Exploring dataset...")
    stats = explore_dataset(df, show_top_users=5, show_top_products=5)
    
    # 3. Get top-N recommendations
    print("\n3️⃣  Getting top 10 products (last 10000 days)...")
    top_products = get_top_n_products(df, days=10000, n=10)
    print("\n📈 Top 10 Products:")
    print(top_products.to_string(index=False))
    
    # 4. Get user-specific recommendations
    print("\n4️⃣  Generating personalized recommendations...")
    
    # Find a user with sufficient ratings
    user_counts = df.groupby('user_id').size()
    sample_user = user_counts[user_counts >= 10].index[0]
    
    print(f"\n👤 Recommendations for User {sample_user}:")
    try:
        recommendations = recommend_for_user(
            df, 
            user_id=sample_user, 
            n=5,
            min_similarity=0.1,
            use_normalized=True
        )
        print(recommendations.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")
    
    # 5. Evaluation (optional - takes time)
    print("\n5️⃣  Model Evaluation (optional)...")
    response = input("Run evaluation? This may take a few minutes. (y/n): ")
    
    if response.lower() == 'y':
        print("\nSplitting data into train/test...")
        train_df, test_df = split_train_test(df, test_size=0.2, min_ratings_per_user=10)
        
        print("\nEvaluating recommendations...")
        
        def recommend_wrapper(train_df, user_id, n):
            try:
                return recommend_for_user(train_df, user_id, n, min_similarity=0.1)
            except:
                return []
        
        eval_results = evaluate_recommendations(
            train_df, 
            test_df, 
            recommend_wrapper,
            k_values=[5, 10],
            relevance_threshold=4.0
        )
        
        print("\n✅ Evaluation complete!")
    
    print("\n" + "="*70)
    print("🎉 Demo completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
