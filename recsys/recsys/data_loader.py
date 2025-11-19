import pandas as pd
import os
from pathlib import Path


def load_ratings(path, min_rating=0.5, max_rating=5.0, validate=True):
    """
    Load and validate ratings data from CSV file.
    
    Parameters:
    -----------
    path : str
        Path to the CSV file containing ratings
    min_rating : float
        Minimum valid rating value (default: 0.5)
    max_rating : float
        Maximum valid rating value (default: 5.0)
    validate : bool
        Whether to perform data validation (default: True)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and validated ratings dataframe
    
    Raises:
    -------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If data validation fails
    """
    # Check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ratings file not found: {path}")
    
    # Load data
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Check required columns
    required_cols = ["user_id", "product_id", "rating", "timestamp"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with missing critical values
    initial_rows = len(df)
    df = df.dropna(subset=["user_id", "product_id", "rating"])
    dropped_na = initial_rows - len(df)
    if dropped_na > 0:
        print(f"⚠️  Removed {dropped_na} rows with missing values")
    
    # Type conversions with error handling
    try:
        df["user_id"] = df["user_id"].astype(int)
        df["product_id"] = df["product_id"].astype(int)
        df["rating"] = df["rating"].astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error converting data types: {e}")
    
    if validate:
        # Validate rating range
        invalid_ratings = df[(df["rating"] < min_rating) | (df["rating"] > max_rating)]
        if len(invalid_ratings) > 0:
            print(f"⚠️  Found {len(invalid_ratings)} ratings outside valid range [{min_rating}, {max_rating}]")
            df = df[(df["rating"] >= min_rating) & (df["rating"] <= max_rating)]
        
        # Check for negative IDs
        negative_users = df[df["user_id"] < 0]
        negative_products = df[df["product_id"] < 0]
        if len(negative_users) > 0 or len(negative_products) > 0:
            print(f"⚠️  Found {len(negative_users)} negative user IDs and {len(negative_products)} negative product IDs")
            df = df[(df["user_id"] >= 0) & (df["product_id"] >= 0)]
        
        # Check for duplicates
        duplicates = df.duplicated(subset=["user_id", "product_id"], keep="last")
        num_duplicates = duplicates.sum()
        if num_duplicates > 0:
            print(f"⚠️  Found {num_duplicates} duplicate user-product pairs, keeping most recent")
            df = df[~duplicates]
    
    # Convert timestamp
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    except Exception as e:
        print(f"⚠️  Warning: Could not parse timestamps: {e}")
        # If timestamp conversion fails, use current time as fallback
        df["timestamp"] = pd.Timestamp.now()
    
    print(f"✅ Loaded {len(df)} valid ratings")
    print(f"   Users: {df['user_id'].nunique()}, Products: {df['product_id'].nunique()}")
    
    return df
