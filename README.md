# Recommendation Engine

A production-ready recommendation engine with comprehensive data validation, normalization, and evaluation metrics.

## 🚀 Features

### Core Functionality
- **Top-N recommendations** - Best-rated products within a time window
- **User-based collaborative filtering** - Personalized recommendations using item-based cosine similarity
- **Cold start handling** - Fallback to popular items for new users

### Data Quality & Safety
- **Robust data validation** - Rating range checks, duplicate detection, missing value handling
- **Rating normalization** - Per-user normalization to handle rating bias (mean-centering, z-score, min-max)
- **Type safety** - Automatic type conversion and validation

### Evaluation & Analysis
- **Comprehensive metrics** - RMSE, MAE, Precision@K, Recall@K, F1@K
- **Data exploration** - Sparsity analysis, user/item statistics, rating distribution
- **Train/test splitting** - Proper evaluation with temporal awareness

### Production Ready
- Modern Python packaging (pyproject.toml)
- CLI interface
- Unit tests
- Demo script

---

## 📁 Project Structure

```
recsys/
├── pyproject.toml              # Modern Python packaging
├── setup.cfg                   # Additional metadata
├── README.md
├── requirements.txt
│
├── recsys/
│   ├── __init__.py            # Package exports
│   ├── cli.py                 # Command-line interface
│   ├── data_loader.py         # Data loading with validation
│   ├── recommended_top.py     # Top-N recommendations
│   ├── recommend_user.py      # User collaborative filtering
│   ├── utils.py               # Rating normalization, statistics
│   ├── data_exploration.py    # Dataset analysis tools
│   ├── evaluation.py          # Metrics (RMSE, Precision@K, etc.)
│   └── demo.py                # Demo script
│
└── tests/
    ├── __init__.py
    ├── test_top.py            # Tests for top-N recommendations
    └── test_user.py           # Tests for user recommendations
```

---

## 📦 Installation

### Option 1: Install from source (recommended for development)

```bash
cd /path/to/recommendation_machine/recsys
pip install -e .
```

### Option 2: Install dependencies only

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### **Quick Start with Demo**

```bash
cd recsys
python demo.py ratings.csv
```

### **Top N Products**

```bash
# Using the module
python -m recsys.cli ratings.csv top --days 10000 --n 10

# Or if installed
recsys ratings.csv top --days 10000 --n 10
```

### **User Recommendations**

```bash
# Using the module
python -m recsys.cli ratings.csv user --user_id 42 --n 5

# Or if installed
recsys ratings.csv user --user_id 42 --n 5
```

---

## 🧪 Running Tests

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test file
python -m unittest tests.test_top -v
```

---

## 🛠️ Development

To work on this project:

1. Create a virtual environment:
```bash
python3 -m venv env
source env/bin/activate  # On macOS/Linux
```

2. Install in editable mode:
```bash
pip install -e .
```

3. Make your changes and run tests:
```bash
python -m unittest discover tests -v
```

---

## 📝 API Usage

### Basic Usage

```python
from recsys import load_ratings, get_top_n_products, recommend_for_user

# Load data with validation
df = load_ratings('ratings.csv', validate=True)

# Get top products
top_products = get_top_n_products(df, days=30, n=10)
print(top_products)

# Get user recommendations (with improvements)
recommendations = recommend_for_user(
    df, 
    user_id=42, 
    n=5,
    min_similarity=0.1,        # Filter out weak similarities
    use_normalized=True,        # Use rating normalization
    normalization_method='mean_center'
)
print(recommendations)
```

### Data Exploration

```python
from recsys import explore_dataset, analyze_user, analyze_product

# Get comprehensive dataset statistics
stats = explore_dataset(df, show_top_users=10, show_top_products=10)

# Analyze specific user
analyze_user(df, user_id=123)

# Analyze specific product
analyze_product(df, product_id=456)
```

### Model Evaluation

```python
from recsys import split_train_test, evaluate_recommendations

# Split data for evaluation
train_df, test_df = split_train_test(df, test_size=0.2)

# Evaluate recommendation quality
def recommend_wrapper(train_df, user_id, n):
    return recommend_for_user(train_df, user_id, n, min_similarity=0.1)

metrics = evaluate_recommendations(
    train_df, 
    test_df, 
    recommend_wrapper,
    k_values=[5, 10, 20],
    relevance_threshold=4.0
)

print(f"Precision@10: {metrics[10]['precision']:.4f}")
print(f"Recall@10: {metrics[10]['recall']:.4f}")
```

### Rating Normalization

```python
from recsys import normalize_ratings_per_user, get_user_statistics

# Normalize ratings to handle user bias
df_normalized = normalize_ratings_per_user(df, method='mean_center')

# Get per-user statistics
user_stats = get_user_statistics(df)
print(user_stats.head())
```

---

## 📊 Data Format

The input CSV should have the following columns:
- `user_id` (int): User identifier (must be >= 0)
- `product_id` (int): Product identifier (must be >= 0)
- `rating` (float): Rating value (default range: 0.5-5.0)
- `timestamp` (int): Unix timestamp

### Data Validation Features

The `load_ratings()` function automatically:
- ✅ Removes rows with missing critical values
- ✅ Validates rating ranges (configurable)
- ✅ Detects and removes duplicates (keeps most recent)
- ✅ Checks for negative IDs
- ✅ Converts timestamps from Unix format
- ✅ Reports all data quality issues

---

## 🔬 Key Improvements

### 1. Rating Normalization
Different users have different rating scales. Normalization handles this:
- **Mean-centered**: Removes user bias by subtracting mean rating
- **Z-score**: Standardizes ratings to mean=0, std=1
- **Min-max**: Scales ratings to 0-1 range per user

### 2. Improved Collaborative Filtering
- **Similarity threshold**: Filters out weak item-item similarities
- **Weighted scoring**: Normalizes by sum of similarities
- **Cold start handling**: Falls back to popular items for new users
- **Estimated ratings**: Provides predicted rating values

### 3. Comprehensive Evaluation
- **Precision@K**: What % of recommendations are relevant?
- **Recall@K**: What % of relevant items were recommended?
- **F1@K**: Harmonic mean of precision and recall
- **RMSE/MAE**: Rating prediction accuracy

### 4. Dataset Analysis
- **Sparsity metrics**: Understand data density
- **User/item statistics**: Rating patterns and distributions
- **Cold start identification**: Find users/items with few ratings
- **Rating bias detection**: Identify lenient vs. strict raters

---

## 🤝 Contributing

1. Make sure tests pass before submitting changes
2. Add tests for new features
3. Follow PEP 8 style guidelines
