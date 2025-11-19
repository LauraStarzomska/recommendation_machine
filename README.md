# Recommendation Engine

A simple but complete recommendation engine supporting:

1. **Top-N best-rated products** within a time window  
2. **User-specific recommendations** using item-based collaborative filtering  

---

## 🚀 Features

- Data cleaning and validation  
- Time-window filtering  
- Cosine similarity recommendation model  
- CLI interface  
- Modern Python packaging with pyproject.toml
- Unit tests

---

## 📁 Project Structure

```
recsys/
├── pyproject.toml         # Modern Python packaging
├── setup.cfg              # Additional metadata
├── README.md
├── requirements.txt
│
├── recsys/
│   ├── __init__.py
│   ├── cli.py             # Command-line interface
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── recommended_top.py # Top-N recommendations
│   ├── recommend_user.py  # User-specific recommendations
│   └── utils.py           # Utility functions
│
└── tests/
    ├── __init__.py
    ├── test_top.py        # Tests for top-N recommendations
    └── test_user.py       # Tests for user recommendations
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

You can also use the recommendation engine programmatically:

```python
from recsys import load_ratings, get_top_n_products, recommend_for_user

# Load data
df = load_ratings('ratings.csv')

# Get top products
top_products = get_top_n_products(df, days=30, n=10)
print(top_products)

# Get user recommendations
recommendations = recommend_for_user(df, user_id=42, n=5)
print(recommendations)
```

---

## 📊 Data Format

The input CSV should have the following columns:
- `user_id` (int): User identifier
- `product_id` (int): Product identifier  
- `rating` (float): Rating value
- `timestamp` (int): Unix timestamp

---

## 🤝 Contributing

1. Make sure tests pass before submitting changes
2. Add tests for new features
3. Follow PEP 8 style guidelines
