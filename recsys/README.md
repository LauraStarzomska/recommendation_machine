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
- Extensible project structure  

---

## 📁 Project Structure

recsys/
│── cli.py
│── requirements.txt
│── README.md
│
└── recsys/
├── data_loader.py
├── recommend_top.py
├── recommend_user.py
└── utils.py


---

## ▶️ Usage

### **Install dependencies**

```bash
pip install -r requirements.txt
## Top N Products
python cli.py ratings.csv top --days 30 --n 10
