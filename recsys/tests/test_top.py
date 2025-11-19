import unittest
import pandas as pd
from datetime import datetime
from recsys.recommended_top import get_top_n_products


class TestTopRecommendations(unittest.TestCase):
    
    def setUp(self):
        """Create sample data for testing"""
        self.df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
            'product_id': [101, 102, 101, 103, 102, 103, 101, 104],
            'rating': [5.0, 4.0, 5.0, 3.0, 4.5, 4.0, 5.0, 2.0],
            'timestamp': [datetime.now()] * 8
        })
    
    def test_get_top_n_products(self):
        """Test that top N products are returned"""
        result = get_top_n_products(self.df, days=30, n=3)
        
        self.assertEqual(len(result), 3)
        self.assertTrue('product_id' in result.columns)
        self.assertTrue('avg_rating' in result.columns)
        self.assertTrue('rating_count' in result.columns)
    
    def test_top_product_has_highest_rating(self):
        """Test that the top product has the highest average rating"""
        result = get_top_n_products(self.df, days=30, n=5)
        
        top_product = result.iloc[0]
        self.assertEqual(top_product['product_id'], 101)
        self.assertEqual(top_product['avg_rating'], 5.0)


if __name__ == '__main__':
    unittest.main()
