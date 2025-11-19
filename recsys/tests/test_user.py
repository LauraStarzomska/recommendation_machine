import unittest
import pandas as pd
from datetime import datetime
from recsys.recommend_user import recommend_for_user


class TestUserRecommendations(unittest.TestCase):
    
    def setUp(self):
        """Create sample data for testing"""
        self.df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
            'product_id': [101, 102, 101, 103, 102, 103, 101, 104],
            'rating': [5.0, 4.0, 5.0, 3.0, 4.5, 4.0, 5.0, 2.0],
            'timestamp': [datetime.now()] * 8
        })
    
    def test_recommend_for_user(self):
        """Test that recommendations are generated for a user"""
        result = recommend_for_user(self.df, user_id=1, n=2)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('product_id' in result.columns)
    
    def test_no_duplicate_recommendations(self):
        """Test that a user doesn't get products they already rated"""
        result = recommend_for_user(self.df, user_id=1, n=5)
        
        user_1_products = self.df[self.df['user_id'] == 1]['product_id'].values
        recommended_products = result['product_id'].values
        
        overlap = set(user_1_products) & set(recommended_products)
        self.assertEqual(len(overlap), 0, "User should not get products they already rated")


if __name__ == '__main__':
    unittest.main()
