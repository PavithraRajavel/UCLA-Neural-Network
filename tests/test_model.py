import unittest
import pandas as pd
from src.model import split_data, train_model, evaluate_model


class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'GRE_Score': [320, 300, 340],
            'TOEFL_Score': [110, 100, 120],
            'University_Rating': [5, 3, 4],
            'SOP': [4.5, 3.5, 5],
            'LOR': [4.5, 3.0, 5.0],
            'CGPA': [9.65, 8.56, 9.8],
            'Research': [1, 0, 1],
            'Admit_Chance': [1, 0, 1]
        })
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.data, 'Admit_Chance')

    def test_split_data(self):
        self.assertEqual(len(self.X_train), 2)
        self.assertEqual(len(self.X_test), 1)
        self.assertIn('GRE_Score', self.X_train.columns)
        self.assertNotIn('Admit_Chance', self.X_train.columns)

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        model = train_model(self.X_train, self.y_train)
        accuracy = evaluate_model(model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0)


if __name__ == '__main__':
    unittest.main()
