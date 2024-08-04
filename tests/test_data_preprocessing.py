import unittest
from src.data_preprocessing import load_data, preprocess_data


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.filepath = 'data/admission.csv'
        self.data = load_data(self.filepath)

    def test_load_data(self):
        self.assertIsNotNone(self.data)
        self.assertFalse(self.data.empty)

    def test_preprocess_data(self):
        processed_data = preprocess_data(self.data)
        self.assertIn('Admit_Chance', processed_data.columns)
        self.assertNotIn('Serial_No', processed_data.columns)


if __name__ == '__main__':
    unittest.main()
