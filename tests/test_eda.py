import unittest
import pandas as pd
from src.eda import plot_distributions, plot_correlations


class TestEDA(unittest.TestCase):

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

    def test_plot_distributions(self):
        try:
            plot_distributions(self.data)
        except Exception as e:
            self.fail(f'plot_distributions(): {e}')

    def test_plot_correlations(self):
        try:
            plot_correlations(self.data)
        except Exception as e:
            self.fail(f'plot_correlations(): {e}')


if __name__ == '__main__':
    unittest.main()
