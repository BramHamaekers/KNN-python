import unittest
import pandas as pd
from src.D_TREE import D_TREE


class test_D_TREE(unittest.TestCase):
    def test_entropy(self):
        # Create Data
        case_1 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        case_2 = pd.Series([1, 1, 1, 1, 1, 0])
        case_3 = pd.Series([1, 1, 1, 1])
        case_4 = pd.Series([1, 1, 0, 0])

        # Test cases
        self.assertEqual(round(D_TREE.entropy(case_1), 3), 0.989)
        self.assertEqual(round(D_TREE.entropy(case_2), 3), 0.650)
        self.assertEqual(D_TREE.entropy(case_3), 0)
        self.assertEqual(D_TREE.entropy(case_4), 1)



