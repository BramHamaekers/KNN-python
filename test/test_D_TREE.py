import numpy as np
import unittest
import pandas as pd

from src import datasets
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

    def test_variance(self):
        None
        # TODO

    def test_information_gain(self):
        # Create data
        X_train, Y_train, X_test, Y_test = datasets.load_Obesity()
        X = pd.concat([X_train, X_test], axis=0)
        Y = pd.concat([Y_train, Y_test], axis=0)

        # Test cases
        ig = D_TREE.information_gain(X, Y, X['Gender'] == 'Male')
        self.assertEqual(round(ig, 7), 0.0005507)



