import operator

import numpy as np
import unittest
import pandas as pd

from src import datasets
from src.D_TREE import D_TREE
from src.D_TREE.D_TREE import D_TREE_model, D_TREE_test


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
        ig = D_TREE.information_gain(X, Y, D_TREE_test('Gender', 'Male', operator.eq))
        self.assertEqual(round(ig, 7), 0.0005507)

    def test_get_best_split(self):
        # Create data
        X_train, Y_train, X_test, Y_test = datasets.load_Obesity()
        X = pd.concat([X_train, X_test], axis=0)
        Y = pd.concat([Y_train, Y_test], axis=0)

        # Test cases
        best_test, best_IG = D_TREE.get_best_split(X, Y)
        self.assertEqual(round(best_IG, 7), 0.3824541)

    def test_make_split(self):
        # Create data
        X_train, Y_train, X_test, Y_test = datasets.load_Obesity()
        X = pd.concat([X_train, X_test], axis=0).head()  ####### HEAD
        Y = pd.concat([Y_train, Y_test], axis=0).head()

        D_TREE.make_split(X, Y, D_TREE_test('Weight', 90, operator.lt))

        # Test cases
        # TODO test does nothing

    def test_fit_D_Tree(self):
        X_train, Y_train, X_test, Y_test = datasets.load_Obesity()
        X = pd.concat([X_train, X_test], axis=0)
        Y = pd.concat([Y_train, Y_test], axis=0)

        model = D_TREE_model()
        model.fit(X, Y)
        # print(model.root.left.left.left.X)

        # TODO test growing tree somehow and test predictions made by tree

    def test_make_prediction(self):
        # Create data
        X_train, Y_train, X_test, Y_test = datasets.load_Obesity()
        X = pd.concat([X_train, X_test], axis=0).head(5)  ####### HEAD
        Y = pd.concat([Y_train, Y_test], axis=0).head(5)

        ind_X = X_test.iloc[20]
        ind_Y = Y_test.iloc[20]

        print(ind_X)
        print(ind_Y)

        print(X)
        print('=================================')

        print(D_TREE_test('Height', 100, operator.lt).test(ind_X))






