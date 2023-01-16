import unittest

import pandas as pd
from matplotlib import pyplot as plt

from src.KNN.KNN import KNN_model

#  https://medium.com/@rangavamsi5/k-nearest-neighbors-algorithm-in-python-64f38792193


class KNN_tests(unittest.TestCase):
    def test_KNN_model_init(self):
        model = KNN_model(5)
        self.assertEqual(model.k, 5)
        self.assertEqual(model.distance_type, 'euclidean')
        self.assertEqual(model.normalization, False)

    def test_KNN(self):

        # Create data
        x = [12, 15, 21, 23, 59, 29, 73, 56, 74]
        y = [0, 3, 27, 39, 41, 56, 74, 62, 51]
        target = ['small', 'small', 'small', 'medium', 'medium', 'medium', 'large', 'large', 'large']
        dataset = pd.DataFrame(list(zip(x, y, target)),
                               columns=['x', 'y', 'target'])
        X_train = dataset[['x', 'y']]
        Y_train = dataset.target

        # Create model
        model = KNN_model(5)
        model.fit(X_train, Y_train)

        # Test target of nearest neighbours
        self.assertEqual(model.KNN([30, 40]), ('medium', 'small', 'medium', 'medium', 'large'))

    def test_predict(self):
        # Create data
        x = [12, 15, 21, 23, 59, 29, 73, 56, 74]
        y = [0, 3, 27, 39, 41, 56, 74, 62, 51]
        target = ['small', 'small', 'small', 'medium', 'medium', 'medium', 'large', 'large', 'large']
        dataset = pd.DataFrame(list(zip(x, y, target)), columns=['x', 'y', 'target'])
        X_train = dataset[['x', 'y']]
        Y_train = dataset.target

        # Create model
        model = KNN_model(5, normalization=True)
        model.fit(X_train, Y_train)

        # Test target of nearest neighbours
        test_set = pd.DataFrame((list(zip([30, 10, 20], [40, 20, 70]))), columns=['x', 'y'])
        self.assertEqual(model.predict(test_set), ['medium', 'small', 'medium'])

    # TODO: Test preprocessing