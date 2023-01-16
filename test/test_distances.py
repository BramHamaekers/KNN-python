import unittest
from src.KNN import distances


class test_distances(unittest.TestCase):
    def test_euclidean(self):
        self.assertEqual(round(distances.euclidean([1, 2], [4, 7]), 3), 5.831)
        self.assertEqual(round(distances.euclidean([30, 40], [23, 39]), 3), 7.071)

        x = [1, 51.7, 20.3, 194.0, 3775.0, 0, 1, 1, 0]
        y = [1, 47.5, 16.8, 199.0, 3900.0, 1, 0, 0, 1]
        self.assertEqual(round(distances.euclidean(x, y), 3), 125.235)

    def test_manhattan(self):
        self.assertEqual(distances.manhattan([0, 0], [1, 1]), 2)
        self.assertEqual(distances.manhattan([0, 0], [-1, -1]), 2)

        x, y = [1, 2, 3, 4, 5, 6], [10, 20, 30, 1, 2, 3]
        self.assertEqual(distances.manhattan(x, y), 63)

        x = [1, 51.7, 20.3, 194.0, 3775.0, 0, 1, 1, 0]
        y = [1, 47.5, 16.8, 199.0, 3900.0, 1, 0, 0, 1]
        self.assertEqual(round(distances.manhattan(x, y), 3), 141.7)

    def test_chebyshev(self):
        self.assertEqual(distances.chebyshev([1, 0, 0], [0, 1, 0]), 1)

        x = [1.5, 2.7, 4, 1, 1.4, 7]
        y = [2.7, 35, 0, -1, 2, 4]
        self.assertEqual(distances.chebyshev(x, y), 32.30)
