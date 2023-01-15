from src.KNN import distances


class KNN_model:
    def __init__(self, k, distance_type='euclidean'):
        self.k = k
        self.distance_type = distance_type
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def distance(self, x, y):
        match self.distance_type:
            case 'euclidean': return distances.euclidean(x, y)
            case 'manhattan': return distances.manhattan(x, y)
            case 'chebyshev': return distances.chebyshev(x, y)
        return 0

    def KNN(self, sample):
        d_list = []
        for idx, instance in enumerate(self.x_train.values):
            d_list.append((self.y_train.values[idx], self.distance(sample, instance), instance))
        d_list.sort(key=lambda x: x[1])

        return list(zip(*d_list))[0][:self.k]

    def predict(self, test_set):
        predictions = []
        for test_sample in test_set.values:
            neighbors = self.KNN(test_sample)
            labels = [sample for sample in neighbors]
            prediction = max(labels, key=labels.count)
            predictions.append(prediction)
        return predictions
