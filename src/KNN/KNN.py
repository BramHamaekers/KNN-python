import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.KNN import distances


class KNN_model:
    def __init__(self, k, distance_type='euclidean', normalization=False):
        self.k = k
        self.distance_type = distance_type
        self.normalization = normalization
        self.scaler = MinMaxScaler()
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = self.preprocess(x_train)
        self.y_train = y_train

    def preprocess(self, dataset, is_test_data=False):
        dataset = pd.get_dummies(dataset)  # applies OneHotEncoding
        if not self.normalization: return dataset

        # Min-max normalization
        if is_test_data: dataset[dataset.columns] = self.scaler.transform(dataset[dataset.columns])
        else: dataset[dataset.columns] = self.scaler.fit_transform(dataset[dataset.columns])
        return dataset

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
        d_list.sort(key=lambda x: x[1])  # x[1] = distance -> so sorted on smallest distance

        return list(zip(*d_list))[0][:self.k]

    def predict(self, test_set):
        test_set = self.preprocess(test_set, is_test_data=True)
        predictions = []
        for test_sample in test_set.values:
            neighbors = self.KNN(test_sample)
            labels = [sample for sample in neighbors]
            prediction = max(labels, key=labels.count)
            predictions.append(prediction)
        return predictions
