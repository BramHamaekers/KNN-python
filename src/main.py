import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from src.KNN.KNN import KNN_model


def load_iris():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    dataset = pd.read_csv(url, names=attributes)
    dataset.columns = attributes

    return dataset


def split_dataset(dataset):
    train, test = train_test_split(dataset, test_size=0.4, stratify=dataset['species'], random_state=55)

    X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    Y_train = train.species
    X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    Y_test = test.species

    return X_train, Y_train, X_test, Y_test


def KNN(X_train, Y_train, X_test, Y_test, k=5, distance_type='euclidean'):
    model = KNN_model(k, distance_type=distance_type)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    labels = np.unique(Y_test)
    cm = confusion_matrix(Y_test, predictions, labels=labels)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    accuracy = accuracy_score(Y_test, predictions)

    # log KNN results
    print('-------------------')
    print("k-nearest neighbors:")
    print('-------------------')
    print('k:', k)
    print('distance_type:', distance_type)
    print('-------------------')
    print('accuracy:', accuracy)
    print('confusion_matrix:')
    print(cm)

    print('-------------------')


def main():
    dataset = load_iris()  # load dataset iris
    X_train, Y_train, X_test, Y_test = split_dataset(dataset)  # split dataset into train and test set
    KNN(X_train, Y_train, X_test, Y_test)  # apply nearest neighbour


if __name__ == "__main__":
    main()
