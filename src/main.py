import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from src.D_TREE.D_TREE import D_TREE_model
from src.KNN.KNN import KNN_model
import src.datasets.data_loader as data_loader


def compute_metrics(model, X_test, Y_test):
    # Convert values to string to fix potential type mismatches
    Y_test = [str(value) for value in list(Y_test.values)]
    predictions = [str(value) for value in model.predict(X_test)]

    labels = np.unique(Y_test)
    cm = confusion_matrix(list(Y_test), list(predictions), labels=labels)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    accuracy = accuracy_score(Y_test, predictions)

    return accuracy, cm


def KNN(X_train, Y_train, X_test, Y_test, k=5, distance_type='euclidean', normalization=False):
    model = KNN_model(k, distance_type=distance_type, normalization=normalization)
    model.fit(X_train, Y_train)
    accuracy, cm = compute_metrics(model, X_test, Y_test)

    # log KNN results
    print("k-nearest neighbors:")
    print('-------------------')
    print('k:', k)
    print('distance_type:', distance_type)
    print('Min-max normalization:', normalization)
    print('-------------------')
    print('accuracy:', accuracy)
    print('confusion_matrix:')
    print(cm)
    print('-------------------')


def DECISION_TREE(X_train, Y_train, X_test, Y_test):
    model = D_TREE_model()
    model.fit(X_train, Y_train)
    accuracy, cm = compute_metrics(model, X_test, Y_test)

    # log decision tree results
    print("decision tree:")
    print('-------------------')
    print('accuracy:', accuracy)
    print('confusion_matrix:')
    print(cm)
    print('-------------------')


def main():
    X_train, Y_train, X_test, Y_test = data_loader.load_Obesity()  # load dataset iris and split into train and test set
    print('-------------------')
    KNN(X_train, Y_train, X_test, Y_test, k=3, distance_type='chebyshev', normalization=True)
    DECISION_TREE(X_train, Y_train, X_test, Y_test)  # apply decision tree learner


if __name__ == "__main__":
    main()
