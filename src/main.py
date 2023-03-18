import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from src.D_TREE.D_TREE import D_TREE_model
from src.KNN.KNN import KNN_model
import src.datasets.data_loader as data_loader


def KNN(X_train, Y_train, X_test, Y_test, k=5, distance_type='euclidean', normalization=False):
    model = KNN_model(k, distance_type=distance_type, normalization=normalization)
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
    print('Min-max normalization:', normalization)
    print('-------------------')
    print('accuracy:', accuracy)
    print('confusion_matrix:')
    print(cm)
    print('-------------------')


def DECISION_TREE(X_train, Y_train, X_test, Y_test):
    model = D_TREE_model()
    model.fit(X_train, Y_train)


def main():
    X_train, Y_train, X_test, Y_test = data_loader.load_BMI()  # load dataset iris and split into train and test set

    KNN(X_train, Y_train, X_test, Y_test, k=3, distance_type='chebyshev', normalization=True)  # apply k nearest neighbour
    DECISION_TREE(X_train, Y_train, X_test, Y_test)  # apply decision tree learner


if __name__ == "__main__":
    main()
