import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from src.KNN.KNN import KNN_model
import src.datasets as datasets


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
    X_train, Y_train, X_test, Y_test = datasets.load_iris()  # load dataset iris and split into train and test set
    KNN(X_train, Y_train, X_test, Y_test, k=3, distance_type='manhattan')  # apply nearest neighbour


if __name__ == "__main__":
    main()
