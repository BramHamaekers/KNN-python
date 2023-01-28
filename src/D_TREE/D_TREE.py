from math import log2

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype


def variance(Y_train):
    return Y_train.var()


def entropy(Y_train):
    p_list = Y_train.value_counts() / Y_train.shape[0]
    return -sum(p * log2(p) for p in p_list)


def information_gain(X_train, Y_train, test):
    # test must be binary test
    # si = |S_i| / |S|
    s1 = sum(test)/X_train.shape[0]
    s2 = 1 - s1

    # IG_Regression(S,t) = Var(S) - sum((|S_i| / |S|)*Var(S_i)
    # IG_Classification(S,t) = CE(S) - sum((|S_i| / |S|)*CE(S_i)
    if s1 == 0 or s2 == 0: return 0
    if is_numeric_dtype(Y_train.dtypes): variance(Y_train) - (s1 * variance(Y_train[test])) - (s2 * variance(Y_train[-test]))
    return entropy(Y_train) - (s1 * entropy(Y_train[test])) - (s2 * entropy(Y_train[-test]))


def get_best_split(X_train, Y_train):
    values_dict = X_train.apply(lambda column: column.unique()).to_dict()
    best_IG, best_test = 0, None
    for col in values_dict:
        for value in values_dict[col]:
            if is_numeric_dtype(values_dict[col].dtype): test = X_train[col] < value
            else: test = X_train[col] == value
            IG = information_gain(X_train, Y_train, test)
            if IG > best_IG:
                best_IG = IG
                best_test = test
    return best_test, best_IG


def make_split(X_train, Y_train, test):
    X_1 = X_train[test]
    Y_1 = Y_train[test]

    X_2 = X_train[-test]
    Y_2 = Y_train[-test]
    return X_1, Y_1, X_2, Y_2


def make_prediction(X, Y):
    # IDFK do something i guess TODO
    return 0


class D_TREE_node:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.test = None
        self.left = None
        self.right = None


class D_TREE_model:
    def __init__(self, min_information_gain = 1e-20):
        self.min_IG = min_information_gain
        self.root = None

    def fit(self, X_train, Y_train):
        self.root = D_TREE_node(X_train, Y_train)
        self.grow_tree(self.root)

    def grow_tree(self, node):
        X, Y = node.X, node.Y

        # Stop splitting if no more splits can be made
        if X.shape[0] < 2:
            return

        # Do split if information gain of split is high enough
        test, IG = get_best_split(X, Y)
        if IG > self.min_IG:
            X_1, Y_1, X_2, Y_2 = make_split(X, Y, test)
            node.test = test
            node.left = D_TREE_node(X_1, Y_1)
            node.right = D_TREE_node(X_2, Y_2)

            # Recursively grow tree
            self.grow_tree(node.left)
            self.grow_tree(node.right)



        #TODO: learning algorithm
        return 0